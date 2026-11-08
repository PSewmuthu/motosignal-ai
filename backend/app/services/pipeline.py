from __future__ import annotations

from pathlib import Path
import uuid

import cv2
import numpy as np

from app.core.config import settings
from app.services.association.vehicle_matcher import match_rear_track
from app.services.detectors import HybridPlateOCR, UltralyticsDetector
from app.services.yolo_tracking import YOLOTrackingConfig, UltralyticsTrackerFactory
from app.services.events.classifier import classify_event
from app.services.plates.consensus import multi_frame_consensus
from app.services.plates.image_quality import crop_box, quality_metrics
from app.services.traffic.crossing_detector import StopLineCrossingDetector
from app.services.traffic.signal_verifier import SignalVerifier
from app.services.tracking import IoUTracker


def _appearance(frame, bbox):
    crop = crop_box(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8],
                        [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)

    return (hist / norm).astype(float).tolist() if norm else None


def _mean_vectors(vectors):
    valid = [np.asarray(v, dtype=float) for v in vectors if v is not None]
    if not valid:
        return None

    mean = np.mean(valid, axis=0)
    norm = np.linalg.norm(mean)

    return (mean / norm).tolist() if norm else None


def _save_frame(frame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _detect_models(diagnostics: dict):
    motorcycle = plate = ocr = None
    if settings.motorcycle_model:
        try:
            tracker_yaml = Path(settings.tracker_config).expanduser()

            if not tracker_yaml.is_absolute():
                app_relative = Path(
                    __file__).resolve().parents[1] / tracker_yaml
                repo_relative = Path(
                    __file__).resolve().parents[3] / tracker_yaml
                tracker_yaml = app_relative if app_relative.is_file() else repo_relative

            motorcycle = UltralyticsTrackerFactory(
                YOLOTrackingConfig(
                    weights=settings.motorcycle_model,
                    class_ids=(settings.motorcycle_class_id,),
                    tracker_yaml=str(tracker_yaml),
                    confidence=settings.motorcycle_confidence,
                    iou=settings.motorcycle_nms_iou,
                    image_size=settings.motorcycle_image_size,
                    device=settings.yolo_device,
                )
            )
        except Exception as exc:
            diagnostics["motorcycle_detector"] = f"unavailable: {exc}"
    else:
        diagnostics["motorcycle_detector"] = "not_configured"

    if settings.plate_model:
        try:
            plate = UltralyticsDetector(
                settings.plate_model, class_ids=None, confidence=0.20)
        except Exception as exc:
            diagnostics["plate_detector"] = f"unavailable: {exc}"
    else:
        diagnostics["plate_detector"] = "not_configured"

    if settings.enable_ocr:
        try:
            ocr = HybridPlateOCR()
        except Exception as exc:
            diagnostics["ocr"] = f"unavailable: {exc}"
    else:
        diagnostics["ocr"] = "disabled"

    diagnostics.setdefault("motorcycle_detector",
                           "ready" if motorcycle else "not_configured")
    diagnostics.setdefault(
        "plate_detector", "ready" if plate else "not_configured")
    diagnostics.setdefault(
        "ocr", {"status": "ready", "engines": ocr.available_engines} if ocr else "disabled")

    return motorcycle, plate, ocr


def _assign_plate_to_track(plate_box, active_tracks):
    px = (plate_box[0] + plate_box[2]) / 2
    py = (plate_box[1] + plate_box[3]) / 2
    best, best_score = None, 0.0

    for track in active_tracks:
        x1, y1, x2, y2 = track.bbox
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        inside = x1 - 0.08 * width <= px <= x2 + 0.08 * \
            width and y1 - 0.08 * height <= py <= y2 + 0.08 * height

        if inside:
            score = 1.0 - min(1.0, abs(px - (x1 + x2) / 2) / width)
            if score > best_score:
                best, best_score = track, score

    return best


def _scan_video(video_path: str, role: str, job_dir: Path, config: dict, models, diagnostics: dict, progress_callback=None, progress_start: float = 0.05, progress_span: float = 0.4):
    motorcycle_detector, plate_detector, ocr_engine = models
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
        diagnostics.setdefault("warnings", []).append(
            f"{role}: invalid FPS metadata; used 25 FPS fallback")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(config.get("frame_stride") or settings.frame_stride))

    # Each camera/video stream receives its own persistent YOLO tracker state.
    # The deterministic IoU tracker remains only as a lightweight test/fallback adapter.
    yolo_tracker = motorcycle_detector.create() if motorcycle_detector and hasattr(
        motorcycle_detector, "create") else None
    fallback_tracker = IoUTracker() if motorcycle_detector and yolo_tracker is None else None
    line = config.get("stop_line", [])
    crossing_detector = StopLineCrossingDetector(line, config.get("approach_side", "either"),
                                                 float(config.get("max_track_gap_seconds", 1.5))) if role in ("traffic", "combined") and len(line) == 2 else None
    signal_verifier = SignalVerifier(config.get(
        "signal_timeline", []), config.get("direction", "all"))
    tracks: dict[str, dict] = {}
    crossings = []
    frame_index = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if progress_callback and total_frames > 0:
            step = max(stride, total_frames // 40)
            if frame_index % step == 0:
                fraction = min(1.0, frame_index / total_frames)
                progress_callback(progress_start + progress_span * fraction,
                                  f"{role.capitalize()} stream: frame {frame_index}/{total_frames}")

        if frame_index % stride == 0:
            processed_frames += 1
            timestamp = frame_index / fps
            if yolo_tracker is not None:
                active_tracks = yolo_tracker.track(frame)
            elif fallback_tracker is not None:
                motorcycle_detections = motorcycle_detector.detect(frame)
                active_tracks = fallback_tracker.update(motorcycle_detections)
            else:
                active_tracks = []

            for track in active_tracks:
                item = tracks.setdefault(track.track_id, {"track_id": track.track_id, "timestamps": [
                ], "appearances": [], "plate_predictions": [], "plate_evidence": [], "frame_evidence": []})
                item["timestamps"].append(timestamp)
                appearance = _appearance(frame, track.bbox)

                if appearance is not None:
                    item["appearances"].append(appearance)

                if crossing_detector:
                    normalized_bbox = [track.bbox[0] / max(width, 1), track.bbox[1] / max(height, 1),
                                       track.bbox[2] / max(width, 1), track.bbox[3] / max(height, 1)]
                    crossing = crossing_detector.update(
                        track.track_id, normalized_bbox, timestamp)

                    if crossing and track.hits >= int(config.get("min_track_observations", 3)):
                        filename = f"traffic_{track.track_id}_{crossing.timestamp_seconds:.3f}".replace(
                            ".", "_") + ".jpg"
                        _save_frame(frame, job_dir / "evidence" / filename)
                        signal = signal_verifier.at(crossing.timestamp_seconds)

                        crossings.append({
                            "timestamp_seconds": round(crossing.timestamp_seconds, 4),
                            "track_id": track.track_id,
                            "direction": crossing.direction,
                            "previous_signed_distance": crossing.previous_signed_distance,
                            "current_signed_distance": crossing.current_signed_distance,
                            "signal": {"color": signal.color, "status": signal.status, "reason": signal.reason,
                                       "phase_start_seconds": signal.phase_start_seconds,
                                       "phase_end_seconds": signal.phase_end_seconds,
                                       "direction": signal.direction},
                            "frame_file": filename,
                            "appearance": _mean_vectors(item["appearances"])
                        })

            # Plate detection is performed directly on the supplied rear-view frame, never on user-cropped inputs.
            if plate_detector and role in ("rear", "combined"):
                plate_detections = plate_detector.detect(frame)

                for plate_detection in plate_detections:
                    crop = crop_box(frame, plate_detection.bbox)

                    if crop is None or crop.size == 0:
                        continue

                    assigned = _assign_plate_to_track(
                        plate_detection.bbox, active_tracks)
                    track_id = assigned.track_id if assigned else None
                    item = tracks.get(track_id) if track_id else None
                    evidence_suffix = uuid.uuid4().hex[:6]
                    crop_name = f"plate_{track_id or 'unassigned'}_{frame_index}_{evidence_suffix}.jpg"
                    frame_name = f"rear_{track_id or 'unassigned'}_{frame_index}_{evidence_suffix}.jpg"
                    _save_frame(crop, job_dir / "evidence" / crop_name)
                    _save_frame(frame, job_dir / "evidence" / frame_name)
                    prediction_items = ocr_engine.recognize(
                        crop) if ocr_engine else []

                    if item is not None:
                        if prediction_items:
                            # The hybrid reader returns one crop-level candidate after comparing
                            # pretrained engines and preprocessing variants.
                            best_prediction = max(prediction_items, key=lambda prediction: float(
                                prediction.get("confidence", 0.0)))
                            item["plate_predictions"].append(
                                {**best_prediction, "frame_index": frame_index})

                        item["plate_evidence"].append({"file": crop_name, "frame_file": frame_name,
                                                       "timestamp_seconds": timestamp,
                                                       "detector_confidence": plate_detection.confidence,
                                                       "quality": quality_metrics(crop), "ocr": prediction_items})

        frame_index += 1
    cap.release()

    if yolo_tracker is not None:
        diagnostics.setdefault("tracking", {})[role] = {
            "backend": "ultralytics_yolo_track",
            "tracker": motorcycle_detector.tracker_type,
            "tracker_config": Path(motorcycle_detector.config.tracker_yaml).name,
            "tracking_frames": yolo_tracker.frame_index,
            "unique_track_ids": len(yolo_tracker.hits_by_id),
            "reid_enabled": motorcycle_detector.reid_enabled,
            "detector_confidence_floor": motorcycle_detector.config.confidence,
            "nms_iou": motorcycle_detector.config.iou,
            "image_size": motorcycle_detector.config.image_size,
            "frame_stride": stride,
        }
    elif motorcycle_detector is not None:
        diagnostics.setdefault("tracking", {})[role] = {
            "backend": "iou_fallback",
            "warning": "YOLO built-in tracking factory unavailable; fallback is intended for tests only",
            "frame_stride": stride,
        }

    if progress_callback:
        progress_callback(progress_start + progress_span,
                          f"{role.capitalize()} stream scan complete")

    for item in tracks.values():
        item["appearance"] = _mean_vectors(item["appearances"])
        consensus = multi_frame_consensus(item["plate_predictions"])
        item["plate_consensus"] = {"text": consensus.text, "confidence": consensus.confidence,
                                   "status": consensus.status, "observations": consensus.observations,
                                   "conflicts": consensus.conflicts}
        item.pop("appearances", None)

    return {"role": role, "fps": fps, "total_frames": total_frames, "processed_frames": processed_frames,
            "width": width, "height": height, "tracks": list(tracks.values()), "crossings": crossings}


def run_video_analysis(traffic_video: str, rear_video: str | None, job_id: str, config: dict,
                       progress_callback=None) -> dict:
    job_dir = settings.data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict = {"notes": [
        "Automated outputs are decision-support evidence, not a legal determination."]}
    models = _detect_models(diagnostics)

    if config.get("mode", "same_camera") == "same_camera":
        scan = _scan_video(traffic_video, "combined", job_dir, config, models, diagnostics,
                           progress_callback, 0.05, 0.85)
        traffic_scan, rear_scan = scan, scan
        association_mode = "same_camera"
    else:
        traffic_scan = _scan_video(traffic_video, "traffic", job_dir, config, models, diagnostics,
                                   progress_callback, 0.05, 0.40)
        rear_scan = _scan_video(rear_video, "rear", job_dir, config, models, diagnostics,
                                progress_callback, 0.45, 0.40) if rear_video else {"tracks": []}
        association_mode = "separate_cameras"

    if not config.get("stop_line"):
        diagnostics.setdefault("warnings", []).append(
            "No stop line configured; crossing analysis was not run")

    if not config.get("signal_timeline"):
        diagnostics.setdefault("warnings", []).append(
            "No signal timeline configured; all crossing phase decisions are insufficient evidence")

    if not settings.motorcycle_model:
        diagnostics.setdefault("warnings", []).append(
            "Motorcycle model is not configured; no genuine motorcycle detections were produced")

    if not settings.plate_model:
        diagnostics.setdefault("warnings", []).append(
            "Rear-plate detector is not configured; automatic plate localization is unavailable")

    if progress_callback:
        progress_callback(
            0.92, "Matching vehicles and compiling event evidence")

    rear_tracks = rear_scan.get("tracks", [])
    rear_by_id = {item["track_id"]: item for item in rear_tracks}
    events = []

    for crossing in traffic_scan.get("crossings", []):
        if association_mode == "same_camera":
            matched_track = rear_by_id.get(crossing["track_id"])
            association = {"status": "same_camera_track", "track_id": crossing["track_id"], "confidence": 1.0,
                           "reason": "Traffic and rear-plate observations share one tracker and source stream"}
        else:
            association = match_rear_track(crossing, rear_tracks,
                                           time_offset=float(config.get(
                                               "rear_time_offset_seconds", 0.0)),
                                           min_travel=float(config.get(
                                               "min_travel_seconds", 0.0)),
                                           max_travel=float(config.get("max_travel_seconds", 15.0)))
            matched_track = rear_by_id.get(association.get(
                "track_id")) if association.get("track_id") else None

        signal_status = crossing["signal"]["status"]
        classification, reason = classify_event(
            signal_status, association["status"], True)
        plate = matched_track.get(
            "plate_consensus", {}) if matched_track else {}
        plate_text = plate.get("text")
        event_id = str(uuid.uuid4())

        events.append({
            "id": event_id,
            "timestamp_seconds": crossing["timestamp_seconds"],
            "track_id": crossing["track_id"],
            "rear_track_id": matched_track.get("track_id") if matched_track else None,
            "direction": crossing["direction"],
            "signal": crossing["signal"],
            "classification": classification,
            "classification_reason": reason,
            "association": association,
            "association_confidence": association.get("confidence", 0.0),
            "plate": {"text": plate_text, "status": plate.get("status", "unreadable"),
                      "confidence": plate.get("confidence", 0.0), "observations": plate.get("observations", 0),
                      "conflicts": plate.get("conflicts", 0)},
            "traffic_frame_url": f"/evidence/{job_id}/evidence/{crossing['frame_file']}",
            "plate_evidence": [f"/evidence/{job_id}/evidence/{p['file']}" for p in (matched_track or {}).get("plate_evidence", [])],
            "rear_frame_urls": [f"/evidence/{job_id}/evidence/{p['frame_file']}" for p in (matched_track or {}).get("plate_evidence", []) if p.get("frame_file")],
            "plate_observations": (matched_track or {}).get("plate_evidence", []),
            "evidence": {"crossing_frame": crossing["frame_file"], "crossing_frame_time_seconds": crossing["timestamp_seconds"]},
        })

    diagnostics["video"] = {"traffic": {k: traffic_scan.get(k) for k in ("fps", "total_frames", "processed_frames", "width", "height")},
                            "rear": {k: rear_scan.get(k) for k in ("fps", "total_frames", "processed_frames", "width", "height")}}
    diagnostics["counts"] = {"traffic_tracks": len(traffic_scan.get("tracks", [])), "rear_tracks": len(rear_tracks),
                             "crossings": len(traffic_scan.get("crossings", [])), "events": len(events)}
    diagnostics["analysis_mode"] = association_mode
    diagnostics["timestamp_reconciliation"] = {"rear_time_offset_seconds": float(config.get("rear_time_offset_seconds", 0.0)),
                                               "min_travel_seconds": float(config.get("min_travel_seconds", 0.0)),
                                               "max_travel_seconds": float(config.get("max_travel_seconds", 15.0))}

    return {"events": events, "diagnostics": diagnostics}
