# Architecture and component contracts

## System boundaries

MotoSignal AI is a local-first modular monolith. FastAPI accepts uploads and job configuration; a single background worker performs decoding and inference; SQLite stores job metadata and event payloads; local storage holds source videos and evidence; React reads progress and results through the API.

The current worker is an in-process `ThreadPoolExecutor`. It prevents long-running video analysis from blocking API handlers, but queued tasks are not durable across process restarts. This is acceptable for a single-user assignment demonstration, not for a multi-instance production service.

## Processing stages

1. **Ingest:** validate extension and byte limit, stream upload to disk, check that OpenCV can open the file, store FPS, dimensions, frame count-derived duration and the source role.
2. **Decode:** read each camera stream once, process every Nth frame according to `frame_stride`, and compute a video-relative timestamp from frame index/FPS. Use `frame_stride: 1` for the primary accuracy run so a crossing is not skipped between widely spaced samples.
3. **Motorcycle detection and tracking:** Ultralytics `YOLO.track(frame, persist=True, tracker=...)` runs detector and tracker together using the configured BoT-SORT YAML. The selected profile uses standard BoT-SORT settings, disables optional appearance ReID, and keeps track state across consecutive frames. Each camera stream gets a separate tracker instance. The simple IoU tracker remains only as a deterministic test adapter.
4. **Stop-line crossing:** convert the bottom-centre of the motorcycle box to normalized image coordinates. Keep signed side and timestamp per track. Confirm a sign transition and a geometric intersection of the vehicle-anchor trajectory with the finite configured line segment. Interpolate crossing time between sampled observations.
5. **Signal verification:** look up the crossing time in a direction-specific phase timeline. No interval, overlapping intervals, amber, and unknown yield insufficient evidence.
6. **Rear plate localization:** run the plate detector directly on frames from the rear-view stream (the combined source in same-camera mode). Associate a plate box with the enclosing/nearby motorcycle box; save the original plate crop and quality metrics.
7. **OCR and consensus:** optional PaddleOCR + optional EasyOCR on each detected plate crop. Preserve per-frame predictions; aggregate exact normalized strings by confidence. The system abstains when confidence/support is insufficient or predictions conflict.
8. **Vehicle association:** same-camera observations share a tracker ID. Separate-camera mode uses an offset-adjusted time window and a colour-histogram cosine descriptor; only a unique candidate above score/margin thresholds is accepted. The colour descriptor is a baseline and must not be represented as validated person/vehicle re-identification.
9. **Classification:** combine crossing, signal and association states. Keep event state distinct from OCR readability.
10. **Persistence and review:** store a typed event row plus JSON evidence metadata; expose timestamped traffic frames, plate crops, diagnostics and CSV/JSON exports.

## Key interfaces

- `UltralyticsTrackerFactory.create() -> UltralyticsYOLOTracker`
- `UltralyticsYOLOTracker.track(frame) -> list[Track]` (calls Ultralytics `model.track` with `persist=True`)
- `IoUTracker.update(detections) -> list[Track]` (test/fallback adapter only)
- `StopLineCrossingDetector.update(track_id, normalized_bbox, timestamp) -> Crossing | None`
- `SignalVerifier.at(timestamp_seconds) -> SignalDecision`
- `HybridPlateOCR.recognize(plate_image) -> list[{text, confidence, engines, variants}]`
- `multi_frame_consensus(predictions) -> ConsensusResult`
- `match_rear_track(crossing, rear_tracks, ...) -> association result`
- `classify_event(signal_status, association_status) -> (classification, reason)`

## Tracking configuration and accuracy notes

The default tracker profile is `backend/app/configs/trackers/botsort_standard.yaml`. It uses BoT-SORT with `with_reid: False` and `gmc_method: none`. The YOLO detector confidence floor is `0.08`, allowing weak detections to reach the tracker's low-confidence association stage; new tracks still require the higher tracker threshold. ReID adds inference cost and is not a guarantee against ID switches. The selected values are a starting profile, not claimed optimal thresholds.

`track_buffer` is counted in frames passed to the tracker, not necessarily original video frames. Increasing `frame_stride` means fewer updates and a longer real-time buffer; use stride 1 for crossing-sensitive analysis. If the model is custom-trained for a single motorcycle class, set `MOTOSIGNAL_MOTORCYCLE_CLASS_ID=0` (or the corresponding class index from its `data.yaml`). Standard COCO models use class index 3 for motorcycle.

Tune on a labelled validation clip. Compare `botsort_standard.yaml` against `botsort_standard.yaml` and `bytetrack_baseline.yaml`, keeping the detector checkpoint, source frames, confidence threshold, image size, and frame stride fixed. Report IDF1/HOTA (or MOTA), ID switches, fragmentation, track recall, crossing event precision/recall, and processing FPS. Do not select a configuration from visual smoothness alone.

## Normalized stop-line geometry

Both line endpoints and the bounding-box anchor are expressed in normalized image coordinates. For line endpoints `A` and `B` and anchor `P`, the signed side measure is the 2D cross product `(B-A) × (P-A)`. A candidate requires a sign change, a non-zero dead-band separation, and a segment intersection between the previous and current anchor positions and the finite stop-line segment. This avoids treating an unrelated bounding-box overlap as a crossing.

The current approach is image-plane geometry. It does not estimate road-plane distance, speed, or braking behaviour. Camera perspective and calibration can affect the anchor trajectory, so the configured line must be checked visually for each camera.

## Time model

All timestamps are seconds from the beginning of each encoded video stream. In separate-camera mode, `rear_time_offset_seconds` is added to rear-stream observation timestamps before comparing against the traffic-stream crossing timestamp. The offset sign convention is: `corrected_rear_time = rear_video_time + rear_time_offset_seconds`. Signal intervals are evaluated against the traffic stream's video-relative timestamp.

## Failure and abstention behaviour

Model load failures are surfaced in job diagnostics. A missing detector produces no synthetic detections. Signal gaps/overlaps are insufficient evidence. A plate with no OCR is unreadable. Conflicting OCR can remain `conflicting` with no emitted text. A cross-camera match with no usable descriptors, weak score, or insufficient uniqueness margin is unmatched/ambiguous. Missing evidence never upgrades a case to a suspected violation.

## Production hardening still required

Durable queue (e.g. Celery/RQ), migrations, authenticated users, authorization checks, upload isolation, retention/deletion jobs, object storage, request quotas, structured audit logs, model registry, calibrated confidence thresholds, robust learned tracking/ReID, GPU resource scheduling, metrics/tracing, load tests, security review, and jurisdiction-specific legal review.
