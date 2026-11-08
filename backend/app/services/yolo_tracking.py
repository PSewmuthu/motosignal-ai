from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.tracking import Track


class YOLOTrackingUnavailable(RuntimeError):
    """Raised when the configured Ultralytics tracking runtime is unavailable."""


@dataclass(frozen=True)
class YOLOTrackingConfig:
    weights: str
    class_ids: tuple[int, ...] | None
    tracker_yaml: str
    confidence: float = 0.08
    iou: float = 0.65
    image_size: int = 1280
    device: str | None = None


class UltralyticsTrackerFactory:
    """Creates one isolated Ultralytics tracker state per video stream.

    The factory is safe to use for both same-camera and separate-camera jobs:
    each call to ``create`` loads a fresh model/tracker state, so track IDs and
    Kalman/ReID state cannot leak from one camera stream into another.
    """

    def __init__(self, config: YOLOTrackingConfig):
        weights = Path(config.weights).expanduser()

        if not weights.is_file():
            raise YOLOTrackingUnavailable(
                f"Motorcycle model weights not found: {weights}")

        tracker_yaml = Path(config.tracker_yaml).expanduser()
        if not tracker_yaml.is_file():
            raise YOLOTrackingUnavailable(
                f"Tracker YAML not found: {tracker_yaml}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise YOLOTrackingUnavailable(
                "Install the ultralytics package to use YOLO tracking: pip install ultralytics"
            ) from exc

        if not 0.0 <= config.confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")

        if not 0.0 <= config.iou <= 1.0:
            raise ValueError("YOLO NMS IoU must be between 0 and 1")

        if config.image_size < 320:
            raise ValueError("YOLO image_size must be at least 320")

        self.config = YOLOTrackingConfig(
            weights=str(weights.resolve()),
            class_ids=config.class_ids,
            tracker_yaml=str(tracker_yaml.resolve()),
            confidence=config.confidence,
            iou=config.iou,
            image_size=config.image_size,
            device=config.device,
        )

        # The tracker type and ReID flag are read from the same YAML passed to
        # Ultralytics and are included in job diagnostics for reproducibility.
        tracker_values: dict[str, str] = {}
        for raw_line in tracker_yaml.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                tracker_values[key.strip()] = value.strip().strip("\"'")

        self.tracker_type = tracker_values.get("tracker_type", "unknown")
        self.reid_enabled = tracker_values.get(
            "with_reid", "False").lower() == "true"
        self._model_class = YOLO

        try:
            # Load the first model during job setup so missing/corrupt/incompatible
            # checkpoints are reported as detector-unavailable diagnostics early.
            self._first_model = YOLO(self.config.weights)
        except Exception as exc:
            raise YOLOTrackingUnavailable(
                f"Could not load motorcycle YOLO weights: {exc}") from exc

    def create(self) -> "UltralyticsYOLOTracker":
        model = self._first_model
        self._first_model = None

        if model is None:
            try:
                model = self._model_class(self.config.weights)
            except Exception as exc:
                raise YOLOTrackingUnavailable(
                    f"Could not create isolated YOLO tracker for video stream: {exc}") from exc

        return UltralyticsYOLOTracker(self.config, model=model)


class UltralyticsYOLOTracker:
    """Frame-by-frame YOLO tracking using ``model.track(..., persist=True)``.

    The detector and tracker run together, so the built-in Ultralytics tracker
    receives the detection scores before association. Do not call ``detect``
    followed by a second independent tracker for this backend.
    """

    def __init__(self, config: YOLOTrackingConfig, model: Any | None = None):
        try:
            if model is None:
                from ultralytics import YOLO
                model = YOLO(config.weights)
        except ImportError as exc:
            raise YOLOTrackingUnavailable(
                "Ultralytics is not installed") from exc

        self.model = model
        self.config = config
        self.hits_by_id: dict[str, int] = {}
        self.frame_index = 0

    def track(self, frame) -> list[Track]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "tracker": self.config.tracker_yaml,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "verbose": False,
        }

        if self.config.class_ids is not None:
            kwargs["classes"] = list(self.config.class_ids)

        if self.config.device:
            kwargs["device"] = self.config.device

        results = self.model.track(**kwargs)
        self.frame_index += 1
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            return []

        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        tracks: list[Track] = []
        for track_id, bbox, confidence, class_id in zip(ids, coords, confidences, classes):
            if self.config.class_ids is not None and int(class_id) not in self.config.class_ids:
                continue
            key = str(int(track_id))
            hits = self.hits_by_id.get(key, 0) + 1
            self.hits_by_id[key] = hits

            tracks.append(
                Track(
                    track_id=key,
                    bbox=[float(value) for value in bbox],
                    confidence=float(confidence),
                    age=0,
                    hits=hits,
                    class_id=int(class_id),
                )
            )

        return tracks
