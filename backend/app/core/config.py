from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Environment-backed application settings; no secrets are required for local use."""

    app_name = "MotoSignal AI"
    data_dir = Path(os.getenv("MOTOSIGNAL_DATA_DIR", "./data")).resolve()
    database_url = os.getenv(
        "DATABASE_URL", f"sqlite:///{(data_dir / 'motosignal.db').as_posix()}")
    motorcycle_model = os.getenv("MOTOSIGNAL_MOTORCYCLE_MODEL", "").strip()
    motorcycle_class_id = int(os.getenv("MOTOSIGNAL_MOTORCYCLE_CLASS_ID", "3"))
    motorcycle_confidence = float(
        os.getenv("MOTOSIGNAL_MOTORCYCLE_CONFIDENCE", "0.08"))
    motorcycle_nms_iou = float(
        os.getenv("MOTOSIGNAL_MOTORCYCLE_NMS_IOU", "0.65"))
    motorcycle_image_size = int(
        os.getenv("MOTOSIGNAL_MOTORCYCLE_IMAGE_SIZE", "1280"))
    yolo_device = os.getenv("MOTOSIGNAL_YOLO_DEVICE", "").strip() or None
    tracker_config = os.getenv(
        "MOTOSIGNAL_TRACKER_CONFIG", "configs/trackers/botsort_standard.yaml").strip()
    plate_model = os.getenv("MOTOSIGNAL_PLATE_MODEL", "").strip()
    enable_ocr = os.getenv("MOTOSIGNAL_ENABLE_OCR", "false").lower() == "true"
    frame_stride = max(1, int(os.getenv("MOTOSIGNAL_FRAME_STRIDE", "1")))
    max_upload_mb = max(10, int(os.getenv("MOTOSIGNAL_MAX_UPLOAD_MB", "500")))
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

    @classmethod
    def prepare(cls) -> None:
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        (cls.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (cls.data_dir / "jobs").mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.prepare()
