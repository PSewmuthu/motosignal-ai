import cv2
import numpy as np

from app.core.config import settings
from app.services import pipeline
from app.services.tracking import Detection


class FakeMotorcycleDetector:
    def __init__(self):
        self.index = 0

    def detect(self, frame):
        y2 = 35 + 4 * self.index
        self.index += 1
        return [Detection([60, y2 - 25, 130, y2], 0.95, 3)]


class FakePlateDetector:
    def __init__(self):
        self.index = 0

    def detect(self, frame):
        y2 = 35 + 4 * self.index
        self.index += 1
        return [Detection([95, y2 - 9, 115, y2 - 3], 0.93, 0)]


class FakeOCR:
    def recognize(self, image):
        return [{"text": "ABC1234", "confidence": 0.94}]


def test_same_camera_pipeline_emits_evidence_backed_red_event(tmp_path, monkeypatch):
    video_path = tmp_path / "synthetic-intersection.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 100))

    assert writer.isOpened()

    for index in range(8):
        frame = np.full((100, 200, 3), 35 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    monkeypatch.setattr(settings, "data_dir", tmp_path / "runtime")
    monkeypatch.setattr(settings, "motorcycle_model", "/test/motorcycle.pt")
    monkeypatch.setattr(settings, "enable_ocr", True)

    def fake_models(diagnostics):
        diagnostics.update({"motorcycle_detector": "ready_test_double",
                           "plate_detector": "ready_test_double", "ocr": "ready_test_double"})
        return FakeMotorcycleDetector(), FakePlateDetector(), FakeOCR()

    monkeypatch.setattr(pipeline, "_detect_models", fake_models)

    result = pipeline.run_video_analysis(
        str(video_path), None, "integration-job",
        {
            "mode": "same_camera",
            "stop_line": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
            "approach_side": "either",
            "direction": "all",
            "signal_timeline": [{"start_seconds": 0, "end_seconds": 10, "color": "red", "direction": "all"}],
            "frame_stride": 1,
            "min_track_observations": 3
        }
    )

    assert len(result["events"]) == 1

    event = result["events"][0]

    assert event["classification"] == "SUSPECTED_VIOLATION"
    assert event["signal"]["color"] == "red"
    assert event["plate"]["text"] == "ABC1234"
    assert event["plate"]["status"] == "readable"
    assert event["association"]["status"] == "same_camera_track"
    assert event["traffic_frame_url"].endswith(".jpg")
    assert event["rear_frame_urls"] and event["plate_evidence"]

    evidence_dir = tmp_path / "runtime" / "jobs" / "integration-job" / "evidence"

    assert any(evidence_dir.glob("traffic_*.jpg"))
    assert any(evidence_dir.glob("rear_*.jpg"))
    assert any(evidence_dir.glob("plate_*.jpg"))
