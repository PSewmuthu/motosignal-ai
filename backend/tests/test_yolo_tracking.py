from types import SimpleNamespace

from app.services.yolo_tracking import YOLOTrackingConfig, UltralyticsYOLOTracker


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            id=FakeTensor([7, 8]),
            xyxy=FakeTensor([[10, 20, 40, 70], [50, 22, 90, 74]]),
            conf=FakeTensor([0.91, 0.82]),
            cls=FakeTensor([3, 3]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_ultralytics_tracker_uses_persistent_builtin_tracker_and_returns_tracks():
    model = FakeModel()
    config = YOLOTrackingConfig(
        weights="motorcycle.pt",
        class_ids=(3,),
        tracker_yaml="botsort.yaml",
        confidence=0.08,
        iou=0.65,
        image_size=1280,
        device="cpu"
    )
    tracker = UltralyticsYOLOTracker(config, model=model)

    first = tracker.track(object())
    second = tracker.track(object())

    assert [item.track_id for item in first] == ["7", "8"]
    assert [item.hits for item in first] == [1, 1]
    assert [item.hits for item in second] == [2, 2]
    assert first[0].bbox == [10.0, 20.0, 40.0, 70.0]
    assert first[0].class_id == 3
    assert len(model.calls) == 2
    assert all(call["persist"] is True for call in model.calls)
    assert all(call["tracker"] == "botsort.yaml" for call in model.calls)
    assert all(call["classes"] == [3] for call in model.calls)
    assert all(call["conf"] == 0.08 for call in model.calls)


def test_ultralytics_tracker_handles_empty_tracking_result():
    class EmptyModel:
        def track(self, **kwargs):
            return [SimpleNamespace(boxes=SimpleNamespace(id=None))]

    config = YOLOTrackingConfig("motorcycle.pt", (3,), "botsort.yaml")
    tracker = UltralyticsYOLOTracker(config, model=EmptyModel())

    assert tracker.track(object()) == []


def test_factory_creates_isolated_model_state_for_each_stream(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from app.services.yolo_tracking import UltralyticsTrackerFactory

    weights = tmp_path / "motorcycle.pt"
    weights.write_bytes(b"test-checkpoint-placeholder")
    tracker_yaml = tmp_path / "botsort.yaml"
    tracker_yaml.write_text(
        "tracker_type: botsort\nwith_reid: True\n", encoding="utf-8")
    created_models = []

    def fake_yolo(path):
        model = FakeModel()
        created_models.append((path, model))
        return model

    fake_module = ModuleType("ultralytics")
    fake_module.YOLO = fake_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    factory = UltralyticsTrackerFactory(
        YOLOTrackingConfig(str(weights), (3,), str(tracker_yaml))
    )
    first_stream = factory.create()
    second_stream = factory.create()

    assert factory.tracker_type == "botsort"
    assert factory.reid_enabled is True
    assert first_stream.model is not second_stream.model
    assert len(created_models) == 2
