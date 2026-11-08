from app.services.traffic.crossing_detector import StopLineCrossingDetector


def test_detects_temporal_crossing_with_anchor_transition():
    detector = StopLineCrossingDetector(
        [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}])

    assert detector.update("1", [0.4, 0.30, 0.5, 0.45], 1.0) is None

    crossing = detector.update("1", [0.4, 0.55, 0.5, 0.65], 1.5)

    assert crossing is not None
    assert crossing.direction == "negative_to_positive"
    assert 1.0 < crossing.timestamp_seconds < 1.5


def test_ignores_crossing_of_infinite_extension_outside_segment():
    detector = StopLineCrossingDetector(
        [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}])
    detector.update("1", [0.0, 0.3, 0.1, 0.45], 1.0)

    assert detector.update("1", [0.0, 0.55, 0.1, 0.7], 1.5) is None


def test_respects_configured_approach_side():
    detector = StopLineCrossingDetector(
        [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}], approach_side="negative")
    detector.update("1", [0.4, 0.55, 0.5, 0.65], 1.0)

    assert detector.update("1", [0.4, 0.3, 0.5, 0.45], 1.5) is None


def test_does_not_infer_crossing_across_long_observation_gap():
    detector = StopLineCrossingDetector(
        [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}], max_gap_seconds=1.0
    )
    detector.update("1", [0.4, 0.30, 0.5, 0.40], 1.0)

    assert detector.update("1", [0.4, 0.55, 0.5, 0.65], 2.5) is None
