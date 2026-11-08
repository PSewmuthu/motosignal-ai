from app.services.plates.consensus import multi_frame_consensus
from app.services.events.classifier import classify_event


def test_consensus_uses_multiple_predictions():
    result = multi_frame_consensus([
        {"text": "ABC1234", "confidence": 0.95},
        {"text": "ABC1234", "confidence": 0.90},
        {"text": "ABC123A", "confidence": 0.55},
    ], min_confidence=0.55)

    assert result.text == "ABC1234"
    assert result.observations == 3


def test_conflicting_or_missing_ocr_abstains():
    assert multi_frame_consensus([]).text is None

    result = multi_frame_consensus([{"text": "AB123", "confidence": 0.7}, {
                                   "text": "XY789", "confidence": 0.7}])

    assert result.text is None
    assert result.status == "conflicting"


def test_classification_requires_supported_association_for_red_phase():
    assert classify_event("red_at_crossing", "same_camera_track")[
        0] == "SUSPECTED_VIOLATION"
    assert classify_event("red_at_crossing", "unmatched")[
        0] == "INSUFFICIENT_EVIDENCE"
    assert classify_event("green_at_crossing", "same_camera_track")[
        0] == "NOT_A_VIOLATION"


def test_plate_requires_distinct_frame_support():
    single = multi_frame_consensus(
        [{"text": "ABC1234", "confidence": 0.99, "frame_index": 12}])

    assert single.text is None
    assert single.status == "unreadable"

    duplicate_frame = multi_frame_consensus([
        {"text": "ABC1234", "confidence": 0.99, "frame_index": 12},
        {"text": "ABC1234", "confidence": 0.98, "frame_index": 12},
    ])

    assert duplicate_frame.text is None
    assert duplicate_frame.observations == 1
