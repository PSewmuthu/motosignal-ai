from app.services.association.vehicle_matcher import match_rear_track


def test_match_requires_time_and_visual_evidence():
    crossing = {"timestamp_seconds": 10.0, "appearance": [1.0, 0.0]}
    tracks = [{"track_id": "r1", "timestamps": [
        13.0, 14.0], "appearance": [0.99, 0.05]}]
    result = match_rear_track(
        crossing, tracks, min_travel=1, max_travel=8, min_score=0.5, min_margin=0.01)

    assert result["status"] == "matched"
    assert result["track_id"] == "r1"


def test_match_abstains_without_appearance_or_time_candidate():
    crossing = {"timestamp_seconds": 10.0, "appearance": None}
    tracks = [{"track_id": "r1", "timestamps": [
        13.0], "appearance": [1.0, 0.0]}]
    result = match_rear_track(crossing, tracks, min_travel=1, max_travel=8)

    assert result["status"] == "unmatched"


def test_match_abstains_when_candidate_is_not_unique():
    crossing = {"timestamp_seconds": 10.0, "appearance": [1.0, 0.0]}
    tracks = [
        {"track_id": "r1", "timestamps": [13.0], "appearance": [1.0, 0.0]},
        {"track_id": "r2", "timestamps": [13.0], "appearance": [1.0, 0.0]},
    ]
    result = match_rear_track(
        crossing, tracks, min_travel=1, max_travel=8, min_score=0.1, min_margin=0.1)

    assert result["status"] == "ambiguous"
