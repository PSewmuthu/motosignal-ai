from app.services.traffic.signal_verifier import SignalVerifier


def test_red_phase_is_detected_at_crossing_time():
    verifier = SignalVerifier(
        [{"start_seconds": 0, "end_seconds": 10, "color": "red", "direction": "north"}], "north")
    result = verifier.at(5)

    assert result.status == "red_at_crossing"
    assert result.phase_start_seconds == 0
    assert result.phase_end_seconds == 10
    assert result.direction == "north"


def test_missing_or_overlapping_phase_is_insufficient():
    assert SignalVerifier([], "all").at(3).status == "insufficient_evidence"

    phases = [{"start_seconds": 0, "end_seconds": 10, "color": "red", "direction": "all"},
              {"start_seconds": 2, "end_seconds": 8, "color": "green", "direction": "all"}]

    assert SignalVerifier(phases).at(4).status == "insufficient_evidence"


def test_amber_is_not_adjudicated_as_red_or_green():
    verifier = SignalVerifier(
        [{"start_seconds": 0, "end_seconds": 10, "color": "amber", "direction": "all"}])

    assert verifier.at(5).status == "insufficient_evidence"
