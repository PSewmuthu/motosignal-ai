def classify_event(signal_status: str, association_status: str, crossing_confirmed: bool = True) -> tuple[str, str]:
    if not crossing_confirmed:
        return "INSUFFICIENT_EVIDENCE", "A temporal stop-line crossing was not confirmed"

    if signal_status == "red_at_crossing":
        if association_status in ("same_camera_track", "matched"):
            return "SUSPECTED_VIOLATION", "A tracked motorcycle crossed during the configured red phase; this is an analysis flag, not a legal determination"

        return "INSUFFICIENT_EVIDENCE", "Red phase detected, but the rear-view vehicle association is not sufficiently supported"

    if signal_status == "green_at_crossing":
        return "NOT_A_VIOLATION", "The configured signal timeline is green at the crossing time"

    return "INSUFFICIENT_EVIDENCE", "Signal state or crossing evidence is incomplete or ambiguous"
