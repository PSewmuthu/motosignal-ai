from dataclasses import dataclass


@dataclass
class SignalDecision:
    color: str
    status: str
    reason: str
    phase_start_seconds: float | None = None
    phase_end_seconds: float | None = None
    direction: str | None = None


class SignalVerifier:
    def __init__(self, timeline: list[dict], direction: str = "all"):
        self.timeline = sorted(timeline, key=lambda x: x["start_seconds"])
        self.direction = direction

    def at(self, timestamp_seconds: float) -> SignalDecision:
        matches = [p for p in self.timeline
                   if p["start_seconds"] <= timestamp_seconds < p["end_seconds"]
                   and p.get("direction", "all") in ("all", self.direction)]

        if len(matches) != 1:
            reason = "No signal phase covers the crossing time" if not matches else "Overlapping signal phases make the state ambiguous"
            return SignalDecision("unknown", "insufficient_evidence", reason)

        phase = matches[0]
        color = phase["color"]
        common = (float(phase["start_seconds"]), float(
            phase["end_seconds"]), phase.get("direction", "all"))

        if color == "red":
            return SignalDecision(color, "red_at_crossing", "Configured direction-specific timeline is red at the crossing time", *common)

        if color == "green":
            return SignalDecision(color, "green_at_crossing", "Configured direction-specific timeline is green at the crossing time", *common)

        return SignalDecision(color, "insufficient_evidence", "Amber or unknown phase is not automatically adjudicated", *common)
