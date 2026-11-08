from dataclasses import dataclass
from typing import Optional


@dataclass
class Crossing:
    timestamp_seconds: float
    direction: str
    previous_signed_distance: float
    current_signed_distance: float


class StopLineCrossingDetector:
    """Detect a time-ordered anchor transition across a finite, normalized line segment."""

    def __init__(self, line: list[dict], approach_side: str = "either", max_gap_seconds: float = 1.5,
                 min_anchor_displacement: float = 0.005):
        self.line = [(float(p["x"]), float(p["y"])) for p in line]
        self.approach_side = approach_side
        self.max_gap_seconds = max_gap_seconds
        self.min_anchor_displacement = min_anchor_displacement
        self.previous: dict[str, tuple[float, float, float, float]] = {}

    def _signed_distance(self, point: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.line

        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def _intersection_on_segment(self, a, b) -> tuple[bool, float]:
        (x1, y1), (x2, y2) = self.line
        rx, ry = b[0] - a[0], b[1] - a[1]
        sx, sy = x2 - x1, y2 - y1
        denom = rx * sy - ry * sx

        if abs(denom) < 1e-9:
            return False, 0.0

        qpx, qpy = x1 - a[0], y1 - a[1]
        t = (qpx * sy - qpy * sx) / denom
        u = (qpx * ry - qpy * rx) / denom

        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0, t

    def update(self, track_id: str, bbox: list[float], timestamp_seconds: float) -> Optional[Crossing]:
        if len(self.line) != 2:
            return None

        x1, y1, x2, y2 = bbox
        anchor = ((x1 + x2) / 2.0, y2)
        distance = self._signed_distance(anchor)
        previous = self.previous.get(track_id)
        self.previous[track_id] = (
            anchor[0], anchor[1], distance, timestamp_seconds)

        if previous is None:
            return None

        px, py, prev_distance, prev_time = previous
        gap = timestamp_seconds - prev_time
        displacement = ((anchor[0] - px) ** 2 + (anchor[1] - py) ** 2) ** 0.5
        if gap <= 0 or gap > self.max_gap_seconds or displacement < self.min_anchor_displacement:
            return None

        # The dead band avoids oscillation around a noisy line estimate.
        eps = 0.0015
        if abs(prev_distance) <= eps or abs(distance) <= eps or prev_distance * distance >= 0:
            return None

        intersects, fraction = self._intersection_on_segment((px, py), anchor)
        if not intersects:
            return None

        before_side = "positive" if prev_distance > 0 else "negative"
        after_side = "positive" if distance > 0 else "negative"
        if self.approach_side != "either" and before_side != self.approach_side:
            return None

        crossing_time = prev_time + max(0.0, min(1.0, fraction)) * gap

        return Crossing(crossing_time, f"{before_side}_to_{after_side}", prev_distance, distance)
