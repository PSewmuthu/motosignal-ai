from dataclasses import dataclass


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int = 0


@dataclass
class Track:
    track_id: str
    bbox: list[float]
    confidence: float
    age: int = 0
    hits: int = 1
    class_id: int = 0


class IoUTracker:
    """Greedy IoU tracker retained only as a deterministic test/fallback adapter."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 12):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[str, Track] = {}
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self.tracks.values():
            track.age += 1

        candidates = []
        for tid, track in self.tracks.items():
            for index, det in enumerate(detections):
                score = iou(track.bbox, det.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, tid, index))

        candidates.sort(reverse=True)
        assigned_tracks, assigned_dets = set(), set()
        for score, tid, index in candidates:
            if tid in assigned_tracks or index in assigned_dets:
                continue

            det = detections[index]
            track = self.tracks[tid]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.class_id = det.class_id
            track.age = 0
            track.hits += 1
            assigned_tracks.add(tid)
            assigned_dets.add(index)

        for index, det in enumerate(detections):
            if index in assigned_dets:
                continue

            tid = str(self.next_id)
            self.next_id += 1
            self.tracks[tid] = Track(
                tid, det.bbox, det.confidence, class_id=det.class_id)

        expired = [tid for tid, track in self.tracks.items()
                   if track.age > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        return [track for track in self.tracks.values() if track.age == 0]
