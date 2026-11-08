from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
class ConsensusResult:
    text: str | None
    confidence: float
    status: str
    observations: int
    conflicts: int


def normalize_plate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def multi_frame_consensus(predictions: list[dict], min_confidence: float = 0.65,
                          min_observations: int = 2) -> ConsensusResult:
    """Confidence-weighted exact-string vote across distinct frames; abstains on weak evidence."""
    per_frame: dict[str, tuple[str, float]] = {}

    for index, item in enumerate(predictions):
        text = normalize_plate(str(item.get("text") or ""))
        confidence = float(item.get("confidence", 0.0))

        if not text or not 0.0 <= confidence <= 1.0:
            continue

        frame_key = str(item.get("frame_index", item.get(
            "frame_id", f"observation-{index}")))
        previous = per_frame.get(frame_key)

        if previous is None or confidence > previous[1]:
            per_frame[frame_key] = (text, confidence)

    usable = list(per_frame.values())
    if len(usable) < min_observations:
        return ConsensusResult(None, 0.0, "unreadable", len(usable), 0)

    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for text, confidence in usable:
        scores[text] += confidence
        counts[text] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner, score = ranked[0]
    total = sum(scores.values()) or 1.0
    support = score / total
    mean_conf = score / counts[winner]
    confidence = support * mean_conf
    conflicts = sum(1 for text, _ in usable if text != winner)

    if confidence < min_confidence or (len(ranked) > 1 and support < 0.60):
        return ConsensusResult(None, round(confidence, 4), "conflicting" if conflicts else "unreadable", len(usable), conflicts)

    return ConsensusResult(winner, round(confidence, 4), "readable", len(usable), conflicts)
