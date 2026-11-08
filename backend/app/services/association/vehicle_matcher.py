from __future__ import annotations
import numpy as np


def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.shape != vb.shape:
        return 0.0

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))

    return float(np.dot(va, vb) / denom) if denom else 0.0


def match_rear_track(crossing: dict, rear_tracks: list[dict], *, time_offset: float = 0.0,
                     min_travel: float = 0.0, max_travel: float = 15.0,
                     min_score: float = 0.72, min_margin: float = 0.10) -> dict:
    """Conservative cross-camera candidate matcher; abstains unless time and appearance agree."""
    scored = []
    event_time = float(crossing["timestamp_seconds"])

    for track in rear_tracks:
        times = [float(t) + time_offset for t in track.get("timestamps", [])]
        deltas = [t - event_time for t in times if min_travel <=
                  t - event_time <= max_travel]

        if not deltas:
            continue

        delta = min(deltas, key=lambda d: abs(
            d - (min_travel + max_travel) / 2))
        span = max(max_travel - min_travel, 1e-6)
        temporal = max(0.0, 1.0 - abs(delta - (min_travel +
                       max_travel) / 2) / (span / 2 + 1e-6))
        appearance = cosine_similarity(crossing.get(
            "appearance"), track.get("appearance"))

        if crossing.get("appearance") is None or track.get("appearance") is None:
            continue

        score = 0.65 * max(0.0, appearance) + 0.35 * temporal
        scored.append((score, track, delta, appearance, temporal))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return {"status": "unmatched", "track_id": None, "confidence": 0.0, "reason": "No candidate had both a valid travel-time window and appearance descriptor"}

    best = scored[0]
    margin = best[0] - scored[1][0] if len(scored) > 1 else best[0]
    if best[0] < min_score or margin < min_margin:
        return {"status": "ambiguous", "track_id": None, "confidence": round(best[0], 4),
                "reason": "Candidate score or uniqueness margin is below the configured threshold"}

    return {"status": "matched", "track_id": best[1]["track_id"], "confidence": round(best[0], 4),
            "travel_seconds": round(best[2], 3), "appearance_similarity": round(best[3], 4),
            "temporal_score": round(best[4], 4), "reason": "Unique candidate passed conservative time and appearance thresholds"}
