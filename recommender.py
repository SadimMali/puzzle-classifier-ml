"""Content-based puzzle ranking. No training, user ratings or other users' data."""
from collections import Counter
from datetime import datetime, timezone
from math import isfinite, log, sqrt


def _tokens(puzzle):
    return {f"theme:{theme}" for theme in puzzle["themes"]} | (
        {f"type:{puzzle['puzzle_type']}"} if puzzle.get("puzzle_type") else set()
    )


def _normalize(vector):
    norm = sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def _centroid(vectors):
    total = Counter()
    for vector in vectors:
        total.update(vector)
    return _normalize(total)


def cosine_similarity(a, b):
    a, b = _normalize(a), _normalize(b)
    return max(0.0, min(1.0, sum(value * b.get(key, 0) for key, value in a.items())))


def _attempt_time(puzzle):
    value = puzzle.get("lastAttemptAt")
    if not value:
        return float("-inf")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()


def _history_profile(history, vector):
    # Each puzzle counts once at its most recent interaction. A retry refreshes
    # recency; its number of attempts does not multiply its weight.
    unique = {}
    for puzzle in history:
        if puzzle["id"] not in unique or _attempt_time(puzzle) > _attempt_time(unique[puzzle["id"]]):
            unique[puzzle["id"]] = puzzle
    ordered = sorted(unique.values(), key=lambda p: (-_attempt_time(p), p["id"]))
    if not any(p.get("lastAttemptAt") for p in ordered):
        return _centroid(vector(p) for p in ordered), len(ordered)
    total = Counter()
    for rank, puzzle in enumerate(ordered):
        # Recent practice dominates: weights 1, .4, .16, ... (about 60%, 24%, 10%).
        # This tracks content interest, never skill or puzzle success probability.
        weight = 0.4 ** rank
        total.update({key: weight * value for key, value in vector(puzzle).items()})
    return _normalize(total), len(ordered)


def recommend(catalog, history, candidates, offset=0, limit=20):
    """Fit global content statistics, average own history, rank before pagination.

    Inputs contain puzzle content only. Backend enforces ownership and eligibility.
    Stored difficulty ratings come from the existing predictor/provider pipeline.
    """
    frequencies = Counter(key for puzzle in catalog for key in _tokens(puzzle))
    ratings = [p["difficulty_rating"] for p in catalog
               if p.get("difficulty_rating") is not None and isfinite(p["difficulty_rating"])]
    low, high = (min(ratings), max(ratings)) if ratings else (0, 0)

    def vector(puzzle):
        content = _normalize({key: log((len(catalog) + 1) / (frequencies[key] + 1)) + 1
                              for key in _tokens(puzzle) if key in frequencies})
        rating = puzzle.get("difficulty_rating")
        if high > low and rating is not None and isfinite(rating):
            d = max(0, min(1, (rating - low) / (high - low)))
            content.update({"difficulty:low": 0.5 * (1 - d), "difficulty:high": 0.5 * d})
        return _normalize(content)

    profile, history_count = _history_profile(history, vector)
    mode = "history" if profile else "global"
    if not profile:
        profile = _centroid(vector(p) for p in catalog)
    ranked = [{"id": p["id"], "recommendation": {
        "algorithm": "cosine_similarity", "profile": mode,
        "score": cosine_similarity(profile, vector(p)),
        "sharedThemes": sorted({t for t in p["themes"] if profile.get(f"theme:{t}", 0) > 0}),
        "historyPuzzleCount": history_count, "globalPuzzleCount": len(catalog),
    }} for p in candidates]
    ranked.sort(key=lambda p: (-p["recommendation"]["score"], p["id"]))
    return ranked[offset:offset + limit]
