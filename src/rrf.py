# src/rrf.py
from collections import defaultdict

def rrf_merge(*rankings, k: int = 60):
    """Reciprocal Rank Fusion. rankings = list of ranklists of ES hits."""
    scores = defaultdict(float)
    id2hit = {}
    for ranklist in rankings:
        for rank, hit in enumerate(ranklist, start=1):
            _id = hit["_id"]
            id2hit[_id] = hit
            scores[_id] += 1.0 / (k + rank)
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [id2hit[_id] for _id, _ in merged]
