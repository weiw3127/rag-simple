from typing import List, Tuple
from collections import defaultdict

def rrf_fuse(rank_lists: list[list[Tuple[str, float]]], k: int = 60, top_k: int = 10) -> list[Tuple[str, float]]:
    scores = defaultdict(float)
    for ranked in rank_lists:
        for r, (doc_id, _s) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
