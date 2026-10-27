from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence


def rrf_fuse(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion: merge several ranked id lists into one.

    score(id) = sum over lists of 1 / (k + rank), rank starting at 1.
    Returns (id, score) sorted by score desc; ties keep first-seen order.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    order: dict[Hashable, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
