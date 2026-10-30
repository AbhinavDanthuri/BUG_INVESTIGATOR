"""Path-based priors for ranked search results.

An eval run on pallets/flask showed what their absence costs: of twelve evidence chunks handed to
the investigator, eleven were `pyproject.toml`, `tests/test_basic.py` and `examples/` - with
`tests/test_basic.py` taking four slots on its own. One real source file made it through. The
verdict that run produced was not reasoning about the bug; there was nothing there to reason about.

Two priors fix most of it, both cheap and both applied after fusion so recall is unaffected:

* tests, examples and packaging files are demoted, not dropped - a test is real evidence of
  intended behaviour, it just should not outrank the implementation;
* no single file may take more than a couple of slots, because four chunks of one test file say
  little more than two do.
"""
from __future__ import annotations

import re

# Demotions are multiplicative on the fused score and deliberately coarse; the ordering between
# these groups matters, the exact values do not.
_SUPPORTING = (
    (re.compile(r"(^|/)(tests?|testing|spec|specs)(/|$)|(^|/)(test_[^/]*|[^/]*_test)\.[a-z]+$"), 0.35),
    (re.compile(r"(^|/)(examples?|samples?|demos?|benchmarks?|fixtures?)(/|$)"), 0.25),
    (re.compile(r"(^|/)(pyproject\.toml|setup\.py|setup\.cfg|tox\.ini|noxfile\.py|package\.json"
                r"|requirements[^/]*\.txt|Makefile|Dockerfile|[^/]*\.ya?ml|[^/]*\.cfg|[^/]*\.ini)$"), 0.15),
)

MAX_PER_FILE = 2


def path_weight(path: str) -> float:
    """Multiplier for a chunk's fused score, from its path alone. 1.0 means implementation code."""
    p = (path or "").strip().lstrip("./")
    for pattern, weight in _SUPPORTING:
        if pattern.search(p):
            return weight
    return 1.0


def rerank(scored: list[tuple[int, float]], path_of, max_per_file: int = MAX_PER_FILE) -> list[tuple[int, float]]:
    """Apply the path prior, then cap how many chunks any one file may contribute.

    Capping is done after weighting so the cap keeps each file's best chunks, and ties keep their
    incoming order - fusion already decided those.
    """
    weighted = [(cid, score * path_weight(path_of(cid))) for cid, score in scored]
    order = {cid: i for i, (cid, _) in enumerate(weighted)}
    weighted.sort(key=lambda cs: (-cs[1], order[cs[0]]))

    seen: dict[str, int] = {}
    kept: list[tuple[int, float]] = []
    for cid, score in weighted:
        path = path_of(cid) or ""
        if seen.get(path, 0) >= max_per_file:
            continue
        seen[path] = seen.get(path, 0) + 1
        kept.append((cid, score))
    return kept
