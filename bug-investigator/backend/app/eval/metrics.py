"""File-level localization metrics.

The question each answers: did we point at the files the real fix actually touched?
"""
from __future__ import annotations


def normalize(path: str) -> str:
    return path.strip().lstrip("./")


def hit_at_k(predicted: list[str], gold: list[str], k: int) -> bool:
    """True if any of the top-k predicted files is in the gold set."""
    goldset = {normalize(g) for g in gold}
    return any(normalize(p) in goldset for p in predicted[:k])


def file_recall(predicted: list[str], gold: list[str]) -> float:
    """Fraction of gold files that appear anywhere in the prediction."""
    goldset = {normalize(g) for g in gold}
    if not goldset:
        return 0.0
    predset = {normalize(p) for p in predicted}
    return len(goldset & predset) / len(goldset)


def verdict_is_correct(predicted: str, kind: str) -> bool:
    """A fixed issue should be called a defect; a won't-fix issue should not.

    'unclear' counts as correct on a won't-fix case: declining to patch is the behaviour that
    matters. On a case with a real fix, 'unclear' is a miss - the tool failed to find the defect.
    """
    if kind == "wont_fix":
        return predicted in ("no", "unclear")
    return predicted == "yes"


def summarize(rows: list[dict]) -> dict:
    """Localization is averaged over positive cases only; a won't-fix case has no gold files.

    `verdict` is averaged over every case that carries one, because knowing when NOT to patch is
    the thing a localization score cannot see.
    """
    positives = [r for r in rows if r.get("kind", "fixed") == "fixed"]
    judged = [r for r in rows if r.get("verdict_correct") is not None]
    n = len(positives)
    out = {
        "n": len(rows),
        "localized": n,
        "hit@1": (sum(r["hit_at_1"] for r in positives) / n) if n else 0.0,
        "hit@3": (sum(r["hit_at_3"] for r in positives) / n) if n else 0.0,
        "recall": (sum(r["file_recall"] for r in positives) / n) if n else 0.0,
    }
    out["verdict"] = (
        sum(bool(r["verdict_correct"]) for r in judged) / len(judged) if judged else None
    )
    return out
