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


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "recall": 0.0}
    return {
        "n": n,
        "hit@1": sum(r["hit_at_1"] for r in rows) / n,
        "hit@3": sum(r["hit_at_3"] for r in rows) / n,
        "recall": sum(r["file_recall"] for r in rows) / n,
    }
