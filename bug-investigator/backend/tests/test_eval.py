"""Eval metrics and the harness wiring (GitHub stubbed, index and retrieval real)."""

from app.db import models as m
from app.eval.harness import run_baseline, run_eval, score
from app.eval.metrics import file_recall, hit_at_k, summarize
from tests.test_runner import ScriptedLLM


def test_hit_at_k():
    pred = ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]
    assert hit_at_k(pred, ["src/a.py"], 1)
    assert not hit_at_k(pred, ["src/c.py"], 1)
    assert hit_at_k(pred, ["src/c.py"], 3)
    assert not hit_at_k(pred, ["src/zzz.py"], 3)
    assert not hit_at_k([], ["src/a.py"], 3)


def test_hit_at_k_normalizes_leading_dot_slash():
    assert hit_at_k(["./src/a.py"], ["src/a.py"], 1)


def test_file_recall():
    assert file_recall(["a", "b"], ["a", "b"]) == 1.0
    assert file_recall(["a"], ["a", "b"]) == 0.5
    assert file_recall(["z"], ["a"]) == 0.0
    assert file_recall(["a"], []) == 0.0


def test_summarize_averages_and_handles_empty():
    rows = [
        {"hit_at_1": True, "hit_at_3": True, "file_recall": 1.0},
        {"hit_at_1": False, "hit_at_3": True, "file_recall": 0.5},
    ]
    s = summarize(rows)
    assert s == {"n": 2, "hit@1": 0.5, "hit@3": 1.0, "recall": 0.75}
    assert summarize([])["n"] == 0


def test_score_shape():
    assert set(score(["a"], ["a"])) == {"hit_at_1", "hit_at_3", "file_recall"}


def test_baseline_predicts_files_from_issue_text(db, indexed_repo):
    repo, _ = indexed_repo
    issue = m.Issue(
        repo_id=repo.id, number=1, title="average crashes on empty list",
        body="average(values) divides by len(values)", state="closed", html_url="u",
    )
    db.add(issue)
    db.commit()
    files = run_baseline(db, repo, issue, k=10)
    assert "src/stats.py" in files
    assert len(files) == len(set(files))  # distinct, order preserved


class StubGH:
    """Only the calls run_eval makes for an already-cached issue."""

    def get_issue(self, ref):  # pragma: no cover - issue is pre-seeded in these tests
        raise AssertionError("should not be called")


def test_run_eval_scores_agent_against_baseline(db, settings, indexed_repo):
    repo, _ = indexed_repo
    db.add(
        m.Issue(
            repo_id=repo.id, number=5, title="average crashes on empty list",
            body="average([]) raises ZeroDivisionError", state="closed", html_url="u",
        )
    )
    db.add(m.EvalCase(repo_id=repo.id, issue_number=5, gold_pr_number=6, gold_files=["src/stats.py"]))
    db.commit()

    summary = run_eval(db, repo, settings, ScriptedLLM(), StubGH())

    assert summary["agent"]["n"] == 1
    assert summary["agent"]["hit@1"] == 1.0  # scripted agent predicts src/stats.py
    assert summary["bm25_baseline"]["n"] == 1
    assert 0.0 <= summary["bm25_baseline"]["hit@3"] <= 1.0

    rows = db.query(m.EvalResult).all()
    assert {r.system for r in rows} == {"agent", "bm25_baseline"}
    assert db.query(m.Investigation).count() == 1


def test_run_eval_marks_a_miss_when_the_agent_names_the_wrong_file(db, settings, indexed_repo):
    repo, _ = indexed_repo
    db.add(m.Issue(repo_id=repo.id, number=7, title="t", body="b", state="closed", html_url="u"))
    db.add(m.EvalCase(repo_id=repo.id, issue_number=7, gold_files=["src/stats.py"]))
    db.commit()

    summary = run_eval(
        db, repo, settings, ScriptedLLM(affected=["src/network.py"]), StubGH(), systems=("agent",)
    )
    assert summary["agent"]["hit@1"] == 0.0
    assert summary["agent"]["recall"] == 0.0


def test_format_summary_is_readable():
    from app.eval.harness import format_summary

    text = format_summary({"agent": {"n": 2, "hit@1": 0.5, "hit@3": 1.0, "recall": 0.75}})
    assert "agent" in text and "0.50" in text and "hit@1" in text
