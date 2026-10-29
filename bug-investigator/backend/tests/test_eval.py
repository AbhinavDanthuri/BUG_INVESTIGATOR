"""Eval metrics and the harness wiring (GitHub stubbed, index and retrieval real)."""

import pytest

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
    assert (s["n"], s["localized"], s["hit@1"], s["hit@3"], s["recall"]) == (2, 2, 0.5, 1.0, 0.75)
    assert s["verdict"] is None  # nothing judged
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
    db.add(
        m.EvalCase(
            repo_id=repo.id,
            issue_number=5,
            kind=m.EvalCaseKind.fixed,
            gold_pr_number=6,
            gold_files=["src/stats.py"],
        )
    )
    db.commit()

    summary = run_eval(db, repo, settings, ScriptedLLM(), StubGH())

    assert summary["agent"]["n"] == 1
    assert summary["agent"]["hit@1"] == 1.0  # scripted agent predicts src/stats.py
    assert summary["agent"]["verdict"] == 1.0  # called a fixed issue a defect
    assert summary["bm25_baseline"]["n"] == 1
    assert 0.0 <= summary["bm25_baseline"]["hit@3"] <= 1.0

    rows = db.query(m.EvalResult).all()
    assert {r.system for r in rows} == {"agent", "bm25_baseline"}
    assert db.query(m.Investigation).count() == 1


def test_run_eval_marks_a_miss_when_the_agent_names_the_wrong_file(db, settings, indexed_repo):
    repo, _ = indexed_repo
    db.add(m.Issue(repo_id=repo.id, number=7, title="t", body="b", state="closed", html_url="u"))
    db.add(
        m.EvalCase(
            repo_id=repo.id, issue_number=7, kind=m.EvalCaseKind.fixed, gold_files=["src/stats.py"]
        )
    )
    db.commit()

    summary = run_eval(
        db, repo, settings, ScriptedLLM(affected=["src/network.py"]), StubGH(), systems=("agent",)
    )
    assert summary["agent"]["hit@1"] == 0.0
    assert summary["agent"]["recall"] == 0.0


def test_format_summary_is_readable():
    from app.eval.harness import format_summary

    text = format_summary(
        {"agent": {"n": 2, "localized": 2, "hit@1": 0.5, "hit@3": 1.0, "recall": 0.75, "verdict": 0.5}}
    )
    assert "agent" in text and "0.50" in text and "hit@1" in text and "verdict" in text

    no_verdict = format_summary(
        {"bm25": {"n": 1, "localized": 1, "hit@1": 0.0, "hit@3": 0.0, "recall": 0.0, "verdict": None}}
    )
    assert "n/a" in no_verdict  # the baseline has no notion of a defect


# ---------- negative (won't-fix) cases ----------
@pytest.mark.parametrize(
    "predicted,kind,expected",
    [
        ("yes", "fixed", True),
        ("no", "fixed", False),
        ("unclear", "fixed", False),      # failed to find a real defect
        ("no", "wont_fix", True),
        ("unclear", "wont_fix", True),    # declining to patch is what matters
        ("yes", "wont_fix", False),       # the pallets/click#3887 failure
    ],
)
def test_verdict_is_correct(predicted, kind, expected):
    from app.eval.metrics import verdict_is_correct

    assert verdict_is_correct(predicted, kind) is expected


def test_summarize_separates_localization_from_verdict():
    """Won't-fix cases have no gold files, so they must not drag the localization average down."""
    rows = [
        {"kind": "fixed", "hit_at_1": True, "hit_at_3": True, "file_recall": 1.0, "verdict_correct": True},
        {"kind": "wont_fix", "hit_at_1": False, "hit_at_3": False, "file_recall": 0.0, "verdict_correct": False},
    ]
    s = summarize(rows)
    assert s["n"] == 2
    assert s["localized"] == 1
    assert s["hit@1"] == 1.0   # scored on the one positive case only
    assert s["verdict"] == 0.5  # scored across both


def test_run_eval_scores_a_wont_fix_case(db, settings, indexed_repo):
    """The regression that matters: patching intended behaviour must score as wrong."""
    repo, _ = indexed_repo
    db.add(
        m.Issue(
            repo_id=repo.id, number=9, title="average should return None for empty input",
            body="I would prefer a different design", state="closed", html_url="u",
        )
    )
    db.add(m.EvalCase(repo_id=repo.id, issue_number=9, kind=m.EvalCaseKind.wont_fix, gold_files=[]))
    db.commit()

    # a tool that insists everything is a defect
    wrong = run_eval(db, repo, settings, ScriptedLLM(is_defect="yes"), StubGH(), systems=("agent",))
    assert wrong["agent"]["verdict"] == 0.0
    assert wrong["agent"]["localized"] == 0  # no positive cases to localize

    row = db.query(m.EvalResult).filter_by(system="agent").one()
    assert row.defect_verdict == "yes"
    assert row.verdict_correct is False


def test_run_eval_rewards_declining_to_patch(db, settings, indexed_repo):
    repo, _ = indexed_repo
    db.add(
        m.Issue(
            repo_id=repo.id, number=11, title="works as documented but I disagree",
            body="please change it", state="closed", html_url="u",
        )
    )
    db.add(m.EvalCase(repo_id=repo.id, issue_number=11, kind=m.EvalCaseKind.wont_fix, gold_files=[]))
    db.commit()

    right = run_eval(db, repo, settings, ScriptedLLM(is_defect="no"), StubGH(), systems=("agent",))
    assert right["agent"]["verdict"] == 1.0


def test_baseline_carries_no_verdict(db, settings, indexed_repo):
    """BM25 retrieves files; it cannot judge intent, so it must not be scored as if it could."""
    repo, _ = indexed_repo
    db.add(
        m.Issue(repo_id=repo.id, number=13, title="average crashes", body="on empty", state="closed", html_url="u")
    )
    db.add(m.EvalCase(repo_id=repo.id, issue_number=13, kind=m.EvalCaseKind.fixed, gold_files=["src/stats.py"]))
    db.commit()

    summary = run_eval(db, repo, settings, ScriptedLLM(), StubGH(), systems=("bm25_baseline",))
    assert summary["bm25_baseline"]["verdict"] is None


class _DiscoveryGH:
    """A repo with one fixed issue, one won't-fix, and one closed for an unscoreable reason."""

    def __init__(self):
        self.issues = [
            {"number": 1, "title": "crash", "body": "", "is_pr": False},
            {"number": 2, "title": "please redesign this", "body": "", "is_pr": False},
            {"number": 3, "title": "closed as duplicate", "body": "", "is_pr": False},
            {"number": 4, "title": "a pull request", "body": "", "is_pr": True},
        ]

    def list_issues(self, owner, repo, limit=300, state="all"):
        return self.issues

    def issue_timeline_closers(self, owner, repo, number):
        return [99] if number == 1 else []

    def issue_state_reason(self, owner, repo, number):
        return "not_planned" if number == 2 else "completed"

    def pr_files(self, owner, repo, number, limit=100):
        return ["src/stats.py", "tests/test_stats.py"]


def test_build_eval_cases_discovers_both_kinds(db, indexed_repo):
    from app.eval.harness import build_eval_cases

    repo, _ = indexed_repo
    created = build_eval_cases(db, repo, _DiscoveryGH(), limit=5, wont_fix_limit=5)

    by_number = {c.issue_number: c for c in created}
    assert set(by_number) == {1, 2}  # the PR and the ambiguous close are skipped

    fixed = by_number[1]
    assert fixed.kind is m.EvalCaseKind.fixed
    assert fixed.gold_pr_number == 99
    assert fixed.gold_files == ["src/stats.py"]  # test files excluded from the gold set

    wont_fix = by_number[2]
    assert wont_fix.kind is m.EvalCaseKind.wont_fix
    assert wont_fix.gold_files == []
    assert wont_fix.gold_pr_number is None


def test_build_eval_cases_skips_closes_that_are_not_wont_fix(db, indexed_repo):
    """An issue closed as a duplicate tells us nothing about whether it was a defect."""
    from app.eval.harness import build_eval_cases

    repo, _ = indexed_repo
    created = build_eval_cases(db, repo, _DiscoveryGH(), limit=5, wont_fix_limit=5)
    assert 3 not in {c.issue_number for c in created}


def test_build_eval_cases_is_idempotent(db, indexed_repo):
    from app.eval.harness import build_eval_cases

    repo, _ = indexed_repo
    gh = _DiscoveryGH()
    first = build_eval_cases(db, repo, gh, limit=5, wont_fix_limit=5)
    second = build_eval_cases(db, repo, gh, limit=5, wont_fix_limit=5)
    assert len(first) == 2 and second == []


# ---------- ground truth: found by reading real eval cases ----------
class _MixedGH(_DiscoveryGH):
    """Closes that look like cases but are not: a docs-only fix and a process rejection."""

    def __init__(self):
        self.issues = [
            {"number": 10, "title": "docs fix", "body": "", "is_pr": False, "labels": []},
            {"number": 11, "title": "ai report", "body": "", "is_pr": False, "labels": ["rejected AI"]},
            {"number": 12, "title": "real wont fix", "body": "", "is_pr": False, "labels": ["wontfix"]},
        ]

    def issue_timeline_closers(self, owner, repo, number):
        return [77] if number == 10 else []

    def issue_state_reason(self, owner, repo, number):
        return "not_planned"

    def pr_files(self, owner, repo, number, limit=100):
        return ["docs/faqs.md", "CHANGES.rst"]


def test_a_docs_only_fix_is_not_a_defect_case(db, indexed_repo):
    """pallets/click#3887 was answered by an FAQ entry; scoring it as a missed bug is wrong."""
    from app.eval.harness import build_eval_cases

    repo, _ = indexed_repo
    created = build_eval_cases(db, repo, _MixedGH(), limit=5, wont_fix_limit=5)
    assert 10 not in {c.issue_number for c in created}


def test_a_process_rejection_is_not_a_wont_fix_case(db, indexed_repo):
    """pallets/click#3892 described a plausible crash and was closed as 'rejected AI'."""
    from app.eval.harness import build_eval_cases

    repo, _ = indexed_repo
    created = build_eval_cases(db, repo, _MixedGH(), limit=5, wont_fix_limit=5)
    assert {c.issue_number for c in created} == {12}  # the genuine wontfix survives


def _timeline_event(number, *, merged=True, repo="o/r", body="Fixes #5"):
    return {
        "event": "cross-referenced",
        "source": {
            "issue": {
                "number": number,
                "title": "a change",
                "body": body,
                "repository_url": f"https://api.github.com/repos/{repo}",
                "pull_request": {"merged_at": "2026-01-01T00:00:00Z" if merged else None},
            }
        },
    }


def _closers(events):
    from app.github.client import GitHubClient

    gh = GitHubClient("t")
    gh._get = lambda path, **kw: events  # type: ignore[method-assign]
    return gh.issue_timeline_closers("o", "r", 5)


def test_a_merged_pr_that_says_it_fixes_the_issue_counts():
    assert _closers([_timeline_event(9)]) == [9]


def test_a_pr_from_another_repository_does_not_count():
    """pallets/click#3846 'was fixed by PR 1', which was a different repository's PR 1."""
    assert _closers([_timeline_event(1, repo="someone/else")]) == []


def test_a_pr_that_merely_mentions_the_issue_does_not_count():
    assert _closers([_timeline_event(9, body="see also #5 for context")]) == []


def test_an_unmerged_pr_does_not_count():
    assert _closers([_timeline_event(9, merged=False)]) == []


def test_closing_keyword_for_a_different_issue_does_not_count():
    assert _closers([_timeline_event(9, body="Fixes #55")]) == []
