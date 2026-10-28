"""End to end: real index, real hybrid retrieval, real persistence; only the LLM is faked."""
import json

import pytest

from app import runner
from app.db import models as m

ISSUE_TITLE = "ZeroDivisionError when averaging an empty list"
ISSUE_BODY = "Calling average([]) raises ZeroDivisionError instead of returning 0."


class ScriptedLLM:
    """Answers by ROLE, citing whichever evidence the retriever actually returned."""

    def __init__(
        self,
        verdicts=("accept",),
        cite_real=True,
        affected=("src/stats.py",),
        is_defect="yes",
    ):
        self.verdicts = list(verdicts)
        self.cite_real = cite_real
        self.affected = list(affected)
        self.is_defect = is_defect
        self.seen_evidence_ids: list[str] = []

    def invoke_json(self, system, user):
        role = system.split("\n")[0].replace("ROLE:", "").strip()
        if role == "planner":
            return {
                "hypotheses": ["average() does not guard an empty list"],
                "queries": [{"q": "average divides by len", "sources": ["code"]}],
            }
        if role == "investigator":
            ids = [line.split("]")[0][1:] for line in user.splitlines() if line.startswith("[E")]
            self.seen_evidence_ids = ids
            return {
                "is_defect": self.is_defect,
                "verdict_reason": (
                    "documented to raise on empty input"
                    if self.is_defect == "no"
                    else "crashes on input the docs say is valid"
                ),
                "root_cause": "average() computes sum/len without checking for an empty list",
                "confidence": 0.85,
                "affected_files": self.affected,
                "reasoning": "the retrieved source shows an unguarded division",
                "evidence_ids": (ids[:1] if self.cite_real else ["E999"]),
            }
        if role == "fix":
            return {
                "fix": {
                    "file": "src/stats.py",
                    "diff": "@@\n-    return sum(values) / len(values)\n+    if not values:\n+        return 0\n+    return sum(values) / len(values)",
                    "explanation": "return 0 for an empty list",
                },
                "tests": [
                    {"name": "test_average_empty", "code": "assert average([]) == 0", "rationale": "covers the crash"}
                ],
            }
        if role == "critic":
            v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return {"verdict": v, "scores": {"groundedness": 0.9}, "feedback": "" if v == "accept" else "look at callers"}
        raise AssertionError(role)


@pytest.fixture
def queued(db, indexed_repo):
    repo, _ = indexed_repo
    issue = m.Issue(
        repo_id=repo.id, number=12, title=ISSUE_TITLE, body=ISSUE_BODY,
        state="open", labels=["bug"], html_url="https://github.com/acme/sample/issues/12",
    )
    db.add(issue)
    db.commit()
    inv = m.Investigation(repo_id=repo.id, issue_id=issue.id, max_iterations=3)
    db.add(inv)
    db.commit()
    return inv


def test_full_run_completes_and_persists_everything(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(), settings)

    assert inv.status is m.RunStatus.completed, inv.error
    assert inv.iterations == 1
    assert inv.confidence == pytest.approx(0.85)
    assert inv.index_version_id is not None

    steps = db.query(m.AgentStep).filter_by(investigation_id=inv.id).order_by(m.AgentStep.seq).all()
    assert [s.agent.value for s in steps] == ["planner", "investigator", "fix", "critic"]
    assert all(s.latency_ms is not None for s in steps)

    reviews = db.query(m.CriticReview).filter_by(investigation_id=inv.id).all()
    assert len(reviews) == 1 and reviews[0].verdict is m.Verdict.accept

    report = db.query(m.Report).filter_by(investigation_id=inv.id).one()
    assert "empty list" in report.root_cause
    assert report.affected_files == ["src/stats.py"]
    assert report.low_confidence is False

    assert db.query(m.FixSuggestion).filter_by(report_id=report.id).count() == 1
    assert db.query(m.TestSuggestion).filter_by(report_id=report.id).count() == 1


def test_evidence_rows_link_to_real_chunks_and_mark_citations(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(), settings)
    rows = db.query(m.Evidence).filter_by(investigation_id=inv.id).all()
    assert rows, "evidence should be recorded"
    assert all(db.get(m.Chunk, r.chunk_id) is not None for r in rows)
    assert sum(r.cited_in_report for r in rows) == 1
    assert [r.rank for r in rows] == sorted(r.rank for r in rows)


def test_retrieved_evidence_actually_contains_the_buggy_file(db, settings, queued):
    llm = ScriptedLLM()
    inv = runner.run(db, queued.id, llm, settings)
    refs = {
        db.get(m.Chunk, r.chunk_id).source_ref
        for r in db.query(m.Evidence).filter_by(investigation_id=inv.id)
    }
    assert "src/stats.py" in refs


def test_markdown_report_has_the_expected_sections(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(), settings)
    md = db.query(m.Report).filter_by(investigation_id=inv.id).one().report_md
    for heading in ("# Investigation:", "## Root cause", "## Evidence", "## Suggested fix", "## Suggested tests"):
        assert heading in md
    assert "```diff" in md
    assert "acme/sample#12" in md


def test_critic_revision_loops_and_records_each_pass(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(verdicts=("revise", "accept")), settings)
    assert inv.status is m.RunStatus.completed
    assert inv.iterations == 2
    assert db.query(m.CriticReview).filter_by(investigation_id=inv.id).count() == 2
    assert db.query(m.AgentStep).filter_by(investigation_id=inv.id).count() == 8  # 4 agents x 2 passes


def test_ungrounded_answer_is_forced_low_confidence(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(cite_real=False), settings)
    report = db.query(m.Report).filter_by(investigation_id=inv.id).one()
    assert report.low_confidence is True
    assert inv.iterations == inv.max_iterations  # never accepted
    assert all(r.verdict is m.Verdict.revise for r in db.query(m.CriticReview).filter_by(investigation_id=inv.id))


def test_missing_index_fails_the_run_with_a_clear_error(db, settings):
    repo = m.Repository(owner="acme", name="noindex", clone_url="x")
    db.add(repo)
    db.commit()
    issue = m.Issue(repo_id=repo.id, number=1, title="t", state="open", html_url="u")
    db.add(issue)
    db.commit()
    inv = m.Investigation(repo_id=repo.id, issue_id=issue.id)
    db.add(inv)
    db.commit()

    out = runner.run(db, inv.id, ScriptedLLM(), settings)
    assert out.status is m.RunStatus.failed
    assert "no ready index" in out.error
    assert out.finished_at is not None


def test_llm_failure_is_recorded_not_raised(db, settings, queued):
    class Boom:
        def invoke_json(self, system, user):
            raise RuntimeError("provider exploded")

    out = runner.run(db, queued.id, Boom(), settings)
    assert out.status is m.RunStatus.failed
    assert "provider exploded" in out.error


def test_agent_step_output_is_queryable_json(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(), settings)
    planner = db.query(m.AgentStep).filter_by(investigation_id=inv.id, agent=m.AgentName.planner).one()
    assert json.loads(json.dumps(planner.output))["hypotheses"]


# ---------- the defect verdict, end to end ----------
def test_non_defect_run_persists_a_verdict_and_no_fix(db, settings, queued):
    """The pallets/click#3887 failure: v1 would have written a patch for intended behaviour."""
    inv = runner.run(db, queued.id, ScriptedLLM(is_defect="no"), settings)

    assert inv.status is m.RunStatus.completed
    report = db.query(m.Report).filter_by(investigation_id=inv.id).one()
    assert report.is_defect is m.DefectVerdict.no
    assert report.verdict_reason
    assert db.query(m.FixSuggestion).filter_by(report_id=report.id).count() == 0
    assert db.query(m.TestSuggestion).filter_by(report_id=report.id).count() == 0

    steps = [s.agent.value for s in db.query(m.AgentStep).filter_by(investigation_id=inv.id)]
    assert "fix" not in steps

    # still explains the mechanism, and says plainly why no patch
    assert "## Why no fix is proposed" in report.report_md
    assert "Does not look like a defect" in report.report_md
    assert report.root_cause


def test_defect_run_still_records_a_fix(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(is_defect="yes"), settings)
    report = db.query(m.Report).filter_by(investigation_id=inv.id).one()
    assert report.is_defect is m.DefectVerdict.yes
    assert db.query(m.FixSuggestion).filter_by(report_id=report.id).count() == 1
    assert "## Root cause" in report.report_md


def test_unclear_verdict_is_reported_honestly(db, settings, queued):
    inv = runner.run(db, queued.id, ScriptedLLM(is_defect="unclear"), settings)
    report = db.query(m.Report).filter_by(investigation_id=inv.id).one()
    assert report.is_defect is m.DefectVerdict.unclear
    assert "Cannot tell from the evidence" in report.report_md
