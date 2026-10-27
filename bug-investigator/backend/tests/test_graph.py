"""Graph logic tests with a scripted fake LLM and fake retriever (no network, no API key)."""
from app.agents.graph import run_investigation
from app.retrieval.base import Evidence

ISSUE = {"number": 7, "title": "Crash on empty list", "body": "calling total([]) raises IndexError", "labels": ["bug"]}


class FakeRetriever:
    def search(self, query, source_types=None, k=8):
        return [
            Evidence(id="E1", source_type="code", ref="src/math.py", text="def total(xs): return xs[0]", start_line=1, end_line=2),
            Evidence(id="E2", source_type="issue", ref="#3", text="similar crash earlier"),
        ]


class FakeLLM:
    """Replies by ROLE tag. `critic_verdicts` is consumed one per critic call (last one repeats)."""

    def __init__(self, critic_verdicts, cited=("E1",)):
        self.critic_verdicts = list(critic_verdicts)
        self.cited = list(cited)
        self.calls = []

    def invoke_json(self, system, user):
        role = system.split("\n")[0].replace("ROLE:", "").strip()
        self.calls.append(role)
        if role == "planner":
            return {"hypotheses": ["empty list not handled"], "queries": [{"q": "total empty list", "sources": ["code"]}]}
        if role == "investigator":
            return {
                "root_cause": "total() indexes xs[0] without checking for an empty list",
                "confidence": 0.8,
                "affected_files": ["src/math.py"],
                "reasoning": "E1 shows xs[0]",
                "evidence_ids": self.cited + ["E999"],  # E999 was never retrieved
            }
        if role == "fix":
            return {"fix": {"file": "src/math.py", "diff": "-...", "explanation": "guard empty"}, "tests": [{"name": "test_empty", "code": "assert total([]) == 0", "rationale": "covers crash"}]}
        if role == "critic":
            verdict = self.critic_verdicts.pop(0) if len(self.critic_verdicts) > 1 else self.critic_verdicts[0]
            return {"verdict": verdict, "scores": {"groundedness": 0.9}, "feedback": "search for callers" if verdict == "revise" else ""}
        raise AssertionError(role)


def test_critic_revise_then_accept_loops_once():
    llm = FakeLLM(["revise", "accept"])
    out = run_investigation(llm, FakeRetriever(), ISSUE, max_iterations=3)
    assert llm.calls == ["planner", "investigator", "fix", "critic"] * 2
    assert out["iteration"] == 2
    assert out["report"]["low_confidence"] is False
    assert [t["agent"] for t in out["trace"]] == ["planner", "investigator", "fix", "critic"] * 2


def test_always_revise_stops_at_max_iterations_with_low_confidence():
    llm = FakeLLM(["revise"])
    out = run_investigation(llm, FakeRetriever(), ISSUE, max_iterations=2)
    assert out["iteration"] == 2
    assert out["report"]["low_confidence"] is True


def test_uncited_evidence_is_dropped_and_report_cites_only_retrieved():
    llm = FakeLLM(["accept"])
    out = run_investigation(llm, FakeRetriever(), ISSUE)
    assert out["analysis"]["evidence_ids"] == ["E1"]
    assert [e["id"] for e in out["report"]["cited_evidence"]] == ["E1"]


def test_no_valid_citation_forces_revise_even_if_llm_accepts():
    llm = FakeLLM(["accept"], cited=())  # only the hallucinated E999 gets cited -> filtered out
    out = run_investigation(llm, FakeRetriever(), ISSUE, max_iterations=2)
    assert out["iteration"] == 2
    assert out["report"]["low_confidence"] is True
