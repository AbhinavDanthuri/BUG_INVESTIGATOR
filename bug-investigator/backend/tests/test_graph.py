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

    def __init__(self, critic_verdicts, cited=("E1",), is_defect="yes"):
        self.critic_verdicts = list(critic_verdicts)
        self.cited = list(cited)
        self.is_defect = is_defect
        self.calls = []

    def invoke_json(self, system, user):
        role = system.split("\n")[0].replace("ROLE:", "").strip()
        self.calls.append(role)
        if role == "planner":
            return {"hypotheses": ["empty list not handled"], "queries": [{"q": "total empty list", "sources": ["code"]}]}
        if role == "investigator":
            return {
                "is_defect": self.is_defect,
                "verdict_reason": "documented to raise" if self.is_defect == "no" else "crashes on valid input",
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


# ---------- the defect verdict (v2) ----------
def test_non_defect_skips_the_fix_agent_entirely():
    """v1 always produced a patch. Patching intended behaviour is the failure this prevents."""
    llm = FakeLLM(["accept"], is_defect="no")
    out = run_investigation(llm, FakeRetriever(), ISSUE)

    assert "fix" not in llm.calls
    assert llm.calls == ["planner", "investigator", "critic"]
    assert out["report"]["is_defect"] == "no"
    assert out["report"]["fix"] == {}
    assert out["report"]["tests"] == []
    assert out["report"]["verdict_reason"]


def test_unclear_verdict_also_skips_the_fix():
    llm = FakeLLM(["accept"], is_defect="unclear")
    out = run_investigation(llm, FakeRetriever(), ISSUE)
    assert "fix" not in llm.calls
    assert out["report"]["is_defect"] == "unclear"


def test_defect_verdict_still_produces_a_fix():
    llm = FakeLLM(["accept"], is_defect="yes")
    out = run_investigation(llm, FakeRetriever(), ISSUE)
    assert "fix" in llm.calls
    assert out["report"]["fix"]["file"] == "src/math.py"
    assert out["report"]["tests"]


def test_an_unrecognised_verdict_defaults_to_unclear_not_defect():
    """Defaulting a malformed verdict to 'yes' would reintroduce the exact bias we removed."""
    llm = FakeLLM(["accept"], is_defect="probably?")
    out = run_investigation(llm, FakeRetriever(), ISSUE)
    assert out["analysis"]["is_defect"] == "unclear"
    assert "fix" not in llm.calls


def test_report_still_explains_the_mechanism_when_not_a_defect():
    """A 'no' verdict should still be useful: say what is happening and why."""
    llm = FakeLLM(["accept"], is_defect="no")
    out = run_investigation(llm, FakeRetriever(), ISSUE)
    assert out["report"]["root_cause"]
    assert out["report"]["cited_evidence"]


# ---------- direct lookups ----------
class RecordingRetriever(FakeRetriever):
    """Tracks what was searched versus looked up directly."""

    def __init__(self):
        self.searches: list[str] = []
        self.lookups: list[tuple[str, int | None]] = []

    def search(self, query, source_types=None, k=8):
        self.searches.append(query)
        return super().search(query, source_types, k)

    def fetch_file(self, path, line=None, span=40):
        self.lookups.append((path, line))
        return [
            Evidence(
                id="L1",
                source_type="code",
                ref=path,
                text="def total(xs): return xs[0]",
                start_line=line or 1,
                end_line=(line or 1) + 5,
                score=1.0,
            )
        ]


ISSUE_WITH_LOCATION = {
    "number": 8,
    "title": "IndexError in total()",
    "body": "The bug is at src/math.py:42 - it indexes without a guard.",
    "labels": ["bug"],
}


def test_a_location_named_in_the_issue_is_looked_up_not_searched_for():
    """pallets/click#3832: the issue named core.py:1391 and the retriever never found it."""
    retriever = RecordingRetriever()
    run_investigation(FakeLLM(["accept"]), retriever, ISSUE_WITH_LOCATION)

    assert ("src/math.py", 42) in retriever.lookups


def test_looked_up_evidence_reaches_the_investigator():
    retriever = RecordingRetriever()
    out = run_investigation(FakeLLM(["accept"]), retriever, ISSUE_WITH_LOCATION)
    assert any(e["id"] == "L1" for e in out["evidence"])


def test_the_step_log_separates_lookups_from_searches():
    """Worth seeing in the timeline: was this found, or did we already know where to look?"""
    retriever = RecordingRetriever()
    out = run_investigation(FakeLLM(["accept"]), retriever, ISSUE_WITH_LOCATION)
    step = next(t for t in out["trace"] if t["agent"] == "investigator")
    assert step["output"]["looked_up"] >= 1


def test_an_issue_with_no_location_still_searches():
    retriever = RecordingRetriever()
    run_investigation(FakeLLM(["accept"]), retriever, ISSUE)  # no path in the body
    assert retriever.lookups == []
    assert retriever.searches


def test_a_failing_lookup_does_not_sink_the_run():
    """A stale or malformed path is common in old issues; it must not abort the investigation."""

    class BrokenLookup(FakeRetriever):
        def fetch_file(self, path, line=None, span=40):
            raise RuntimeError("index is confused")

    out = run_investigation(FakeLLM(["accept"]), BrokenLookup(), ISSUE_WITH_LOCATION)
    assert out["report"]["root_cause"]  # search still carried the run


def test_lookups_cannot_starve_search():
    """A flood of lookups must leave room for search results.

    The regression this guards: three files were looked up, four chunks each, filling all twelve
    evidence slots with import blocks. Search contributed nothing and the verdict collapsed.
    """
    from app.agents.nodes import MAX_LOOKUP_EVIDENCE

    class FloodingRetriever(FakeRetriever):
        def __init__(self):
            self.searched = False

        def search(self, query, source_types=None, k=8):
            self.searched = True
            return super().search(query, source_types, k)

        def fetch_file(self, path, line=None, span=40):
            return [
                Evidence(id=f"L{path}{i}", source_type="code", ref=path, text="import os",
                         start_line=i, end_line=i + 10, score=1.0)
                for i in range(20)
            ]

    retriever = FloodingRetriever()
    body = "see a.py:1 and b.py:2 and c.py:3 and d.py:4 and e.py:5"
    out = run_investigation(
        FakeLLM(["accept"]), retriever, {**ISSUE, "body": body}
    )

    step = next(t for t in out["trace"] if t["agent"] == "investigator")
    assert step["output"]["looked_up"] <= MAX_LOOKUP_EVIDENCE
    assert retriever.searched
    assert any(e["id"] in ("E1", "E2") for e in out["evidence"])  # search still got through
