import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import models as m
from app.github.client import parse_issue_url
from app.llm import extract_json
from app.retrieval.fusion import rrf_fuse


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = [i for i, _ in rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])]
    assert fused[:2] == ["a", "b"] or fused[:2] == ["b", "a"]
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused.index("c") > fused.index("a") and fused.index("d") > fused.index("b")


def test_rrf_item_in_one_list_only_still_returned():
    assert [i for i, _ in rrf_fuse([["x"], []])] == ["x"]


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go:\n{"a": 1}\nHope that helps'],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


class _Reply:
    def __init__(self, content):
        self.content = content


class _FlakyModel:
    """Raises a 429-style error `fails` times, then returns JSON."""

    def __init__(self, fails, exc=RuntimeError("429 RESOURCE_EXHAUSTED")):
        self.fails, self.exc, self.calls = fails, exc, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return _Reply('{"ok": true}')


def test_llm_retries_rate_limit_then_succeeds():
    from app.llm import LangChainLLM

    delays = []
    model = _FlakyModel(fails=2)
    llm = LangChainLLM(model, base_delay=1.0, sleep=delays.append)
    assert llm.invoke_json("s", "u") == {"ok": True}
    assert model.calls == 3 and delays == [1.0, 2.0]


def test_llm_does_not_retry_other_errors():
    from app.llm import LangChainLLM

    model = _FlakyModel(fails=5, exc=ValueError("bad request"))
    with pytest.raises(ValueError):
        LangChainLLM(model, sleep=lambda s: None).invoke_json("s", "u")
    assert model.calls == 1


def test_parse_issue_url():
    ref = parse_issue_url("https://github.com/psf/requests/issues/6710#issuecomment-1")
    assert (ref.owner, ref.repo, ref.number) == ("psf", "requests", 6710)
    with pytest.raises(ValueError):
        parse_issue_url("https://github.com/psf/requests/pull/1")


def test_models_create_and_relate_on_sqlite():
    engine = create_engine("sqlite://")
    m.Base.metadata.create_all(engine)
    with Session(engine) as s:
        repo = m.Repository(owner="o", name="r", clone_url="https://github.com/o/r.git")
        s.add(repo)
        s.flush()
        issue = m.Issue(repo_id=repo.id, number=1, title="t", state="open", html_url="u")
        s.add(issue)
        s.flush()
        inv = m.Investigation(repo_id=repo.id, issue_id=issue.id)
        s.add(inv)
        s.flush()
        s.add(m.AgentStep(investigation_id=inv.id, iteration=0, seq=1, agent=m.AgentName.planner, output={"k": 1}))
        s.add(m.Report(investigation_id=inv.id, root_cause="x", confidence=0.5, report_md="# r"))
        s.commit()
        got = s.scalar(select(m.Investigation))
        assert got.status == m.RunStatus.queued
        assert got.steps[0].output == {"k": 1}
        assert got.report.root_cause == "x"
