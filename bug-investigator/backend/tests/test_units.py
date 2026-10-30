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


def test_env_file_is_found_from_any_working_directory(tmp_path, monkeypatch):
    """`.env` lives at the project root but commands run from backend/ — both must work."""
    from app.config import _BACKEND_DIR, _PROJECT_ROOT, Settings

    env_files = Settings.model_config["env_file"]
    assert _PROJECT_ROOT / ".env" in env_files
    assert _BACKEND_DIR / ".env" in env_files

    monkeypatch.chdir(tmp_path)  # a directory with no .env at all
    monkeypatch.setenv("GITHUB_TOKEN", "from-the-environment")
    assert Settings().github_token == "from-the-environment"


# ---------- embedder backends ----------
def test_hashing_embedder_is_deterministic_and_normalized():
    import numpy as np

    from app.retrieval.embedder import get_embedder

    e = get_embedder("hashing")
    a = e.encode(["def average(values): return sum(values) / len(values)"])
    b = e.encode(["def average(values): return sum(values) / len(values)"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hashing_embedder_ranks_related_text_above_unrelated():
    from app.retrieval.embedder import get_embedder

    e = get_embedder("hashing")
    v = e.encode(["average divides by len", "average of an empty list", "socket connect timeout"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_gemini_embedder_requires_an_api_key():
    from app.retrieval.embedder import GeminiEmbedder

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiEmbedder("gemini:models/text-embedding-004", api_key="")


def test_estimate_tokens_scales_with_length():
    from app.retrieval.embedder import estimate_tokens

    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 100


def test_batch_by_tokens_splits_on_the_token_budget_not_item_count():
    """The bug this fixes: 64 big chunks in one request blew a 30k-tokens-per-minute quota."""
    from app.retrieval.embedder import batch_by_tokens

    big = ["x" * 4000] * 5  # ~1000 tokens each
    batches = batch_by_tokens(big, max_tokens=2500, max_items=64)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [i for b in batches for i in b] == [0, 1, 2, 3, 4]  # nothing lost or reordered


def test_batch_by_tokens_still_honours_an_item_cap():
    from app.retrieval.embedder import batch_by_tokens

    assert [len(b) for b in batch_by_tokens(["a"] * 7, max_tokens=10_000, max_items=3)] == [3, 3, 1]


def test_batch_by_tokens_never_drops_an_oversized_single_item():
    from app.retrieval.embedder import batch_by_tokens

    assert batch_by_tokens(["x" * 100_000], max_tokens=100, max_items=64) == [[0]]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("429 please retry in 51.1s", 51.1),
        ("{'retryDelay': '51s'}", 51.0),
        ("no delay mentioned", None),
    ],
)
def test_parse_retry_delay(message, expected):
    from app.retrieval.embedder import parse_retry_delay

    assert parse_retry_delay(RuntimeError(message)) == expected


def test_token_budget_waits_once_the_window_is_full():
    from app.retrieval.embedder import TokenBudget

    now = [0.0]
    slept: list[float] = []

    def clock():
        return now[0]

    def sleep(sec):
        slept.append(sec)
        now[0] += sec

    budget = TokenBudget(limit=1000, sleep=sleep, clock=clock)
    budget.consume(600)
    assert slept == []          # fits
    budget.consume(300)
    assert slept == []          # still fits (900 of 1000)
    budget.consume(400)         # would exceed; must wait for the window to roll
    assert slept and slept[0] > 60 - 1


def test_token_budget_penalty_blocks_further_sends():
    from app.retrieval.embedder import TokenBudget

    now, slept = [0.0], []

    def sleep(sec):
        slept.append(sec)
        now[0] += sec

    budget = TokenBudget(limit=1000, sleep=sleep, clock=lambda: now[0])
    budget.penalize(30.0)
    budget.consume(10)
    assert slept, "a penalty should force a wait before the next request"


class _FakeEmbedClient:
    def __init__(self, fail_times=0, delay_message="429 please retry in 2s", dim=3):
        self.fail_times = fail_times
        self.delay_message = delay_message
        self.dim = dim
        self.batch_sizes: list[int] = []

    def embed_documents(self, batch):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.delay_message)
        self.batch_sizes.append(len(batch))
        return [[1.0] + [0.0] * (self.dim - 1) for _ in batch]


def _install_fake_client(monkeypatch, client):
    import sys

    class FakeModule:
        GoogleGenerativeAIEmbeddings = staticmethod(lambda **kw: client)

    monkeypatch.setitem(sys.modules, "langchain_google_genai", FakeModule)


class _FakeClock:
    """A clock that only moves when sleep() is called, so throttling is testable instantly."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def test_gemini_embedder_discovers_dimension_and_normalizes(monkeypatch):
    import numpy as np

    from app.retrieval.embedder import GeminiEmbedder

    client = _FakeEmbedClient(dim=4)
    _install_fake_client(monkeypatch, client)

    clock = _FakeClock()
    e = GeminiEmbedder("gemini:gemini-embedding-001", api_key="k", sleep=clock.sleep, clock=clock)
    assert e.dim == 4  # read from the API, not hardcoded

    out = e.encode(["a", "b", "c"])
    assert out.shape == (3, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_gemini_embedder_keeps_requests_under_the_token_budget(monkeypatch):
    from app.retrieval.embedder import GeminiEmbedder

    client = _FakeEmbedClient()
    _install_fake_client(monkeypatch, client)

    clock = _FakeClock()
    e = GeminiEmbedder(
        "gemini:gemini-embedding-001",
        api_key="k",
        max_tokens_per_request=1000,
        sleep=clock.sleep,
        clock=clock,
    )
    client.batch_sizes.clear()
    e.encode(["x" * 2000] * 6)  # ~500 tokens each -> 2 per request
    assert client.batch_sizes == [2, 2, 2]


def test_gemini_embedder_retries_rate_limits_and_preserves_order(monkeypatch):
    from app.retrieval.embedder import GeminiEmbedder

    client = _FakeEmbedClient()
    _install_fake_client(monkeypatch, client)
    clock = _FakeClock()
    e = GeminiEmbedder("gemini:gemini-embedding-001", api_key="k", sleep=clock.sleep, clock=clock)

    client.fail_times = 2
    out = e.encode(["a", "b"])
    assert out.shape == (2, 3)
    assert 2.0 in clock.slept  # waited the delay the API asked for


def test_gemini_embedder_gives_up_on_non_rate_limit_errors(monkeypatch):
    from app.retrieval.embedder import GeminiEmbedder

    client = _FakeEmbedClient()
    _install_fake_client(monkeypatch, client)
    clock = _FakeClock()
    e = GeminiEmbedder("gemini:gemini-embedding-001", api_key="k", sleep=clock.sleep, clock=clock)

    client.fail_times = 1
    client.delay_message = "400 invalid request"
    with pytest.raises(RuntimeError, match="invalid request"):
        e.encode(["a"])


def test_get_embedder_dispatches_on_the_model_name(monkeypatch):
    from app.retrieval.embedder import HashingEmbedder, get_embedder

    get_embedder.cache_clear()
    assert isinstance(get_embedder("hashing"), HashingEmbedder)


def test_token_budget_gives_up_waiting_if_the_clock_never_moves():
    """A sleep that does not advance time must not hang the indexer forever."""
    from app.retrieval.embedder import TokenBudget

    slept: list[float] = []
    budget = TokenBudget(limit=100, sleep=slept.append, clock=lambda: 0.0)
    budget.consume(100)
    budget.consume(100)  # would block forever without the bounded retry
    assert len(slept) == TokenBudget.MAX_WAITS


def test_gemini_embedder_refuses_an_absurd_retry_delay(monkeypatch):
    """A 22-hour 'retry in' means the daily quota is gone: fail loudly, never sleep through it."""
    from app.retrieval.embedder import GeminiEmbedder, QuotaExhausted

    client = _FakeEmbedClient()
    _install_fake_client(monkeypatch, client)
    clock = _FakeClock()
    e = GeminiEmbedder("gemini:gemini-embedding-001", api_key="k", sleep=clock.sleep, clock=clock)

    client.fail_times = 1
    client.delay_message = "429 RESOURCE_EXHAUSTED. Please retry in 81807s"
    with pytest.raises(QuotaExhausted, match="daily free-tier quota"):
        e.encode(["a"])
    assert not any(x > 300 for x in clock.slept), "must not sleep for hours"


def test_llm_logs_and_honours_the_api_retry_delay(caplog):
    import logging as _logging

    from app.llm import LangChainLLM

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 please retry in 7s")
            return _Reply('{"ok": true}')

    slept: list[float] = []
    with caplog.at_level(_logging.WARNING):
        out = LangChainLLM(Model(), sleep=slept.append).invoke_json("s", "u")
    assert out == {"ok": True}
    assert slept == [7.0]  # the delay the API asked for, not 4s
    assert "rate-limited" in caplog.text  # a retry must never look like a hang


def test_llm_refuses_a_daily_quota_wait():
    from app.llm import LangChainLLM, QuotaExhausted

    class Model:
        def invoke(self, messages):
            raise RuntimeError("429 RESOURCE_EXHAUSTED retry in 81807s")

    slept: list[float] = []
    with pytest.raises(QuotaExhausted, match="daily"):
        LangChainLLM(Model(), sleep=slept.append).invoke_json("s", "u")
    assert slept == []


@pytest.mark.parametrize(
    "message",
    [
        "503 UNAVAILABLE. This model is currently experiencing high demand.",
        "500 internal error",
        "ServerError: 502 Bad Gateway",
        "read timed out",
        "connection reset by peer",
    ],
)
def test_transient_server_errors_are_retried(message):
    """A busy provider is not a bug in our request: retry rather than failing the whole run."""
    from app.llm import LangChainLLM, is_retryable

    assert is_retryable(RuntimeError(message))

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(message)
            return _Reply('{"ok": true}')

    model = Model()
    assert LangChainLLM(model, sleep=lambda s: None).invoke_json("s", "u") == {"ok": True}
    assert model.calls == 2


def test_a_genuine_client_error_is_not_retried():
    from app.llm import LangChainLLM, is_retryable

    assert not is_retryable(ValueError("400 invalid argument"))

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise ValueError("400 invalid argument")

    model = Model()
    with pytest.raises(ValueError):
        LangChainLLM(model, sleep=lambda s: None).invoke_json("s", "u")
    assert model.calls == 1


def test_groq_provider_is_wired(monkeypatch):
    """Switching provider must not require touching agent code."""
    import sys

    from app.config import Settings
    from app.llm import LangChainLLM, get_llm

    captured = {}

    class FakeChatGroq:
        def __init__(self, **kw):
            captured.update(kw)

    class FakeModule:
        ChatGroq = FakeChatGroq

    monkeypatch.setitem(sys.modules, "langchain_groq", FakeModule)
    llm = get_llm(Settings(llm_provider="groq", llm_model="some-model", groq_api_key="k"))

    assert isinstance(llm, LangChainLLM)
    assert captured["model"] == "some-model"
    assert captured["api_key"] == "k"
    assert captured["temperature"] == 0


def test_unknown_provider_fails_clearly():
    from app.config import Settings
    from app.llm import get_llm

    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_llm(Settings(llm_provider="nonesuch"))


def test_fastembed_backend_is_selected_by_prefix(monkeypatch):
    """torch would not load on the development machine (Application Control blocks the DLL), which
    is why the index was built with `hashing` and BM25 ended up doing all the retrieval. ONNX
    Runtime serves the same weights without torch."""
    import sys
    import types

    import numpy as np

    from app.retrieval.embedder import FASTEMBED_PREFIX, FastEmbedEmbedder, get_embedder

    asked: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, name):
            asked.append(name)

        def embed(self, texts, batch_size=32):
            for i, _ in enumerate(texts):
                yield np.array([float(i + 1), 0.0, 0.0], dtype="float32") * 5  # unnormalized

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    get_embedder.cache_clear()

    emb = get_embedder(f"{FASTEMBED_PREFIX}BAAI/bge-small-en-v1.5")
    assert isinstance(emb, FastEmbedEmbedder)
    assert asked == ["BAAI/bge-small-en-v1.5"]  # the prefix is stripped before the model is loaded
    assert emb.dim == 3

    out = emb.encode(["a", "b"])
    assert out.shape == (2, 3) and out.dtype == np.dtype("float32")
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)  # normalized for the cosine index
    assert emb.encode([]).shape == (0, 3)
    get_embedder.cache_clear()
