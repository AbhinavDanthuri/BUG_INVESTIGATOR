"""Embeddings.

Three backends, chosen by the EMBEDDING_MODEL setting:

- `gemini:<model>`  — Google's embedding API. No local model, no torch, no download. Needs the
  same free API key the LLM uses. Pick this when a machine cannot run torch.
- `hashing`         — deterministic bag-of-tokens, no dependencies at all. Used by the test suite
  and offline smoke runs. Retrieval quality is poor; not for real investigations.
- anything else     — a sentence-transformers model name, run locally (best quality, needs torch).
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from functools import lru_cache

import numpy as np

HASHING_MODEL = "hashing"
GEMINI_PREFIX = "gemini:"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
log = logging.getLogger(__name__)


class Embedder:
    """`.dim` and `.encode(texts) -> (n, dim) float32, L2-normalized`."""

    model_name: str
    dim: int

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class HashingEmbedder(Embedder):
    """Hashed bag-of-tokens: deterministic, offline, numpy only."""

    def __init__(self, dim: int = 256):
        self.model_name = HASHING_MODEL
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in _TOKEN.findall(text):
            out.append(tok.lower())
            parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", tok) if p]
            if len(parts) > 1:
                out.extend(p.lower() for p in parts)
        return out

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = 2166136261  # FNV-1a: stable across processes, unlike hash()
                for ch in tok.encode("utf-8"):
                    h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
                mat[row, h % self.dim] += 1.0
        return _normalize(mat)


class QuotaExhausted(RuntimeError):
    """The API asked us to wait longer than any sane run should: a daily quota, not a burst."""


def estimate_tokens(text: str) -> int:
    """Rough token count. ~4 characters per token is the usual approximation for English + code."""
    return max(1, len(text) // 4)


_RETRY_AFTER = re.compile(r"retry(?:Delay|\s+in)['\":\s]*([0-9.]+)\s*s", re.IGNORECASE)


def parse_retry_delay(exc: Exception) -> float | None:
    """Gemini tells us exactly how long to wait ('Please retry in 51.1s'). Prefer it to guessing."""
    match = _RETRY_AFTER.search(str(exc))
    return float(match.group(1)) if match else None


class TokenBudget:
    """Sliding-window throttle: never let more than `limit` tokens go out in any 60s window.

    The free embedding tier is metered in tokens per minute, so waiting only after a 429 is too
    late — one oversized request can blow the whole minute. This paces requests before sending.
    """

    def __init__(self, limit: int, window: float = 60.0, sleep=time.sleep, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._sleep = sleep
        self._clock = clock
        self._spent: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> int:
        while self._spent and now - self._spent[0][0] >= self.window:
            self._spent.popleft()
        return sum(tokens for _, tokens in self._spent)

    MAX_WAITS = 4  # safety net: never spin forever if the clock does not advance

    def consume(self, tokens: int) -> None:
        """Block until `tokens` fit inside the window, then record them as spent."""
        for _ in range(self.MAX_WAITS):
            now = self._clock()
            used = self._prune(now)
            if not self._spent or used + tokens <= self.limit:
                self._spent.append((now, tokens))
                return
            oldest = self._spent[0][0]
            self._sleep(max(0.1, self.window - (now - oldest) + 0.5))

        # Waited the full window and the budget still looks full: the recorded spend must be
        # stale (a clock that did not move, or a penalty longer than the window). Reset and go,
        # rather than hanging; a real 429 is retried by the caller anyway.
        self._spent.clear()
        self._spent.append((self._clock(), tokens))

    def penalize(self, seconds: float) -> None:
        """After a 429, treat the window as full for `seconds` so we stop pushing."""
        self._spent.append((self._clock() + seconds - self.window, self.limit))


def batch_by_tokens(texts: list[str], max_tokens: int, max_items: int) -> list[list[int]]:
    """Group indices into requests small enough to fit the per-minute budget.

    Batching by item count is what broke: 64 chunks of code can be 100k+ tokens, far past a
    30k/minute quota, so every request 429'd no matter how long we waited.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, text in enumerate(texts):
        tokens = estimate_tokens(text)
        if current and (current_tokens + tokens > max_tokens or len(current) >= max_items):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


class GeminiEmbedder(Embedder):
    """Google's embedding API, paced to stay inside the free tier's tokens-per-minute quota."""

    MAX_CHARS = 8000  # per chunk; longer text is truncated before embedding

    def __init__(
        self,
        model_name: str,
        api_key: str,
        tokens_per_minute: int = 25_000,  # under the 30k free-tier limit, with headroom
        max_tokens_per_request: int = 4_000,
        max_items_per_request: int = 64,
        max_retries: int = 6,
        max_wait_seconds: float = 300.0,  # a longer delay means a daily quota; stop, do not nap
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for gemini embeddings")
        self.model_name = model_name
        self._max_tokens_per_request = max_tokens_per_request
        self._max_items_per_request = max_items_per_request
        self._max_retries = max_retries
        self._max_wait_seconds = max_wait_seconds
        self._sleep = sleep
        self._budget = TokenBudget(tokens_per_minute, sleep=sleep, clock=clock)
        self._client = GoogleGenerativeAIEmbeddings(
            model=model_name.removeprefix(GEMINI_PREFIX), google_api_key=api_key
        )
        self.dim = len(self._request(["dimension probe"])[0])

    def _request(self, batch: list[str]) -> list[list[float]]:
        from app.llm import is_retryable

        tokens = sum(estimate_tokens(t) for t in batch)
        for attempt in range(self._max_retries + 1):
            self._budget.consume(tokens)
            try:
                return self._client.embed_documents(batch)
            except Exception as exc:  # noqa: BLE001
                if attempt == self._max_retries or not is_retryable(exc):
                    raise
                # The API states how long to wait; honour it instead of a blind doubling.
                delay = parse_retry_delay(exc) or min(60.0, 4.0 * 2**attempt)
                if delay > self._max_wait_seconds:
                    raise QuotaExhausted(
                        f"the embedding API wants a {delay / 3600:.1f} hour wait, which means the "
                        f"daily free-tier quota is used up, not a short burst limit. Options: wait "
                        f"for the quota to reset, use a different API key, or set "
                        f"EMBEDDING_MODEL to a local sentence-transformers model or to 'hashing'."
                    ) from exc
                self._budget.penalize(delay)
                log.warning("embedding rate-limited, waiting %.0fs", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        clipped = [t[: self.MAX_CHARS] for t in texts]
        batches = batch_by_tokens(
            clipped,
            self._max_tokens_per_request,
            batch_size or self._max_items_per_request,
        )
        total_tokens = sum(estimate_tokens(t) for t in clipped)
        log.info(
            "embedding %d chunks in %d requests (~%d tokens, roughly %.0f min at the free-tier rate)",
            len(clipped),
            len(batches),
            total_tokens,
            total_tokens / max(1, self._budget.limit),
        )

        out = np.zeros((len(clipped), self.dim), dtype="float32")
        for n, indices in enumerate(batches, start=1):
            vectors = self._request([clipped[i] for i in indices])
            for i, vec in zip(indices, vectors, strict=True):
                out[i] = vec
            if n % 10 == 0 or n == len(batches):
                log.info("  embedded %d/%d requests", n, len(batches))
        return _normalize(out)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")


@lru_cache(maxsize=2)
def get_embedder(model_name: str, api_key: str = "") -> Embedder:
    """Cached: loading a local model, or probing the API for its dimension, is slow."""
    if model_name == HASHING_MODEL:
        return HashingEmbedder()
    if model_name.startswith(GEMINI_PREFIX):
        return GeminiEmbedder(model_name, api_key)
    return SentenceTransformerEmbedder(model_name)