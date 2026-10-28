"""Provider-agnostic LLM access. Agents only depend on `LLM.invoke_json`."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

from app.config import Settings


class LLM(Protocol):
    def invoke_json(self, system: str, user: str) -> dict:
        """Call the model and return a parsed JSON object."""
        ...


log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRY_AFTER = re.compile(r"retry(?:Delay|\s+in)['\":\s]*([0-9.]+)\s*s", re.IGNORECASE)


def parse_retry_delay(exc: Exception) -> float | None:
    """Providers often state the wait ('Please retry in 51.1s'); prefer it to a blind doubling."""
    match = _RETRY_AFTER.search(str(exc))
    return float(match.group(1)) if match else None


def extract_json(text: str) -> dict:
    """Parse a JSON object out of model output (handles code fences and surrounding prose)."""
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output") from None
        return json.loads(text[start : end + 1])


def is_rate_limit(exc: Exception) -> bool:
    """True for 429 / quota errors from any provider. Shared with the embedder."""
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in ("429", "rate limit", "ratelimit", "resource_exhausted", "quota"))


def is_transient_server_error(exc: Exception) -> bool:
    """True for 5xx / overload / timeout: the provider is busy, not us. Worth another try."""
    s = f"{type(exc).__name__} {exc}".lower()
    return any(
        t in s
        for t in ("503", "500", "502", "504", "unavailable", "overloaded", "high demand",
                  "internal error", "timeout", "timed out", "connection reset")
    )


def is_retryable(exc: Exception) -> bool:
    return is_rate_limit(exc) or is_transient_server_error(exc)


class QuotaExhausted(RuntimeError):
    """The API asked for a wait longer than any sane run: a daily quota, not a burst limit."""


class LangChainLLM:
    """Wraps a LangChain chat model. Retries rate limits with backoff (free tiers hit these often),
    says so in the log rather than going silent, and gives invalid JSON one repair retry."""

    def __init__(
        self,
        chat_model,
        max_retries: int = 5,
        base_delay: float = 4.0,
        max_wait_seconds: float = 300.0,
        sleep=time.sleep,
    ):
        self._model = chat_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_wait_seconds = max_wait_seconds
        self._sleep = sleep

    def _call(self, messages):
        for attempt in range(self._max_retries + 1):
            try:
                return self._model.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                if attempt == self._max_retries or not is_retryable(exc):
                    raise
                delay = parse_retry_delay(exc) or self._base_delay * 2**attempt
                if delay > self._max_wait_seconds:
                    raise QuotaExhausted(
                        f"the model API wants a {delay / 3600:.1f} hour wait, so the daily "
                        f"free-tier quota is used up. Wait for the reset or use another key."
                    ) from exc
                # Without this line a retry looks like a hang.
                reason = "rate-limited" if is_rate_limit(exc) else "unavailable"
                log.warning(
                    "model %s (attempt %d/%d), waiting %.0fs",
                    reason,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)

    def invoke_json(self, system: str, user: str) -> dict:
        messages = [("system", system), ("human", user)]
        reply = self._call(messages)
        try:
            return extract_json(_content(reply))
        except ValueError:
            messages += [
                ("ai", _content(reply)),
                ("human", "Your reply was not valid JSON. Reply again with only the JSON object."),
            ]
            return extract_json(_content(self._call(messages)))


def _content(reply) -> str:
    c = reply.content
    if isinstance(c, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.google_api_key, temperature=0)
    elif provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=settings.llm_model, api_key=settings.anthropic_api_key, temperature=0, max_tokens=4096
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)
    else:
        raise ValueError(f"unknown LLM_PROVIDER: {provider}")
    return LangChainLLM(model)
