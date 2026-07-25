"""Central helper for logging LLM calls to the monitoring store."""

from __future__ import annotations

import time
from typing import Optional


def log_llm_call(
    *,
    backend: str,
    model: str,
    prompt: str,
    response: str = "",
    success: bool = True,
    error: Optional[str] = None,
    context: str = "rollout",
    agent: str = "unknown",
    latency_ms: Optional[float] = None,
) -> None:
    from monitoring.store import get_store

    get_store().record_llm_call(
        backend=backend,
        model=model,
        latency_ms=latency_ms or 0.0,
        prompt_chars=len(prompt),
        response_chars=len(response),
        success=success,
        error=error,
        context=context,
        agent=agent,
    )


class LLMCallTimer:
    """Context manager to time and log an LLM call."""

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        prompt: str,
        context: str = "rollout",
        agent: str = "unknown",
    ) -> None:
        self.backend = backend
        self.model = model
        self.prompt = prompt
        self.context = context
        self.agent = agent
        self._t0 = 0.0
        self.response = ""

    def __enter__(self) -> "LLMCallTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        latency_ms = (time.perf_counter() - self._t0) * 1000
        log_llm_call(
            backend=self.backend,
            model=self.model,
            prompt=self.prompt,
            response=self.response,
            success=exc is None,
            error=str(exc) if exc else None,
            context=self.context,
            agent=self.agent,
            latency_ms=latency_ms,
        )
