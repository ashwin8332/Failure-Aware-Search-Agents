"""Ollama HTTP API backend for local CPU/GPU inference."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaLLM:
    """Call a local Ollama model via /api/generate (async-friendly)."""

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.timeout = timeout
        logger.info("Using Ollama model %s at %s", self.model, self.host)

    async def _apost(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Async POST so FastAPI event loop is never blocked."""
        url = f"{self.host}/api/generate"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _post_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sync fallback used by offline scripts (not the web server)."""
        url = f"{self.host}/api/generate"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def agenerate(
        self,
        prompt: str,
        temperature: float = 1.0,
        *,
        context: str = "live",
        agent: str = "unknown",
    ) -> str:
        """Async generate — use this from FastAPI/async callers."""
        from shared.llm_logging import log_llm_call

        t0 = time.perf_counter()
        try:
            data = await self._apost({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            })
            text = str(data.get("response", "")).strip()
            latency_ms = (time.perf_counter() - t0) * 1000
            log_llm_call(
                backend="ollama", model=self.model, prompt=prompt,
                response=text, success=True, context=context, agent=agent,
                latency_ms=latency_ms,
            )
            try:
                from app.metrics import record_llm_metrics
                record_llm_metrics(backend="ollama", success=True, latency_ms=latency_ms)
            except Exception:
                pass
            return text
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log_llm_call(
                backend="ollama", model=self.model, prompt=prompt,
                response="", success=False, error=str(exc), context=context,
                agent=agent, latency_ms=latency_ms,
            )
            try:
                from app.metrics import record_llm_metrics
                record_llm_metrics(backend="ollama", success=False, latency_ms=latency_ms)
            except Exception:
                pass
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        *,
        context: str = "rollout",
        agent: str = "unknown",
    ) -> str:
        """Sync generate — kept for offline scripts (generate_rollouts.py etc.)."""
        from shared.llm_logging import log_llm_call

        t0 = time.perf_counter()
        try:
            data = self._post_sync({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            })
            text = str(data.get("response", "")).strip()
            latency_ms = (time.perf_counter() - t0) * 1000
            log_llm_call(
                backend="ollama", model=self.model, prompt=prompt,
                response=text, success=True, context=context, agent=agent,
                latency_ms=latency_ms,
            )
            try:
                from app.metrics import record_llm_metrics
                record_llm_metrics(backend="ollama", success=True, latency_ms=latency_ms)
            except Exception:
                pass
            return text
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log_llm_call(
                backend="ollama", model=self.model, prompt=prompt,
                response="", success=False, error=str(exc), context=context,
                agent=agent, latency_ms=latency_ms,
            )
            try:
                from app.metrics import record_llm_metrics
                record_llm_metrics(backend="ollama", success=False, latency_ms=latency_ms)
            except Exception:
                pass
            raise
