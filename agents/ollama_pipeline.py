"""Optional Ollama-backed agents for live /query inference."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from agents.composer import MockComposerAgent
from agents.search_agent import MockSearchAgent
from agents.verification_agent import MockVerificationAgent
from shared.ollama_llm import OllamaLLM

logger = logging.getLogger(__name__)


class OllamaSearchAgent(MockSearchAgent):
    """Search agent that adds an Ollama synthesis step on top of web retrieval."""

    name = "search_agent"

    def __init__(self, instruction: str, llm: OllamaLLM) -> None:
        super().__init__(instruction)
        self.llm = llm

    async def run(self, query: str, search_mode: str = "general") -> dict:
        base = await super().run(query, search_mode=search_mode)
        snippets = base.get("snippets") or []
        prompt = (
            "You are the search agent in a failure-aware QA pipeline.\n"
            f"User question: {query}\n"
            f"Retrieved snippets: {json.dumps(snippets)}\n"
            "Write a concise draft answer citing the snippets. "
            "End with a line: Confidence: N% — brief reason.\n"
        )
        try:
            raw = await self.llm.agenerate(
                prompt,
                temperature=0.3,
                context="live",
                agent="search_agent",
            )
            base["llm_raw"] = raw
            base["llm_backend"] = "ollama"
            base["llm_model"] = self.llm.model
            if raw.strip():
                base["search_draft"] = (
                    f"Question: {query}\n"
                    f"Search backend: {base.get('search_backend', 'unknown')}\n"
                    f"Retrieval quality: {base.get('retrieval_quality')}\n"
                    f"Snippets: {snippets}\n"
                    f"LLM draft:\n{raw.strip()}"
                )
                from shared.rewards import extract_confidence

                base["expressed_confidence"] = extract_confidence(base["search_draft"])
        except Exception as exc:
            logger.warning("Ollama search synthesis failed: %s — keeping heuristic draft", exc)
            base["llm_error"] = str(exc)
        return base


class OllamaVerificationAgent(MockVerificationAgent):
    """Verification agent with optional Ollama interpretation of Wikipedia + snippets."""

    name = "verification_agent"

    def __init__(self, instruction: str, llm: OllamaLLM) -> None:
        super().__init__(instruction)
        self.llm = llm

    async def run(self, search_draft: str, query: str) -> dict:
        base = await super().run(search_draft, query)
        prompt = (
            "You are the verification agent. Given the search draft and query, "
            "output ONLY valid JSON with keys: verdict (confirmed|conflicted|unverified), "
            "confidence_adjustment (float -0.3 to 0.1), conflict_summary (string or null), "
            "verified_facts (list of strings).\n"
            f"Query: {query}\n"
            f"Search draft:\n{search_draft}\n"
            f"Baseline verdict JSON:\n{base.get('verification_verdict', '{}')}\n"
        )
        try:
            raw = await self.llm.agenerate(
                prompt,
                temperature=0.1,
                context="live",
                agent="verification_agent",
            )
            base["llm_raw"] = raw
            base["llm_backend"] = "ollama"
            base["llm_model"] = self.llm.model
            from agents.verification_agent import parse_verdict

            parsed = parse_verdict(raw)
            if parsed.get("verdict"):
                base["verification_verdict"] = json.dumps(parsed)
        except Exception as exc:
            logger.warning("Ollama verification failed: %s — keeping Wikipedia verdict", exc)
            base["llm_error"] = str(exc)
        return base


class OllamaComposerAgent(MockComposerAgent):
    """Composer with optional Ollama narrative; policy still from rule-based composer."""

    name = "response_composer"

    def __init__(self, instruction: str, llm: OllamaLLM) -> None:
        super().__init__(instruction)
        self.llm = llm

    async def run(
        self,
        query: str,
        search_draft: str,
        verification_verdict: str,
        search_out: Optional[dict] = None,
        verify_out: Optional[dict] = None,
    ) -> dict:
        base = await super().run(query, search_draft, verification_verdict, search_out=search_out, verify_out=verify_out)
        prompt = (
            "You are the response composer. Rewrite the final user-facing answer in 2-3 sentences. "
            "Respect the policy and confidence from the baseline.\n"
            f"Query: {query}\n"
            f"Policy: {base.get('policy')}\n"
            f"Confidence: {base.get('confidence')}\n"
            f"Search draft:\n{search_draft}\n"
            f"Verification:\n{verification_verdict}\n"
            f"Baseline response:\n{base.get('final_response', '')}\n"
        )
        try:
            raw = await self.llm.agenerate(
                prompt,
                temperature=0.4,
                context="live",
                agent="composer_agent",
            )
            base["llm_raw"] = raw
            base["llm_backend"] = "ollama"
            base["llm_model"] = self.llm.model
            if raw.strip():
                conf_pct = int((base.get("confidence") or 0.5) * 100)
                base["final_response"] = (
                    f"Policy: {base.get('policy')}\n"
                    f"Final Response: {raw.strip()}\n"
                    f"Confidence: {conf_pct}% — verification-adjusted (Ollama composer)."
                )
        except Exception as exc:
            logger.warning("Ollama composer failed: %s — keeping rule-based output", exc)
            base["llm_error"] = str(exc)
        return base


def build_ollama_pipeline(cfg: dict) -> tuple[Any, Any, Any]:
    """Build search → verify → composer using Ollama models from inference config."""
    default_model = cfg.get("ollama_model", "phi3:mini")
    host = cfg.get("ollama_host")
    agents_cfg = cfg.get("agents") or {}

    def _llm(agent_key: str) -> OllamaLLM:
        model = agents_cfg.get(agent_key) or default_model
        return OllamaLLM(model=model, host=host)

    from agents.prompts_loader import load_prompt

    search = OllamaSearchAgent(load_prompt("search_agent_prompt.txt"), _llm("search_model"))
    verify = OllamaVerificationAgent(load_prompt("verify_agent_prompt.txt"), _llm("verification_model"))
    composer = OllamaComposerAgent(load_prompt("composer_prompt.txt"), _llm("composer_model"))
    return search, verify, composer
