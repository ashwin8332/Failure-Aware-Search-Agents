"""Search sub-agent for failure-aware retrieval."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from agents.prompts_loader import load_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash")


def build_search_agent(tools: Optional[List[Any]] = None, model: Optional[str] = None):
    """
    Build ADK LlmAgent for search with MCP tools.

    Falls back to a lightweight mock agent when google-adk is unavailable.
    """
    model = model or DEFAULT_MODEL
    instruction = load_prompt("search_agent_prompt.txt")

    try:
        from google.adk.agents import LlmAgent

        return LlmAgent(
            model=model,
            name="search_agent",
            description="Searches the web, checks retrieval quality, and drafts an answer with confidence.",
            instruction=instruction,
            tools=tools or [],
            output_key="search_draft",
        )
    except ImportError:
        logger.warning("google-adk not installed — returning MockSearchAgent")
        return MockSearchAgent(instruction=instruction)


def _confidence_for_policy(policy: str) -> int:
    return {"confident": 78, "hedge": 50, "re_search": 38}.get(policy, 45)


class MockSearchAgent:
    """Offline mock for development without Google ADK."""

    name = "search_agent"

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    async def run(self, query: str, search_mode: str = "general") -> dict:
        from shared.rewards import extract_confidence
        from shared.search_tools import (
            search_quality_check,
            synthesize_answer_from_snippets,
            web_search,
        )

        result = web_search(query, 5, search_mode)
        snippets = [r.get("snippet", "") for r in result.get("results", []) if r.get("snippet")]
        quality = search_quality_check(query, snippets)
        policy = quality["recommendation"]

        # Retry once with cleaned query (never prefix "rephrase:" — that hurts scoring)
        if policy == "re_search" and quality["quality_score"] < 0.4:
            clean_query = query.strip().rstrip("?").strip()
            if clean_query.lower() != query.strip().lower():
                retry = web_search(clean_query, 5, search_mode)
                retry_snippets = [
                    r.get("snippet", "") for r in retry.get("results", []) if r.get("snippet")
                ]
                retry_quality = search_quality_check(query, retry_snippets)
                if retry_quality["quality_score"] > quality["quality_score"]:
                    result = retry
                    snippets = retry_snippets
                    quality = retry_quality
                    policy = quality["recommendation"]

        answer = synthesize_answer_from_snippets(query, snippets)
        conf_pct = _confidence_for_policy(policy)
        snippet_preview = snippets[:3]

        draft = (
            f"Question: {query}\n"
            f"Search backend: {result.get('search_backend', 'unknown')}\n"
            f"Retrieval quality: {quality['quality_score']}\n"
            f"Snippets: {snippet_preview}\n"
            f"Final Answer: {answer}\n"
            f"Confidence: {conf_pct}% — "
            f"{'strong' if policy == 'confident' else 'mixed' if policy == 'hedge' else 'limited'} evidence."
        )

        return {
            "search_draft": draft,
            "retrieval_quality": quality["quality_score"],
            "policy_hint": policy,
            "expressed_confidence": extract_confidence(draft),
            "search_backend": result.get("search_backend"),
            "quality_reasons": quality.get("reasons", []),
            "snippets": snippet_preview,
            "results": result.get("results", []),
            # Source trust
            "source_trust_scores": result.get("source_trust_scores", []),
            "avg_source_trust": result.get("avg_source_trust", 0.5),
            "avg_trust_pct": result.get("avg_trust_pct", 50),
            "llm_backend": "heuristic",
            "llm_model": None,
            "llm_raw": None,
        }
