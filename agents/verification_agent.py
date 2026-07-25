"""Verification sub-agent for cross-checking evidence."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Optional

from agents.prompts_loader import load_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash")


def build_verification_agent(tools: Optional[List[Any]] = None, model: Optional[str] = None):
    """Build ADK verification agent."""
    model = model or DEFAULT_MODEL
    instruction = load_prompt("verify_agent_prompt.txt")

    try:
        from google.adk.agents import LlmAgent

        return LlmAgent(
            model=model,
            name="verification_agent",
            description="Cross-checks search draft against Wikipedia and coherence tools.",
            instruction=instruction,
            tools=tools or [],
            output_key="verification_verdict",
        )
    except ImportError:
        logger.warning("google-adk not installed — returning MockVerificationAgent")
        return MockVerificationAgent(instruction=instruction)


class MockVerificationAgent:
    """Offline mock verification using shared search tools."""

    name = "verification_agent"

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    async def run(self, search_draft: str, query: str) -> dict:
        from shared.search_tools import coherence_score, wikipedia_lookup

        entity = query.split("?")[0].strip()[:80]
        wiki = wikipedia_lookup(entity)
        snippets = re.findall(r"Snippets: (\[.*?\])", search_draft)
        texts = [wiki.get("summary", "")]
        if snippets:
            try:
                texts.extend(json.loads(snippets[0].replace("'", '"')))
            except json.JSONDecodeError:
                pass
        coh = coherence_score([t for t in texts if t])

        if coh["conflict_detected"]:
            verdict = {
                "verdict": "conflicted",
                "confidence_adjustment": -0.25,
                "conflict_summary": "Retrieved snippets may contradict Wikipedia or each other.",
                "verified_facts": [],
            }
        elif wiki.get("found"):
            verdict = {
                "verdict": "confirmed",
                "confidence_adjustment": 0.05,
                "conflict_summary": None,
                "verified_facts": [wiki["summary"][:200]],
            }
        else:
            verdict = {
                "verdict": "unverified",
                "confidence_adjustment": -0.1,
                "conflict_summary": None,
                "verified_facts": [],
            }
        return {
            "verification_verdict": json.dumps(verdict),
            "wiki_found": wiki.get("found", False),
            "wiki_url": wiki.get("url", ""),
            "coherence_score": coh.get("score", 1.0),
            "conflict_pairs": coh.get("conflict_pairs", []),
            "llm_backend": "wikipedia",
            "llm_model": None,
            "llm_raw": None,
        }


def parse_verdict(raw: str) -> dict:
    """Parse JSON verdict from agent output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {
        "verdict": "unverified",
        "confidence_adjustment": 0.0,
        "conflict_summary": None,
        "verified_facts": [],
    }
