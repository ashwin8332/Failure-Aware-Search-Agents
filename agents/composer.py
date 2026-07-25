"""Final response composer — maps verification to user-facing policy.

New in this version:
- Blended confidence: 0.45·retrieval + 0.30·source_trust + 0.25·verification
- Structured explanation block with ✔/✖ factors, breakdown, and policy reason
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from agents.prompts_loader import load_prompt
from agents.verification_agent import parse_verdict
from shared.rewards import extract_confidence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash")

# Blended confidence weights
_W_RETRIEVAL = 0.45
_W_TRUST = 0.30
_W_VERIFICATION = 0.25


def build_composer_agent(model: Optional[str] = None):
    """Build final response composer agent."""
    model = model or DEFAULT_MODEL
    instruction = load_prompt("composer_prompt.txt")

    try:
        from google.adk.agents import LlmAgent

        return LlmAgent(
            model=model,
            name="response_composer",
            description="Composes final confident, hedged, or re-search response.",
            instruction=instruction,
            tools=[],
            output_key="final_response",
        )
    except ImportError:
        return MockComposerAgent(instruction=instruction)


def _extract_answer(draft: str) -> Optional[str]:
    m = re.search(r"Final Answer:\s*(.+?)(?:\nConfidence:|\Z)", draft, re.S | re.I)
    return m.group(1).strip() if m else None


def _usable_draft_answer(draft: str) -> Optional[str]:
    answer = _extract_answer(draft)
    if not answer:
        return None
    lowered = answer.lower()
    if "could not find" in lowered or "insufficient" in lowered or "uncertain" in lowered:
        return None
    return answer


def _format_verified_response(
    verified_facts: list,
    search_draft: str,
) -> str:
    draft_answer = _usable_draft_answer(search_draft)
    if draft_answer:
        return draft_answer
    return str(verified_facts[0]).strip()


def _verification_score(verdict_str: str) -> float:
    """Map verification verdict to a 0-1 score for the blended formula."""
    v = verdict_str.lower()
    if v == "confirmed":
        return 0.85
    if v == "unverified":
        return 0.40
    if v == "conflicted":
        return 0.10
    return 0.40


def _quality_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def build_explanation(
    *,
    query: str,
    num_results: int,
    retrieval_quality: float,
    avg_source_trust: float,
    avg_trust_pct: int,
    wiki_found: bool,
    conflict_detected: bool,
    verdict: str,
    policy: str,
    base_conf: float,
    blended_conf: float,
    final_conf: float,
    search_backend: str,
    source_trust_scores: List[Dict[str, Any]],
    quality_reasons: List[str],
    policy_reason: str,
) -> Dict[str, Any]:
    """Build structured explainability block."""
    ver_score = _verification_score(verdict)

    # Conflict penalty (negative SHAP contribution)
    conflict_penalty = -0.10 if conflict_detected else 0.0

    factors = [
        {
            "label": f"Retrieved {num_results} source{'s' if num_results != 1 else ''} via {search_backend}",
            "passed": num_results > 0,
            "icon": "✔" if num_results > 0 else "✖",
        },
        {
            "label": f"Wikipedia verification: {'found' if wiki_found else 'not found'}",
            "passed": wiki_found,
            "icon": "✔" if wiki_found else "✖",
        },
        {
            "label": f"Source conflict: {'detected' if conflict_detected else 'none'}",
            "passed": not conflict_detected,
            "icon": "✖" if conflict_detected else "✔",
        },
        {
            "label": f"Retrieval quality = {_quality_label(retrieval_quality)} ({retrieval_quality:.2f})",
            "passed": retrieval_quality >= 0.45,
            "icon": "✔" if retrieval_quality >= 0.45 else "✖",
        },
        {
            "label": f"Avg source trust = {avg_trust_pct}%",
            "passed": avg_trust_pct >= 60,
            "icon": "✔" if avg_trust_pct >= 60 else "✖",
        },
        {
            "label": f"Verification verdict = {verdict}",
            "passed": verdict == "confirmed",
            "icon": "✔" if verdict == "confirmed" else ("⚠" if verdict == "unverified" else "✖"),
        },
    ]

    if quality_reasons and quality_reasons != ["Acceptable retrieval quality"]:
        for reason in quality_reasons:
            factors.append({
                "label": f"Note: {reason}",
                "passed": False,
                "icon": "⚠",
            })

    # ── Failure reasons (Why-Policy panel) ────────────────────────────────
    failure_reasons: List[Dict[str, Any]] = []

    if policy in ("hedge", "re_search"):
        failure_reasons.append({
            "label": f"Retrieval quality {'adequate' if retrieval_quality >= 0.45 else 'low'} ({_quality_label(retrieval_quality)})",
            "passed": retrieval_quality >= 0.45,
        })
        failure_reasons.append({
            "label": f"{'Sufficient' if num_results >= 3 else 'Only ' + str(num_results)} source{'s' if num_results != 1 else ''} found",
            "passed": num_results >= 3,
        })
        failure_reasons.append({
            "label": f"Verification {'complete (Wikipedia found)' if wiki_found else 'incomplete (Wikipedia not found)'}",
            "passed": wiki_found,
        })
        failure_reasons.append({
            "label": f"Sources {'agree' if not conflict_detected else 'disagree (conflict detected)'}",
            "passed": not conflict_detected,
        })
        failure_reasons.append({
            "label": f"Source trust {'acceptable' if avg_trust_pct >= 60 else 'low'} ({avg_trust_pct}%)",
            "passed": avg_trust_pct >= 60,
        })
        failure_reasons.append({
            "label": f"Verdict: {verdict} ({'confirms answer' if verdict == 'confirmed' else 'does not confirm'})",
            "passed": verdict == "confirmed",
        })

    # ── SHAP-style confidence breakdown ───────────────────────────────────
    retrieval_contribution = round(_W_RETRIEVAL * retrieval_quality * 100)
    trust_contribution = round(_W_TRUST * avg_source_trust * 100)
    verification_contribution = round(_W_VERIFICATION * ver_score * 100)
    conflict_contribution = round(conflict_penalty * 100)

    confidence_breakdown = {
        "retrieval_quality_score": round(retrieval_quality, 3),
        "retrieval_weight": _W_RETRIEVAL,
        "retrieval_contribution_pct": retrieval_contribution,
        "source_trust_score": round(avg_source_trust, 3),
        "source_trust_weight": _W_TRUST,
        "trust_contribution_pct": trust_contribution,
        "verification_score": round(ver_score, 3),
        "verification_weight": _W_VERIFICATION,
        "verification_contribution_pct": verification_contribution,
        "conflict_penalty": conflict_penalty,
        "conflict_penalty_pct": conflict_contribution,
        "blended_raw": round(blended_conf, 3),
        "final": round(final_conf, 3),
        "final_pct": int(final_conf * 100),
        "formula": (
            f"{_W_RETRIEVAL}×retrieval({retrieval_quality:.2f}) "
            f"+ {_W_TRUST}×trust({avg_source_trust:.2f}) "
            f"+ {_W_VERIFICATION}×verification({ver_score:.2f}) "
            f"= {blended_conf:.2f}"
        ),
    }

    return {
        "factors": factors,
        "failure_reasons": failure_reasons,
        "confidence_breakdown": confidence_breakdown,
        "policy_reason": policy_reason,
        "policy_chosen": policy,
        "source_trust_scores": source_trust_scores,
    }



class MockComposerAgent:
    """Deterministic composer without LLM — now with blended confidence and explainability."""

    name = "response_composer"

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    async def run(
        self,
        query: str,
        search_draft: str,
        verification_verdict: str,
        search_out: Optional[Dict[str, Any]] = None,
        verify_out: Optional[Dict[str, Any]] = None,
    ) -> dict:
        search_out = search_out or {}
        verify_out = verify_out or {}

        verdict = parse_verdict(verification_verdict)
        v = verdict.get("verdict", "unverified")
        verified_facts = [
            str(f).strip() for f in (verdict.get("verified_facts") or []) if f and str(f).strip()
        ]

        # Raw component scores
        retrieval_quality = float(search_out.get("retrieval_quality", 0.5))
        avg_source_trust = float(search_out.get("avg_source_trust", 0.5))
        avg_trust_pct = int(search_out.get("avg_trust_pct", 50))
        ver_score = _verification_score(v)
        adj = float(verdict.get("confidence_adjustment", 0.0))

        # Blended confidence formula
        blended_conf = (
            _W_RETRIEVAL * retrieval_quality
            + _W_TRUST * avg_source_trust
            + _W_VERIFICATION * ver_score
        )
        # Apply verification adjustment on top (small nudge)
        blended_conf = max(0.05, min(0.95, blended_conf + adj * 0.3))

        insufficient = "INSUFFICIENT_EVIDENCE" in search_draft

        # ── Policy selection ──────────────────────────────────────────────
        if v == "confirmed" and verified_facts:
            policy = "confident"
            response = _format_verified_response(verified_facts, search_draft)
            final_conf = max(blended_conf, 0.75)
            policy_reason = (
                "Wikipedia confirmed the answer and retrieved verified facts. "
                f"Blended confidence ({blended_conf:.0%}) elevated to confident threshold."
            )
        elif v == "confirmed" and blended_conf >= 0.65:
            policy = "confident"
            response = _usable_draft_answer(search_draft) or "Answer supported by retrieved evidence."
            final_conf = blended_conf
            policy_reason = (
                f"Verification confirmed, blended confidence {blended_conf:.0%} ≥ 65% → confident."
            )
        elif v == "conflicted":
            policy = "re_search"
            response = (
                f"Sources conflict or are insufficient for: {query}. "
                f"{verdict.get('conflict_summary') or 'Consider rephrasing your question.'}"
            )
            final_conf = min(blended_conf, 0.35)
            policy_reason = (
                "Verification detected conflicting evidence across sources. "
                f"Confidence capped at {int(final_conf * 100)}% → re_search triggered."
            )
        elif insufficient:
            policy = "re_search"
            response = (
                f"Sources conflict or are insufficient for: {query}. "
                f"{verdict.get('conflict_summary') or 'Consider rephrasing your question.'}"
            )
            final_conf = min(blended_conf, 0.35)
            policy_reason = (
                "Search returned insufficient evidence. "
                f"Blended confidence {blended_conf:.0%} capped at {int(final_conf * 100)}% → re_search."
            )
        else:
            policy = "hedge"
            answer = _usable_draft_answer(search_draft)
            if v == "confirmed" and not answer:
                answer = "Answer partially supported by retrieved evidence."
            response = f"Based on limited evidence: {answer or 'uncertain'}."
            final_conf = min(blended_conf, 0.62)
            policy_reason = (
                f"Verification returned '{v}' and blended confidence {blended_conf:.0%} < 65%. "
                f"Confidence capped at {int(final_conf * 100)}% → hedge policy applied."
            )

        final_conf = max(0.05, min(0.95, final_conf))
        conf_pct = int(final_conf * 100)

        # ── Build explanation ──────────────────────────────────────────────
        wiki_found = verify_out.get("wiki_found", False)
        conflict_detected = len(verify_out.get("conflict_pairs", [])) > 0
        source_trust_scores = search_out.get("source_trust_scores", [])
        quality_reasons = search_out.get("quality_reasons", [])
        search_backend = search_out.get("search_backend", "unknown")
        num_results = len(search_out.get("results", search_out.get("snippets", [])))

        explanation = build_explanation(
            query=query,
            num_results=num_results,
            retrieval_quality=retrieval_quality,
            avg_source_trust=avg_source_trust,
            avg_trust_pct=avg_trust_pct,
            wiki_found=wiki_found,
            conflict_detected=conflict_detected,
            verdict=v,
            policy=policy,
            base_conf=extract_confidence(search_draft, default=0.5),
            blended_conf=blended_conf,
            final_conf=final_conf,
            search_backend=search_backend,
            source_trust_scores=source_trust_scores,
            quality_reasons=quality_reasons,
            policy_reason=policy_reason,
        )

        text = (
            f"Policy: {policy}\n"
            f"Final Response: {response}\n"
            f"Confidence: {conf_pct}% — verification={v}, retrieval-adjusted."
        )
        return {
            "final_response": text,
            "policy": policy,
            "confidence": conf_pct / 100.0,
            "explanation": explanation,
            "llm_backend": "rules",
            "llm_model": None,
            "llm_raw": None,
        }
