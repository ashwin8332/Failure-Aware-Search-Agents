"""Multi-Agent Voting Pipeline.

Architecture:
  Search Agent
       │
       ├──→ Verifier 1  (Wikipedia + coherence — same as MockVerificationAgent)
       └──→ Verifier 2  (source overlap + keyword heuristic)
                │
           Reasoning Agent  (weighs both verdicts, detects disagreement)
                │
             Vote  (V1×0.45 + V2×0.30 + Reasoning×0.25)
                │
           Composer  (receives vote_summary instead of single verdict)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from shared.search_tools import coherence_score, wikipedia_lookup
from shared.rewards import extract_confidence

logger = logging.getLogger(__name__)


# ── Verifier 1 — Wikipedia + coherence (same logic as MockVerificationAgent) ──

async def run_verifier1(search_draft: str, query: str, snippets: list) -> Dict[str, Any]:
    """Wikipedia-backed verifier."""
    entity = query.split("?")[0].strip()[:80]
    wiki = wikipedia_lookup(entity)
    texts = [wiki.get("summary", "")] + snippets[:3]
    coh = coherence_score([t for t in texts if t])

    if coh["conflict_detected"]:
        verdict = "conflicted"
        confidence = 0.30
        reasoning = "Wikipedia content conflicts with retrieved snippets."
        adjustment = -0.25
    elif wiki.get("found"):
        verdict = "confirmed"
        confidence = 0.85
        reasoning = f"Wikipedia confirmed: {wiki['summary'][:120]}…"
        adjustment = 0.05
    else:
        verdict = "unverified"
        confidence = 0.45
        reasoning = "Wikipedia did not find a matching article."
        adjustment = -0.10

    return {
        "agent": "Verifier 1 (Wikipedia)",
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "adjustment": adjustment,
        "wiki_found": wiki.get("found", False),
        "wiki_url": wiki.get("url", ""),
        "coherence": coh["score"],
    }


# ── Verifier 2 — Source overlap + keyword heuristic ──────────────────────

def _keyword_overlap(text1: str, text2: str) -> float:
    stop = {"the", "a", "an", "is", "are", "was", "of", "in", "on", "to", "for"}
    def kw(t):
        return set(re.findall(r"[a-zA-Z]{3,}", t.lower())) - stop
    t1, t2 = kw(text1), kw(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / max(len(t1), len(t2))


async def run_verifier2(search_draft: str, query: str, snippets: list) -> Dict[str, Any]:
    """Source-overlap heuristic verifier."""
    if not snippets:
        return {
            "agent": "Verifier 2 (Heuristic)",
            "verdict": "unverified",
            "confidence": 0.30,
            "reasoning": "No snippets available for cross-validation.",
            "adjustment": -0.15,
        }

    # Cross-snippet overlap
    overlaps = []
    for i in range(len(snippets)):
        for j in range(i + 1, len(snippets)):
            overlaps.append(_keyword_overlap(snippets[i], snippets[j]))
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Query-snippet relevance
    query_relevances = [_keyword_overlap(query, s) for s in snippets]
    avg_relevance = sum(query_relevances) / len(query_relevances)

    if avg_relevance >= 0.4 and avg_overlap >= 0.25:
        verdict = "confirmed"
        confidence = 0.75
        reasoning = f"Sources show good overlap ({avg_overlap:.0%}) and relevance ({avg_relevance:.0%})."
        adjustment = 0.05
    elif avg_relevance >= 0.2:
        verdict = "unverified"
        confidence = 0.50
        reasoning = f"Moderate relevance ({avg_relevance:.0%}) but limited cross-source agreement ({avg_overlap:.0%})."
        adjustment = -0.05
    else:
        verdict = "conflicted"
        confidence = 0.25
        reasoning = f"Low keyword relevance ({avg_relevance:.0%}) to query — sources may not address it."
        adjustment = -0.20

    return {
        "agent": "Verifier 2 (Heuristic)",
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "adjustment": adjustment,
        "avg_overlap": round(avg_overlap, 3),
        "avg_relevance": round(avg_relevance, 3),
    }


# ── Reasoning Agent — weighs both verdicts ────────────────────────────────

_VERDICT_SCORE = {"confirmed": 1.0, "unverified": 0.4, "conflicted": 0.0}

async def run_reasoning_agent(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize two verifier outputs into a consensus verdict."""
    s1 = _VERDICT_SCORE.get(v1["verdict"], 0.4)
    s2 = _VERDICT_SCORE.get(v2["verdict"], 0.4)
    disagreement = v1["verdict"] != v2["verdict"]

    weighted = 0.55 * s1 + 0.45 * s2

    if weighted >= 0.75:
        consensus = "confirmed"
        adjustment = 0.05
    elif weighted >= 0.35:
        consensus = "unverified"
        adjustment = -0.08 if disagreement else -0.05
    else:
        consensus = "conflicted"
        adjustment = -0.20

    reasoning = (
        f"V1 ({v1['verdict']}, {v1['confidence']:.0%}) + V2 ({v2['verdict']}, {v2['confidence']:.0%}). "
        f"Weighted score: {weighted:.2f}. "
        + ("⚠ Verifiers disagree — applying uncertainty penalty." if disagreement else "Verifiers agree.")
    )

    return {
        "agent": "Reasoning Agent",
        "consensus_verdict": consensus,
        "confidence": round(weighted, 3),
        "reasoning": reasoning,
        "disagreement_detected": disagreement,
        "adjustment": adjustment,
        "weighted_score": round(weighted, 3),
    }


# ── Vote aggregation ──────────────────────────────────────────────────────

_W_V1 = 0.45
_W_V2 = 0.30
_W_REASON = 0.25

async def aggregate_vote(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
    reasoning: Dict[str, Any],
) -> Dict[str, Any]:
    """Weighted vote combining all three agents."""
    s1 = _VERDICT_SCORE.get(v1["verdict"], 0.4)
    s2 = _VERDICT_SCORE.get(v2["verdict"], 0.4)
    sr = _VERDICT_SCORE.get(reasoning["consensus_verdict"], 0.4)

    final_score = _W_V1 * s1 + _W_V2 * s2 + _W_REASON * sr
    adj = _W_V1 * v1.get("adjustment", 0) + _W_V2 * v2.get("adjustment", 0) + _W_REASON * reasoning.get("adjustment", 0)

    if final_score >= 0.70:
        final_verdict = "confirmed"
    elif final_score >= 0.30:
        final_verdict = "unverified"
    else:
        final_verdict = "conflicted"

    vote_breakdown = [
        {
            "agent": v1["agent"],
            "verdict": v1["verdict"],
            "confidence": v1["confidence"],
            "weight": _W_V1,
            "contribution": round(_W_V1 * s1, 3),
            "reasoning": v1.get("reasoning", ""),
        },
        {
            "agent": v2["agent"],
            "verdict": v2["verdict"],
            "confidence": v2["confidence"],
            "weight": _W_V2,
            "contribution": round(_W_V2 * s2, 3),
            "reasoning": v2.get("reasoning", ""),
        },
        {
            "agent": reasoning["agent"],
            "verdict": reasoning["consensus_verdict"],
            "confidence": reasoning["confidence"],
            "weight": _W_REASON,
            "contribution": round(_W_REASON * sr, 3),
            "reasoning": reasoning.get("reasoning", ""),
        },
    ]

    return {
        "final_verdict": final_verdict,
        "final_score": round(final_score, 3),
        "confidence_adjustment": round(adj, 3),
        "disagreement_detected": reasoning["disagreement_detected"],
        "vote_breakdown": vote_breakdown,
    }


# ── VotingOrchestrator ────────────────────────────────────────────────────

class VotingOrchestrator:
    """Full 4-stage voting pipeline."""

    name = "voting_orchestrator"

    def __init__(self, search, composer, backend: str = "mock") -> None:
        self.search = search
        self.composer = composer
        self.backend = backend

    async def run(self, query: str, search_mode: str = "general") -> dict:
        # Stage 1: Search
        search_out = await self.search.run(query, search_mode=search_mode)
        snippets = list(search_out.get("snippets", []))

        # Stage 2: Parallel verification
        v1_out = await run_verifier1(search_out.get("search_draft", ""), query, snippets)
        v2_out = await run_verifier2(search_out.get("search_draft", ""), query, snippets)

        # Stage 3: Reasoning
        reasoning_out = await run_reasoning_agent(v1_out, v2_out)

        # Stage 4: Vote
        vote_summary = await aggregate_vote(v1_out, v2_out, reasoning_out)

        # Build synthetic verdict for composer
        synthetic_verdict = json.dumps({
            "verdict": vote_summary["final_verdict"],
            "confidence_adjustment": vote_summary["confidence_adjustment"],
            "conflict_summary": (
                "Verifiers disagree on this answer." if vote_summary["disagreement_detected"] else None
            ),
            "verified_facts": [],
        })

        # Augment verify_out for composer
        verify_out = {
            "verification_verdict": synthetic_verdict,
            "wiki_found": v1_out.get("wiki_found", False),
            "wiki_url": v1_out.get("wiki_url", ""),
            "coherence_score": v1_out.get("coherence", 1.0),
            "conflict_pairs": [] if not vote_summary["disagreement_detected"] else [[0, 1]],
        }

        final = await self.composer.run(
            query,
            search_out.get("search_draft", ""),
            synthetic_verdict,
            search_out=search_out,
            verify_out=verify_out,
        )

        return {
            "query": query,
            "inference_backend": f"{self.backend}+voting",
            "search": search_out,
            "verification": verify_out,
            "final": final,
            "vote_summary": vote_summary,
            "voting_stages": {
                "verifier1": v1_out,
                "verifier2": v2_out,
                "reasoning": reasoning_out,
                "vote": vote_summary,
            },
        }
