"""Source reliability scoring — domain trust database.

Maps domains to trust scores (0–100) and tiers:
  authoritative  >= 90
  credible       >= 70
  average        >= 45
  low            <  45
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Domain trust database
# ---------------------------------------------------------------------------
_TRUST_DB: Dict[str, int] = {
    # Academic / research
    "wikipedia.org": 95,
    "pubmed.ncbi.nlm.nih.gov": 98,
    "nih.gov": 98,
    "ncbi.nlm.nih.gov": 97,
    "scholar.google.com": 90,
    "jstor.org": 92,
    "nature.com": 95,
    "science.org": 95,
    "sciencedirect.com": 90,
    "springer.com": 90,
    "wiley.com": 88,
    "tandfonline.com": 86,
    "arxiv.org": 85,
    "ssrn.com": 82,
    "researchgate.net": 78,
    "academia.edu": 72,
    # Government / official
    "cdc.gov": 97,
    "who.int": 96,
    "gov": 90,           # TLD catch-all for .gov
    "nasa.gov": 96,
    "fda.gov": 96,
    "nist.gov": 95,
    "census.gov": 93,
    "bls.gov": 93,
    "europa.eu": 90,
    "un.org": 88,
    # Reputable news / journalism
    "reuters.com": 88,
    "apnews.com": 88,
    "bbc.com": 85,
    "bbc.co.uk": 85,
    "theguardian.com": 82,
    "nytimes.com": 82,
    "washingtonpost.com": 80,
    "economist.com": 84,
    "ft.com": 83,
    "bloomberg.com": 82,
    "wsj.com": 80,
    "npr.org": 84,
    "pbs.org": 82,
    # Tech reference
    "developer.mozilla.org": 92,
    "docs.python.org": 95,
    "stackoverflow.com": 74,
    "github.com": 72,
    "docs.microsoft.com": 88,
    "learn.microsoft.com": 88,
    "support.google.com": 85,
    "developers.google.com": 88,
    "aws.amazon.com": 86,
    "docs.aws.amazon.com": 88,
    # Encyclopedias / reference
    "britannica.com": 90,
    "merriam-webster.com": 88,
    "dictionary.com": 75,
    "wolframalpha.com": 85,
    # Medical
    "mayoclinic.org": 92,
    "webmd.com": 70,
    "healthline.com": 65,
    "medicalnewstoday.com": 62,
    # Social / UGC (lower trust)
    "reddit.com": 48,
    "quora.com": 45,
    "medium.com": 50,
    "substack.com": 50,
    "twitter.com": 40,
    "x.com": 40,
    "facebook.com": 38,
    "youtube.com": 45,
    "tiktok.com": 30,
    # Generic / unknown blogs
    "blogspot.com": 38,
    "wordpress.com": 40,
    "wix.com": 35,
    "squarespace.com": 38,
    # Mock / dev fallbacks
    "example.com": 20,
}

# Tier thresholds
_TIER_AUTHORITATIVE = 90
_TIER_CREDIBLE = 70
_TIER_AVERAGE = 45


def _extract_domain(url: str) -> str:
    """Extract root domain from a URL."""
    url = url.lower().strip()
    # Strip scheme
    url = re.sub(r"^https?://", "", url)
    # Strip path/query
    url = url.split("/")[0].split("?")[0].split("#")[0]
    # Strip www.
    url = re.sub(r"^www\.", "", url)
    return url


def _score_domain(domain: str) -> int:
    """Look up trust score for a domain, with fallback heuristics."""
    if not domain:
        return 40

    # Exact match
    if domain in _TRUST_DB:
        return _TRUST_DB[domain]

    # Suffix match (e.g. sub.wikipedia.org → wikipedia.org)
    for known, score in _TRUST_DB.items():
        if domain.endswith("." + known) or domain == known:
            return score

    # TLD-based heuristics
    if domain.endswith(".gov"):
        return 88
    if domain.endswith(".edu"):
        return 82
    if domain.endswith(".org"):
        return 62
    if domain.endswith(".ac.uk") or domain.endswith(".ac."):
        return 80

    # Default for unknown domains
    return 42


def _tier(score: int) -> str:
    if score >= _TIER_AUTHORITATIVE:
        return "authoritative"
    if score >= _TIER_CREDIBLE:
        return "credible"
    if score >= _TIER_AVERAGE:
        return "average"
    return "low"


def score_source(url: str) -> Dict[str, Any]:
    """Return trust info for a single URL."""
    domain = _extract_domain(url)
    score = _score_domain(domain)
    return {
        "url": url,
        "domain": domain,
        "trust_score": score,
        "trust_pct": score,          # alias for UI convenience
        "trust_tier": _tier(score),
    }


def score_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach trust scores to a list of search results and compute aggregate.

    Mutates each result dict in-place (adds trust_score / trust_tier / domain).
    Returns aggregate dict with avg_source_trust and per-result list.
    """
    if not results:
        return {
            "source_trust_scores": [],
            "avg_source_trust": 0.5,        # neutral default
            "avg_trust_pct": 50,
            "min_trust_pct": 50,
            "max_trust_pct": 50,
        }

    trust_list: List[Dict[str, Any]] = []
    scores: List[int] = []

    for r in results:
        url = r.get("url", "")
        info = score_source(url)
        r["trust_score"] = info["trust_score"]
        r["trust_tier"] = info["trust_tier"]
        r["domain"] = info["domain"]
        trust_list.append(info)
        scores.append(info["trust_score"])

    avg_pct = round(sum(scores) / len(scores))
    return {
        "source_trust_scores": trust_list,
        "avg_source_trust": avg_pct / 100.0,   # 0-1 float for formula
        "avg_trust_pct": avg_pct,
        "min_trust_pct": min(scores),
        "max_trust_pct": max(scores),
    }
