"""
FastMCP search server for Failure-Aware Search Agents.

Exposes web_search, wikipedia_lookup, coherence_score, and search_quality_check.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from shared.rate_limiter import TokenBucketRateLimiter
from shared.sanitize import sanitize_query
from shared.search_tools import (
    coherence_score as _coherence_score,
    search_quality_check as _search_quality_check,
    web_search as _web_search,
    wikipedia_lookup as _wikipedia_lookup,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("failure-aware-search")

_rate_limit = int(os.environ.get("MCP_RATE_LIMIT_PER_MINUTE", "10"))
limiter = TokenBucketRateLimiter(max_calls=_rate_limit, window_seconds=60.0)


def _check_rate(tool_name: str) -> None:
    key = f"tool:{tool_name}"
    if not limiter.allow(key):
        wait = limiter.wait_time(key)
        raise RuntimeError(
            f"Rate limit exceeded for {tool_name}. Retry in {wait:.1f}s "
            f"(max {_rate_limit} calls/min)."
        )


@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> dict:
    """
    Search the web. Returns results with quality metadata.

    Args:
        query: Search query string (max 500 chars).
        num_results: Number of results (1-10).
    """
    try:
        _check_rate("web_search")
        query = sanitize_query(query)
        num_results = max(1, min(int(num_results), 10))
        return _web_search(query, num_results)
    except Exception as exc:
        logger.exception("web_search error")
        return {
            "error": str(exc),
            "results": [],
            "coherence_score": 0.0,
            "conflict_detected": False,
            "retrieval_quality": 0.0,
        }


@mcp.tool()
async def wikipedia_lookup(entity: str) -> dict:
    """
    Fetch Wikipedia summary for authoritative grounding.

    Args:
        entity: Entity or topic name to look up.
    """
    try:
        _check_rate("wikipedia_lookup")
        entity = sanitize_query(entity)
        return _wikipedia_lookup(entity)
    except Exception as exc:
        logger.exception("wikipedia_lookup error")
        return {
            "found": False,
            "summary": "",
            "url": "",
            "categories": [],
            "error": str(exc),
        }


@mcp.tool()
async def coherence_score(texts: List[str]) -> dict:
    """
    Score semantic coherence of text passages.

    Args:
        texts: List of snippet strings to compare.
    """
    try:
        _check_rate("coherence_score")
        sanitized = [sanitize_query(t, max_length=2000) for t in texts[:20]]
        return _coherence_score(sanitized)
    except Exception as exc:
        logger.exception("coherence_score error")
        return {"score": 0.0, "conflict_pairs": [], "error": str(exc)}


@mcp.tool()
async def search_quality_check(query: str, snippets: List[str]) -> dict:
    """
    Assess retrieval quality and recommend policy action.

    Args:
        query: Original search query.
        snippets: Retrieved snippet texts.

    Returns recommendation in {confident, hedge, re_search}.
    """
    try:
        _check_rate("search_quality_check")
        query = sanitize_query(query)
        sanitized = [sanitize_query(s, max_length=2000) for s in snippets[:20]]
        return _search_quality_check(query, sanitized)
    except Exception as exc:
        logger.exception("search_quality_check error")
        return {
            "quality_score": 0.0,
            "reasons": [str(exc)],
            "recommendation": "re_search",
            "error": str(exc),
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
