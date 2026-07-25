"""Input sanitization helpers."""

from __future__ import annotations

import re

MAX_QUERY_LENGTH = 500
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_query(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Strip, remove control chars, and truncate search queries."""
    cleaned = _CONTROL_CHARS.sub("", text.strip())
    if not cleaned:
        raise ValueError("Query must not be empty after sanitization.")
    return cleaned[:max_length]
