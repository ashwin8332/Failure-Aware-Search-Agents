"""Token-bucket rate limiter for MCP tool calls."""

from __future__ import annotations

import threading
import time
from typing import Dict


class TokenBucketRateLimiter:
    """Simple per-key token bucket rate limiter."""

    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._buckets: Dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "default") -> bool:
        """Return True if the call is allowed under the rate limit."""
        now = time.monotonic()
        with self._lock:
            timestamps = self._buckets.setdefault(key, [])
            cutoff = now - self.window_seconds
            self._buckets[key] = [t for t in timestamps if t > cutoff]
            if len(self._buckets[key]) >= self.max_calls:
                return False
            self._buckets[key].append(now)
            return True

    def wait_time(self, key: str = "default") -> float:
        """Seconds until the next call would be allowed (0 if allowed now)."""
        now = time.monotonic()
        with self._lock:
            timestamps = self._buckets.get(key, [])
            if len(timestamps) < self.max_calls:
                return 0.0
            oldest = min(timestamps)
            return max(0.0, self.window_seconds - (now - oldest))
