"""
Prometheus + system metrics module for FASA.

Exposes:
  /metrics   — Prometheus scrape endpoint (standard + custom)

Custom metrics tracked:
  fasa_requests_total          — counter, labelled by policy/status
  fasa_request_latency_seconds — histogram, end-to-end query latency
  fasa_confidence_gauge        — gauge, rolling avg confidence
  fasa_llm_calls_total         — counter, labelled by backend
  fasa_llm_latency_seconds     — histogram, LLM generation latency
  fasa_error_rate_gauge        — gauge, rolling error rate (0–1)
  fasa_retrieval_quality_gauge — gauge, rolling avg retrieval quality
  fasa_active_users_gauge      — gauge, unique IPs in last 5 min
  system_cpu_percent           — gauge
  system_memory_percent        — gauge
  system_uptime_seconds        — gauge
"""

from __future__ import annotations

import time
import os
from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry, REGISTRY,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ── Custom business metrics ──────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "fasa_requests_total",
    "Total search queries processed",
    ["policy", "status"],
)

REQUEST_LATENCY = Histogram(
    "fasa_request_latency_seconds",
    "End-to-end query latency in seconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60],
)

CONFIDENCE_GAUGE = Gauge(
    "fasa_confidence_gauge",
    "Rolling average expressed confidence (0–1)",
)

LLM_CALLS_TOTAL = Counter(
    "fasa_llm_calls_total",
    "Total LLM generation calls",
    ["backend", "status"],
)

LLM_LATENCY = Histogram(
    "fasa_llm_latency_seconds",
    "LLM generation latency in seconds",
    ["backend"],
    buckets=[0.5, 1, 2, 5, 10, 20, 40, 60, 120],
)

ERROR_RATE_GAUGE = Gauge(
    "fasa_error_rate_gauge",
    "Rolling error rate fraction (0–1)",
)

RETRIEVAL_QUALITY_GAUGE = Gauge(
    "fasa_retrieval_quality_gauge",
    "Rolling average retrieval quality score (0–1)",
)

ACTIVE_USERS_GAUGE = Gauge(
    "fasa_active_users_gauge",
    "Approximate unique active users (last 5 min)",
)

# ── System metrics ───────────────────────────────────────────────────────────

CPU_GAUGE    = Gauge("system_cpu_percent",    "CPU utilisation percent")
MEMORY_GAUGE = Gauge("system_memory_percent", "Memory utilisation percent")
MEMORY_USED_GB = Gauge("system_memory_used_gb",  "Memory used in GB")
MEMORY_TOTAL_GB = Gauge("system_memory_total_gb", "Total memory in GB")
DISK_GAUGE   = Gauge("system_disk_percent",   "Disk utilisation percent")
DISK_FREE_GB = Gauge("system_disk_free_gb",   "Free disk space in GB")
DISK_TOTAL_GB = Gauge("system_disk_total_gb", "Total disk space in GB")
UPTIME_GAUGE = Gauge("system_uptime_seconds", "Process uptime in seconds")

_START_TIME = time.time()


def update_system_metrics() -> None:
    """Called on each scrape to refresh live system stats."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        CPU_GAUGE.set(cpu)

        vm = psutil.virtual_memory()
        MEMORY_GAUGE.set(vm.percent)
        MEMORY_USED_GB.set(round(vm.used / 1e9, 2))
        MEMORY_TOTAL_GB.set(round(vm.total / 1e9, 2))

        du = psutil.disk_usage("/")
        DISK_GAUGE.set(du.percent)
        DISK_FREE_GB.set(round(du.free / 1e9, 2))
        DISK_TOTAL_GB.set(round(du.total / 1e9, 2))
    except Exception:
        pass
    UPTIME_GAUGE.set(time.time() - _START_TIME)


def refresh_from_store() -> None:
    """Pull latest aggregate stats from the monitoring SQLite store."""
    try:
        from monitoring.store import get_store
        store = get_store()
        summary = store.live_summary()
        live = summary

        conf = live.get("avg_confidence")
        if conf is not None:
            CONFIDENCE_GAUGE.set(float(conf))

        err = live.get("error_rate")
        if err is not None:
            ERROR_RATE_GAUGE.set(float(err))

        # active users approximation from unique query count in last 5 min
        # (placeholder — SQLite doesn't track IPs, use total as proxy)
        total = live.get("total_queries", 0)
        ACTIVE_USERS_GAUGE.set(min(total, 99))  # caps at 99 for display

        # retrieval quality from latest queries
        try:
            rows = store.recent_queries(limit=20)
            quals = [r.get("retrieval_quality") for r in rows if r.get("retrieval_quality") is not None]
            if quals:
                RETRIEVAL_QUALITY_GAUGE.set(sum(quals) / len(quals))
        except Exception:
            pass

    except Exception:
        pass


def record_query_metrics(
    *,
    policy: str,
    status: str,
    latency_ms: float,
    confidence: float | None = None,
) -> None:
    """Call after every /query response to update counters."""
    REQUESTS_TOTAL.labels(policy=policy, status=status).inc()
    REQUEST_LATENCY.observe(latency_ms / 1000.0)
    if confidence is not None:
        CONFIDENCE_GAUGE.set(float(confidence))


def record_llm_metrics(
    *,
    backend: str,
    success: bool,
    latency_ms: float,
) -> None:
    """Call after every LLM generation."""
    status = "ok" if success else "error"
    LLM_CALLS_TOTAL.labels(backend=backend, status=status).inc()
    LLM_LATENCY.labels(backend=backend).observe(latency_ms / 1000.0)
