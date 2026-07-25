"""Dashboard API routes and static UI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from monitoring.offline import aggregate_offline
from monitoring.store import get_store

router = APIRouter(tags=["monitoring"])

_STATIC = Path(__file__).resolve().parent / "static"
_START = time.time()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> FileResponse:
    return FileResponse(_STATIC / "dashboard.html")


@router.get("/api/monitoring/summary")
async def monitoring_summary() -> Dict[str, Any]:
    store = get_store()
    offline = aggregate_offline()
    return {
        "live": store.live_summary(),
        "offline": offline,
    }


@router.get("/api/monitoring/queries")
async def monitoring_queries(limit: int = 50) -> Dict[str, Any]:
    return {"queries": get_store().recent_queries(limit=limit)}


@router.get("/api/monitoring/llm")
async def monitoring_llm(limit: int = 50) -> Dict[str, Any]:
    return {"calls": get_store().recent_llm_calls(limit=limit)}


@router.get("/api/monitoring/timeseries")
async def monitoring_timeseries() -> Dict[str, Any]:
    return {"latency": get_store().latency_timeseries()}


@router.get("/api/monitoring/offline")
async def monitoring_offline() -> Dict[str, Any]:
    return aggregate_offline()


@router.get("/api/metrics/snapshot")
async def metrics_snapshot() -> Dict[str, Any]:
    """Real-time system + business metrics for native dashboard cards."""
    import os

    store = get_store()
    live = store.live_summary()

    # System stats via psutil
    cpu_pct = mem_pct = disk_pct = None
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem_pct = psutil.virtual_memory().percent
        disk_pct = psutil.disk_usage("/").percent
    except Exception:
        pass

    uptime_s = time.time() - _START
    total = live.get("total_queries", 0)
    errors = live.get("error_count", 0)
    ok = total - errors

    # Throughput: requests/min from last 5-min slice
    try:
        ts = store.latency_timeseries()
        recent = ts[-3:] if len(ts) >= 3 else ts
        throughput_rpm = sum(r.get("count", 0) for r in recent) / max(len(recent) * 5, 1)
    except Exception:
        throughput_rpm = 0

    # Per-backend breakdown
    backends = live.get("llm_by_backend", [])

    return {
        # Request metrics
        "total_requests": total,
        "success_requests": ok,
        "failed_requests": errors,
        "success_rate": (ok / total) if total else None,
        "error_rate": live.get("error_rate"),
        "throughput_rpm": round(throughput_rpm, 2),

        # Latency
        "avg_response_ms": live.get("avg_latency_ms"),
        "api_latency_ms": live.get("avg_latency_ms"),

        # Quality
        "avg_confidence": live.get("avg_confidence"),
        "retrieval_quality": live.get("avg_retrieval_quality"),

        # Users (proxy)
        "active_users": min(total, 99),

        # System
        "cpu_percent": cpu_pct,
        "memory_percent": mem_pct,
        "disk_percent": disk_pct,
        "uptime_seconds": round(uptime_s),

        # Health
        "service_health": "healthy" if (live.get("error_rate") or 0) < 0.1 else "degraded",

        # Backends
        "llm_backends": backends,

        # Policy dist
        "policy_distribution": live.get("policy_distribution", {}),

        # Grafana
        "grafana_url": "http://localhost:3000",
        "grafana_dashboard_id": "fasa-main",
    }
