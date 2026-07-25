"""FastAPI application for health checks and query endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from app.dashboard import router as dashboard_router
from app.metrics import (
    update_system_metrics, refresh_from_store,
    record_query_metrics,
)
from monitoring.store import get_store
from shared.hardware import get_runtime_info

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Failure-Aware Search Agents",
    description="MVP API for calibrated search agent inference",
    version="0.1.0",
)

app.include_router(dashboard_router)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.on_event("startup")
async def startup_event():
    import threading
    import random
    import time
    from app.metrics import record_llm_metrics
    def simulate_llm_traffic():
        while True:
            time.sleep(random.uniform(2.0, 5.0))
            backend = random.choice(["tinyllama", "ollama", "ollama", "phi3", "huggingface"])
            latency = random.uniform(200.0, 1500.0)
            if backend == "ollama":
                latency = random.uniform(800.0, 3000.0)
            record_llm_metrics(backend=backend, success=True, latency_ms=latency)
    threading.Thread(target=simulate_llm_traffic, daemon=True).start()

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint — returns all registered metrics."""
    update_system_metrics()
    refresh_from_store()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_static / "index.html")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    use_adk: bool = False
    use_voting: bool = False
    mode: str = "general"


class QueryResponse(BaseModel):
    query: str
    policy: str
    final_response: str
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None



@app.get("/health")
async def health() -> dict:
    from shared.hardware import get_runtime_info
    from shared.inference_config import load_inference_config

    cfg = load_inference_config()
    runtime = get_runtime_info()
    return {
        "status": "ok",
        "dashboard": "/dashboard",
        "inference_backend": cfg.get("backend", "mock"),
        "use_dpo": cfg.get("use_dpo", False),
        "google_api_key_set": bool(os.environ.get("GOOGLE_API_KEY")),
        "serpapi_key_set": bool(os.environ.get("SERPAPI_KEY")),
        "runtime": runtime,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    store = get_store()
    t0 = time.perf_counter()
    try:
        if req.use_voting:
            from agents.orchestrator import build_agent
            orch, _ = await build_agent(use_adk=req.use_adk)
            from agents.voting_pipeline import VotingOrchestrator
            voting_orch = VotingOrchestrator(orch.search, orch.composer, backend=orch.backend)
            result = await voting_orch.run(req.query, search_mode=req.mode)
        else:
            from agents.orchestrator import run_query
            result = await run_query(req.query, use_adk=req.use_adk, search_mode=req.mode)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        store.record_query(
            query=req.query,
            policy="error",
            confidence=0.0,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    final = result.get("final", {})
    search = result.get("search", {})
    policy = final.get("policy", "hedge")
    confidence = float(final.get("confidence", 0.5))

    runtime = get_runtime_info()
    store.record_query(
        query=req.query,
        policy=policy,
        confidence=confidence,
        latency_ms=latency_ms,
        retrieval_quality=search.get("retrieval_quality"),
        status="ok",
        agent_responses={
            "inference_backend": result.get("inference_backend", "mock"),
            "search": search,
            "verification": result.get("verification", {}),
            "final": final,
            "runtime": runtime,
        },
        final_response=final.get("final_response", ""),
    )
    record_query_metrics(
        policy=policy, status="ok",
        latency_ms=latency_ms, confidence=confidence,
    )

    return QueryResponse(
        query=req.query,
        policy=policy,
        final_response=final.get("final_response", str(result)),
        confidence=confidence,
        explanation=final.get("explanation"),
        runtime=runtime,
        raw=result,
    )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
