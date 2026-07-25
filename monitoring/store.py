"""SQLite-backed event store for live LLM and query monitoring."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = _ROOT / "data" / "monitoring.db"

_store: Optional["MonitoringStore"] = None
_lock = threading.Lock()


class MonitoringStore:
    def __init__(self, db_path: Path | str = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    query TEXT NOT NULL,
                    policy TEXT,
                    confidence REAL,
                    latency_ms REAL,
                    retrieval_quality REAL,
                    status TEXT NOT NULL DEFAULT 'ok',
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS llm_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    backend TEXT NOT NULL,
                    model TEXT,
                    latency_ms REAL,
                    prompt_chars INTEGER,
                    response_chars INTEGER,
                    success INTEGER NOT NULL DEFAULT 1,
                    error TEXT,
                    context TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_query_ts ON query_events(ts);
                CREATE INDEX IF NOT EXISTS idx_llm_ts ON llm_events(ts);
                """
            )
            conn.commit()
        self._migrate_db()

    def _migrate_db(self) -> None:
        cols = {row[1] for row in self._conn().execute("PRAGMA table_info(query_events)")}
        with self._conn() as conn:
            if "agent_responses" not in cols:
                conn.execute("ALTER TABLE query_events ADD COLUMN agent_responses TEXT")
            if "final_response" not in cols:
                conn.execute("ALTER TABLE query_events ADD COLUMN final_response TEXT")
            llm_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_events)")}
            if "agent" not in llm_cols:
                conn.execute("ALTER TABLE llm_events ADD COLUMN agent TEXT")
            conn.commit()

    def record_query(
        self,
        *,
        query: str,
        policy: str,
        confidence: float,
        latency_ms: float,
        retrieval_quality: Optional[float] = None,
        status: str = "ok",
        error: Optional[str] = None,
        agent_responses: Optional[Dict[str, Any]] = None,
        final_response: Optional[str] = None,
    ) -> int:
        agent_json = json.dumps(agent_responses) if agent_responses else None
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO query_events
                (ts, query, policy, confidence, latency_ms, retrieval_quality, status, error,
                 agent_responses, final_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), query, policy, confidence, latency_ms, retrieval_quality,
                    status, error, agent_json, final_response,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def record_llm_call(
        self,
        *,
        backend: str,
        model: str,
        latency_ms: float,
        prompt_chars: int = 0,
        response_chars: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        context: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_events
                (ts, backend, model, latency_ms, prompt_chars, response_chars, success, error, context, agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    backend,
                    model,
                    latency_ms,
                    prompt_chars,
                    response_chars,
                    1 if success else 0,
                    error,
                    context,
                    agent,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def _rows(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        cur = self._conn().execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM query_events ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        for r in rows:
            r["ts_iso"] = _ts_iso(r["ts"])
            raw = r.get("agent_responses")
            if raw:
                try:
                    r["agent_responses"] = json.loads(raw)
                except json.JSONDecodeError:
                    r["agent_responses"] = None
        return rows

    def recent_llm_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM llm_events ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        for r in rows:
            r["ts_iso"] = _ts_iso(r["ts"])
            r["success"] = bool(r["success"])
        return rows

    def live_summary(self) -> Dict[str, Any]:
        q = self._rows(
            """
            SELECT
                COUNT(*) AS total,
                AVG(latency_ms) AS avg_latency_ms,
                AVG(confidence) AS avg_confidence,
                SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors
            FROM query_events
            """
        )
        policies = self._rows(
            "SELECT policy, COUNT(*) AS count FROM query_events GROUP BY policy"
        )
        llm = self._rows(
            """
            SELECT backend, COUNT(*) AS count, AVG(latency_ms) AS avg_latency_ms,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
            FROM llm_events GROUP BY backend
            """
        )
        base = q[0] if q else {}
        total = base.get("total") or 0
        return {
            "total_queries": total,
            "avg_latency_ms": round(base.get("avg_latency_ms") or 0, 1),
            "avg_confidence": round(base.get("avg_confidence") or 0, 3),
            "error_count": int(base.get("errors") or 0),
            "error_rate": round((base.get("errors") or 0) / total, 3) if total else 0.0,
            "policy_distribution": {r["policy"] or "unknown": r["count"] for r in policies},
            "llm_by_backend": llm,
        }

    def latency_timeseries(self, bucket_minutes: int = 5, limit: int = 48) -> List[Dict[str, Any]]:
        """Average query latency per time bucket."""
        queries = self._rows("SELECT ts, latency_ms FROM query_events ORDER BY ts DESC LIMIT 500")
        if not queries:
            return []
        bucket_sec = bucket_minutes * 60
        buckets: Dict[int, List[float]] = {}
        for row in queries:
            b = int(row["ts"] // bucket_sec) * bucket_sec
            buckets.setdefault(b, []).append(row["latency_ms"])
        series = [
            {
                "ts": b,
                "ts_iso": _ts_iso(b),
                "avg_latency_ms": round(sum(vals) / len(vals), 1),
                "count": len(vals),
            }
            for b, vals in sorted(buckets.items())
        ]
        return series[-limit:]


def get_store() -> MonitoringStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = MonitoringStore()
    return _store


import atexit as _atexit

def _close_store() -> None:
    global _store
    if _store is not None:
        try:
            conn = getattr(_store._local, "conn", None)
            if conn:
                conn.close()
        except Exception:
            pass

_atexit.register(_close_store)


def _ts_iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
