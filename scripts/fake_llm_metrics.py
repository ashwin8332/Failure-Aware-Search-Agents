import sqlite3
import time
import random
import os
from pathlib import Path

db_path = Path(__file__).resolve().parents[1] / "data" / "monitoring.db"

with sqlite3.connect(db_path, timeout=5.0) as conn:
    for i in range(50):
        backend = random.choice(["ollama", "ollama", "ollama", "tinyllama", "tinyllama", "phi3", "huggingface"])
        latency_ms = random.uniform(200.0, 1500.0)
        conn.execute(
            """
            INSERT INTO llm_events
            (ts, backend, model, latency_ms, prompt_chars, response_chars, success, error, context, agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time() - random.uniform(0, 300),
                backend, "fake-model", latency_ms, 50, 150, True, None, "rollout", "fake_agent"
            )
        )
    conn.commit()

print("Fake LLM data inserted into monitoring.db!")
