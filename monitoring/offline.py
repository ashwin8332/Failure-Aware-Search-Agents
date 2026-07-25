"""Aggregate offline pipeline artifacts for dashboard panels."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path, limit: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scored_trajectories() -> List[Dict[str, Any]]:
    return _read_jsonl(_ROOT / "data" / "scored_trajectories.jsonl")


def load_preference_summary() -> Dict[str, Any]:
    path = _ROOT / "data" / "preference_pairs" / "summary_stats.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_results() -> Dict[str, Any]:
    path = _ROOT / "results" / "eval_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_rollout_files() -> int:
    rollout_dir = _ROOT / "data" / "rollouts"
    if not rollout_dir.exists():
        return 0
    return len(list(rollout_dir.glob("*.jsonl")))


def aggregate_offline() -> Dict[str, Any]:
    trajectories = load_scored_trajectories()
    pref = load_preference_summary()
    eval_data = load_eval_results()

    if not trajectories:
        return {
            "has_data": False,
            "rollout_query_count": count_rollout_files(),
            "preference_summary": pref,
            "eval_metrics": eval_data.get("metrics", {}),
        }

    n = len(trajectories)
    rewards = [t.get("composite_reward", 0) for t in trajectories]
    confidences = [t.get("expressed_confidence", 0) for t in trajectories]
    policies_hint = Counter()
    for t in trajectories:
        c = t.get("expressed_confidence", 0.5)
        if c >= 0.65:
            policies_hint["confident"] += 1
        elif c < 0.4:
            policies_hint["re_search"] += 1
        else:
            policies_hint["hedge"] += 1

    by_ambiguity: Dict[str, List[float]] = defaultdict(list)
    for t in trajectories:
        by_ambiguity[t.get("ambiguity_type", "unknown")].append(t.get("composite_reward", 0))

    return {
        "has_data": True,
        "trajectory_count": n,
        "rollout_query_count": count_rollout_files(),
        "mean_composite_reward": round(sum(rewards) / n, 4),
        "mean_confidence": round(sum(confidences) / n, 3),
        "mean_r_fact": round(sum(t.get("r_fact", 0) for t in trajectories) / n, 4),
        "mean_r_calib": round(sum(t.get("r_calib", 0) for t in trajectories) / n, 4),
        "mean_r_recovery": round(sum(t.get("r_recovery", 0) for t in trajectories) / n, 4),
        "accuracy": round(sum(1 for t in trajectories if t.get("is_correct")) / n, 3),
        "research_rate": round(
            sum(1 for t in trajectories if t.get("triggered_research")) / n, 3
        ),
        "confidence_histogram": _histogram(confidences, bins=10),
        "reward_histogram": _histogram(rewards, bins=10),
        "policy_from_confidence": dict(policies_hint),
        "reward_by_ambiguity": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_ambiguity.items())
        },
        "preference_summary": pref,
        "eval_metrics": eval_data.get("metrics", {}),
    }


def _histogram(values: List[float], bins: int = 10) -> List[Dict[str, Any]]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [{"bin": f"{lo:.2f}", "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    return [
        {
            "bin": f"{lo + i * width:.2f}–{lo + (i + 1) * width:.2f}",
            "count": counts[i],
        }
        for i in range(bins)
    ]
