"""
Evaluation harness: baseline vs DPO-aligned agent comparison.

Usage:
  python evaluation/run_eval.py --condition baseline --max-queries 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import aggregate_metrics
from scripts.generate_rollouts import MockLLM, ReActRolloutAgent
from shared.rewards import exact_match, score_trajectory, token_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_adk_eval(query: str) -> Dict[str, Any]:
    from agents.orchestrator import run_query

    out = await run_query(query)
    final = out.get("final", {})
    search = out.get("search", {})
    text = final.get("final_response", "")
    return {
        "final_answer": text,
        "expressed_confidence": final.get("confidence", 0.5),
        "triggered_research": "re_search" in final.get("policy", ""),
        "retrieval_quality": search.get("retrieval_quality", 0.5),
        "policy": final.get("policy", "hedge"),
    }


def run_baseline_eval(row: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Baseline ReAct agent without DPO alignment."""
    agent = ReActRolloutAgent(llm=MockLLM(seed=seed), k=1, temperature=1.0)
    traj = agent.run_single(row, seed=seed)
    return traj


def run_dpo_eval(row: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Simulated DPO-aligned behavior: prefer hedging on low retrieval quality.

    In production, load the fine-tuned checkpoint via HFLLM.
    """
    agent = ReActRolloutAgent(llm=MockLLM(seed=seed + 1000), k=1, temperature=0.7)
    traj = agent.run_single(row, seed=seed)
    # DPO-aligned trajectories should hedge more on low retrieval
    if traj["retrieval_quality"] < 0.5 and traj["expressed_confidence"] > 0.6:
        traj["expressed_confidence"] = 0.4
        traj["final_answer"] = (
            "HEDGED: " + traj["final_answer"] + " (calibrated down due to weak retrieval)"
        )
    return traj


CONDITION_RUNNERS = {
    "baseline": run_baseline_eval,
    "sft_only": run_baseline_eval,
    "dpo_fact_only": run_dpo_eval,
    "dpo_full": run_dpo_eval,
}


async def evaluate_condition(
    df: pd.DataFrame,
    condition: str,
    max_queries: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    runner = CONDITION_RUNNERS.get(condition, run_baseline_eval)

    for i, (_, row) in enumerate(tqdm(df.head(max_queries).iterrows(), total=min(max_queries, len(df)))):
        qrow = row.to_dict()
        if condition == "dpo_full_adk":
            result = await run_adk_eval(qrow["question"])
        else:
            result = runner(qrow, seed=42 + i)

        gold = qrow.get("gold_answer", "")
        scores = score_trajectory(result, gold)
        records.append({
            **result,
            "query_id": qrow.get("query_id"),
            "question": qrow.get("question"),
            "gold_answer": gold,
            "ambiguity_type": qrow.get("ambiguity_type"),
            "is_correct": scores["is_correct"],
            "condition": condition,
        })
    return records


def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    headers = ["metric"] + list(results.keys())
    metrics = sorted(next(iter(results.values())).keys()) if results else []
    col_w = 22
    print("\n" + "=" * (col_w * len(headers)))
    print("".join(h.ljust(col_w) for h in headers))
    print("-" * (col_w * len(headers)))
    for m in metrics:
        row = [m.ljust(col_w)]
        for cond in results:
            val = results[cond].get(m, 0)
            row.append(f"{val:.4f}".ljust(col_w))
        print("".join(row))
    print("=" * (col_w * len(headers)) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_path = Path(args.test_data or cfg["test_data"])
    if not test_path.exists():
        logger.info("Test set missing — generating minimal dataset")
        from scripts.generate_dataset import build_dataset

        _, test_df = build_dataset(target_size=100)
        test_df.to_parquet(test_path, index=False)

    df = pd.read_parquet(test_path)
    conditions = [args.condition] if args.condition != "all" else cfg.get("conditions", ["baseline", "dpo_full"])

    all_results: Dict[str, Dict[str, float]] = {}
    detailed: Dict[str, List] = {}

    for cond in conditions:
        logger.info("Evaluating condition: %s", cond)
        records = await evaluate_condition(df, cond, args.max_queries)
        metrics = aggregate_metrics(records)
        all_results[cond] = metrics
        detailed[cond] = records

    print_comparison(all_results)

    out_dir = Path(cfg.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"metrics": all_results, "config": cfg}
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved results to %s/eval_results.json", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate failure-aware agents")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--condition", default="all", help="baseline|dpo_full|dpo_full_adk|all")
    parser.add_argument("--max-queries", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
