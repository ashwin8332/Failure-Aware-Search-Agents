"""
Score trajectories with composite reward: factuality + calibration + recovery.

Output: data/scored_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.rewards import score_trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_rollout_files(rollout_dir: Path) -> Iterator[Dict[str, Any]]:
    for path in sorted(rollout_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score agent trajectories")
    parser.add_argument("--rollout-dir", default="data/rollouts")
    parser.add_argument("--output", default="data/scored_trajectories.jsonl")
    parser.add_argument("--gold-data", default="data/ambiguous_queries.parquet")
    args = parser.parse_args()

    gold_map: Dict[str, str] = {}
    try:
        import pandas as pd

        df = pd.read_parquet(args.gold_data)
        gold_map = dict(zip(df["query_id"], df["gold_answer"]))
    except Exception as exc:
        logger.warning("Could not load gold data: %s", exc)

    scored: List[Dict[str, Any]] = []
    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        logger.error("Rollout dir %s does not exist. Run generate_rollouts.py first.", rollout_dir)
        return

    for traj in iter_rollout_files(rollout_dir):
        gold = traj.get("gold_answer") or gold_map.get(traj["query_id"], "")
        scores = score_trajectory(traj, gold)
        traj.update(scores)
        scored.append(traj)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(record) + "\n")

    if scored:
        mean_r = sum(t["composite_reward"] for t in scored) / len(scored)
        logger.info("Scored %d trajectories. Mean composite reward: %.4f", len(scored), mean_r)
    else:
        logger.warning("No trajectories found in %s", rollout_dir)


if __name__ == "__main__":
    main()
