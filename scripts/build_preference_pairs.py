"""
Build DPO preference pairs from scored trajectories.

Output: data/preference_pairs/train_pairs.jsonl, val_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.trajectory import format_trajectory_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_scored(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_pairs(
    trajectories: List[Dict[str, Any]],
    margin_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trajectories:
        by_query[t["query_id"]].append(t)

    pairs: List[Dict[str, Any]] = []
    kept, discarded = 0, 0

    for query_id, trajs in by_query.items():
        if len(trajs) < 2:
            continue
        sorted_t = sorted(trajs, key=lambda x: x.get("composite_reward", 0), reverse=True)
        chosen, rejected = sorted_t[0], sorted_t[-1]
        gap = chosen.get("composite_reward", 0) - rejected.get("composite_reward", 0)
        if gap <= margin_threshold:
            discarded += 1
            continue
        kept += 1
        pairs.append({
            "prompt": chosen.get("question", ""),
            "chosen": format_trajectory_text(chosen),
            "rejected": format_trajectory_text(rejected),
            "reward_gap": round(gap, 4),
            "ambiguity_type": chosen.get("ambiguity_type", "unknown"),
            "query_id": query_id,
        })

    logger.info("Pairs kept: %d, discarded: %d", kept, discarded)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument("--input", default="data/scored_trajectories.jsonl")
    parser.add_argument("--output-dir", default="data/preference_pairs")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trajectories = load_scored(Path(args.input))
    pairs = build_pairs(trajectories, margin_threshold=args.margin)

    if not pairs:
        logger.error("No preference pairs created. Check rollout/scoring pipeline.")
        return

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=args.val_ratio, random_state=args.seed
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train_pairs.jsonl", train_pairs), ("val_pairs.jsonl", val_pairs)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(p) + "\n")

    summary = {
        "total_pairs": len(pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "mean_reward_gap": sum(p["reward_gap"] for p in pairs) / len(pairs),
        "by_ambiguity_type": {},
    }
    for p in pairs:
        t = p["ambiguity_type"]
        summary["by_ambiguity_type"][t] = summary["by_ambiguity_type"].get(t, 0) + 1

    with open(out_dir / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved %d train / %d val pairs to %s", len(train_pairs), len(val_pairs), out_dir)


if __name__ == "__main__":
    main()
