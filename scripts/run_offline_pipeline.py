"""Run the full offline training data pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline pipeline")
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--model-id", default=None, help="HF model for rollouts")
    parser.add_argument("--ollama-model", default=None, help="Ollama model for rollouts")
    args = parser.parse_args()

    py = sys.executable
    env = {"PYTHONPATH": str(ROOT)}

    import os

    os.environ.update(env)

    dataset_cmd = [py, "scripts/generate_dataset.py", "--target-size", str(args.target_size)]
    if args.synthetic_only:
        dataset_cmd.append("--synthetic-only")
    run(dataset_cmd)

    rollout_cmd = [
        py, "scripts/generate_rollouts.py",
        "--max-queries", str(args.max_queries),
        "--k", str(args.k),
    ]
    if args.model_id:
        rollout_cmd.extend(["--model-id", args.model_id])
    if args.ollama_model:
        rollout_cmd.extend(["--ollama-model", args.ollama_model])
    run(rollout_cmd)
    run([py, "scripts/score_trajectories.py"])
    run([py, "scripts/build_preference_pairs.py", "--margin", str(args.margin)])
    print("\nOffline pipeline complete.")


if __name__ == "__main__":
    main()
