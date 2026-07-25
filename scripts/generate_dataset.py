"""
Build ambiguous query dataset from HotpotQA, TriviaQA, and AmbigQA.

Output: data/ambiguous_queries.parquet, data/ambiguous_test_set.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AMBIGUITY_TYPES = [
    "temporal",
    "conflicting_sources",
    "multi_hop",
    "out_of_scope",
    "near_miss",
]

SOURCE_TYPE_MAP = {
    "hotpot_qa": "multi_hop",
    "trivia_qa": "near_miss",
    "ambig_qa": "conflicting_sources",
}


def _normalize_hotpot(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "query_id": str(uuid.uuid4()),
        "question": example["question"],
        "gold_answer": example.get("answer", ""),
        "source": "hotpot_qa",
        "ambiguity_type": "multi_hop",
    }


def _normalize_trivia(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    answer = example.get("answer", {})
    if isinstance(answer, dict):
        gold = answer.get("value", answer.get("normalized_value", ""))
    else:
        gold = str(answer)
    return {
        "query_id": str(uuid.uuid4()),
        "question": example["question"],
        "gold_answer": gold,
        "source": "trivia_qa",
        "ambiguity_type": "near_miss",
    }


def _normalize_ambig(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    gold = ""
    annotations = example.get("annotations")
    if isinstance(annotations, list) and annotations:
        ann = annotations[0] if isinstance(annotations[0], dict) else {}
        answers = ann.get("answer", [])
        if answers:
            gold = answers[0] if isinstance(answers[0], str) else str(answers[0])
    elif example.get("nq_answers"):
        nq = example["nq_answers"]
        if isinstance(nq, list) and nq:
            texts = nq[0].get("text", []) if isinstance(nq[0], dict) else []
            gold = texts[0] if texts else ""
    return {
        "query_id": str(uuid.uuid4()),
        "question": example["question"],
        "gold_answer": gold,
        "source": "ambig_qa",
        "ambiguity_type": "conflicting_sources",
    }


def _load_with_fallback(name: str, config: str, split: str, max_rows: int) -> List[Dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(name, config, split=split)
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))
        return list(ds)
    except Exception as exc:
        logger.warning("Could not load %s/%s: %s — using synthetic fallback", name, config, exc)
        return []


def _synthetic_fallback(n: int) -> List[Dict[str, Any]]:
    """Deterministic synthetic queries when HF datasets unavailable."""
    samples = [
        ("Who became president after JFK?", "Lyndon B. Johnson", "temporal"),
        ("What is the capital of the country where the Eiffel Tower is located?", "Paris", "multi_hop"),
        ("When did Apple release the first iPhone?", "2007", "near_miss"),
        ("What drug treats both malaria and lupus?", "hydroxychloroquine", "conflicting_sources"),
        ("Who wrote the novel about the whale Moby Dick?", "Herman Melville", "out_of_scope"),
    ]
    rows = []
    for i in range(n):
        q, a, t = samples[i % len(samples)]
        rows.append({
            "query_id": str(uuid.uuid4()),
            "question": q,
            "gold_answer": a,
            "source": "synthetic",
            "ambiguity_type": t,
        })
    return rows


def build_dataset(target_size: int = 5000, test_ratio: float = 0.1) -> pd.DataFrame:
    per_source = max(100, target_size // 3)
    rows: List[Dict[str, Any]] = []

    hotpot = _load_with_fallback("hotpotqa/hotpot_qa", "distractor", "train", per_source)
    rows.extend(_normalize_hotpot(e, i) for i, e in enumerate(hotpot))

    trivia = _load_with_fallback("mandarjoshi/trivia_qa", "unfiltered.nocontext", "train", per_source)
    rows.extend(_normalize_trivia(e, i) for i, e in enumerate(trivia))

    ambig = _load_with_fallback("sewon/ambig_qa", "light", "train", per_source)
    rows.extend(_normalize_ambig(e, i) for i, e in enumerate(ambig))

    if len(rows) < 50:
        rows = _synthetic_fallback(target_size)

    df = pd.DataFrame(rows)
    df = df[df["gold_answer"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    # Stratified sample to target size
    if len(df) > target_size:
        df = (
            df.groupby("ambiguity_type", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, target_size // len(AMBIGUITY_TYPES)), random_state=42))
            .reset_index(drop=True)
        )

    split_kwargs: Dict[str, Any] = {"test_size": test_ratio, "random_state": 42}
    counts = df["ambiguity_type"].value_counts()
    if counts.min() >= 2 and len(df) >= 10:
        split_kwargs["stratify"] = df["ambiguity_type"]
    train_df, test_df = train_test_split(df, **split_kwargs)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    return pd.concat([train_df, test_df], ignore_index=True), test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ambiguous query dataset")
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip HuggingFace downloads; use built-in sample queries only",
    )
    parser.add_argument("--output", default="data/ambiguous_queries.parquet")
    parser.add_argument("--test-output", default="data/ambiguous_test_set.parquet")
    args = parser.parse_args()

    if args.synthetic_only:
        full_df = pd.DataFrame(_synthetic_fallback(args.target_size))
        full_df["split"] = "train"
        test_df = full_df.sample(n=max(1, int(args.target_size * 0.1)), random_state=42)
        test_df = test_df.copy()
        test_df["split"] = "test"
        train_df = full_df[~full_df["query_id"].isin(test_df["query_id"])].copy()
        full_df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        full_df, test_df = build_dataset(target_size=args.target_size)
    train_only = full_df[full_df["split"] == "train"]

    os.makedirs("data", exist_ok=True)
    train_only.to_parquet(args.output, index=False)
    test_df.to_parquet(args.test_output, index=False)

    logger.info("Saved %d train queries to %s", len(train_only), args.output)
    logger.info("Saved %d test queries to %s", len(test_df), args.test_output)
    logger.info("Ambiguity distribution:\n%s", train_only["ambiguity_type"].value_counts())


if __name__ == "__main__":
    main()
