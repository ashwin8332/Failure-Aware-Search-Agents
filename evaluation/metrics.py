"""Evaluation metrics for failure-aware agents."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from shared.rewards import exact_match, token_f1


def expected_calibration_error(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> float:
    """Compute ECE over equal-width confidence bins."""
    if not confidences:
        return 0.0
    conf = np.array(confidences)
    acc = np.array(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def hallucination_rate(
    confidences: List[float],
    correct: List[bool],
    threshold: float = 0.7,
) -> float:
    """Fraction of wrong answers with high confidence."""
    if not confidences:
        return 0.0
    wrong_high = sum(
        1 for c, ok in zip(confidences, correct) if not ok and c > threshold
    )
    return wrong_high / len(confidences)


def research_rate(
    triggered: List[bool],
    low_quality: List[bool],
) -> float:
    """Re-search rate on low-quality retrieval queries."""
    eligible = sum(low_quality)
    if eligible == 0:
        return 0.0
    return sum(1 for t, lq in zip(triggered, low_quality) if lq and t) / eligible


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate evaluation records into report metrics."""
    confidences = [r["expressed_confidence"] for r in records]
    correct = [r["is_correct"] for r in records]
    triggered = [r.get("triggered_research", False) for r in records]
    low_q = [r.get("retrieval_quality", 1.0) < 0.5 for r in records]
    f1_scores = [
        token_f1(r.get("final_answer", ""), r.get("gold_answer", "")) for r in records
    ]

    return {
        "ece": expected_calibration_error(confidences, correct),
        "hallucination_rate": hallucination_rate(confidences, correct),
        "overconfidence_rate": hallucination_rate(confidences, correct, threshold=0.7),
        "task_success_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "research_rate_low_quality": research_rate(triggered, low_q),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "n_queries": len(records),
    }
