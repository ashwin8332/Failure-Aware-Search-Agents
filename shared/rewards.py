"""Reward scoring for agent trajectories."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BERT_SCORE_AVAILABLE = False
try:
    from bert_score import score as bert_score_fn

    _BERT_SCORE_AVAILABLE = True
except ImportError:
    bert_score_fn = None  # type: ignore


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def factuality_score(pred: str, gold: str) -> float:
    """BERTScore F1 with token_f1 fallback."""
    if _BERT_SCORE_AVAILABLE and bert_score_fn is not None:
        try:
            _, _, f1 = bert_score_fn(
                [pred], [gold], lang="en", verbose=False, device="cpu"
            )
            return float(f1.item())
        except Exception as exc:
            logger.warning("BERTScore failed, using token_f1: %s", exc)
    em = 1.0 if exact_match(pred, gold) else 0.0
    return max(em, token_f1(pred, gold))


def calibration_score(expressed_conf: float, is_correct: bool) -> float:
    """Penalize miscalibrated confidence."""
    target = 1.0 if is_correct else 0.0
    conf = max(0.0, min(1.0, expressed_conf))
    return 1.0 - abs(conf - target)


def recovery_score(triggered_research: bool, retrieval_quality: float) -> float:
    """Reward re-search when retrieval was poor."""
    return float(triggered_research) * (1.0 - max(0.0, min(1.0, retrieval_quality)))


def composite_reward(
    r_fact: float,
    r_calib: float,
    r_recovery: float,
    w_fact: float = 0.5,
    w_calib: float = 0.3,
    w_recovery: float = 0.2,
) -> float:
    return w_fact * r_fact + w_calib * r_calib + w_recovery * r_recovery


def score_trajectory(traj: Dict[str, Any], gold_answer: str) -> Dict[str, float]:
    """Score a trajectory dict and return component rewards."""
    final_answer = traj.get("final_answer", "")
    expressed_conf = float(traj.get("expressed_confidence", 0.75))
    triggered = bool(traj.get("triggered_research", False))
    retrieval_q = float(traj.get("retrieval_quality", 0.5))

    r_fact = factuality_score(final_answer, gold_answer)
    is_correct = exact_match(final_answer, gold_answer) or r_fact >= 0.8
    r_calib = calibration_score(expressed_conf, is_correct)
    r_recovery = recovery_score(triggered, retrieval_q)
    total = composite_reward(r_fact, r_calib, r_recovery)

    return {
        "r_fact": round(r_fact, 4),
        "r_calib": round(r_calib, 4),
        "r_recovery": round(r_recovery, 4),
        "composite_reward": round(total, 4),
        "is_correct": is_correct,
    }


def extract_confidence(text: str, default: float = 0.75) -> float:
    """Extract expressed confidence from agent output."""
    patterns = [
        r"(?:I am|confidence:|certainty:)\s*(\d{1,3})\s*%",
        r"confidence:\s*(\d{1,3})",
        r"(\d{1,3})\s*%\s*confident",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return min(1.0, max(0.0, int(m.group(1)) / 100.0))
    if "low" in text.lower() and "confidence" in text.lower():
        return 0.35
    if "insufficient_evidence" in text.lower():
        return 0.3
    return default
