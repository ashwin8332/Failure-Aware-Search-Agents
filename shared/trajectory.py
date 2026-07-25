"""Trajectory serialization for DPO preference pairs."""

from __future__ import annotations

from typing import Any, Dict, List


def format_trajectory_text(traj: Dict[str, Any]) -> str:
    """Format a trajectory dict as training text for DPO."""
    question = traj.get("question", traj.get("prompt", ""))
    lines = [f"Question: {question}", ""]
    for step in traj.get("steps", []):
        lines.append(f"Thought: {step.get('thought', '')}")
        lines.append(f"Action: {step.get('action', '')}[{step.get('action_input', '')}]")
        lines.append(f"Observation: {step.get('observation', '')}")
        lines.append("")
    conf_pct = int(float(traj.get("expressed_confidence", 0.75)) * 100)
    lines.append(f"Final Answer: {traj.get('final_answer', '')}")
    lines.append(f"Confidence: {conf_pct}%")
    return "\n".join(lines)


def trajectory_from_rollout(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize rollout JSONL record."""
    return {
        "trajectory_id": record["trajectory_id"],
        "query_id": record["query_id"],
        "question": record.get("question", ""),
        "steps": record.get("steps", []),
        "final_answer": record.get("final_answer", ""),
        "expressed_confidence": record.get("expressed_confidence", 0.75),
        "triggered_research": record.get("triggered_research", False),
        "retrieval_quality": record.get("retrieval_quality", 0.5),
        "gold_answer": record.get("gold_answer"),
        "ambiguity_type": record.get("ambiguity_type"),
        "composite_reward": record.get("composite_reward"),
    }
