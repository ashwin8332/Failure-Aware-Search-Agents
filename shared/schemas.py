"""Shared data schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class Trajectory:
    trajectory_id: str
    query_id: str
    question: str
    steps: List[TrajectoryStep]
    final_answer: str
    expressed_confidence: float
    triggered_research: bool
    retrieval_quality: float
    gold_answer: Optional[str] = None
    ambiguity_type: Optional[str] = None
    r_fact: Optional[float] = None
    r_calib: Optional[float] = None
    r_recovery: Optional[float] = None
    composite_reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    position: int = 0
    quality_score: float = 0.5


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    reward_gap: float
    ambiguity_type: str
    query_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationVerdict:
    verdict: str  # confirmed | conflicted | unverified
    confidence_adjustment: float
    conflict_summary: Optional[str] = None
    verified_facts: List[str] = field(default_factory=list)
