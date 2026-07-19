from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskKind = Literal["qa", "math", "coding", "legal", "custom"]


@dataclass(slots=True)
class TaskSpec:
    """Defines a task family without constraining the actual output schema."""

    name: str
    kind: TaskKind
    instructions: str
    capabilities: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] = field(default_factory=dict)
    target_difficulty: Literal["easy", "medium", "hard", "adaptive"] = "adaptive"
    verifier: str | None = None
    profile: Literal["generic", "cs_research", "legal_reasoning", "scientific_reasoning"] = "generic"
    max_rounds: int = 8
    constraints: list[str] = field(default_factory=list)
    forbidden_content: list[str] = field(default_factory=list)
    source_policy: Literal["grounded_only", "grounded_with_explicit_assumptions", "creative"] = "grounded_only"
    environment: dict[str, Any] = field(default_factory=dict)
    require_rubric: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceDocument:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    """A normalized dataset record. `payload` supports any task-specific shape."""

    id: str
    task_name: str
    source_id: str
    payload: dict[str, Any]
    capabilities: list[str] = field(default_factory=list)
    difficulty: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Evaluation:
    valid: bool
    weak_score: float | None
    strong_score: float | None
    judge_score: float
    reasons: list[str] = field(default_factory=list)
    weak_rollouts: list[float] = field(default_factory=list)
    strong_rollouts: list[float] = field(default_factory=list)
    verdict: str = "improve"
    suggestion_for_writer: str | None = None
    failure_kind: str | None = None
    quality_verifier: dict[str, Any] | None = None

    @property
    def gap(self) -> float | None:
        if self.weak_score is None or self.strong_score is None:
            return None
        return self.strong_score - self.weak_score

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "weak_score": self.weak_score, "strong_score": self.strong_score,
                "judge_score": self.judge_score, "gap": self.gap, "reasons": self.reasons,
                "weak_rollouts": self.weak_rollouts, "strong_rollouts": self.strong_rollouts,
                "verdict": self.verdict, "suggestion_for_writer": self.suggestion_for_writer,
                "failure_kind": self.failure_kind, "quality_verifier": self.quality_verifier}


@dataclass(slots=True)
class DatasetReport:
    total: int
    accepted: int
    duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    capability_counts: dict[str, int] = field(default_factory=dict)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    quality_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    attempted: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
