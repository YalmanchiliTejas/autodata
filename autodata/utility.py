"""Calibration and held-out improvement gates for useful RL data."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean

from .models import Evaluation


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


@dataclass(slots=True)
class UtilityDecision:
    ready: bool
    accepted: bool
    reason: str


class AdaptiveUtilityGate:
    """Learns reward-quality thresholds from probe rollouts, not task labels.

    The gate is intentionally distributional: it rejects all-zero/all-saturated
    reward groups and then selects examples whose variance and strong-vs-weak gap
    are high relative to the observed source/task population.
    """

    def __init__(self, *, calibration_examples: int = 20, selection_quantile: float = 0.60):
        self.calibration_examples = calibration_examples
        self.selection_quantile = selection_quantile
        self.observations: list[Evaluation] = []

    def assess(self, evaluation: Evaluation) -> UtilityDecision:
        if not evaluation.valid or evaluation.weak_score is None or evaluation.strong_score is None:
            return UtilityDecision(self.ready, False, "invalid solver/judge evaluation")
        self.observations.append(evaluation)
        if not self.ready:
            return UtilityDecision(False, False, "calibrating reward distribution; generate diverse probe tasks")
        weak_std = _std(evaluation.weak_rollouts)
        gap = evaluation.gap or 0.0
        baseline_std = _quantile([_std(item.weak_rollouts) for item in self.observations], self.selection_quantile)
        baseline_gap = _quantile([item.gap or 0.0 for item in self.observations], self.selection_quantile)
        if weak_std == 0:
            return UtilityDecision(True, False, "all weak rollouts have the same reward; no within-group RL signal")
        if weak_std < baseline_std or gap < baseline_gap:
            return UtilityDecision(True, False, "reward variance or capability gap is below calibrated population level")
        return UtilityDecision(True, True, "above calibrated variance and capability-gap baselines")

    @property
    def ready(self) -> bool:
        return len(self.observations) >= self.calibration_examples


@dataclass(slots=True)
class PilotResult:
    baseline_scores: list[float]
    trained_scores: list[float]

    @property
    def improvement(self) -> float:
        return mean(self.trained_scores) - mean(self.baseline_scores)

    @property
    def paired_standard_error(self) -> float:
        deltas = [trained - base for base, trained in zip(self.baseline_scores, self.trained_scores)]
        return _std(deltas) / sqrt(max(1, len(deltas)))

    def useful(self) -> bool:
        """Require positive held-out gain larger than one paired standard error."""
        return self.improvement > self.paired_standard_error
