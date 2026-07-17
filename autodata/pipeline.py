from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from .evaluation import CandidateEvaluator
from .models import Candidate, DatasetReport, SourceDocument, TaskSpec
from .providers import TextModel
from .quality import DatasetQualityController
from .tasks import ADAPTERS
from .utility import AdaptiveUtilityGate


class DatasetBuilder:
    """Agentic generation with cumulative, dataset-level feedback."""

    def __init__(self, generator: TextModel, evaluator: CandidateEvaluator,
                 quality: DatasetQualityController | None = None, utility_gate: AdaptiveUtilityGate | None = None):
        self.generator, self.evaluator = generator, evaluator
        self.quality = quality or DatasetQualityController()
        self.utility_gate = utility_gate
        self.last_attempts: list[Candidate] = []

    def build(self, specs: Iterable[TaskSpec], sources: Iterable[SourceDocument], *,
              rounds_per_source: int = 3) -> tuple[list[Candidate], DatasetReport]:
        sources, specs = list(sources), list(specs)
        candidates: list[Candidate] = []
        self.last_attempts = []
        feedback: list[str] = []
        requested_capabilities = [cap for spec in specs for cap in spec.capabilities]
        for spec in specs:
            adapter = ADAPTERS[spec.kind]
            for source in sources:
                for _ in range(min(rounds_per_source, spec.max_rounds)):
                    try:
                        raw = self.generator.complete(adapter.generation_prompt(spec, source, feedback))
                        candidate = adapter.parse(raw, spec, source, uuid4().hex[:12])
                        structural_issues = adapter.validate(candidate)
                        if structural_issues:
                            feedback.extend(structural_issues)
                            candidate.evaluation = {"status": "rejected_structure", "reasons": structural_issues}
                            self.last_attempts.append(candidate)
                            continue
                        evaluation = self.evaluator.evaluate(spec, candidate)
                        candidate.evaluation = evaluation.as_dict()
                        utility_decision = self.utility_gate.assess(evaluation) if self.utility_gate else None
                        if not self.evaluator.accepts(spec, evaluation):
                            feedback.extend(self._failure_feedback(evaluation))
                            self.last_attempts.append(candidate)
                            continue
                        if utility_decision and not utility_decision.accepted:
                            feedback.append(f"UTILITY GATE: {utility_decision.reason}")
                            candidate.evaluation["utility_gate"] = utility_decision.reason
                            self.last_attempts.append(candidate)
                            continue
                        keep, reason = self.quality.keep(candidate, candidates)
                        if not keep:
                            feedback.append(reason or "dataset redundancy")
                            candidate.evaluation["status"] = "rejected_dataset_quality"
                            self.last_attempts.append(candidate)
                            continue
                        candidate.accepted = True
                        candidates.append(candidate)
                        self.last_attempts.append(candidate)
                    except ValueError as exc:
                        feedback.append(str(exc))
                report = self.quality.analyze(candidates, requested_capabilities)
                feedback.extend(report.recommendations)
        return candidates, self.quality.analyze(candidates, requested_capabilities)

    @staticmethod
    def _failure_feedback(evaluation) -> list[str]:
        """Paper-style, actionable buckets fed verbatim to the next challenger round."""
        if evaluation.suggestion_for_writer:
            return [f"IMPROVE: {evaluation.suggestion_for_writer}"]
        if evaluation.weak_score is not None and evaluation.weak_score > 0.65:
            return ["TOO EASY: weak solver was too successful. Use an entirely new reasoning angle requiring deeper multi-step reasoning."]
        if evaluation.strong_score is not None and evaluation.strong_score < 0.60:
            return ["FAILED ON STRONG: make the task more tractable and ensure the reference/rubric are grounded and unambiguous."]
        return [f"FAILED QUALITY: {reason}" for reason in evaluation.reasons] or ["FAILED QUALITY: generate a distinct, better-specified task."]
