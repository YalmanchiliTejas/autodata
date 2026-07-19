"""Outer-loop optimization for the data-scientist prompt harness.

The inner loop improves individual tasks. This module improves the instructions
that govern that loop, using held-out source documents as the acceptance gate.
It deliberately mutates versioned prompt guidance rather than executing model-
generated code diffs.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import Protocol

from .json_output import extract_json_object
from .models import SourceDocument, TaskSpec
from .pipeline import DatasetBuilder
from .providers import TextModel


@dataclass(slots=True)
class HarnessVariant:
    id: str
    guidance: str
    score: float = float("-inf")
    parent_id: str | None = None
    rationale: str = ""


@dataclass(slots=True)
class MetaIteration:
    iteration: int
    parent_id: str
    candidate_id: str
    parent_score: float
    candidate_score: float
    accepted: bool
    rationale: str


class HarnessEvaluator(Protocol):
    def score(self, spec: TaskSpec, sources: list[SourceDocument]) -> tuple[float, str]: ...


class BuilderHarnessEvaluator:
    """Scores prompt variants by accepted-data rate on held-out source material."""

    def __init__(self, builder: DatasetBuilder, *, rounds_per_source: int = 3, repeats: int = 2):
        self.builder, self.rounds_per_source, self.repeats = builder, rounds_per_source, repeats

    def score(self, spec: TaskSpec, sources: list[SourceDocument]) -> tuple[float, str]:
        scores: list[float] = []
        diagnostics: list[str] = []
        for _ in range(self.repeats):
            _, report = self.builder.build([spec], sources, rounds_per_source=self.rounds_per_source)
            # Acceptance rate rewards discriminative examples; issue penalties
            # make a superficially high rate from redundant data less attractive.
            value = report.accepted / max(1, report.total) - 0.05 * len(report.quality_issues)
            scores.append(value)
            diagnostics.extend(report.quality_issues)
        return sum(scores) / len(scores), "; ".join(diagnostics[-12:]) or "No dataset-level issues reported."


class MetaOptimizer:
    """Prompt-harness evolution modeled after the paper's meta-optimization.

    It selects a parent from a population, asks an analyzer/mutator for a narrow
    guidance revision based on trajectory diagnostics, and evaluates parent and
    child on held-out sources. A child enters the population only on a strict
    held-out improvement, avoiding training-set-only prompt overfitting.
    """

    def __init__(self, mutator: TextModel, evaluator: HarnessEvaluator, *, temperature: float = 0.1,
                 seed: int = 0):
        self.mutator, self.evaluator, self.temperature = mutator, evaluator, temperature
        self.random = random.Random(seed)
        self.history: list[MetaIteration] = []

    def optimize(self, spec: TaskSpec, training_sources: list[SourceDocument], validation_sources: list[SourceDocument],
                 *, iterations: int = 20) -> tuple[TaskSpec, list[HarnessVariant]]:
        baseline = HarnessVariant("baseline", "")
        baseline.score, _ = self.evaluator.score(spec, validation_sources)
        population = [baseline]
        for iteration in range(1, iterations + 1):
            parent = self._select(population)
            parent_spec = self._apply(spec, parent.guidance)
            _, training_diagnostics = self.evaluator.score(parent_spec, training_sources)
            revision, rationale = self._propose(parent_spec, parent, training_diagnostics)
            child = HarnessVariant(f"iter-{iteration}", self._merge(parent.guidance, revision), parent_id=parent.id, rationale=rationale)
            # Re-evaluate parent and child on validation because model sampling is noisy.
            parent_score, _ = self.evaluator.score(parent_spec, validation_sources)
            child.score, _ = self.evaluator.score(self._apply(spec, child.guidance), validation_sources)
            accepted = child.score > parent_score
            self.history.append(MetaIteration(iteration, parent.id, child.id, parent_score, child.score, accepted, rationale))
            if accepted:
                population.append(child)
        best = max(population, key=lambda item: item.score)
        return self._apply(spec, best.guidance), population

    def _select(self, population: list[HarnessVariant]) -> HarnessVariant:
        ceiling = max(item.score for item in population)
        weights = [math.exp((item.score - ceiling) / max(self.temperature, 1e-6)) for item in population]
        return self.random.choices(population, weights=weights, k=1)[0]

    def _propose(self, spec: TaskSpec, parent: HarnessVariant, diagnostics: str) -> tuple[str, str]:
        prompt = f"""You improve a synthetic-data scientist's generation strategy. Diagnose systematic failure patterns and propose ONE narrow, testable prompt-guidance change. Do not weaken acceptance criteria, alter solver prompts, reveal answers, or change the task's safety/environment constraints.
Task instructions: {spec.instructions}
Current added guidance: {parent.guidance or '(none)'}
Observed training diagnostics: {diagnostics}
Return ONLY JSON: {{"revision": "imperative guidance to append", "rationale": "why this addresses the diagnostic"}}."""
        try:
            response = extract_json_object(self.mutator.complete(prompt), {"revision", "rationale"}, producer="mutator")
            revision = str(response["revision"]).strip()
            rationale = str(response.get("rationale", "")).strip()
            if not revision:
                raise ValueError("empty revision")
            return revision, rationale
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return "Preserve all existing task, environment, and rubric constraints.", f"invalid mutator response: {exc}"

    @staticmethod
    def _merge(existing: str, revision: str) -> str:
        return "\n".join(item for item in (existing, revision) if item)

    @staticmethod
    def _apply(spec: TaskSpec, guidance: str) -> TaskSpec:
        if not guidance:
            return spec
        return replace(spec, instructions=f"{spec.instructions}\n\nDATA-SCIENTIST STRATEGY:\n{guidance}")
