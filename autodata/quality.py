from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from .models import Candidate, DatasetReport


def _terms(candidate: Candidate) -> set[str]:
    text = " ".join(str(value) for value in candidate.payload.values()).lower()
    return set(re.findall(r"[a-z0-9_]{3,}", text))


class DatasetQualityController:
    """Dataset-level controls: redundancy, coverage, distribution, and leakage signals."""

    def __init__(self, *, duplicate_threshold: float = 0.82, min_capability_examples: int = 2,
                 max_source_share: float = 0.35):
        self.duplicate_threshold = duplicate_threshold
        self.min_capability_examples = min_capability_examples
        self.max_source_share = max_source_share

    def analyze(self, candidates: list[Candidate], requested_capabilities: list[str] = ()) -> DatasetReport:
        accepted = [item for item in candidates if item.accepted]
        capability_counts = Counter(cap for item in accepted for cap in item.capabilities)
        difficulty_counts = Counter(item.difficulty for item in accepted)
        source_counts = Counter(item.source_id for item in accepted)
        pairs = self._duplicates(accepted)
        issues: list[str] = []
        recommendations: list[str] = []
        for capability in requested_capabilities:
            if capability_counts[capability] < self.min_capability_examples:
                issues.append(f"underrepresented capability: {capability}")
                recommendations.append(f"Generate more examples targeting capability '{capability}'.")
        if pairs:
            issues.append(f"{len(pairs)} near-duplicate candidate pairs")
            recommendations.append("Use distinct reasoning patterns, entities, and solution approaches; avoid paraphrases.")
        if accepted:
            for source_id, count in source_counts.items():
                if count / len(accepted) > self.max_source_share:
                    issues.append(f"source concentration: {source_id} supplies {count}/{len(accepted)} examples")
                    recommendations.append("Sample underused sources before generating more examples.")
        if len(difficulty_counts) <= 1 and len(accepted) > 3:
            issues.append("difficulty distribution lacks variety")
            recommendations.append("Generate a calibrated mix of easy, medium, and hard tasks.")
        return DatasetReport(len(candidates), len(accepted), pairs, dict(capability_counts),
                             dict(difficulty_counts), dict(source_counts), issues, recommendations)

    def keep(self, candidate: Candidate, existing: list[Candidate]) -> tuple[bool, str | None]:
        candidate_terms = _terms(candidate)
        for prior in existing:
            if not prior.accepted:
                continue
            prior_terms = _terms(prior)
            similarity = len(candidate_terms & prior_terms) / max(1, len(candidate_terms | prior_terms))
            if similarity >= self.duplicate_threshold:
                return False, f"near duplicate of {prior.id} (Jaccard {similarity:.2f})"
        return True, None

    def _duplicates(self, candidates: list[Candidate]) -> list[tuple[str, str]]:
        result = []
        for left, right in combinations(candidates, 2):
            a, b = _terms(left), _terms(right)
            if len(a & b) / max(1, len(a | b)) >= self.duplicate_threshold:
                result.append((left.id, right.id))
        return result
