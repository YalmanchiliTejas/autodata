"""Local rubric checks before expensive model rollouts.

This does not pretend lexical checks prove correctness. It catches malformed,
redundant, and unexecutable rewards; the model-based rubric verifier supplies
the semantic, source-grounded check in the next gate.
"""
from __future__ import annotations

import re


GENERIC_CRITERIA = ("is correct", "is helpful", "is clear", "provides a good", "uses a structured")


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", text.lower()))


def validate_rubric(payload: dict, capabilities: list[str], environment: dict) -> list[str]:
    rubric = payload.get("rubric")
    if rubric is None:
        return []
    if not isinstance(rubric, list) or not rubric:
        return ["rubric must be a non-empty array"]
    issues: list[str] = []
    texts: list[str] = []
    for index, item in enumerate(rubric):
        if not isinstance(item, dict) or not isinstance(item.get("criterion"), str):
            issues.append(f"rubric item {index} needs a string criterion")
            continue
        criterion = item["criterion"].strip()
        texts.append(criterion)
        if len(criterion) < 12 or any(phrase in criterion.lower() for phrase in GENERIC_CRITERIA):
            issues.append(f"rubric item {index} is not an observable task-specific check")
        if ("weight" not in item or isinstance(item.get("weight"), bool)
                or not isinstance(item.get("weight"), int) or item["weight"] == 0):
            issues.append(f"rubric item {index} needs a non-zero integer weight")
        if item.get("category") not in (None, "positive", "negative"):
            issues.append(f"rubric item {index} has invalid category")
    for left in range(len(texts)):
        for right in range(left):
            a, b = _terms(texts[left]), _terms(texts[right])
            if a and b and len(a & b) / len(a | b) >= 0.80:
                issues.append(f"rubric criteria {right} and {left} are near-duplicates")
    tagged = {item.get("capability") for item in rubric if isinstance(item, dict)}
    missing = [capability for capability in capabilities if capability not in tagged]
    if capabilities and any("capability" in item for item in rubric if isinstance(item, dict)) and missing:
        issues.append(f"rubric does not cover requested capabilities: {', '.join(missing)}")
    if not environment or not environment.get("success_conditions"):
        issues.append("rubric-bearing task needs an environment contract with success_conditions")
    return issues
