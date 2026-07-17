from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .models import Candidate, SourceDocument, TaskSpec
from .prompts import challenger_contract, rubric_authoring_contract
from .rubrics import validate_rubric


class TaskAdapter(ABC):
    """Task-specific prompting and structural validation, shared pipeline semantics."""

    kind: str
    required_fields: tuple[str, ...] = ("input", "answer")

    @abstractmethod
    def generation_prompt(self, spec: TaskSpec, source: SourceDocument, feedback: list[str]) -> str: ...

    def parse(self, raw: str, spec: TaskSpec, source: SourceDocument, candidate_id: str) -> Candidate:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("generator must return one JSON object") from exc
        if not isinstance(payload, dict):
            raise ValueError("generator output must be a JSON object")
        missing = [key for key in self.required_fields if not payload.get(key)]
        if missing:
            raise ValueError(f"candidate missing required fields: {', '.join(missing)}")
        capabilities = payload.pop("capabilities", spec.capabilities)
        difficulty = payload.pop("difficulty", spec.target_difficulty)
        return Candidate(candidate_id, spec.name, source.id, payload, list(capabilities), difficulty,
                         provenance={"source_metadata": source.metadata, "profile": spec.profile,
                                     "forbidden_content": spec.forbidden_content, "environment": spec.environment,
                                     "require_rubric": spec.require_rubric})

    def validate(self, candidate: Candidate) -> list[str]:
        return self._generic_validation(candidate) + self._rubric_validation(candidate)

    @staticmethod
    def _rubric_validation(candidate: Candidate) -> list[str]:
        issues = validate_rubric(candidate.payload, candidate.capabilities, candidate.provenance.get("environment", {}))
        if candidate.provenance.get("require_rubric") and "rubric" not in candidate.payload:
            issues.append("RL-oriented task requires a challenger-authored rubric")
        return issues

    @staticmethod
    def _generic_validation(candidate: Candidate) -> list[str]:
        if candidate.provenance.get("profile") == "generic":
            audit = candidate.payload.get("self_audit")
            expected = {"grounded", "within_scope", "no_answer_leakage", "fairness_checked", "verifiable"}
            if not isinstance(audit, dict):
                return ["generic task requires a self_audit object"]
            if set(audit) != expected or not all(audit.values()):
                return ["generic self_audit must contain all required checks and every check must be true"]
            serialized = json.dumps(candidate.payload).lower()
            blocked = [term for term in candidate.provenance.get("forbidden_content", []) if term.lower() in serialized]
            if blocked:
                return [f"generic task contains prohibited content or feature: {', '.join(blocked)}"]
        return []


class QuestionAnswerAdapter(TaskAdapter):
    kind = "qa"
    required_fields = ("question", "answer", "rubric")

    def generation_prompt(self, spec, source, feedback):
        return _prompt(spec, source, feedback, "Create one grounded question-answer task. The rubric must make evaluation possible without hidden knowledge.")


class MathAdapter(TaskAdapter):
    kind = "math"
    required_fields = ("problem", "answer", "solution", "verification")

    def generation_prompt(self, spec, source, feedback):
        return _prompt(spec, source, feedback, "Create one solvable mathematical problem. Include a rigorous solution and a machine- or human-checkable verification rule.")


class CodingAdapter(TaskAdapter):
    kind = "coding"
    required_fields = ("prompt", "reference_solution", "tests")

    def generation_prompt(self, spec, source, feedback):
        return _prompt(spec, source, feedback, "Create one programming task. Include an executable reference solution and deterministic tests. Do not depend on network, files, or unspecified libraries.")

    def validate(self, candidate):
        tests = candidate.payload.get("tests")
        generic_issues = self._generic_validation(candidate)
        if not isinstance(tests, list) or not tests:
            return generic_issues + ["coding task needs a non-empty tests array"]
        return generic_issues + self._rubric_validation(candidate)


class CustomAdapter(TaskAdapter):
    kind = "custom"
    required_fields = ()

    def generation_prompt(self, spec, source, feedback):
        schema = json.dumps(spec.output_schema or {"input": "...", "answer": "..."})
        return _prompt(spec, source, feedback, f"Create one task matching this exact JSON payload schema: {schema}.")

    def validate(self, candidate):
        # JSON Schema's `required` is enough for a lightweight generic contract;
        # applications can provide a stricter custom verifier as well.
        required = candidate.provenance.get("output_required", [])
        missing = [field for field in required if field not in candidate.payload]
        schema_issues = [f"custom task missing schema fields: {', '.join(missing)}"] if missing else []
        return self._generic_validation(candidate) + schema_issues + self._rubric_validation(candidate)

    def parse(self, raw, spec, source, candidate_id):
        candidate = super().parse(raw, spec, source, candidate_id)
        candidate.provenance["output_required"] = spec.output_schema.get("required", [])
        return candidate


class LegalAdapter(TaskAdapter):
    kind = "legal"
    required_fields = ("question", "rubric", "target_capabilities")

    def generation_prompt(self, spec, source, feedback):
        return _prompt(spec, source, feedback, "Create an application-style legal reasoning task from a transferable principle in the source.")

    def validate(self, candidate):
        generic_issues = self._generic_validation(candidate)
        rubric = candidate.payload.get("rubric")
        if not isinstance(rubric, list) or not 15 <= len(rubric) <= 25:
            return generic_issues + ["legal rubric must contain 15–25 criteria"]
        required = {"number", "criterion", "category", "capability", "weight_class", "weight"}
        malformed = [str(index) for index, item in enumerate(rubric) if set(item) != required]
        return generic_issues + ([f"legal rubric items must use the exact six-key schema: {', '.join(malformed)}"] if malformed else []) + self._rubric_validation(candidate)


def _prompt(spec: TaskSpec, source: SourceDocument, feedback: list[str], task_instruction: str) -> str:
    prior = "\n".join(f"- {item}" for item in feedback[-8:]) or "- No prior feedback."
    return f"""You generate high-value synthetic training data.
Task family: {spec.name} ({spec.kind})
Task instructions: {spec.instructions}
Desired capabilities: {', '.join(spec.capabilities) or 'infer from instructions'}
Desired difficulty: {spec.target_difficulty}
Source policy: {spec.source_policy}
Hard constraints: {', '.join(spec.constraints) or 'None beyond this task contract.'}
Prohibited content or task features: {', '.join(spec.forbidden_content) or 'None specified.'}
Environment contract: {json.dumps(spec.environment or {'success_conditions': 'derive from task instructions'}, ensure_ascii=False)}
{task_instruction}
Grounding source ({source.id}):
{source.content}
Dataset-level feedback to address:
{prior}
{challenger_contract(spec)}
{rubric_authoring_contract(spec)}
Return ONLY a JSON object. Include `capabilities` and `difficulty` in addition to task fields."""


ADAPTERS: dict[str, TaskAdapter] = {adapter.kind: adapter for adapter in (
    QuestionAnswerAdapter(), MathAdapter(), CodingAdapter(), LegalAdapter(), CustomAdapter()
)}
