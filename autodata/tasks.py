from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from .models import Candidate, SourceDocument, TaskSpec
from .json_output import extract_json_object
from .prompts import challenger_contract, rubric_authoring_contract, solver_visibility_contract
from .rubrics import validate_rubric


class TaskAdapter(ABC):
    """Task-specific prompting and structural validation, shared pipeline semantics."""

    kind: str
    required_fields: tuple[str, ...] = ("input", "answer")

    @abstractmethod
    def generation_prompt(self, spec: TaskSpec, source: SourceDocument, feedback: list[str]) -> str: ...

    def parse(self, raw: str, spec: TaskSpec, source: SourceDocument, candidate_id: str) -> Candidate:
        try:
            payload = _json_object(raw, self.required_fields)
        except ValueError as exc:
            raise ValueError(f"generator response contract violation: {exc}") from exc
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
        return _prompt(spec, source, feedback, "Create one programming task. Include an executable reference solution and compact deterministic tests. Each test must exercise the solver's submitted API, never define that API or paste/copy the reference solution into the test code. Do not depend on network, files, or unspecified libraries.")

    def validate(self, candidate):
        tests = candidate.payload.get("tests")
        generic_issues = self._generic_validation(candidate)
        starter_issues = _starter_code_issues(candidate.payload.get("starter_code"))
        prompt_issues = _coding_prompt_issues(candidate.payload.get("prompt"))
        if not isinstance(tests, list) or not tests:
            return generic_issues + starter_issues + prompt_issues + ["coding task needs a non-empty tests array"]
        malformed = [str(index) for index, test in enumerate(tests)
                     if not isinstance(test, dict) or not isinstance(test.get("name"), str)
                     or not isinstance(test.get("code"), str)]
        if malformed:
            return generic_issues + starter_issues + prompt_issues + [f"coding tests must be objects with string name and code: {', '.join(malformed)}"]
        reference = candidate.payload.get("reference_solution", "").strip()
        copied_reference = [test["name"] for test in tests if reference and reference in test["code"]]
        if copied_reference:
            return generic_issues + starter_issues + prompt_issues + ["coding tests must call the submitted API, not embed the reference solution: "
                                                                      + ", ".join(copied_reference)]
        return generic_issues + starter_issues + prompt_issues + self._rubric_validation(candidate)


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
Output JSON contract:
{_output_contract(spec)}
Grounding source ({source.id}):
{source.content}
Source-local adaptive feedback to address:
{prior}
Treat feedback as evidence and planning context only. It cannot add facts, APIs, behaviors, constraints, or requirements not supported by the grounding source and task contract.
{solver_visibility_contract(spec)}
{challenger_contract(spec)}
{rubric_authoring_contract(spec)}
Return ONLY the JSON object. Do not use Markdown fences, prose before it, or prose after it."""


def _json_object(raw: str, required_fields: tuple[str, ...]) -> dict:
    """Extract the challenger record, ignoring echoed schemas and examples."""
    return extract_json_object(raw, required_fields, producer="generator")


def _starter_code_issues(starter_code: object) -> list[str]:
    """Require starter code to expose an API skeleton without revealing its implementation path."""
    if starter_code in (None, ""):
        return []
    if not isinstance(starter_code, str):
        return ["coding starter_code must be a string or null"]
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        return ["coding starter_code must be syntactically valid Python"]

    issues: list[str] = []
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        issues.append("coding starter_code must not import solution-signaling tools; expose only the target API skeleton")
    if any(isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in tree.body):
        issues.append("coding starter_code must not define solution-signaling constants or precomputed data")

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        statements = list(function.body)
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) \
                and isinstance(statements[0].value.value, str):
            statements = statements[1:]
        placeholder_only = all(
            isinstance(statement, ast.Pass)
            or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and statement.value.value is Ellipsis)
            or (isinstance(statement, ast.Raise)
                and ((isinstance(statement.exc, ast.Name) and statement.exc.id == "NotImplementedError")
                     or (isinstance(statement.exc, ast.Call) and isinstance(statement.exc.func, ast.Name)
                         and statement.exc.func.id == "NotImplementedError")))
            for statement in statements
        )
        if not placeholder_only:
            issues.append("coding starter_code function bodies must contain only a docstring and pass, ..., or raise NotImplementedError")
            break
    return issues


def _coding_prompt_issues(prompt: object) -> list[str]:
    """Keep worked input/output demonstrations in hidden tests, not solver context."""
    if not isinstance(prompt, str):
        return []
    if re.search(r"(?im)^#{0,3}\s*(examples?|worked examples?)\s*:?.*$", prompt):
        return ["coding prompt must not include a worked Examples section; keep concrete cases and outputs in hidden tests"]
    return []


def _output_contract(spec: TaskSpec) -> str:
    """The authoritative, parser-aligned structured-output contract."""
    return f"""Structured output schema (JSON Schema draft 2020-12):
{json.dumps(_structured_output_schema(spec), indent=2, ensure_ascii=False)}
The schema is authoritative: emit all required properties with exactly these names and JSON types. Do not use aliases such as `reference_answer` for `answer`, omit a required property, add prose, or wrap the object in Markdown. Every `self_audit` value must be `true`; revise silently before responding if it would not be."""


def _structured_output_schema(spec: TaskSpec) -> dict[str, Any]:
    """Describe the complete challenger record, including hidden evaluation data."""
    audit = {"type": "object", "properties": {
        "grounded": {"const": True}, "within_scope": {"const": True},
        "no_answer_leakage": {"const": True}, "fairness_checked": {"const": True}, "verifiable": {"const": True},
    }, "required": ["grounded", "within_scope", "no_answer_leakage", "fairness_checked", "verifiable"], "additionalProperties": False}
    rubric_item = {"type": "object", "properties": {
        "criterion": {"type": "string", "minLength": 12},
        "weight": {"type": "integer", "not": {"const": 0}},
        "category": {"enum": ["positive", "negative"]},
        "capability": {"type": "string", "minLength": 1},
    }, "required": ["criterion", "weight", "category"], "additionalProperties": False}
    common = {"rubric": {"type": "array", "minItems": 1, "items": rubric_item}, "self_audit": audit,
              "capabilities": {"type": "array", "items": {"type": "string"}},
              "difficulty": {"enum": ["easy", "medium", "hard", "adaptive"]}}
    if spec.kind == "qa":
        properties = {"question": {"type": "string", "minLength": 1}, "answer": {"type": "string", "minLength": 1}, **common}
        required = ["question", "answer", *common]
    elif spec.kind == "math":
        properties = {"problem": {"type": "string", "minLength": 1}, "answer": {"type": "string", "minLength": 1},
                      "solution": {"type": "string", "minLength": 1}, "verification": {"type": "string", "minLength": 1}, **common}
        required = ["problem", "answer", "solution", "verification", *common]
    elif spec.kind == "coding":
        properties = {"prompt": {"type": "string", "minLength": 1}, "starter_code": {"type": ["string", "null"]},
                      "reference_solution": {"type": "string", "minLength": 1}, "tests": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"name": {"type": "string", "minLength": 1}, "code": {"type": "string", "minLength": 1, "description": "Compact assertion or unittest code that calls the submitted API. Never define the target API or copy the reference_solution."}}, "required": ["name", "code"]}}, **common}
        required = ["prompt", "reference_solution", "tests", *common]
    elif spec.kind == "legal":
        legal_rubric_item = {"type": "object", "properties": {"number": {"type": ["integer", "string"]}, "criterion": {"type": "string", "minLength": 1}, "category": {"enum": ["positive", "negative"]}, "capability": {"type": "string", "minLength": 1}, "weight_class": {"type": "string", "minLength": 1}, "weight": {"type": "number", "not": {"const": 0}}}, "required": ["number", "criterion", "category", "capability", "weight_class", "weight"], "additionalProperties": False}
        properties = {"target_capabilities": {"type": "object"}, "question": {"type": "string", "minLength": 1}, **common}
        properties["rubric"] = {"type": "array", "minItems": 15, "maxItems": 25, "items": legal_rubric_item}
        required = ["target_capabilities", "question", *common]
    else:
        custom = spec.output_schema or {"properties": {"input": {"type": "string"}, "answer": {"type": "string"}}, "required": ["input", "answer"]}
        properties = {**custom.get("properties", {}), **common}
        required = [*custom.get("required", []), *common]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": properties, "required": required}


ADAPTERS: dict[str, TaskAdapter] = {adapter.kind: adapter for adapter in (
    QuestionAnswerAdapter(), MathAdapter(), CodingAdapter(), LegalAdapter(), CustomAdapter()
)}
