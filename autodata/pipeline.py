from __future__ import annotations

from collections.abc import Iterable
from collections import Counter
import json
import logging
from time import perf_counter
from uuid import uuid4

from .evaluation import CandidateEvaluator, solver_visible_payload
from .json_output import extract_json_object
from .models import Candidate, DatasetReport, SourceDocument, TaskSpec
from .prompts import orchestrator_reflection_prompt
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
        self.audit_events: list[dict] = []
        self.logger = logging.getLogger("autodata")

    def build(self, specs: Iterable[TaskSpec], sources: Iterable[SourceDocument], *,
              rounds_per_source: int = 3, stop_after_accept_per_source: bool = False) -> tuple[list[Candidate], DatasetReport]:
        sources, specs = list(sources), list(specs)
        candidates: list[Candidate] = []
        self.last_attempts = []
        self.audit_events = []
        requested_capabilities = [cap for spec in specs for cap in spec.capabilities]
        for spec in specs:
            adapter = ADAPTERS[spec.kind]
            attempt_limit = min(rounds_per_source, spec.max_rounds)
            for source_index, source in enumerate(sources, start=1):
                # Reflection, structure, and evaluation feedback is evidence
                # about this exact grounding chunk. Carrying it into another
                # chunk can import irrelevant facts and failed task designs.
                feedback: list[str] = []
                self.logger.info("task=%s source=%s/%s stage=feedback_scope_initialized scope=source_chunk",
                                 spec.name, source_index, len(sources))
                for attempt in range(1, attempt_limit + 1):
                    try:
                        self._audit(spec, source, attempt, "generating", source_index=source_index, source_total=len(sources))
                        if feedback:
                            self.logger.info("task=%s source=%s attempt=%s stage=challenger_generation_with_feedback feedback_items=%s latest=%s",
                                             spec.name, source_index, attempt, len(feedback), _log_preview(feedback[-1]))
                        started = perf_counter()
                        raw = self.generator.complete(adapter.generation_prompt(spec, source, feedback))
                        # Candidate records can contain a reference solution and
                        # many tests. Logging the full response makes normal runs
                        # unreadable and can expose hidden evaluation material.
                        self.logger.info("task=%s source=%s attempt=%s stage=challenger_complete seconds=%.1f response_chars=%s",
                                         spec.name, source_index, attempt, perf_counter() - started, len(raw))
                        candidate = adapter.parse(raw, spec, source, uuid4().hex[:12])
                        self.logger.info("candidate=%s stage=parsed fields=%s rubric_items=%s tests=%s",
                                         candidate.id, ",".join(sorted(candidate.payload)),
                                         len(candidate.payload.get("rubric", [])), len(candidate.payload.get("tests", [])))
                        task_surface = solver_visible_payload(spec, candidate)
                        serialized_task = json.dumps(task_surface, ensure_ascii=False, separators=(",", ":"))
                        self.logger.info("candidate=%s stage=challenger_task solver_visible_chars=%s fields=%s",
                                         candidate.id, len(serialized_task), ",".join(sorted(task_surface)))
                        self._audit(spec, source, attempt, "challenger_task", candidate=candidate,
                                    solver_visible_task=task_surface)
                        structural_issues = adapter.validate(candidate)
                        if structural_issues:
                            feedback.append(_feedback_packet("STRUCTURE", structural_issues))
                            candidate.evaluation = {"status": "rejected_structure", "reasons": structural_issues}
                            self.last_attempts.append(candidate)
                            self._audit(spec, source, attempt, "rejected_structure", candidate=candidate,
                                        reasons=structural_issues)
                            continue
                        self.logger.info("candidate=%s stage=structure_passed", candidate.id)
                        evaluation = self.evaluator.evaluate(spec, candidate, source_content=source.content)
                        candidate.evaluation = evaluation.as_dict()
                        utility_decision = self.utility_gate.assess(evaluation) if self.utility_gate else None
                        if not self.evaluator.accepts(spec, evaluation):
                            if attempt >= attempt_limit:
                                self.logger.info("candidate=%s stage=challenger_orchestrator_skipped reason=no_remaining_attempts",
                                                 candidate.id)
                                self._audit(spec, source, attempt, "orchestrator_skipped", candidate=candidate,
                                            reasons=["no generation attempt remains to consume orchestrator feedback"],
                                            evaluation={"failure_kind": evaluation.failure_kind,
                                                        "remaining_attempts": 0})
                            elif evaluation.failure_kind == "infrastructure":
                                message = _feedback_packet(
                                    "EVALUATION INFRASTRUCTURE",
                                    evaluation.reasons,
                                    suffix="Preserve the prior task strategy; this failure is not evidence about task quality.",
                                )
                                feedback.append(message)
                                self.logger.info("candidate=%s stage=challenger_orchestrator_skipped reason=infrastructure_failure",
                                                 candidate.id)
                                self._audit(spec, source, attempt, "orchestrator_skipped", candidate=candidate,
                                            reasons=["evaluation infrastructure failure is not challenger feedback"],
                                            evaluation={"failure_kind": "infrastructure"})
                            else:
                                feedback.extend(self._orchestrator_feedback(spec, source, attempt, candidate, evaluation))
                            self.last_attempts.append(candidate)
                            self._audit(spec, source, attempt, "rejected_evaluation", candidate=candidate,
                                        reasons=evaluation.reasons, evaluation=candidate.evaluation)
                            continue
                        if utility_decision and not utility_decision.accepted:
                            feedback.append(f"UTILITY GATE: {utility_decision.reason}")
                            candidate.evaluation["utility_gate"] = utility_decision.reason
                            self.last_attempts.append(candidate)
                            self._audit(spec, source, attempt, "rejected_utility", candidate=candidate,
                                        reasons=[utility_decision.reason], evaluation=candidate.evaluation)
                            continue
                        keep, reason = self.quality.keep(candidate, candidates)
                        if not keep:
                            feedback.append(reason or "dataset redundancy")
                            candidate.evaluation["status"] = "rejected_dataset_quality"
                            self.last_attempts.append(candidate)
                            self._audit(spec, source, attempt, "rejected_dataset_quality", candidate=candidate,
                                        reasons=[reason or "dataset redundancy"], evaluation=candidate.evaluation)
                            continue
                        candidate.accepted = True
                        candidates.append(candidate)
                        self.last_attempts.append(candidate)
                        self._audit(spec, source, attempt, "accepted", candidate=candidate,
                                    evaluation=candidate.evaluation)
                        if stop_after_accept_per_source:
                            break
                    except ValueError as exc:
                        feedback.append(str(exc))
                        self._audit(spec, source, attempt, "generation_or_parse_error", reasons=[str(exc)])
                    except RuntimeError as exc:
                        # A provider outage or rejected request must not discard
                        # the audit trail or terminate an entire multi-source run.
                        feedback.append(f"MODEL ERROR: {exc}")
                        self.logger.warning("task=%s source=%s attempt=%s stage=model_error error=%s",
                                            spec.name, source_index, attempt, exc)
                        self._audit(spec, source, attempt, "inference_error", reasons=[str(exc)])
        report = self.quality.analyze(candidates, requested_capabilities)
        terminal_statuses = {"accepted", "rejected_structure", "rejected_evaluation", "rejected_utility",
                             "rejected_dataset_quality", "generation_or_parse_error", "inference_error"}
        report.attempted = sum(event["status"] in terminal_statuses for event in self.audit_events)
        report.rejection_counts = dict(Counter(
            event["status"] for event in self.audit_events
            if event["status"].startswith("rejected_") or event["status"] in {"generation_or_parse_error", "inference_error"}
        ))
        return candidates, report

    def _audit(self, spec: TaskSpec, source: SourceDocument, attempt: int, status: str, *,
               candidate: Candidate | None = None, reasons: list[str] | None = None,
               evaluation: dict | None = None, source_index: int | None = None,
               source_total: int | None = None, solver_visible_task: dict | None = None) -> None:
        event = {
            "task": spec.name,
            "source_id": source.id,
            "attempt": attempt,
            "status": status,
            "candidate_id": candidate.id if candidate else None,
            "reasons": reasons or [],
            "evaluation": evaluation,
            "candidate_summary": _candidate_summary(candidate) if candidate else None,
            "solver_visible_task": solver_visible_task,
        }
        self.audit_events.append(event)
        position = f" source={source_index}/{source_total}" if source_index and source_total else ""
        reason = f" reason={event['reasons'][0]}" if event["reasons"] else ""
        self.logger.info("task=%s%s attempt=%s status=%s%s", spec.name, position, attempt, status, reason)

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

    def _orchestrator_feedback(self, spec: TaskSpec, source: SourceDocument, attempt: int,
                               candidate: Candidate, evaluation) -> list[str]:
        """Use the challenger instance as the orchestrator for its next round.

        No second orchestrator provider is created: ``self.generator`` first
        reflects on the failed trajectory, then receives its own strategy in
        the next normal challenger-generation call.
        """
        hidden = {"answer", "reference_answer", "reference_solution", "solution", "tests", "verification", "self_audit"}
        task_surface = {key: value for key, value in candidate.payload.items() if key not in hidden}
        try:
            self.logger.info("candidate=%s stage=challenger_orchestrator_start weak_score=%s strong_score=%s reasons=%s",
                             candidate.id, evaluation.weak_score, evaluation.strong_score, len(evaluation.reasons))
            started = perf_counter()
            raw = self.generator.complete(orchestrator_reflection_prompt(
                spec, source.id, source.content, task_surface, evaluation.as_dict()))
            reflection = extract_json_object(
                raw,
                {"failure_summary", "evidence", "avoid", "next_reasoning_angle", "challenger_instruction",
                 "source_quotes", "task_kind", "task_shape"},
                producer="challenger-orchestrator",
            )
            required = ("failure_summary", "next_reasoning_angle", "challenger_instruction")
            if any(not isinstance(reflection[field], str) or not reflection[field].strip() for field in required):
                raise ValueError("challenger-orchestrator reflection requires non-empty summary, angle, and instruction")
            if any(not isinstance(reflection[field], list) or any(not isinstance(item, str) for item in reflection[field])
                   for field in ("evidence", "avoid", "source_quotes")):
                raise ValueError("challenger-orchestrator reflection evidence, avoid, and source_quotes must be string arrays")
            quotes = [quote.strip() for quote in reflection["source_quotes"] if quote.strip()]
            if not quotes or any(quote not in source.content for quote in quotes):
                raise ValueError("challenger-orchestrator source_quotes must be non-empty verbatim excerpts from the source")
            reflection["source_quotes"] = quotes
            _validate_orchestrator_task_kind(spec, reflection)
            strategy = _format_orchestrator_strategy(reflection)
            evaluation.suggestion_for_writer = strategy
            details = {"failure_summary": reflection["failure_summary"], "evidence": reflection["evidence"],
                       "avoid": reflection["avoid"], "next_reasoning_angle": reflection["next_reasoning_angle"],
                       "challenger_instruction": reflection["challenger_instruction"],
                       "source_quotes": reflection["source_quotes"], "task_kind": reflection["task_kind"],
                       "task_shape": reflection["task_shape"]}
            self._audit(spec, source, attempt, "orchestrator_feedback", candidate=candidate,
                        reasons=[f"orchestrator: {reflection['failure_summary']}"], evaluation=details)
            self.logger.info("candidate=%s stage=challenger_orchestrator_complete seconds=%.1f task_kind=%s task_shape=%s failure=%s next_angle=%s",
                             candidate.id, perf_counter() - started, reflection["task_kind"], reflection["task_shape"],
                             _log_preview(reflection["failure_summary"]),
                             _log_preview(reflection["next_reasoning_angle"]))
            return [f"ORCHESTRATOR STRATEGY:\n{strategy}"]
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            self.logger.warning("candidate=%s stage=challenger_orchestrator_invalid error=%s", candidate.id, exc)
            self._audit(spec, source, attempt, "orchestrator_feedback_fallback", candidate=candidate,
                        reasons=[f"orchestrator reflection failed: {exc}"],
                        evaluation={"fallback": "using evaluator evidence without orchestrator strategy"})
            return self._failure_feedback(evaluation)


def _candidate_summary(candidate: Candidate) -> dict:
    """Useful audit metadata that never includes a hidden task body or solution."""
    return {"fields": sorted(candidate.payload), "rubric_items": len(candidate.payload.get("rubric", [])),
            "test_items": len(candidate.payload.get("tests", [])), "capabilities": candidate.capabilities,
            "difficulty": candidate.difficulty}


def _format_orchestrator_strategy(reflection: dict) -> str:
    """Preserve diagnostics while making source-grounding non-negotiable."""
    evidence = "; ".join(reflection["evidence"])
    avoid = "; ".join(reflection["avoid"])
    source_quotes = "; ".join(repr(quote) for quote in reflection["source_quotes"])
    return (f"Failure: {reflection['failure_summary']}\nEvidence: {evidence}\n"
            f"Avoid: {avoid}\nVerbatim source anchors: {source_quotes}\n"
            f"Next source-supported reasoning angle: {reflection['next_reasoning_angle']}\n"
            f"Instruction: {reflection['challenger_instruction']}\n"
            "Hard boundary: the angle and instruction are planning hypotheses, not facts. Preserve documented semantics; "
            "do not turn any behavior into a task requirement unless a verbatim source anchor directly states it. "
            "Do not invent API arguments, flags, behaviors, or requirements.")


def _log_preview(value: str, limit: int = 240) -> str:
    """Keep operational logs useful without dumping prompts or hidden task data."""
    compact = " ".join(value.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def _feedback_packet(label: str, reasons: list[str], *, suffix: str | None = None) -> str:
    """Keep one failed round as one feedback item so repetitive details cannot evict strategy."""
    body = "; ".join(dict.fromkeys(reason.strip() for reason in reasons if reason.strip()))
    packet = f"{label}: {body or 'unspecified failure'}"
    return f"{packet} {suffix}" if suffix else packet


def _validate_orchestrator_task_kind(spec: TaskSpec, reflection: dict) -> None:
    """Reject a strategy that changes the deliverable even when its prose claims otherwise."""
    expected_shape = {"coding": "executable_implementation", "qa": "grounded_question_answer",
                      "math": "solvable_math_problem", "legal": "application_legal_scenario",
                      "custom": "schema_conforming_task"}[spec.kind]
    if reflection.get("task_kind") != spec.kind or reflection.get("task_shape") != expected_shape:
        raise ValueError(f"challenger-orchestrator strategy must preserve task kind {spec.kind} ({expected_shape})")
    if spec.kind == "coding":
        instruction = reflection["challenger_instruction"].lower()
        executable_markers = ("implement", "implementation", "write code", "complete the function",
                              "complete the class", "create a function", "create a class", "executable code")
        if not any(marker in instruction for marker in executable_markers):
            raise ValueError("coding orchestrator strategy must request an executable implementation, not conceptual QA")
