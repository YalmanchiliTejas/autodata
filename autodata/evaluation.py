from __future__ import annotations

import json
from statistics import mean
import logging
from datetime import datetime, timezone
from time import perf_counter

from .models import Candidate, Evaluation, TaskSpec
from .json_output import extract_json_object
from .providers import TextModel
from .prompts import loop_judge_prompt, quality_verifier_prompt, rubric_scoring_prompt, solver_system_prompt


logger = logging.getLogger("autodata")
solver_prompt_logger = logging.getLogger("autodata.solver_prompts")
# Full prompts are intentionally file-only. Applications may attach a handler,
# but they must never propagate into the operational stdout logger.
solver_prompt_logger.propagate = False


_HIDDEN_SOLVER_FIELDS = {
    "answer", "reference_answer", "reference_solution", "solution", "rubric", "tests",
    "verification", "self_audit", "target_capabilities", "reasoning_skills", "question_type",
}


def solver_visible_payload(spec: TaskSpec, candidate: Candidate) -> dict:
    """Return exactly the task surface sent to both weak and strong solvers."""
    payload = {key: value for key, value in candidate.payload.items() if key not in _HIDDEN_SOLVER_FIELDS}
    payload["environment"] = spec.environment
    return payload


class CandidateEvaluator:
    """Evaluates learning value with weak/strong solvers and an independent judge."""

    def __init__(self, weak: TextModel, strong: TextModel, judge: TextModel, *, quality_verifier: TextModel | None = None,
                 weak_rollouts: int = 3, strong_rollouts: int = 3, min_gap: float = 0.20,
                 min_strong: float = 0.60, min_judge: float = 0.70, weak_screen_max: float = 0.65,
                 contract_retries: int = 1):
        self.weak, self.strong, self.judge = weak, strong, judge
        self.quality_verifier = quality_verifier
        self.weak_rollouts, self.strong_rollouts = weak_rollouts, strong_rollouts
        self.min_gap, self.min_strong, self.min_judge = min_gap, min_strong, min_judge
        self.weak_screen_max = weak_screen_max
        self.contract_retries = contract_retries

    def evaluate(self, spec: TaskSpec, candidate: Candidate, *, source_content: str | None = None) -> Evaluation:
        logger.info("candidate=%s stage=evaluation_start weak_rollouts=%s strong_rollouts=%s",
                    candidate.id, self.weak_rollouts, self.strong_rollouts)
        task = json.dumps(candidate.payload, ensure_ascii=False)
        # The challenger owns the answer and rubric. Solvers receive only the task
        # surface, otherwise we would be measuring their ability to read the reward.
        solver_payload = solver_visible_payload(spec, candidate)
        solver_task = json.dumps(solver_payload, ensure_ascii=False)
        # The scoring judge needs the rubric but, following the CS setup in the
        # paper, does not get the challenger reference answer.
        judge_payload = dict(solver_payload)
        judge_payload["rubric"] = candidate.payload.get("rubric", [])
        judge_task = json.dumps(judge_payload, ensure_ascii=False)
        weak_runs = 4 if spec.profile == "scientific_reasoning" else self.weak_rollouts
        strong_runs = 4 if spec.profile == "scientific_reasoning" else self.strong_rollouts
        matched_solver_prompt = solver_system_prompt(spec)
        # Paper-aligned ordering: reject leakage and malformed rewards before
        # exposing the task to either solver or spending rollout compute.
        if self.quality_verifier:
            try:
                started = perf_counter()
                expected_criteria = len(candidate.payload.get("rubric", []))
                logger.info("candidate=%s stage=quality_verifier_start order=pre_solver expected_criteria=%s",
                            candidate.id, expected_criteria)
                required_fields = {"passed", "checks", "solver_context_audit", "criterion_audit",
                                   "intrinsic_reward_eligible", "issues", "feedback"}
                verifier_prompt = quality_verifier_prompt(spec, task, source_content)
                quality = None
                last_quality_error: Exception | None = None
                for retry in range(self.contract_retries + 1):
                    raw_quality = self.quality_verifier.complete(verifier_prompt)
                    try:
                        quality = extract_json_object(
                            raw_quality, required_fields, producer="quality-verifier",
                            validator=lambda value: _quality_response_contract(value, expected_criteria),
                        )
                        break
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                        last_quality_error = exc
                        if retry < self.contract_retries:
                            logger.warning("candidate=%s stage=quality_verifier contract_retry=%s/%s error=%s",
                                           candidate.id, retry + 1, self.contract_retries, exc)
                if quality is None:
                    assert last_quality_error is not None
                    raise last_quality_error
                context_audit = quality["solver_context_audit"]
                logger.info("candidate=%s stage=quality_verifier_complete passed=%s context_passed=%s "
                            "intrinsic_reward_eligible=%s audit_items=%s expected_criteria=%s seconds=%.1f",
                            candidate.id, quality["passed"], context_audit["passed"],
                            quality["intrinsic_reward_eligible"], len(quality["criterion_audit"]),
                            expected_criteria, perf_counter() - started)
                if not context_audit["passed"]:
                    leakage = ", ".join(context_audit["leakage_types"]) or "unspecified"
                    evidence = [f"solver-context leakage ({leakage}): {item}"
                                for item in context_audit["evidence"]]
                    reasons = evidence + list(quality["issues"])
                    if quality["feedback"]:
                        reasons.append(quality["feedback"])
                    logger.info("candidate=%s stage=evaluation_stopped reason=solver_context_leakage types=%s",
                                candidate.id, leakage)
                    return Evaluation(False, None, None, 0.0,
                                      reasons or [f"solver-context leakage detected: {leakage}"])
                if not quality["passed"]:
                    reasons = list(quality["issues"])
                    if quality["feedback"]:
                        reasons.append(quality["feedback"])
                    logger.info("candidate=%s stage=evaluation_stopped reason=quality_verifier_rejected",
                                candidate.id)
                    return Evaluation(False, None, None, 0.0,
                                      reasons or ["quality verification failed"])
                if expected_criteria:
                    required = ("grounded", "observable", "environment_compatible", "discriminative")
                    rubric_failures = []
                    for index, row in enumerate(quality["criterion_audit"], start=1):
                        failed = [field for field in required if not row[field]]
                        if failed:
                            rubric_failures.append(
                                f"rubric criterion {index} failed intrinsic audit ({', '.join(failed)}): {row['evidence']}"
                            )
                    if not quality["intrinsic_reward_eligible"] or rubric_failures:
                        reasons = list(quality["issues"]) + rubric_failures
                        if not quality["intrinsic_reward_eligible"]:
                            reasons.insert(0, "quality verifier marked intrinsic_reward_eligible=false")
                        if quality["feedback"]:
                            reasons.append(quality["feedback"])
                        logger.info("candidate=%s stage=evaluation_stopped reason=rubric_intrinsic_ineligible "
                                    "failed_criteria=%s", candidate.id, len(rubric_failures))
                        return Evaluation(False, None, None, 0.0, reasons)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("candidate=%s stage=quality_verifier_invalid error=%s", candidate.id, exc)
                return Evaluation(False, None, None, 0.0, [f"invalid quality-verifier response: {exc}"],
                                  failure_kind="infrastructure")

        # A task the weak solver handles easily cannot provide useful training
        # signal, so the weak screen remains the compute gate for strong runs.
        started = perf_counter()
        weak_answers = self._solver_rollouts(
            self.weak, spec, candidate, "weak", weak_runs, matched_solver_prompt, solver_payload, solver_task)
        logger.info("candidate=%s stage=weak_rollouts_complete count=%s seconds=%.1f",
                    candidate.id, weak_runs, perf_counter() - started)
        try:
            started = perf_counter()
            weak_result = self._score_judge(
                rubric_scoring_prompt(spec, judge_task, weak_answers, [], weak_only=True),
                "weak-screen", weak_runs, candidate_id=candidate.id)
            weak_scores = _score_response(weak_result, "weak-screen", weak_runs)
            weak_eval = Evaluation(weak_result["valid"], mean(weak_scores), None,
                                   weak_result["judge_score"], weak_result["reasons"], weak_scores, [])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("candidate=%s stage=weak_screen_invalid error=%s", candidate.id, exc)
            return Evaluation(False, None, None, 0.0, [f"invalid weak-screen judge response: {exc}"],
                              failure_kind="infrastructure")
        logger.info("candidate=%s stage=weak_screen_complete valid=%s score=%.3f judge_score=%.3f seconds=%.1f",
                    candidate.id, weak_eval.valid, weak_eval.weak_score, weak_eval.judge_score, perf_counter() - started)
        if not weak_eval.valid:
            logger.info("candidate=%s stage=evaluation_stopped reason=weak_screen_invalid", candidate.id)
            return weak_eval
        if self._weak_is_too_easy(spec, weak_scores):
            weak_eval.valid = False
            weak_eval.reasons.append("TOO EASY: weak solver passed the early screen; strong solver was not called")
            logger.info("candidate=%s stage=evaluation_stopped reason=weak_too_easy", candidate.id)
            return weak_eval
        started = perf_counter()
        strong_answers = self._solver_rollouts(
            self.strong, spec, candidate, "strong", strong_runs, matched_solver_prompt, solver_payload, solver_task)
        logger.info("candidate=%s stage=strong_rollouts_complete count=%s seconds=%.1f",
                    candidate.id, strong_runs, perf_counter() - started)
        prompt = rubric_scoring_prompt(spec, judge_task, weak_answers, strong_answers)
        try:
            started = perf_counter()
            result = self._score_judge(prompt, "final judge", weak_runs, strong_runs, candidate_id=candidate.id)
            weak_scores = _score_response(result, "final judge", weak_runs)
            strong_scores = _score_response(result, "final judge", strong_runs, field="strong_scores")
            evaluation = Evaluation(result["valid"], mean(weak_scores), mean(strong_scores),
                                    result["judge_score"], result["reasons"],
                                    weak_scores, strong_scores)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("candidate=%s stage=final_judge_invalid error=%s", candidate.id, exc)
            return Evaluation(False, None, None, 0.0, [f"invalid judge response: {exc}"],
                              failure_kind="infrastructure")
        logger.info("candidate=%s stage=final_judge_complete valid=%s weak=%.3f strong=%.3f gap=%.3f judge_score=%.3f seconds=%.1f",
                    candidate.id, evaluation.valid, evaluation.weak_score, evaluation.strong_score,
                    evaluation.gap or 0.0, evaluation.judge_score, perf_counter() - started)
        diagnostics = json.dumps(evaluation.as_dict())
        try:
            started = perf_counter()
            verdict = _judge_json(self.judge.complete(loop_judge_prompt(spec, task, diagnostics)),
                                 {"weak_pattern", "strong_pattern", "gap_interpretation", "rubric_concerns",
                                  "grpo_suitability", "verdict", "verdict_reason", "suggestion_for_writer"})
            _object_contract(verdict, "loop judge", {"weak_pattern": str, "strong_pattern": str, "gap_interpretation": str,
                                                       "rubric_concerns": list, "grpo_suitability": str, "verdict": str,
                                                       "verdict_reason": str, "suggestion_for_writer": str})
            evaluation.verdict = verdict.get("verdict", "improve")
            evaluation.suggestion_for_writer = verdict.get("suggestion_for_writer")
            evaluation.reasons.extend(verdict.get("rubric_concerns", []))
            if evaluation.verdict != "accept":
                evaluation.reasons.append(verdict.get("verdict_reason", "loop judge requested improvement"))
            logger.info("candidate=%s stage=loop_judge_complete verdict=%s seconds=%.1f",
                        candidate.id, evaluation.verdict, perf_counter() - started)
        except (ValueError, TypeError, json.JSONDecodeError):
            # Generic judges may only support scoring. The deterministic gates remain safe.
            pass
        if evaluation.strong_score < self.min_strong:
            evaluation.reasons.append("strong solver could not reliably solve it")
        if evaluation.gap is not None and evaluation.gap < self.min_gap and spec.target_difficulty == "adaptive":
            evaluation.reasons.append("insufficient weak/strong separation")
        if evaluation.judge_score < self.min_judge:
            evaluation.reasons.append("judge found task quality too low")
        return evaluation

    def _score_judge(self, prompt: str, stage: str, weak_count: int,
                     strong_count: int | None = None, *, candidate_id: str) -> dict:
        """Retry malformed judge output and select an object satisfying the full score contract."""
        required = {"valid", "weak_scores", "judge_score", "reasons"}
        if strong_count is not None:
            required.add("strong_scores")

        def validate(payload: dict) -> None:
            _score_response(payload, stage, weak_count)
            if strong_count is not None:
                _score_response(payload, stage, strong_count, field="strong_scores")

        last_error: ValueError | None = None
        for retry in range(self.contract_retries + 1):
            raw = self.judge.complete(prompt)
            try:
                return extract_json_object(raw, required, producer="judge", validator=validate)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = ValueError(str(exc))
                if retry < self.contract_retries:
                    logger.warning("candidate=%s stage=%s contract_retry=%s/%s error=%s",
                                   candidate_id, stage, retry + 1, self.contract_retries, exc)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _solver_rollouts(model: TextModel, spec: TaskSpec, candidate: Candidate, role: str, runs: int,
                         system_prompt: str, solver_payload: dict, solver_task: str) -> list[str]:
        answers = []
        for index in range(1, runs + 1):
            logger.info("candidate=%s stage=solver_rollout_start role=%s rollout=%s/%s",
                        candidate.id, role, index, runs)
            solver_prompt_logger.info(_format_solver_prompt_log(
                spec, candidate, role, index, runs, system_prompt, solver_payload))
            started = perf_counter()
            answer = model.complete(solver_task, system=system_prompt)
            answers.append(answer)
            logger.info("candidate=%s stage=solver_rollout_complete role=%s rollout=%s/%s seconds=%.1f response_chars=%s",
                        candidate.id, role, index, runs, perf_counter() - started, len(answer))
        return answers

    def _weak_is_too_easy(self, spec: TaskSpec, scores: list[float]) -> bool:
        if spec.profile == "scientific_reasoning":
            return sum(score >= 0.5 for score in scores) >= 2
        return mean(scores) > self.weak_screen_max

    def accepts(self, spec: TaskSpec, evaluation: Evaluation) -> bool:
        if not evaluation.valid or evaluation.judge_score < self.min_judge:
            return False
        if spec.profile == "legal_reasoning" and evaluation.verdict != "accept":
            return False
        if spec.profile == "cs_research":
            return (evaluation.strong_score is not None and evaluation.weak_score is not None
                    and evaluation.weak_score <= 0.65 and max(evaluation.weak_rollouts) <= 0.75
                    and min(evaluation.weak_rollouts) > 0 and 0.60 <= evaluation.strong_score < 0.95
                    and evaluation.gap is not None and evaluation.gap >= 0.20)
        if spec.profile == "scientific_reasoning":
            return (sum(score >= 0.5 for score in evaluation.weak_rollouts) <= 1
                    and sum(score >= 0.5 for score in evaluation.strong_rollouts) >= 3)
        if spec.target_difficulty != "adaptive":
            return evaluation.strong_score is not None and evaluation.strong_score >= self.min_strong
        return (evaluation.strong_score is not None and evaluation.strong_score >= self.min_strong
                and evaluation.gap is not None and evaluation.gap >= self.min_gap)


def _judge_json(raw: str, required_fields: set[str]) -> dict:
    """Extract the judge result rather than an echoed schema/example object."""
    return extract_json_object(raw, required_fields, producer="judge")


def _object_contract(payload: dict, stage: str, expected: dict[str, type]) -> None:
    missing = [field for field in expected if field not in payload]
    wrong_type = [field for field, expected_type in expected.items() if field in payload and not isinstance(payload[field], expected_type)]
    if missing or wrong_type:
        details = []
        if missing:
            details.append(f"missing required fields: {', '.join(missing)}")
        if wrong_type:
            details.append(f"wrong field types: {', '.join(wrong_type)}")
        raise ValueError(f"{stage} response contract violation ({'; '.join(details)})")


def _score_response(payload: dict, stage: str, expected_count: int, *, field: str = "weak_scores") -> list[float]:
    expected = {"valid": bool, "weak_scores": list, "judge_score": (int, float), "reasons": list}
    if field == "strong_scores":
        expected["strong_scores"] = list
    _object_contract(payload, stage, expected)
    scores = payload[field]
    if len(scores) != expected_count or any(isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1 for score in scores):
        raise ValueError(f"{stage} response contract violation ({field} must contain {expected_count} numeric scores in [0, 1])")
    if not 0 <= payload["judge_score"] <= 1:
        raise ValueError(f"{stage} response contract violation (judge_score must be in [0, 1])")
    if any(not isinstance(reason, str) for reason in payload["reasons"]):
        raise ValueError(f"{stage} response contract violation (reasons must be an array of strings)")
    return [float(score) for score in scores]


def _quality_response_contract(payload: dict, expected_criteria: int) -> None:
    """Validate the complete pre-solver quality response, including every rubric row."""
    _object_contract(payload, "quality-verifier", {
        "passed": bool, "checks": dict, "solver_context_audit": dict, "criterion_audit": list,
        "intrinsic_reward_eligible": bool, "issues": list, "feedback": str,
    })
    context = payload["solver_context_audit"]
    _object_contract(context, "quality-verifier solver_context_audit", {
        "passed": bool, "leakage_types": list, "evidence": list,
    })
    allowed_leakage = {
        "source_conclusion", "worked_example", "exact_output", "reference_answer", "solution_steps",
        "tool_or_library_hint", "constant_or_lookup_hint", "partial_implementation", "other",
    }
    if any(not isinstance(item, str) or item not in allowed_leakage for item in context["leakage_types"]):
        raise ValueError("quality-verifier solver_context_audit leakage_types contains an invalid value")
    if any(not isinstance(item, str) for item in context["evidence"]):
        raise ValueError("quality-verifier solver_context_audit evidence must be an array of strings")
    if any(not isinstance(item, str) for item in payload["issues"]):
        raise ValueError("quality-verifier issues must be an array of strings")
    audit = payload["criterion_audit"]
    if len(audit) != expected_criteria:
        raise ValueError(
            f"quality-verifier criterion_audit must contain exactly {expected_criteria} rows; received {len(audit)}"
        )
    expected_row = {"grounded": bool, "observable": bool, "environment_compatible": bool,
                    "discriminative": bool, "evidence": str}
    for index, row in enumerate(audit, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"quality-verifier criterion_audit row {index} must be an object")
        _object_contract(row, f"quality-verifier criterion_audit row {index}", expected_row)


def _format_solver_prompt_log(spec: TaskSpec, candidate: Candidate, role: str, rollout: int, total: int,
                              system_prompt: str, solver_payload: dict) -> str:
    """Build a readable, append-safe transcript entry for one actual solver call."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    separator = "=" * 96
    return (
        f"{separator}\n"
        f"SOLVER PROMPT\n"
        f"timestamp_utc : {timestamp}\n"
        f"task          : {spec.name}\n"
        f"task_kind     : {spec.kind}\n"
        f"candidate_id  : {candidate.id}\n"
        f"source_id     : {candidate.source_id}\n"
        f"solver_role   : {role}\n"
        f"rollout       : {rollout}/{total}\n"
        f"\n[SYSTEM PROMPT]\n{system_prompt.strip()}\n"
        f"\n[USER PROMPT — SOLVER-VISIBLE TASK JSON]\n"
        f"{json.dumps(solver_payload, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        f"{separator}\n"
    )
