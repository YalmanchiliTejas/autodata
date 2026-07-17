from __future__ import annotations

import json
from statistics import mean

from .models import Candidate, Evaluation, TaskSpec
from .providers import TextModel
from .prompts import loop_judge_prompt, quality_verifier_prompt, rubric_scoring_prompt, solver_system_prompt


class CandidateEvaluator:
    """Evaluates learning value with weak/strong solvers and an independent judge."""

    def __init__(self, weak: TextModel, strong: TextModel, judge: TextModel, *, quality_verifier: TextModel | None = None,
                 weak_rollouts: int = 3, strong_rollouts: int = 3, min_gap: float = 0.20,
                 min_strong: float = 0.60, min_judge: float = 0.70, weak_screen_max: float = 0.65):
        self.weak, self.strong, self.judge = weak, strong, judge
        self.quality_verifier = quality_verifier
        self.weak_rollouts, self.strong_rollouts = weak_rollouts, strong_rollouts
        self.min_gap, self.min_strong, self.min_judge = min_gap, min_strong, min_judge
        self.weak_screen_max = weak_screen_max

    def evaluate(self, spec: TaskSpec, candidate: Candidate) -> Evaluation:
        task = json.dumps(candidate.payload, ensure_ascii=False)
        # The challenger owns the answer and rubric. Solvers receive only the task
        # surface, otherwise we would be measuring their ability to read the reward.
        hidden_fields = {"answer", "reference_answer", "reference_solution", "solution", "rubric", "tests",
                         "verification", "self_audit", "target_capabilities", "reasoning_skills", "question_type"}
        solver_payload = {key: value for key, value in candidate.payload.items() if key not in hidden_fields}
        solver_payload["environment"] = spec.environment
        solver_task = json.dumps(solver_payload, ensure_ascii=False)
        # The scoring judge needs the rubric but, following the CS setup in the
        # paper, does not get the challenger reference answer.
        judge_payload = dict(solver_payload)
        judge_payload["rubric"] = candidate.payload.get("rubric", [])
        judge_task = json.dumps(judge_payload, ensure_ascii=False)
        weak_runs = 4 if spec.profile == "scientific_reasoning" else self.weak_rollouts
        strong_runs = 4 if spec.profile == "scientific_reasoning" else self.strong_rollouts
        matched_solver_prompt = solver_system_prompt(spec)
        # Cost ordering: weak solver and a cheap rubric score are the first gate.
        # A task it solves easily cannot be useful for training it, so do not call
        # the strong, verifier, or loop judge for that candidate.
        weak_answers = [self.weak.complete(solver_task, system=matched_solver_prompt) for _ in range(weak_runs)]
        try:
            weak_result = json.loads(self.judge.complete(rubric_scoring_prompt(spec, judge_task, weak_answers, [])))
            weak_scores = [float(value) for value in weak_result.get("weak_scores", [weak_result.get("weak_score", 0)])]
            weak_eval = Evaluation(bool(weak_result["valid"]), mean(weak_scores), None,
                                   float(weak_result.get("judge_score", 0)), list(weak_result.get("reasons", [])), weak_scores, [])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return Evaluation(False, None, None, 0.0, [f"invalid weak-screen judge response: {exc}"])
        if not weak_eval.valid:
            return weak_eval
        if self._weak_is_too_easy(spec, weak_scores):
            weak_eval.valid = False
            weak_eval.reasons.append("TOO EASY: weak solver passed the early screen; strong solver was not called")
            return weak_eval
        if self.quality_verifier:
            try:
                quality = json.loads(self.quality_verifier.complete(quality_verifier_prompt(spec, task)))
                if not quality.get("passed", False):
                    return Evaluation(False, None, None, 0.0, list(quality.get("issues", [])) + [quality.get("feedback", "quality verification failed")])
                if candidate.payload.get("rubric"):
                    audit = quality.get("criterion_audit", [])
                    required = {"grounded", "observable", "environment_compatible", "discriminative"}
                    if (not quality.get("intrinsic_reward_eligible", False) or not isinstance(audit, list)
                            or len(audit) != len(candidate.payload["rubric"])
                            or any(not required.issubset(row) or not all(row[key] for key in required) for row in audit)):
                        return Evaluation(False, None, None, 0.0, ["rubric verifier did not establish criterion-level intrinsic reward eligibility"])
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return Evaluation(False, None, None, 0.0, [f"invalid quality-verifier response: {exc}"])
        strong_answers = [self.strong.complete(solver_task, system=matched_solver_prompt) for _ in range(strong_runs)]
        prompt = rubric_scoring_prompt(spec, judge_task, weak_answers, strong_answers)
        try:
            result = json.loads(self.judge.complete(prompt))
            weak_scores = [float(value) for value in result.get("weak_scores", [result.get("weak_score", 0)])]
            strong_scores = [float(value) for value in result.get("strong_scores", [result.get("strong_score", 0)])]
            evaluation = Evaluation(bool(result["valid"]), mean(weak_scores), mean(strong_scores),
                                    float(result.get("judge_score", 0)), list(result.get("reasons", [])),
                                    weak_scores, strong_scores)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return Evaluation(False, None, None, 0.0, [f"invalid judge response: {exc}"])
        diagnostics = json.dumps(evaluation.as_dict())
        try:
            verdict = json.loads(self.judge.complete(loop_judge_prompt(spec, task, diagnostics)))
            evaluation.verdict = verdict.get("verdict", "improve")
            evaluation.suggestion_for_writer = verdict.get("suggestion_for_writer")
            evaluation.reasons.extend(verdict.get("rubric_concerns", []))
            if evaluation.verdict != "accept":
                evaluation.reasons.append(verdict.get("verdict_reason", "loop judge requested improvement"))
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
