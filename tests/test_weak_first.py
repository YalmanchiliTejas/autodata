import json
import pytest

from autodata.evaluation import CandidateEvaluator
from autodata.models import Candidate, SourceDocument, TaskSpec
from autodata.pipeline import DatasetBuilder, _validate_orchestrator_task_kind


class Model:
    def __init__(self, response="answer"):
        self.response, self.calls = response, 0

    def complete(self, prompt, *, system=""):
        self.calls += 1
        return self.response


class Judge(Model):
    def complete(self, prompt, *, system=""):
        self.calls += 1
        return json.dumps({"valid": True, "weak_scores": [0.95, 0.95], "judge_score": 1.0, "reasons": []})


def _quality_response(*, context_passed=True, passed=True):
    return json.dumps({
        "passed": passed,
        "checks": {},
        "solver_context_audit": {
            "passed": context_passed,
            "leakage_types": [] if context_passed else ["tool_or_library_hint"],
            "evidence": [] if context_passed else ["starter code imports the intended solution library"],
        },
        "criterion_audit": [{"grounded": True, "observable": True, "environment_compatible": True,
                             "discriminative": True, "evidence": "checks the required result"}],
        "intrinsic_reward_eligible": True,
        "issues": [],
        "feedback": "",
    })


def test_quality_verifier_runs_before_easy_weak_screen_and_strong_is_skipped():
    weak, strong, judge, verifier = Model(), Model(), Judge(), Model(_quality_response())
    evaluator = CandidateEvaluator(weak, strong, judge, quality_verifier=verifier, weak_rollouts=2)
    candidate = Candidate("c", "t", "s", {"question": "easy", "answer": "x", "rubric": [{"criterion": "returns x", "weight": 1, "category": "positive"}]})
    result = evaluator.evaluate(TaskSpec("t", "qa", "x"), candidate)
    assert not result.valid
    assert strong.calls == 0
    assert verifier.calls == 1
    assert "TOO EASY" in result.reasons[-1]
    assert judge.calls == 1


def test_solver_context_leakage_is_rejected_before_any_solver_call():
    weak, strong, judge = Model(), Model(), Judge()
    verifier = Model(_quality_response(context_passed=False, passed=False))
    evaluator = CandidateEvaluator(weak, strong, judge, quality_verifier=verifier, weak_rollouts=2)
    candidate = Candidate("c", "t", "s", {
        "prompt": "Implement it.", "starter_code": "import hashlib\ndef solve(value): ...",
        "reference_solution": "def solve(value): return hashlib.sha256(value).hexdigest()",
        "rubric": [{"criterion": "returns the digest", "weight": 1, "category": "positive"}],
    })

    result = evaluator.evaluate(TaskSpec("t", "coding", "x"), candidate, source_content="hidden source")

    assert not result.valid
    assert weak.calls == strong.calls == judge.calls == 0
    assert verifier.calls == 1
    assert "solver-context leakage" in result.reasons[0]


class ChallengerOrchestrator(Model):
    def __init__(self):
        super().__init__()
        self.generation_prompts = []

    def complete(self, prompt, *, system=""):
        self.calls += 1
        if "Analyze this rejected round" in prompt:
            return json.dumps({"failure_summary": "direct recall", "evidence": ["weak answer copied the documented path"],
                               "avoid": ["happy-path restatement"], "next_reasoning_angle": "a documented conditional interaction",
                               "challenger_instruction": "Write a task about a documented conditional behavior only.",
                               "source_quotes": ["documented conditional behavior"],
                               "task_kind": "qa", "task_shape": "grounded_question_answer"})
        self.generation_prompts.append(prompt)
        return json.dumps({"question": "What is 2 + 2?", "answer": "4", "capabilities": ["arithmetic"],
                           "rubric": [{"criterion": "Returns the exact arithmetic result four", "weight": 1, "category": "positive"}],
                           "self_audit": {"grounded": True, "within_scope": True, "no_answer_leakage": True,
                                           "fairness_checked": True, "verifiable": True}})


def test_same_challenger_instance_orchestrates_the_next_round():
    challenger = ChallengerOrchestrator()
    evaluator = CandidateEvaluator(Model(), Model(), Judge(), weak_rollouts=2)
    builder = DatasetBuilder(challenger, evaluator)
    builder.build(
        [TaskSpec("t", "qa", "x", ["arithmetic"], environment={"success_conditions": ["derive the exact sum"]})],
        [SourceDocument("s", "documented conditional behavior")], rounds_per_source=2
    )
    assert challenger.calls == 3  # generate → reflect → final generate; no unusable final reflection
    assert "Next source-supported reasoning angle: a documented conditional interaction" in challenger.generation_prompts[1]
    assert "Do not invent API arguments" in challenger.generation_prompts[1]
    reflection_events = [event for event in builder.audit_events if event["status"] == "orchestrator_feedback"]
    assert len(reflection_events) == 1
    assert reflection_events[0]["evaluation"]["next_reasoning_angle"] == "a documented conditional interaction"
    skipped = [event for event in builder.audit_events if event["status"] == "orchestrator_skipped"]
    assert skipped[-1]["reasons"] == ["no generation attempt remains to consume orchestrator feedback"]


def test_coding_orchestrator_rejects_conceptual_qa_strategy():
    reflection = {"task_kind": "coding", "task_shape": "executable_implementation",
                  "challenger_instruction": "Answer four conceptual questions and cite the documentation."}
    with pytest.raises(ValueError, match="executable implementation"):
        _validate_orchestrator_task_kind(TaskSpec("t", "coding", "x"), reflection)


class InvalidJudge(Model):
    def complete(self, prompt, *, system=""):
        self.calls += 1
        return json.dumps({"valid": True, "weak_scores": [], "judge_score": 1.0, "reasons": []})


def test_infrastructure_failure_skips_challenger_orchestration():
    challenger = ChallengerOrchestrator()
    evaluator = CandidateEvaluator(Model(), Model(), InvalidJudge(), weak_rollouts=1, contract_retries=0)
    builder = DatasetBuilder(challenger, evaluator)
    builder.build(
        [TaskSpec("t", "qa", "x", ["arithmetic"], environment={"success_conditions": ["exact sum"]})],
        [SourceDocument("s", "documented conditional behavior")], rounds_per_source=1,
    )
    assert challenger.calls == 1
    statuses = [event["status"] for event in builder.audit_events]
    assert "orchestrator_skipped" in statuses
    assert "orchestrator_feedback" not in statuses
