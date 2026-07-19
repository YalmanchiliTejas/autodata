import json

from autodata.evaluation import CandidateEvaluator
from autodata.models import SourceDocument, TaskSpec
from autodata.pipeline import DatasetBuilder


class Generator:
    def complete(self, prompt, *, system=""):
        return json.dumps({"problem": "What is 2 + 2?", "answer": "4", "solution": "Add two and two.",
                           "verification": "Evaluate 2+2", "capabilities": ["arithmetic"], "difficulty": "medium",
                           "rubric": [{"criterion": "Returns the exact arithmetic result four", "weight": 4, "category": "positive"}],
                           "self_audit": {"grounded": True, "within_scope": True, "no_answer_leakage": True,
                                          "fairness_checked": True, "verifiable": True}})


class Solver:
    def complete(self, prompt, *, system=""):
        return "4"


class Judge:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system=""):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"valid": True, "weak_criterion_matches": [[False], [False], [False]],
                               "judge_score": 0.95, "reasons": []})
        if self.calls == 2:
            return json.dumps({"valid": True, "weak_criterion_matches": [[False], [False], [False]],
                               "strong_criterion_matches": [[True], [True], [True]],
                               "judge_score": 0.95, "reasons": []})
        return json.dumps({"weak_pattern": "weak missed the required result", "strong_pattern": "strong solved it",
                           "gap_interpretation": "fertile", "rubric_concerns": [], "grpo_suitability": "high",
                           "verdict": "accept", "verdict_reason": "useful gap", "suggestion_for_writer": ""})


def test_builder_generates_an_accepted_math_record():
    evaluator = CandidateEvaluator(Solver(), Solver(), Judge())
    spec = TaskSpec("math", "math", "Generate arithmetic", ["arithmetic"], environment={"success_conditions": ["derive the exact sum"]})
    candidates, report = DatasetBuilder(Generator(), evaluator).build(
        [spec], [SourceDocument("s1", "Basic arithmetic")], rounds_per_source=1
    )
    assert len(candidates) == report.accepted == 1
    assert candidates[0].accepted
    assert candidates[0].evaluation["gap"] == 1.0
