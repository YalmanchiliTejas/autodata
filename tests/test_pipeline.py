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
    def complete(self, prompt, *, system=""):
        return json.dumps({"valid": True, "weak_score": 0.2, "strong_score": 0.9,
                           "judge_score": 0.95, "reasons": []})


def test_builder_generates_an_accepted_math_record():
    evaluator = CandidateEvaluator(Solver(), Solver(), Judge())
    spec = TaskSpec("math", "math", "Generate arithmetic", ["arithmetic"], environment={"success_conditions": ["derive the exact sum"]})
    candidates, report = DatasetBuilder(Generator(), evaluator).build(
        [spec], [SourceDocument("s1", "Basic arithmetic")], rounds_per_source=1
    )
    assert len(candidates) == report.accepted == 1
    assert candidates[0].accepted
    assert candidates[0].evaluation["gap"] == 0.7
