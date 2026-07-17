import json

from autodata.evaluation import CandidateEvaluator
from autodata.models import Candidate, TaskSpec


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


def test_easy_weak_candidate_does_not_call_strong_or_quality_verifier():
    weak, strong, judge, verifier = Model(), Model(), Judge(), Model()
    evaluator = CandidateEvaluator(weak, strong, judge, quality_verifier=verifier, weak_rollouts=2)
    candidate = Candidate("c", "t", "s", {"question": "easy", "answer": "x", "rubric": [{"criterion": "returns x", "weight": 1, "category": "positive"}]})
    result = evaluator.evaluate(TaskSpec("t", "qa", "x"), candidate)
    assert not result.valid
    assert strong.calls == 0
    assert verifier.calls == 0
    assert "TOO EASY" in result.reasons[-1]
