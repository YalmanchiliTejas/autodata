import json

from autodata.evaluation import CandidateEvaluator
from autodata.models import Candidate, TaskSpec


class RecordingSolver:
    def __init__(self):
        self.prompts = []
        self.systems = []

    def complete(self, prompt, *, system=""):
        self.prompts.append(prompt)
        self.systems.append(system)
        return "candidate answer"


class Judge:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def complete(self, prompt, *, system=""):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return json.dumps({"valid": True, "weak_score": 0.1, "strong_score": 0.9, "judge_score": 1, "reasons": []})
        return json.dumps({"verdict": "accept", "rubric_concerns": []})


def test_solvers_never_receive_challenger_reference_or_rubric():
    weak, strong = RecordingSolver(), RecordingSolver()
    judge = Judge()
    candidate = Candidate("c", "t", "s", {"question": "What is the answer?", "answer": "top-secret-reference-42", "rubric": [{"criterion": "states the correct result", "weight": 1, "category": "positive"}]})
    CandidateEvaluator(weak, strong, judge, weak_rollouts=1, strong_rollouts=1).evaluate(TaskSpec("t", "qa", "x"), candidate)
    solver_prompts = "\n".join(weak.prompts + strong.prompts)
    assert "top-secret-reference-42" not in solver_prompts
    assert "rubric" not in solver_prompts
    assert "top-secret-reference-42" not in judge.prompts[0]
    assert weak.systems == strong.systems
