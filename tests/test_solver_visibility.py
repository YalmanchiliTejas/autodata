import json

from autodata.evaluation import CandidateEvaluator, _format_solver_prompt_log
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
            return json.dumps({"valid": True, "weak_criterion_matches": [[False]], "judge_score": 1, "reasons": []})
        if self.calls == 2:
            return json.dumps({"valid": True, "weak_criterion_matches": [[False]],
                               "strong_criterion_matches": [[True]], "judge_score": 1, "reasons": []})
        return json.dumps({"weak_pattern": "weak missed", "strong_pattern": "strong passed",
                           "gap_interpretation": "fertile", "rubric_concerns": [], "grpo_suitability": "high",
                           "verdict": "accept", "verdict_reason": "good", "suggestion_for_writer": ""})


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


def test_solver_prompt_log_is_human_readable_and_excludes_hidden_fields():
    spec = TaskSpec("code-task", "coding", "x", environment={"network": False})
    candidate = Candidate("candidate-1", "code-task", "source-1", {
        "prompt": "Implement solve(value).", "reference_solution": "secret", "rubric": ["hidden"]})
    visible = {"prompt": "Implement solve(value).", "environment": {"network": False}}

    entry = _format_solver_prompt_log(spec, candidate, "weak", 2, 3, "Solve carefully.", visible)

    assert "SOLVER PROMPT" in entry
    assert "candidate_id  : candidate-1" in entry
    assert "source_id     : source-1" in entry
    assert "solver_role   : weak" in entry
    assert "rollout       : 2/3" in entry
    assert "[SYSTEM PROMPT]" in entry
    assert "[USER PROMPT — SOLVER-VISIBLE TASK JSON]" in entry
    assert '  "prompt": "Implement solve(value)."' in entry
    assert "secret" not in entry
    assert "rubric" not in entry
