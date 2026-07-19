from __future__ import annotations

import json

from autodata.evaluation import CandidateEvaluator
from autodata.models import SourceDocument, TaskSpec
from autodata.pipeline import DatasetBuilder


class Generator:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system=""):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"problem": "x", "answer": "x", "solution": "x", "verification": "x"})
        return json.dumps({"problem": "x", "answer": "x", "solution": "x", "verification": "x",
                           "rubric": [{"criterion": "Returns the required value from the task", "weight": 1,
                                       "category": "positive"}],
                           "self_audit": {"grounded": True, "within_scope": True, "no_answer_leakage": True,
                                          "fairness_checked": True, "verifiable": True}})


class Solver:
    def complete(self, prompt, *, system=""):
        return "x"


class Judge:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system=""):
        self.calls += 1
        if self.calls % 3 == 1:
            return json.dumps({"valid": True, "weak_criterion_matches": [[False]],
                               "judge_score": 0.9, "reasons": []})
        if self.calls % 3 == 2:
            return json.dumps({"valid": True, "weak_criterion_matches": [[False]],
                               "strong_criterion_matches": [[True]], "judge_score": 0.9, "reasons": []})
        return json.dumps({"weak_pattern": "weak missed", "strong_pattern": "strong passed",
                           "gap_interpretation": "fertile", "rubric_concerns": [], "grpo_suitability": "high",
                           "verdict": "accept", "verdict_reason": "good", "suggestion_for_writer": ""})


def test_retries_rejected_source_then_stops_after_acceptance():
    generator = Generator()
    spec = TaskSpec("math", "math", "x", environment={"success_conditions": ["x"]}, max_rounds=3)
    builder = DatasetBuilder(generator, CandidateEvaluator(Solver(), Solver(), Judge(), weak_rollouts=1, strong_rollouts=1))
    candidates, _ = builder.build([spec], [SourceDocument("s", "x")], rounds_per_source=3,
                                  stop_after_accept_per_source=True)
    assert len(candidates) == 1
    assert generator.calls == 2
    assert [event["status"] for event in builder.audit_events
            if event["status"] not in {"generating", "challenger_task"}] == ["rejected_structure", "accepted"]


def test_challenger_task_audit_records_exact_solver_visible_payload():
    generator = Generator()
    spec = TaskSpec("math", "math", "x", environment={"success_conditions": ["x"]}, max_rounds=1)
    builder = DatasetBuilder(generator, CandidateEvaluator(Solver(), Solver(), Judge(),
                                                           weak_rollouts=1, strong_rollouts=1))
    builder.build([spec], [SourceDocument("s", "x")], rounds_per_source=1)

    task_event = next(event for event in builder.audit_events if event["status"] == "challenger_task")
    visible = task_event["solver_visible_task"]
    assert visible["problem"] == "x"
    assert visible["environment"] == {"success_conditions": ["x"]}
    assert "answer" not in visible
    assert "solution" not in visible
    assert "rubric" not in visible
    assert "self_audit" not in visible


def test_structural_failures_are_grouped_into_one_feedback_packet():
    class ZeroWeightGenerator(Generator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def complete(self, prompt, *, system=""):
            self.prompts.append(prompt)
            self.calls += 1
            return json.dumps({"problem": "x", "answer": "x", "solution": "x", "verification": "x",
                               "rubric": [{"criterion": f"Observable criterion number {index}", "weight": 0,
                                           "category": "positive"} for index in range(10)],
                               "self_audit": {"grounded": True, "within_scope": True, "no_answer_leakage": True,
                                              "fairness_checked": True, "verifiable": True}})

    generator = ZeroWeightGenerator()
    spec = TaskSpec("math", "math", "x", environment={"success_conditions": ["x"]}, max_rounds=2)
    DatasetBuilder(generator, CandidateEvaluator(Solver(), Solver(), Judge())).build(
        [spec], [SourceDocument("s", "x")], rounds_per_source=2)
    feedback_section = generator.prompts[1].split("Source-local adaptive feedback to address:", 1)[1]
    assert feedback_section.count("- STRUCTURE:") == 1


def test_feedback_does_not_leak_into_the_next_source_chunk():
    class RecordingGenerator(Generator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def complete(self, prompt, *, system=""):
            self.prompts.append(prompt)
            self.calls += 1
            return json.dumps({"problem": "x", "answer": "x", "solution": "x", "verification": "x"})

    generator = RecordingGenerator()
    spec = TaskSpec("math", "math", "x", environment={"success_conditions": ["x"]}, max_rounds=1)
    DatasetBuilder(generator, CandidateEvaluator(Solver(), Solver(), Judge())).build(
        [spec], [SourceDocument("s1", "first"), SourceDocument("s2", "second")], rounds_per_source=1)
    assert "No prior feedback" in generator.prompts[0]
    assert "No prior feedback" in generator.prompts[1]
    assert "RL-oriented task requires" not in generator.prompts[1]


def test_checkpoint_resume_skips_terminal_attempts_and_preserves_audit(tmp_path):
    class AlwaysInvalidGenerator:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=""):
            self.calls += 1
            return json.dumps({"problem": "x", "answer": "x", "solution": "x", "verification": "x"})

    checkpoint = tmp_path / "run.checkpoint.json"
    spec = TaskSpec("math", "math", "x", environment={"success_conditions": ["x"]}, max_rounds=1)
    first_generator = AlwaysInvalidGenerator()
    first = DatasetBuilder(first_generator, CandidateEvaluator(Solver(), Solver(), Judge()),
                           checkpoint_path=checkpoint)
    first.build([spec], [SourceDocument("s1", "first")], rounds_per_source=1)
    assert first_generator.calls == 1
    assert checkpoint.exists()

    second_generator = AlwaysInvalidGenerator()
    resumed = DatasetBuilder(second_generator, CandidateEvaluator(Solver(), Solver(), Judge()),
                             checkpoint_path=checkpoint, resume=True)
    _, report = resumed.build(
        [spec], [SourceDocument("s1", "first"), SourceDocument("s2", "second")], rounds_per_source=1)

    assert second_generator.calls == 1  # s1 was terminal; only the new s2 source ran
    rejected = [event for event in resumed.audit_events if event["status"] == "rejected_structure"]
    assert {(event["source_id"], event["attempt"]) for event in rejected} == {("s1", 1), ("s2", 1)}
    assert report.attempted == 2
    state = json.loads(checkpoint.read_text())
    assert state["complete"] is True
    assert len(state["last_attempts"]) == 2
