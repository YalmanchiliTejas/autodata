import pytest

from autodata.evaluation import CandidateEvaluator, _judge_json, _score_response
from autodata.models import Candidate, TaskSpec
from autodata.prompts import loop_judge_prompt, quality_verifier_prompt, rubric_scoring_prompt


def test_judge_json_accepts_a_markdown_wrapped_object():
    assert _judge_json("```json\n{\"valid\": true}\n```", {"valid"}) == {"valid": True}


def test_judge_json_can_extract_an_object_after_a_short_preamble():
    assert _judge_json("Result: {\"valid\": true}", {"valid"}) == {"valid": True}


def test_judge_json_skips_an_echoed_schema_for_a_matching_score_object():
    raw = ('Schema: {"type":"object","properties":{"valid":{"type":"boolean"}}} '
           'Score: {"valid":true,"weak_scores":[0.2],"judge_score":0.9,"reasons":[]}')
    assert _judge_json(raw, {"valid", "weak_scores", "judge_score", "reasons"}) == {
        "valid": True, "weak_scores": [0.2], "judge_score": 0.9, "reasons": []
    }


def test_judge_json_rejects_an_echoed_schema_without_a_score_object():
    with pytest.raises(ValueError, match="none matched required fields"):
        _judge_json('{"type":"object","properties":{"valid":{"type":"boolean"}}}',
                    {"valid", "weak_scores", "judge_score", "reasons"})


def test_score_contract_reports_missing_required_field_by_name():
    with pytest.raises(ValueError, match="missing required fields: valid"):
        _score_response({"weak_scores": [0.2], "judge_score": 0.9, "reasons": []}, "weak-screen", 1)


def test_extractor_skips_matching_object_with_invalid_score_values():
    from autodata.json_output import extract_json_object

    raw = ('Draft: {"valid":true,"weak_scores":[],"judge_score":1,"reasons":[]} '
           'Final: {"valid":true,"weak_scores":[0.2],"judge_score":0.9,"reasons":[]}')
    result = extract_json_object(
        raw, {"valid", "weak_scores", "judge_score", "reasons"}, producer="judge",
        validator=lambda value: _score_response(value, "weak-screen", 1),
    )
    assert result["weak_scores"] == [0.2]


class _ResponseModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, prompt, *, system=""):
        self.calls += 1
        return next(self.responses)


def test_evaluator_retries_invalid_judge_contract_without_regenerating_candidate():
    weak = _ResponseModel(["answer"])
    judge = _ResponseModel([
        '{"valid":true,"weak_scores":[],"judge_score":1,"reasons":[]}',
        '{"valid":true,"weak_scores":[0.9],"judge_score":1,"reasons":[]}',
    ])
    evaluator = CandidateEvaluator(weak, _ResponseModel([]), judge, weak_rollouts=1, contract_retries=1)
    candidate = Candidate("c", "t", "s", {"question": "q", "answer": "a", "rubric": []})
    result = evaluator.evaluate(TaskSpec("t", "qa", "x"), candidate)
    assert judge.calls == 2
    assert result.weak_rollouts == [0.9]
    assert result.failure_kind is None


def test_all_evaluator_prompts_include_a_json_schema_contract():
    spec = TaskSpec("x", "qa", "x")
    assert '"required"' in rubric_scoring_prompt(spec, "{}", ["a"], [], weak_only=True)
    assert '"required"' in quality_verifier_prompt(spec, "{}")
    assert '"required"' in loop_judge_prompt(spec, "{}", "{}")


def test_quality_verifier_receives_hidden_source_and_checks_coding_solution_leakage():
    spec = TaskSpec("x", "coding", "x")
    prompt = quality_verifier_prompt(spec, '{"prompt":"implement"}', "private source text")

    assert "Grounding source (hidden from solvers): private source text" in prompt
    assert "solver_context_audit" in prompt
    assert "solution-specific modules, library functions, primitives" in prompt


def test_evaluator_retries_invalid_quality_verifier_contract_before_solver_calls():
    weak = _ResponseModel(["answer"])
    verifier = _ResponseModel([
        '{"passed":true,"checks":{},"criterion_audit":[],"intrinsic_reward_eligible":true,"issues":[],"feedback":""}',
        '{"passed":true,"checks":{},"solver_context_audit":{"passed":true,"leakage_types":[],"evidence":[]},'
        '"criterion_audit":[],"intrinsic_reward_eligible":true,"issues":[],"feedback":""}',
    ])
    judge = _ResponseModel([
        '{"valid":true,"weak_scores":[0.9],"judge_score":1,"reasons":[]}',
    ])
    evaluator = CandidateEvaluator(weak, _ResponseModel([]), judge, quality_verifier=verifier,
                                   weak_rollouts=1, contract_retries=1)
    candidate = Candidate("c", "t", "s", {"question": "q", "answer": "a", "rubric": []})

    result = evaluator.evaluate(TaskSpec("t", "qa", "x"), candidate)

    assert verifier.calls == 2
    assert weak.calls == 1
    assert result.weak_rollouts == [0.9]
