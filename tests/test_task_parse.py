from __future__ import annotations

from autodata.models import SourceDocument, TaskSpec
from autodata.tasks import QuestionAnswerAdapter


def test_adapter_accepts_json_fenced_by_model():
    raw = '''Here is the candidate:\n```json
{"question":"What is OAuth?", "answer":"A protocol", "rubric": [{"criterion":"Identifies OAuth as an authorization protocol"}]}
```'''
    candidate = QuestionAnswerAdapter().parse(raw, TaskSpec("q", "qa", "x"), SourceDocument("s", "x"), "id")
    assert candidate.payload["question"] == "What is OAuth?"


def test_adapter_skips_an_echoed_schema_for_the_challenger_record():
    raw = ('Schema: {"type":"object","properties":{"question":{"type":"string"}}} '
           'Record: {"question":"What is OAuth?","answer":"A protocol",'
           '"rubric":[{"criterion":"Identifies OAuth as an authorization protocol"}]}')
    candidate = QuestionAnswerAdapter().parse(raw, TaskSpec("q", "qa", "x"), SourceDocument("s", "x"), "id")
    assert candidate.payload["answer"] == "A protocol"


def test_adapter_rejects_json_objects_without_the_challenger_contract():
    raw = '{"type":"object","properties":{"question":{"type":"string"}}}'
    try:
        QuestionAnswerAdapter().parse(raw, TaskSpec("q", "qa", "x"), SourceDocument("s", "x"), "id")
    except ValueError as exc:
        assert "none matched required fields" in str(exc)
    else:
        raise AssertionError("schema echo must not be accepted as a challenger record")
