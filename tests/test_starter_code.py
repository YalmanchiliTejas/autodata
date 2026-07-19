from autodata.models import SourceDocument, TaskSpec
from autodata.tasks import ADAPTERS


def _candidate(starter_code: str):
    spec = TaskSpec("code", "coding", "Implement it", require_rubric=False)
    payload = {
        "prompt": "Implement the public API.",
        "starter_code": starter_code,
        "reference_solution": "def solve(value):\n    return value",
        "tests": [{"name": "identity", "code": "assert solve(1) == 1"}],
        "self_audit": {"grounded": True, "within_scope": True, "no_answer_leakage": True,
                       "fairness_checked": True, "verifiable": True},
    }
    import json
    return ADAPTERS["coding"].parse(json.dumps(payload), spec, SourceDocument("s", "source"), "c")


def test_starter_code_accepts_signatures_and_empty_bodies():
    candidate = _candidate('def solve(value: str) -> str:\n    """Return the transformed value."""\n    ...')
    assert ADAPTERS["coding"].validate(candidate) == []


def test_starter_code_rejects_solution_signaling_imports_and_constants():
    candidate = _candidate(
        "import hashlib\nimport base64\nALLOWED_CHARS = 'abc'\n\ndef solve(value):\n    ..."
    )
    issues = ADAPTERS["coding"].validate(candidate)
    assert any("must not import solution-signaling tools" in issue for issue in issues)
    assert any("must not define solution-signaling constants" in issue for issue in issues)


def test_starter_code_rejects_partial_implementation_logic():
    candidate = _candidate("def solve(value):\n    encoded = value.encode()\n    return encoded")
    issues = ADAPTERS["coding"].validate(candidate)
    assert any("function bodies must contain only" in issue for issue in issues)
