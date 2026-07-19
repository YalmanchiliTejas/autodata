from __future__ import annotations

from autodata.models import Candidate
from autodata.tasks import CodingAdapter, _output_contract


def test_coding_contract_names_required_top_level_and_test_fields():
    contract = _output_contract(type("Spec", (), {"kind": "coding"})())
    assert "JSON Schema" in contract
    assert '"reference_solution"' in contract
    assert '"name"' in contract and '"code"' in contract


def test_coding_validation_rejects_unstructured_tests():
    candidate = Candidate("c", "coding", "s", {"prompt": "x", "reference_solution": "x", "tests": ["assert True"]},
                          provenance={"profile": "generic", "environment": {"success_conditions": ["x"]}})
    assert any("coding tests must be objects" in issue for issue in CodingAdapter().validate(candidate))


def test_coding_validation_rejects_tests_that_embed_reference_solution():
    reference = "def target():\n    return 1"
    candidate = Candidate("c", "coding", "s", {
        "prompt": "Implement target", "reference_solution": reference,
        "tests": [{"name": "copies_solution", "code": reference + "\nassert target() == 1"}],
    }, provenance={"profile": "generic", "environment": {"success_conditions": ["x"]}})
    assert any("not embed the reference solution" in issue for issue in CodingAdapter().validate(candidate))
