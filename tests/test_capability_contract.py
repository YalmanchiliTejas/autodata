from autodata.models import TaskSpec
from autodata.prompts import rubric_authoring_contract


def test_rubric_contract_requires_capability_coverage():
    assert "must appear verbatim" in rubric_authoring_contract(TaskSpec("x", "coding", "x"))
