from autodata.rubrics import validate_rubric


def test_rubric_requires_environment_without_guessing_semantic_duplicates():
    payload = {"rubric": [
        {"criterion": "Identifies the off-by-one boundary in the loop", "weight": 4, "category": "positive"},
        {"criterion": "Identifies the loop off-by-one boundary error", "weight": 3, "category": "positive"},
    ]}
    issues = validate_rubric(payload, [], {})
    assert any("environment contract" in issue for issue in issues)
    assert not any("near-duplicates" in issue for issue in issues)


def test_rubric_rejects_effectively_identical_local_duplicates():
    criterion = "Identifies the off-by-one boundary in the loop"
    payload = {"rubric": [
        {"criterion": criterion, "weight": 4, "category": "positive"},
        {"criterion": criterion, "weight": 3, "category": "positive"},
    ]}
    issues = validate_rubric(payload, [], {"success_conditions": ["identify defect"]})
    assert any("near-duplicates" in issue for issue in issues)


def test_rubric_environment_contract_allows_specific_checks():
    payload = {"rubric": [{"criterion": "Returns a sorted list without mutating the input sequence", "weight": 5, "category": "positive"}]}
    assert validate_rubric(payload, [], {"success_conditions": ["return sorted values"]}) == []
