from autodata.models import Candidate
from autodata.quality import DatasetQualityController


def candidate(identifier, payload, capabilities=("reasoning",), difficulty="hard", source="s1"):
    return Candidate(identifier, "task", source, payload, list(capabilities), difficulty, accepted=True)


def test_duplicate_detection_and_coverage_feedback():
    controller = DatasetQualityController(duplicate_threshold=0.6, min_capability_examples=2)
    first = candidate("a", {"question": "derive the graph shortest path algorithm", "answer": "dijkstra"})
    second = candidate("b", {"question": "derive graph shortest path algorithm", "answer": "dijkstra"})
    keep, reason = controller.keep(second, [first])
    assert not keep
    assert "near duplicate" in reason
    report = controller.analyze([first], ["reasoning", "proof"])
    assert "underrepresented capability: proof" in report.quality_issues


def test_source_concentration_is_reported():
    controller = DatasetQualityController(max_source_share=0.5)
    report = controller.analyze([candidate("a", {"question": "one", "answer": "x"}),
                                 candidate("b", {"question": "two", "answer": "y"})])
    assert any("source concentration" in issue for issue in report.quality_issues)
