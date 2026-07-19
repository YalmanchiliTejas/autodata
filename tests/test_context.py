import pytest

from autodata.context import ContextController, ContextOverflow, ContextPolicy, ContextSegment


def test_controller_preserves_protected_segments_and_compacts_old_material():
    controller = ContextController(ContextPolicy(context_window_tokens=200, reserve_output_tokens=40))
    plan = controller.prepare([
        ContextSegment("task", "Solve exactly this task.", protected=True),
        ContextSegment("environment", "No network. Run only listed tools.", protected=True),
        ContextSegment("old observation", "x" * 2000, priority=1, artifact_ref="artifact://run/old-output"),
    ])
    assert "Solve exactly this task." in plan.text
    assert "No network." in plan.text
    assert "old observation" in plan.compacted
    assert "artifact://run/old-output" in plan.text
    assert plan.estimated_tokens <= 160


def test_controller_refuses_to_silently_truncate_protected_surface():
    controller = ContextController(ContextPolicy(context_window_tokens=100, reserve_output_tokens=20))
    with pytest.raises(ContextOverflow):
        controller.prepare([ContextSegment("task", "x" * 500, protected=True)])
