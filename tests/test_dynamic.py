import json

from autodata.discovery import ContractSynthesizer, SourceProfile
from autodata.models import Evaluation
from autodata.utility import AdaptiveUtilityGate, PilotResult


class ContractModel:
    def complete(self, prompt, *, system=""):
        return json.dumps({"name": "derived", "kind": "custom", "instructions": "Use source facts only.", "capabilities": ["comparison"], "environment": {"success_conditions": ["compare supplied values"]}, "require_rubric": True})


def test_contract_is_synthesized_from_profile_not_domain_template():
    profile = SourceProfile("s", True, "unknown", "", ["a"], ["compare"], ["comparison"], {"success_conditions": ["x"]})
    spec = ContractSynthesizer(ContractModel()).synthesize(profile, "Build useful tasks")
    assert spec.name == "derived"
    assert spec.environment["success_conditions"]


def test_adaptive_gate_rejects_zero_variance_and_pilot_requires_heldout_gain():
    gate = AdaptiveUtilityGate(calibration_examples=2)
    varied = Evaluation(True, 0.3, 0.8, 1.0, weak_rollouts=[0.1, 0.5], strong_rollouts=[0.7, 0.9])
    gate.assess(varied)
    gate.assess(varied)
    flat = Evaluation(True, 0.2, 0.8, 1.0, weak_rollouts=[0.2, 0.2], strong_rollouts=[0.8, 0.8])
    assert not gate.assess(flat).accepted
    assert PilotResult([0.1, 0.2, 0.3], [0.3, 0.4, 0.5]).useful()
