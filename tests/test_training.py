from __future__ import annotations

import pytest

from autodata.models import Candidate
from autodata.training import (candidate_to_grpo_record, ensure_disjoint, exact_match_reward,
                               split_by_group, solver_visible_payload)


def candidate(**payload):
    return Candidate("c1", "arithmetic", "source-a", payload, ["arithmetic"], "medium",
                     provenance={"environment": {"success_conditions": ["return the sum"]}, "profile": "generic"},
                     evaluation={"weak_rollouts": [0, 1]}, accepted=True)


def test_training_record_hides_solution_and_uses_exact_reward():
    record = candidate_to_grpo_record(candidate(question="What is 2+2?", answer="4", solution="2 plus 2", rubric=[]))
    assert record.reward_type == "exact_match"
    assert "answer" not in solver_visible_payload(candidate(question="x", answer="4"))
    assert exact_match_reward(["<think>work</think> Final answer: 4"], ["4"]) == [1.0]


def test_split_is_source_disjoint_and_detects_overlap():
    records = [
        {"id": "one", "group_id": "a"}, {"id": "two", "group_id": "b"},
        {"id": "three", "group_id": "c"}, {"id": "four", "group_id": "d"},
    ]
    train, benchmark = split_by_group(records, benchmark_fraction=0.5, seed="test")
    ensure_disjoint(train, benchmark)
    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint([records[0]], [records[0]])
