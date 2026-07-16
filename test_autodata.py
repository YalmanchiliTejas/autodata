"""Self-check for the Agentic Self-Instruct loop, offline (mock LLM).
Run: python test_autodata.py"""
import json
import os
import tempfile
import time

from autodata import Config, TrajectoryLogger, run_doc
from autodata import agents
from autodata.accepted import format_accepted, validate_accepted
from autodata.coverage import CoverageGraph
from autodata.environment import EnvironmentSpec
from autodata.export import export_record
from autodata.loop import _solve_samples, _valid_scores, _weak_rejection
from autodata.trajectory_report import write_trajectory_report


def test_loop_rejects_then_accepts():
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    cfg = Config(mock=True, log_path=log_path, max_rounds=5)
    log = TrajectoryLogger(log_path)

    ex = run_doc(cfg, log, "paper1", "Some grounding text about algorithms.")
    log.close()

    # Mock ramps strong past 0.65 at round 2 -> round 1 rejects, round 2 accepts.
    assert ex is not None, "expected an accepted example"
    assert ex["round"] == 2, f"expected accept on round 2, got {ex['round']}"
    assert cfg.strong_min <= ex["strong_avg"] < cfg.strong_max
    assert ex["weak_avg"] <= cfg.weak_avg_max
    assert ex["gap"] >= cfg.min_gap

    lines = [json.loads(l) for l in open(log_path)]
    agents = {l["agent"] for l in lines}
    assert {"challenger", "quality_verifier", "weak_solver", "weak_screen_judge",
            "strong_solver", "strong_judge"} <= agents
    kinds = {l["kind"] for l in lines}
    assert "reject" in kinds and "accept" in kinds, f"kinds={kinds}"
    # Each round: challenger + verifier + N weak + weak judge + N strong + strong judge.
    turns_r1 = [l for l in lines if l["kind"] == "turn" and l.get("round") == 1]
    assert len(turns_r1) == 4 + 2 * cfg.solver_samples, len(turns_r1)
    print(f"OK: accepted round {ex['round']}, gap {ex['gap']}, {len(lines)} log lines")


def test_global_loop_budget_caps_rounds():
    # Budget of 1 total round: round 1 rejects (mock accepts only at round 2),
    # then the gate stops before round 2 -> no accept, exactly 1 round run.
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    cfg = Config(mock=True, log_path=log_path, max_rounds=5, max_total_rounds=1)
    log = TrajectoryLogger(log_path)

    ex = run_doc(cfg, log, "paper1", "Some grounding text.")
    log.close()

    assert ex is None, "budget should have stopped it before acceptance"
    assert cfg.rounds_run == 1, f"expected 1 round, got {cfg.rounds_run}"
    kinds = {json.loads(l)["kind"] for l in open(log_path)}
    assert "budget_exhausted" in kinds, kinds
    print(f"OK: global budget stopped at {cfg.rounds_run} round")


def test_validation_and_weak_screening():
    cfg = Config(mock=True)
    assert _weak_rejection(cfg, [0.76]) == "weak_max<="
    assert _weak_rejection(cfg, [0.0]) == "weak_no_zeros"
    assert _weak_rejection(cfg, [0.5, 0.5, 0.5]) is None
    assert _valid_scores([0.0, 1.0], 2)
    assert not _valid_scores([0.2, 0.8, 0.5], 2)
    bad = {"type": "x", "context": "x", "user_prompt": "x", "task": {}, "reference_workflow": [], "rubric": []}
    assert agents.validate_example(bad)
    assert "adding its negative weight" in agents.JUDGE_SYS
    print("OK: validation, exact judge cardinality, and weak-first screening")


def test_weak_failure_skips_strong_solver():
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    cfg = Config(mock=True, log_path=log_path, max_rounds=1, weak_avg_max=0.2)
    log = TrajectoryLogger(log_path)
    assert run_doc(cfg, log, "paper1", "Some grounding text.") is None
    log.close()
    lines = [json.loads(line) for line in open(log_path)]
    assert not any(line["agent"] == "strong_solver" for line in lines)
    assert any(line.get("stage") == "weak_screen" for line in lines)
    print("OK: weak failures skip strong-solver spend")


def test_accepted_record_format():
    cfg = Config(mock=True)
    raw = {
        "doc_id": "paper.txt", "round": 2, "type": "analysis", "skill_tags": ["reasoning"],
        "context": "Context",
        "user_prompt": "Please investigate this issue and provide the requested result with supporting evidence.",
        "task": {"objective": "Do the task", "inputs": [], "constraints": [],
                 "required_actions": ["Inspect", "Deliver"],
                 "deliverables": [{"name": "result", "format": "text", "description": "Result"}]},
        "reference_workflow": [{"step": 1, "action": "Inspect", "expected_result": "Facts"},
                               {"step": 2, "action": "Deliver", "expected_result": "Result"}],
        "rubric": [{"criterion": "a", "weight": 1, "category": "positive"}],
        "weak_avg": 0.3, "strong_avg": 0.7, "gap": 0.4,
        "rollouts": {"strong": [{"response": "Completed with evidence.", "score": 0.9}]},
    }
    record = format_accepted(raw, "paper.txt", "Source document", cfg)
    assert not validate_accepted(record)
    assert record["workflow"]["reference_workflow"][1]["action"] == "Deliver"
    assert record["source"]["sha256"]
    assert record["learning"]["preferred_example"]["response"] == "Completed with evidence."
    print("OK: accepted workflow record is versioned and valid")


def test_coverage_graph_caches_and_rotates_cards():
    d = tempfile.mkdtemp()
    graph = CoverageGraph(os.path.join(d, "coverage.json"))
    cfg = Config(mock=True)
    source_sha, entry = graph.get_or_create(cfg, "paper.txt", "Source document")
    preview = graph.preview(source_sha, 1)
    assert preview[0]["attempts"] == 0
    first = graph.select(source_sha, 1)
    second = graph.select(source_sha, 1)
    assert first[0]["id"] != second[0]["id"]
    assert not graph.validate(source_sha, "Source document")
    assert "Coverage graph: 1 document(s)" in graph.report()
    assert "SELECTED EVIDENCE CARDS" in graph.render(entry, first)
    _, cached = graph.get_or_create(cfg, "paper.txt", "Source document")
    assert cached["cards"][0]["attempts"] >= 1
    print("OK: coverage graph persists and rotates underused evidence")


def test_coverage_mapper_chunks_long_documents():
    cfg = Config(mock=True)
    long_document = "x" * (agents.COVERAGE_CHUNK_CHARS + 1)
    mapped, _ = agents.extract_coverage(cfg, long_document)
    assert len(mapped["cards"]) == 4
    assert mapped["cards"][2]["_source_offset"] == agents.COVERAGE_CHUNK_CHARS
    print("OK: coverage mapper chunks long documents")


def test_independent_extractor_chunks_run_concurrently_and_keep_order():
    cfg = Config(mock=True, extractor_workers=3)
    original_complete = agents.complete
    original_chunk_chars = agents.CHUNK_CHARS

    def delayed_extract(_cfg, _model, _system, user, max_tokens=1500):
        time.sleep(0.1)
        return f"extract:{user}"

    agents.complete = delayed_extract
    agents.CHUNK_CHARS = 3
    try:
        started = time.monotonic()
        result = agents.extract(cfg, "aaabbbccc")
        elapsed = time.monotonic() - started
    finally:
        agents.complete = original_complete
        agents.CHUNK_CHARS = original_chunk_chars

    assert result == "extract:aaa\n\nextract:bbb\n\nextract:ccc"
    assert elapsed < 0.22, f"extractor chunks were serial: {elapsed:.3f}s"
    print(f"OK: extractor chunks completed concurrently in {elapsed:.3f}s")


def test_independent_coverage_chunks_run_concurrently_and_keep_order():
    cfg = Config(mock=True, extractor_workers=3)
    original_complete = agents.complete
    original_chunk_chars = agents.COVERAGE_CHUNK_CHARS

    def delayed_map(_cfg, _model, _system, user, max_tokens=1500):
        time.sleep(0.1)
        return json.dumps({"summary": f"summary:{user}", "cards": [{
            "title": f"card:{user}", "tags": ["test"], "facts": [f"fact:{user}"],
            "evidence_quotes": [user],
        }]})

    agents.complete = delayed_map
    agents.COVERAGE_CHUNK_CHARS = 3
    try:
        started = time.monotonic()
        mapped, raw = agents.extract_coverage(cfg, "aaabbbccc")
        elapsed = time.monotonic() - started
    finally:
        agents.complete = original_complete
        agents.COVERAGE_CHUNK_CHARS = original_chunk_chars

    assert [card["title"] for card in mapped["cards"]] == ["card:aaa", "card:bbb", "card:ccc"]
    assert [card["_source_offset"] for card in mapped["cards"]] == [0, 3, 6]
    assert raw.split("\n\n") == [
        json.dumps({"summary": "summary:aaa", "cards": [{"title": "card:aaa", "tags": ["test"], "facts": ["fact:aaa"], "evidence_quotes": ["aaa"]}]}),
        json.dumps({"summary": "summary:bbb", "cards": [{"title": "card:bbb", "tags": ["test"], "facts": ["fact:bbb"], "evidence_quotes": ["bbb"]}]}),
        json.dumps({"summary": "summary:ccc", "cards": [{"title": "card:ccc", "tags": ["test"], "facts": ["fact:ccc"], "evidence_quotes": ["ccc"]}]}),
    ]
    assert elapsed < 0.22, f"coverage chunks were serial: {elapsed:.3f}s"
    print(f"OK: coverage chunks completed concurrently in {elapsed:.3f}s")


def test_coverage_guided_run_updates_usage_and_provenance():
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    cfg = Config(mock=True, log_path=log_path, max_rounds=2)
    graph = CoverageGraph(os.path.join(d, "coverage.json"))
    log = TrajectoryLogger(log_path)
    ex = run_doc(cfg, log, "paper.txt", "Source document", coverage=graph)
    log.close()
    assert ex and ex["coverage"]["mode"] == "coverage_cards"
    cards = graph.data["documents"][ex["coverage"]["source_sha256"]]["cards"]
    assert all(card["attempts"] == 2 and card["accepted"] == 1 for card in cards)
    lines = [json.loads(line) for line in open(log_path)]
    assert sum(line["kind"] == "coverage_selected" for line in lines) == 2
    print("OK: coverage-guided run updates card usage and accepted provenance")


def test_coverage_escalates_after_strong_failures():
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    cfg = Config(mock=True, log_path=log_path, max_rounds=2, strong_min=0.9,
                 coverage_escalation_rounds=1)
    graph = CoverageGraph(os.path.join(d, "coverage.json"))
    log = TrajectoryLogger(log_path)
    assert run_doc(cfg, log, "paper.txt", "Source document", coverage=graph) is None
    log.close()
    lines = [json.loads(line) for line in open(log_path)]
    assert any(line["kind"] == "coverage_escalated" for line in lines)
    assert any(line["agent"] == "coverage_fallback_extractor" for line in lines)
    print("OK: repeated strong failures escalate to the full compact extract")


def test_trajectory_report_includes_model_context():
    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "traj.jsonl")
    report_path = os.path.join(d, "trace.md")
    cfg = Config(mock=True, log_path=log_path, max_rounds=2)
    log = TrajectoryLogger(log_path)
    run_doc(cfg, log, "paper1", "Grounding document")
    log.close()
    assert write_trajectory_report(log_path, report_path, log.run_id) > 0
    report = open(report_path).read()
    assert "Context Passed To Model" in report
    assert "System Instructions" in report
    assert "claude-sonnet-5" in report
    print("OK: trajectory report includes model context")


def test_solver_receives_user_request_not_hidden_task_contract():
    cfg = Config(mock=True)
    example = {
        "user_prompt": "Please investigate the failed import and tell me which records need attention.",
        "context": "The agent can inspect import status through the available service.",
        "task": {"objective": "INTERNAL: call the import API and verify three hidden assertions."},
    }
    captured = {}
    original_complete = agents.complete

    def capture(_cfg, _model, system, user, max_tokens=1500):
        captured.update({"system": system, "user": user, "max_tokens": max_tokens})
        return "done"

    agents.complete = capture
    try:
        assert agents.solve(cfg, "weak-model", example) == "done"
    finally:
        agents.complete = original_complete

    assert "USER REQUEST" in captured["user"]
    assert example["user_prompt"] in captured["user"]
    assert "TASK SPEC" not in captured["user"]
    assert "INTERNAL:" not in captured["user"]
    assert "Do not return an execution plan" in captured["system"]
    print("OK: solver sees only the user-facing request and visible context")


def test_prompts_do_not_assume_a_local_project_or_tools():
    assert "Do not assume, inspect, target, or\nrefer to a local repository" in agents.CHALLENGER_SYS
    assert "only the supplied context and tools explicitly available" in agents.SOLVER_SYS
    assert "project-specific interfaces" in agents.VERIFIER_SYS
    print("OK: task prompts are source-scoped rather than project-scoped")


def test_environment_contract_and_learning_exports_are_portable():
    spec = EnvironmentSpec.from_mapping({
        "provider": "example", "name": "test-env", "description": "A test service.",
        "capabilities": ["read records"], "tools": [{"name": "lookup"}],
        "constraints": ["No mutations"],
    })
    cfg = Config(mock=True, environment=spec)
    example = {
        "doc_id": "doc", "round": 1, "type": "diagnosis", "skill_tags": [], "context": "facts",
        "user_prompt": "Please inspect the documented issue and provide a complete evidence-backed result.",
        "task": {"objective": "Diagnose", "inputs": [], "constraints": [],
                 "required_actions": ["Inspect", "Report"],
                 "deliverables": [{"name": "result", "format": "text", "description": "result"}]},
        "reference_workflow": [{"step": 1, "action": "Inspect", "expected_result": "facts"},
                               {"step": 2, "action": "Report", "expected_result": "result"}],
        "rubric": [{"criterion": "correct", "weight": 1, "category": "positive"}],
        "weak_avg": 0.2, "strong_avg": 0.8, "gap": 0.6,
        "rollouts": {"strong": [{"response": "Evidence-backed result.", "score": 0.8}]},
    }
    record = format_accepted(example, "doc", "source", cfg)
    assert record["environment"]["name"] == "test-env"
    assert export_record(record, "openai_chat")["messages"][-1]["content"] == "Evidence-backed result."
    assert export_record(record, "prime_verifiers")["answer"] == "Evidence-backed result."
    print("OK: environment contract and learning exports are portable")


def test_solver_rollouts_run_concurrently_and_keep_order():
    cfg = Config(mock=True, solver_samples=3)
    original_solve = agents.solve

    def delayed_solve(_cfg, model, _example):
        time.sleep(0.1)
        return f"result-{model}"

    agents.solve = delayed_solve
    try:
        started = time.monotonic()
        results = _solve_samples(cfg, "weak-model", {"task": {}})
        elapsed = time.monotonic() - started
    finally:
        agents.solve = original_solve

    assert results == ["result-weak-model"] * 3
    assert elapsed < 0.22, f"rollouts were serial: {elapsed:.3f}s"
    print(f"OK: three solver rollouts completed concurrently in {elapsed:.3f}s")


if __name__ == "__main__":
    test_loop_rejects_then_accepts()
    test_global_loop_budget_caps_rounds()
    test_validation_and_weak_screening()
    test_weak_failure_skips_strong_solver()
    test_accepted_record_format()
    test_coverage_graph_caches_and_rotates_cards()
    test_coverage_mapper_chunks_long_documents()
    test_independent_extractor_chunks_run_concurrently_and_keep_order()
    test_independent_coverage_chunks_run_concurrently_and_keep_order()
    test_coverage_guided_run_updates_usage_and_provenance()
    test_coverage_escalates_after_strong_failures()
    test_trajectory_report_includes_model_context()
    test_solver_receives_user_request_not_hidden_task_contract()
    test_prompts_do_not_assume_a_local_project_or_tools()
    test_environment_contract_and_learning_exports_are_portable()
    test_solver_rollouts_run_concurrently_and_keep_order()
    print("PASS")
