# Accepted Workflows And Trajectories

`accepted.jsonl` contains one `autodata.accepted.v1` JSON object per accepted
workflow. It is the portable dataset artifact. A record has:

- `workflow_id`: deterministic ID derived from the source hash and workflow.
- `source`: input document identifier and SHA-256 digest.
- `workflow`: task type, skill tags, execution context, structured task
  specification, reference workflow, and weighted rubric.
- `acceptance`: the accepted round, weak/strong/gap metrics, and the complete
  threshold policy used for the decision.
- `models`: model identifiers for every pipeline role.

The exact prompts, responses, retries, and accept/reject decisions are written to
the JSONL trajectory file as `autodata.trajectory.v1` events. Each event includes a
run ID, document ID, timestamp, level, agent name, and round. Agent `turn` events
contain the full prompt and response; control events contain the reason and metrics.

Run with `--verbose` to print each stage as it completes while preserving the full
content in the trajectory file:

```bash
python -m autodata.cli data/ --out accepted.jsonl --log trajectories.jsonl --verbose
```

Use `--trace-report` to create a readable Markdown review file for only the current
run. Every model turn includes its model ID, system instructions, exact context/user
input, and response; control-flow events include acceptance and rejection details.

```bash
python -m autodata.cli data/ --log trajectories.jsonl --trace-report trajectory.md
```

Do not train directly from trajectory logs. Use only accepted records after checking
their source digest and acceptance policy match the intended dataset version.
