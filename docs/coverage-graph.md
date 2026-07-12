# Coverage-Guided Generation

The coverage graph is used by default during generation and is stored at
`.autodata/coverage.json`.

On a source-hash cache miss, the cheap mapper reads the full document in conservative
100,000-character chunks and emits
a source summary plus non-overlapping evidence cards. Each card contains a stable ID,
topic/reasoning tags, source-grounded facts, and validated character offsets pointing
back to exact source evidence. The graph persists card attempts, accepted uses, and
global tag usage.

At each generation round, the scheduler selects the least-used cards, preferring tags
with lower global coverage. Selection increments card attempts; acceptance increments
accepted uses and writes selected card IDs into the accepted workflow record. After
two consecutive source-fidelity or strong-solver failures, the run escalates to the
legacy full compact extract for the rest of that document. Use `--no-coverage` to
compare against the legacy path.

Build or inspect the map without generation:

```bash
python -m autodata.cli data/ \
  --build-coverage \
  --coverage-cache .autodata/coverage.json \
  --coverage-report coverage-report.txt \
  --verbose
```

Delete the cache when the source corpus changes or when you intentionally want to
rebuild the evidence map. A changed document automatically receives a new source hash.
