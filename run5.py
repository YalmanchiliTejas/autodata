"""Real run: cycle the docs until 5 accepted examples, <=3 rounds each."""
import json, pathlib
from autodata import Config
from autodata import agents
from autodata.accepted import format_accepted, validate_accepted
from autodata.loop import run_doc
from autodata.trajectory import TrajectoryLogger

# Cache extraction so re-running a doc doesn't re-pay the (chunked) extractor.
_raw_extract, _cache = agents.extract, {}
def _cached_extract(cfg, doc):
    if doc not in _cache:
        _cache[doc] = _raw_extract(cfg, doc)
    return _cache[doc]
agents.extract = _cached_extract

cfg = Config(max_rounds=3)
log = TrajectoryLogger("trajectories.jsonl")
docs = [(p.name, p.read_text()) for p in
        (pathlib.Path("data/steel_llms.txt"), pathlib.Path("data/steel_llms_full.txt"))]

accepted, calls = [], 0
while len(accepted) < 5 and calls < 15:
    name, text = docs[calls % len(docs)]
    ex = run_doc(cfg, log, name, text)
    calls += 1
    if ex:
        accepted.append(ex)
        print(f"[ACCEPT {len(accepted)}/5] {name} round {ex['round']} gap {ex['gap']}", flush=True)
    else:
        print(f"[reject] {name} (call {calls})", flush=True)

with open("accepted.jsonl", "w") as f:
    for ex in accepted:
        document = next(text for name, text in docs if name == ex["doc_id"])
        record = format_accepted(ex, ex["doc_id"], document, cfg)
        if errors := validate_accepted(record):
            raise RuntimeError(f"accepted record validation failed: {errors}")
        f.write(json.dumps(record) + "\n")
log.close()
print(f"\nDONE: {len(accepted)} accepted in {calls} run_doc calls -> accepted.jsonl")
