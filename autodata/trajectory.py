"""Trajectory logging. Every subagent call (challenger/solver/judge) is appended
as one JSON line so each agent's full trajectory can be replayed and audited."""
import json
import time
import uuid
from pathlib import Path


class TrajectoryLogger:
    def __init__(self, path, verbose=False):
        self.path = path
        self.doc_id = None
        self.run_id = f"run_{uuid.uuid4().hex[:16]}"
        self.verbose = verbose
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a")

    def start_doc(self, doc_id):
        self.doc_id = doc_id
        self._write("doc_start", "system", {"doc_id": doc_id}, "INFO")

    def log(self, agent, round_i, prompt, response, extra=None):
        """One agent turn: who acted, in which round, with what in/out."""
        self._write("turn", agent, {
            "round": round_i,
            "prompt": prompt,
            "response": response,
            **(extra or {}),
        }, "INFO")

    def event(self, kind, **fields):
        """A control-flow event: accept, reject, feedback, budget_exhausted."""
        level = "ERROR" if kind == "evaluation_error" else "WARN" if kind in {
            "reject", "evaluation_retry", "budget_exhausted"} else "INFO"
        self._write(kind, "orchestrator", fields, level)

    def _write(self, kind, agent, payload, level):
        line = {"schema_version": "autodata.trajectory.v1", "ts": time.time(),
                "run_id": self.run_id, "doc_id": self.doc_id, "level": level,
                "kind": kind, "agent": agent, **payload}
        self._f.write(json.dumps(line) + "\n")
        self._f.flush()
        if self.verbose:
            round_i = payload.get("round", "-")
            if kind == "turn":
                print(f"[{level}] {self.doc_id} round={round_i} {agent}: response recorded", flush=True)
            else:
                details = payload.get("stage") or payload.get("failed") or ""
                print(f"[{level}] {self.doc_id} round={round_i} {kind} {details}", flush=True)

    def close(self):
        self._f.close()
