"""Versioned, portable accepted-workflow records."""
from datetime import datetime, timezone
import hashlib
import json
import math


SCHEMA_VERSION = "autodata.accepted.v3"
PREVIOUS_SCHEMA_VERSION = "autodata.accepted.v2"
LEGACY_SCHEMA_VERSION = "autodata.accepted.v1"


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_accepted(example, document_id, document, cfg):
    """Create the JSONL record consumed by downstream training or review tools."""
    required = {"doc_id", "round", "type", "context", "user_prompt", "task", "reference_workflow", "rubric",
                "weak_avg", "strong_avg", "gap"}
    missing = required - example.keys()
    if missing:
        raise ValueError(f"cannot format accepted example; missing {sorted(missing)}")
    metrics = {key: float(example[key]) for key in ("weak_avg", "strong_avg", "gap")}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        raise ValueError("accepted metrics must be finite values in [0, 1]")

    source_sha256 = _sha256(document)
    workflow = {
        "type": example["type"],
        "skill_tags": list(example.get("skill_tags", [])),
        "context": example["context"],
        "user_prompt": example["user_prompt"],
        "task_spec": example["task"],
        "reference_workflow": example["reference_workflow"],
        "rubric": example["rubric"],
    }
    workflow_id = _sha256(json.dumps({"source_sha256": source_sha256, "workflow": workflow},
                                     sort_keys=True, separators=(",", ":")))[:24]
    record = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": f"adw_{workflow_id}",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"document_id": document_id, "sha256": source_sha256},
        "workflow": workflow,
        "acceptance": {
            "round": example["round"],
            "weak_avg": metrics["weak_avg"],
            "strong_avg": metrics["strong_avg"],
            "gap": metrics["gap"],
            "policy": {
                "weak_avg_max": cfg.weak_avg_max,
                "weak_max_max": cfg.weak_max_max,
                "weak_no_zeros": cfg.weak_no_zeros,
                "strong_min": cfg.strong_min,
                "strong_max": cfg.strong_max,
                "min_gap": cfg.min_gap,
                "solver_samples": cfg.solver_samples,
            },
        },
        "models": {
            "orchestrator": cfg.orchestrator,
            "weak_solver": cfg.weak_solver,
            "strong_solver": cfg.strong_solver,
            "extractor": cfg.extractor,
        },
    }
    if example.get("coverage"):
        record["coverage"] = example["coverage"]
    return record


def validate_accepted(record):
    """Return schema errors for a formatted accepted-workflow record."""
    required = {"schema_version", "workflow_id", "generated_at", "source", "workflow",
                "acceptance", "models"}
    if not isinstance(record, dict):
        return ["record is not an object"]
    errors = [f"missing {key}" for key in sorted(required - record.keys())]
    if record.get("schema_version") not in {SCHEMA_VERSION, PREVIOUS_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        errors.append(f"unsupported schema_version: {record.get('schema_version')}")
    workflow = record.get("workflow", {})
    if not isinstance(workflow, dict):
        return errors + ["workflow is not an object"]
    if record.get("schema_version") == SCHEMA_VERSION:
        workflow_keys = ("type", "context", "user_prompt", "task_spec", "reference_workflow", "rubric")
    elif record.get("schema_version") == PREVIOUS_SCHEMA_VERSION:
        workflow_keys = ("type", "context", "task_spec", "reference_workflow", "rubric")
    else:
        workflow_keys = ("type", "context", "question", "reference_answer", "rubric")
    for key in workflow_keys:
        if key not in workflow:
            errors.append(f"workflow missing {key}")
    if not isinstance(workflow.get("rubric"), list):
        errors.append("workflow.rubric is not a list")
    if record.get("schema_version") == SCHEMA_VERSION and (
        not isinstance(workflow.get("user_prompt"), str) or not workflow["user_prompt"].strip()
    ):
        errors.append("workflow.user_prompt must be a non-empty string")
    acceptance = record.get("acceptance", {})
    for key in ("round", "weak_avg", "strong_avg", "gap", "policy"):
        if key not in acceptance:
            errors.append(f"acceptance missing {key}")
    return errors
