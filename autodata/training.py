"""Dataset export and leakage checks for GRPO pilots.

This module intentionally has no torch/TRL dependency.  It converts accepted
Autodata candidates into records that a trainer can roll out against and score
at *training time*.  The weak/strong scores used to select a candidate are
quality-control metadata, never rewards for a newly generated completion.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import Candidate
from .prompts import solver_system_prompt


HIDDEN_FIELDS = {
    "answer", "reference_answer", "reference_solution", "solution", "rubric", "tests",
    "verification", "self_audit", "target_capabilities", "reasoning_skills", "question_type",
}


@dataclass(frozen=True, slots=True)
class GRPORecord:
    """One prompt plus the material needed to score a fresh rollout."""

    id: str
    group_id: str
    prompt: str
    task: dict[str, Any]
    rubric: list[dict[str, Any]]
    reference: str | None
    reward_type: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "prompt": self.prompt,
            "task": self.task,
            "rubric": self.rubric,
            "reference": self.reference,
            "reward_type": self.reward_type,
            "metadata": self.metadata,
        }


def solver_visible_payload(candidate: Candidate) -> dict[str, Any]:
    """Return exactly the task surface that a rollout model is allowed to see."""
    payload = {key: value for key, value in candidate.payload.items() if key not in HIDDEN_FIELDS}
    payload["environment"] = candidate.provenance.get("environment", {})
    return payload


def candidate_to_grpo_record(candidate: Candidate) -> GRPORecord:
    """Make a training record from an accepted, intrinsically-rewardable task.

    Exact-answer tasks get a deterministic reward.  Other accepted tasks use a
    rubric-judge contract and require an independent online judge during GRPO.
    Coding tasks are deliberately marked ``external_verifier``: executing
    challenger-provided code in the trainer process would be unsafe.
    """
    if not candidate.accepted:
        raise ValueError(f"candidate {candidate.id} was not accepted")
    payload = candidate.payload
    task = solver_visible_payload(candidate)
    reference = _reference(payload)
    if candidate.provenance.get("kind") == "coding" or "reference_solution" in payload or "tests" in payload:
        reward_type = "external_verifier"
    elif reference is not None and _is_short_answer(reference):
        reward_type = "exact_match"
    else:
        reward_type = "rubric_judge"
    prompt = (
        "You are an independent task solver. Solve only the task below using its declared environment. "
        "Give a direct, final answer; do not mention hidden rubrics or evaluation.\n\n"
        + json.dumps(task, ensure_ascii=False)
    )
    profile = str(candidate.provenance.get("profile", "generic"))
    return GRPORecord(
        id=candidate.id,
        # Source-level grouping prevents source excerpts appearing on both sides.
        group_id=f"{candidate.task_name}:{candidate.source_id}",
        prompt=prompt,
        task=task,
        rubric=list(payload.get("rubric", [])),
        reference=reference,
        reward_type=reward_type,
        metadata={
            "task_name": candidate.task_name,
            "source_id": candidate.source_id,
            "capabilities": candidate.capabilities,
            "profile": profile,
            "selection_evaluation": candidate.evaluation,
        },
    )


def export_grpo_records(candidates: Iterable[Candidate], path: str | Path, *, allow_external_verifier: bool = False) -> list[GRPORecord]:
    records = [candidate_to_grpo_record(candidate) for candidate in candidates]
    unsupported = [record.id for record in records if record.reward_type == "external_verifier"]
    if unsupported and not allow_external_verifier:
        raise ValueError(
            "coding/external-verifier records need a sandbox reward adapter; refusing to export them for text-only GRPO: "
            + ", ".join(unsupported[:5])
        )
    if not records:
        raise ValueError("no accepted candidates available for GRPO export")
    Path(path).write_text("".join(json.dumps(record.as_dict(), ensure_ascii=False) + "\n" for record in records))
    return records


def load_grpo_records(path: str | Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    required = {"id", "group_id", "prompt", "task", "rubric", "reference", "reward_type", "metadata"}
    malformed = [str(record.get("id", "<unknown>")) for record in records if not required.issubset(record)]
    if malformed:
        raise ValueError(f"malformed GRPO records: {', '.join(malformed[:5])}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("GRPO record IDs must be unique")
    return records


def split_by_group(records: Iterable[dict[str, Any]], *, benchmark_fraction: float = 0.2, seed: str = "autodata-v1") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stable source-disjoint split, suitable for a small in-distribution pilot."""
    if not 0 < benchmark_fraction < 1:
        raise ValueError("benchmark_fraction must be between zero and one")
    train, benchmark = [], []
    for record in records:
        digest = hashlib.sha256(f"{seed}:{record['group_id']}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (benchmark if bucket < benchmark_fraction else train).append(record)
    if not train or not benchmark:
        raise ValueError("split produced an empty side; export more source groups or change the seed/fraction")
    ensure_disjoint(train, benchmark)
    return train, benchmark


def ensure_disjoint(train: Iterable[dict[str, Any]], benchmark: Iterable[dict[str, Any]]) -> None:
    train_ids = {record["id"] for record in train}
    benchmark_ids = {record["id"] for record in benchmark}
    train_groups = {record["group_id"] for record in train}
    benchmark_groups = {record["group_id"] for record in benchmark}
    if train_ids & benchmark_ids or train_groups & benchmark_groups:
        raise ValueError("train and benchmark overlap by record or source group")


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def normalize_answer(text: str) -> str:
    """Conservative normalizer for short, deterministic answer rewards."""
    final = text.rsplit("</think>", 1)[-1]
    final = re.sub(r"\s+", " ", final.strip().casefold())
    return re.sub(r"^[\s\W]*(?:final answer|answer)\s*[:=-]\s*", "", final)


def exact_match_reward(completions: list[Any], reference: list[str | None], **_: Any) -> list[float]:
    """TRL-compatible reward callback for records declared exact-match."""
    results = []
    for completion, expected in zip(completions, reference):
        if expected is None:
            results.append(0.0)
            continue
        content = _completion_text(completion)
        results.append(float(normalize_answer(content) == normalize_answer(expected)))
    return results


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _reference(payload: dict[str, Any]) -> str | None:
    for key in ("answer", "reference_answer"):
        value = payload.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
    return None


def _is_short_answer(value: str) -> bool:
    return len(value) <= 512 and "\n" not in value.strip()
