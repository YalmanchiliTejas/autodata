from __future__ import annotations

import json
from pathlib import Path

from .models import Candidate, SourceDocument, TaskSpec


def load_specs(path: str | Path) -> list[TaskSpec]:
    data = json.loads(Path(path).read_text())
    return [TaskSpec(**item) for item in data]


def load_sources(path: str | Path) -> list[SourceDocument]:
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            items.append(SourceDocument(**json.loads(line)))
    return items


def write_jsonl(path: str | Path, candidates: list[Candidate]) -> None:
    Path(path).write_text("".join(json.dumps(item.as_dict(), ensure_ascii=False) + "\n" for item in candidates))


def write_records_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write audit records that are not necessarily Candidate objects."""
    Path(path).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
