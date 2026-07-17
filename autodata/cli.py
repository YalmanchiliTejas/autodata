from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .evaluation import CandidateEvaluator
from .io import load_sources, load_specs, write_jsonl
from .pipeline import DatasetBuilder


def _providers(path: str):
    """Load a factory as ``package.module:function``.

    The factory must return a mapping containing `generator`, `weak`, `strong`,
    and `judge` providers, each with a `.complete()` method.
    """
    module_name, separator, name = path.partition(":")
    if not separator:
        raise ValueError("--provider must be MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), name)
    providers = factory()
    missing = {"generator", "weak", "strong", "judge"} - set(providers)
    if missing:
        raise ValueError(f"provider factory is missing roles: {', '.join(sorted(missing))}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a quality-controlled synthetic dataset.")
    parser.add_argument("--tasks", required=True, help="Task-spec JSON array")
    parser.add_argument("--sources", required=True, help="Grounding-source JSONL")
    parser.add_argument("--output", required=True, help="Accepted-candidate JSONL")
    parser.add_argument("--report", default="dataset-report.json", help="Dataset quality report JSON")
    parser.add_argument("--provider", required=True, help="Python provider factory: module:function")
    parser.add_argument("--rounds", type=int, default=3, help="Generation attempts per source/task")
    args = parser.parse_args()

    providers = _providers(args.provider)
    evaluator = CandidateEvaluator(providers["weak"], providers["strong"], providers["judge"])
    candidates, report = DatasetBuilder(providers["generator"], evaluator).build(
        load_specs(args.tasks), load_sources(args.sources), rounds_per_source=args.rounds
    )
    write_jsonl(args.output, candidates)
    Path(args.report).write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(f"accepted {report.accepted}/{report.total} candidates; report: {args.report}")


if __name__ == "__main__":
    main()
