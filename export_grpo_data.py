"""Export accepted Autodata JSONL records and make a source-disjoint pilot split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodata.models import Candidate
from autodata.training import export_grpo_records, load_grpo_records, split_by_group, write_records


def _candidates(path: str) -> list[Candidate]:
    return [Candidate(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Accepted Autodata candidate JSONL")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--benchmark-output", required=True)
    parser.add_argument("--benchmark-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="autodata-v1")
    parser.add_argument("--allow-external-verifier", action="store_true")
    args = parser.parse_args()
    staging = Path(args.train_output).with_suffix(".all.jsonl")
    export_grpo_records(_candidates(args.input), staging, allow_external_verifier=args.allow_external_verifier)
    train, benchmark = split_by_group(load_grpo_records(staging), benchmark_fraction=args.benchmark_fraction, seed=args.seed)
    write_records(args.train_output, train)
    write_records(args.benchmark_output, benchmark)
    staging.unlink()
    print(f"wrote {len(train)} train and {len(benchmark)} source-disjoint benchmark records")


if __name__ == "__main__":
    main()
