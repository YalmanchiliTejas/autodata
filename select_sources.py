"""Make a small, reviewable source subset from an Autodata JSONL corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include", action="append", required=True,
                        help="Keep records whose ID contains this string; repeat for alternatives")
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    selected = [record for record in records if any(term in record["id"] for term in args.include)][:args.limit]
    if len(selected) < args.limit:
        raise ValueError(f"only found {len(selected)} matching records; requested {args.limit}")
    Path(args.output).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected))
    print(f"wrote {len(selected)} selected records to {args.output}")


if __name__ == "__main__":
    main()
