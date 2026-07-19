"""Paired comparison with a bootstrap confidence interval."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def percentile(values, p):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    baseline = {item["id"]: item for item in json.loads(Path(args.baseline).read_text())["results"]}
    trained = {item["id"]: item for item in json.loads(Path(args.trained).read_text())["results"]}
    if set(baseline) != set(trained):
        raise ValueError("baseline and trained evaluation must contain exactly the same benchmark IDs")
    deltas = [trained[key]["mean_reward"] - baseline[key]["mean_reward"] for key in sorted(baseline)]
    rng = random.Random(args.seed)
    bootstrap = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(args.bootstrap_samples)]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    print(json.dumps({
        "paired_examples": len(deltas), "mean_reward_delta": mean(deltas),
        "bootstrap_95_ci": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "trained_wins": wins, "baseline_wins": losses, "ties": len(deltas) - wins - losses,
    }, indent=2))


if __name__ == "__main__":
    main()
