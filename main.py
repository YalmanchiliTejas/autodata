"""Run Autodata with Claude roles and a Modal/vLLM weak solver.

Credentials are read from environment variables named in the JSON config; this
file never reads or writes secrets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodata.evaluation import CandidateEvaluator
from autodata.io import load_sources, load_specs, write_jsonl
from autodata.pipeline import DatasetBuilder
from autodata.providers import AnthropicModel, OpenAICompatibleModel
from autodata.utility import AdaptiveUtilityGate


def provider(config: dict):
    kind = config["provider"]
    options = {key: value for key, value in config.items() if key != "provider"}
    if kind == "anthropic":
        return AnthropicModel(**options)
    if kind == "openai_compatible":
        return OpenAICompatibleModel(**options)
    raise ValueError(f"unsupported provider {kind!r}; use anthropic or openai_compatible")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Autodata dataset with configured model roles.")
    parser.add_argument("--config", required=True, help="JSON run configuration")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    models = config["models"]
    weak = provider(models["weak"])
    strong = provider(models["strong"])
    challenger = provider(models["challenger"])
    judge = provider(models["judge"])
    verifier = provider(models["quality_verifier"]) if models.get("quality_verifier") else None
    evaluation = CandidateEvaluator(weak, strong, judge, quality_verifier=verifier,
                                    **config.get("evaluation", {}))
    utility = AdaptiveUtilityGate(**config["utility_gate"]) if config.get("utility_gate") else None
    candidates, report = DatasetBuilder(challenger, evaluation, utility_gate=utility).build(
        load_specs(config["tasks"]), load_sources(config["sources"]), rounds_per_source=config.get("rounds", 3)
    )
    write_jsonl(config["output"], candidates)
    Path(config.get("report", "dataset-report.json")).write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(f"accepted {report.accepted}/{report.total} candidates")


if __name__ == "__main__":
    main()
