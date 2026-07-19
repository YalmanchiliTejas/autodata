"""Run Autodata with Claude roles and a Modal/vLLM weak solver.

Credentials are read from environment variables named in the JSON config; this
file never reads or writes secrets.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from autodata.evaluation import CandidateEvaluator
from autodata.io import load_sources, load_specs, write_jsonl, write_records_jsonl
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


def configure_logging(solver_prompt_path: str | Path) -> Path:
    """Keep operational events on stdout and full solver prompts in a dedicated file."""
    logging.basicConfig(level=logging.INFO, format="[autodata] %(message)s")
    path = Path(solver_prompt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_logger = logging.getLogger("autodata.solver_prompts")
    for handler in list(prompt_logger.handlers):
        handler.close()
        prompt_logger.removeHandler(handler)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    prompt_logger.addHandler(handler)
    prompt_logger.setLevel(logging.INFO)
    prompt_logger.propagate = False
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Autodata dataset with configured model roles.")
    parser.add_argument("--config", required=True, help="JSON run configuration")
    parser.add_argument("--audit-output", help="Optional JSONL trace; defaults beside the dataset output")
    parser.add_argument("--solver-prompt-log", help="Formatted weak/strong solver prompt transcript")
    parser.add_argument("--source-limit", type=int, help="Run only the first N source chunks; useful for debugging")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    config = json.loads(Path(args.config).read_text())
    default_prompt_log = str(Path(config["output"]).with_suffix(".solver-prompts.log"))
    solver_prompt_path = configure_logging(
        args.solver_prompt_log or config.get("solver_prompt_log") or default_prompt_log)
    if args.source_limit is not None and args.source_limit < 1:
        parser.error("--source-limit must be at least 1")
    models = config["models"]
    weak = provider(models["weak"])
    strong = provider(models["strong"])
    challenger = provider(models["challenger"])
    judge = provider(models["judge"])
    verifier = provider(models["quality_verifier"]) if models.get("quality_verifier") else None
    evaluation = CandidateEvaluator(weak, strong, judge, quality_verifier=verifier,
                                    **config.get("evaluation", {}))
    utility = AdaptiveUtilityGate(**config["utility_gate"]) if config.get("utility_gate") else None
    builder = DatasetBuilder(challenger, evaluation, utility_gate=utility)
    sources = load_sources(config["sources"])
    if args.source_limit is not None:
        sources = sources[:args.source_limit]
    candidates, report = builder.build(load_specs(config["tasks"]), sources,
                                       rounds_per_source=config.get("rounds", 3),
                                       stop_after_accept_per_source=config.get("stop_after_accept_per_source", False))
    write_jsonl(config["output"], candidates)
    audit_path = args.audit_output or config.get("audit_output") or str(Path(config["output"]).with_suffix(".audit.jsonl"))
    write_records_jsonl(audit_path, builder.audit_events)
    Path(config.get("report", "dataset-report.json")).write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(f"attempted {report.attempted}; accepted {report.accepted}; audit: {audit_path}; "
          f"solver prompts: {solver_prompt_path}")


if __name__ == "__main__":
    main()
