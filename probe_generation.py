"""Run only the challenger + local structural checks for one source chunk.

Use this before the expensive weak/strong/judge rollout loop.  It saves the raw
challenger response locally so malformed JSON or an incorrect schema can be
debugged without spending on solver calls.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from autodata.io import load_sources, load_specs
from autodata.providers import AnthropicModel, OpenAICompatibleModel
from autodata.tasks import ADAPTERS


def provider(config: dict):
    options = {key: value for key, value in config.items() if key != "provider"}
    if config["provider"] == "anthropic":
        return AnthropicModel(**options)
    if config["provider"] == "openai_compatible":
        return OpenAICompatibleModel(**options)
    raise ValueError(f"unsupported provider {config['provider']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe challenger generation without solver rollouts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-index", type=int, default=0, help="Zero-based source chunk index")
    parser.add_argument("--output", default="outputs/generation-probe.json")
    parser.add_argument("--max-tokens", type=int, default=1600, help="Bound challenger output for a latency probe")
    parser.add_argument("--timeout", type=float, default=90.0, help="Fail instead of waiting indefinitely")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    config = json.loads(Path(args.config).read_text())
    specs, sources = load_specs(config["tasks"]), load_sources(config["sources"])
    if len(specs) != 1:
        raise ValueError("generation probe expects exactly one task specification")
    if not 0 <= args.source_index < len(sources):
        raise IndexError(f"source index must be in [0, {len(sources) - 1}]")
    spec, source = specs[0], sources[args.source_index]
    adapter = ADAPTERS[spec.kind]
    prompt = adapter.generation_prompt(spec, source, [])
    challenger_config = dict(config["models"]["challenger"])
    challenger_config["max_tokens"] = args.max_tokens
    challenger_config["timeout"] = args.timeout
    started = time.monotonic()
    raw = provider(challenger_config).complete(prompt)
    elapsed = round(time.monotonic() - started, 3)
    result = {
        "task": spec.name,
        "source_id": source.id,
        "prompt_characters": len(prompt),
        "response_characters": len(raw),
        "generation_seconds": elapsed,
        "raw_response": raw,
    }
    try:
        candidate = adapter.parse(raw, spec, source, "probe")
        result["structural_issues"] = adapter.validate(candidate)
        result["parsed_payload"] = candidate.payload
        result["candidate_capabilities"] = candidate.capabilities
    except ValueError as exc:
        result["parse_error"] = str(exc)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    status = "parse_error" if "parse_error" in result else ("structure_failed" if result["structural_issues"] else "structure_passed")
    print(f"{status}; generation={elapsed}s; prompt={len(prompt)} chars; response={len(raw)} chars; output={args.output}")


if __name__ == "__main__":
    main()
