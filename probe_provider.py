"""Measure a minimal authenticated model request before debugging prompts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from probe_generation import provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal provider latency probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", default="challenger", choices=["challenger", "judge", "quality_verifier", "strong", "weak"])
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    config = json.loads(Path(args.config).read_text())
    model_config = dict(config["models"][args.role])
    model_config["max_tokens"] = 16
    model_config["timeout"] = args.timeout
    started = time.monotonic()
    response = provider(model_config).complete("Return exactly: OK")
    elapsed = round(time.monotonic() - started, 3)
    print(json.dumps({"role": args.role, "model": model_config.get("model"),
                      "seconds": elapsed, "response": response}, indent=2))


if __name__ == "__main__":
    main()
