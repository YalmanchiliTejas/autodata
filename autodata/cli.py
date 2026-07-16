"""CLI: run Agentic Self-Instruct over a folder of grounding documents.

    python -m autodata.cli data/ --out accepted.jsonl [--mock]

Writes accepted examples to --out (JSONL) and every agent turn to the trajectory
log (default trajectories.jsonl)."""
import argparse
import json
import pathlib

from .config import Config
from .accepted import format_accepted, validate_accepted
from .coverage import CoverageGraph
from .loop import run_doc
from .trajectory import TrajectoryLogger
from .trajectory_report import write_trajectory_report
from .environment import EnvironmentSpec
from .export import export_jsonl


def _shingles(text, k=5):
    w = text.split()
    return {hash(" ".join(w[i:i + k])) for i in range(max(1, len(w) - k + 1))}


def dedup(files, threshold):
    """Drop near-duplicate docs (shingle-Jaccard >= threshold vs a kept doc).
    ponytail: O(n^2) exact scan; swap in MinHash/LSH beyond a few thousand docs."""
    kept, sigs = [], []
    for f in files:
        s = _shingles(f.read_text())
        if any(len(s & t) / max(1, len(s | t)) >= threshold for t in sigs):
            continue
        kept.append(f)
        sigs.append(s)
    return kept


def main(argv=None):
    p = argparse.ArgumentParser(description="Autodata / Agentic Self-Instruct")
    p.add_argument("docs", help="file or directory of grounding documents (.txt/.md)")
    p.add_argument("--out", default="accepted.jsonl")
    p.add_argument("--mock", action="store_true", help="offline deterministic LLM")
    p.add_argument("--max-rounds", type=int, help="challenger rounds per document")
    p.add_argument("--extractor-workers", type=int, default=4,
                   help="parallel extraction/coverage chunk calls (default: 4)")
    p.add_argument("--budget", type=int, default=0,
                   help="total challenger rounds across ALL docs; stop when hit (0 = unlimited)")
    p.add_argument("--log", default="trajectories.jsonl")
    p.add_argument("--verbose", action="store_true",
                   help="print every pipeline stage while full prompts/responses go to --log")
    p.add_argument("--coverage-cache", default=".autodata/coverage.json",
                   help="persistent evidence-card coverage graph (default: .autodata/coverage.json)")
    p.add_argument("--build-coverage", action="store_true",
                   help="map and validate source evidence cards, then exit without generation")
    p.add_argument("--coverage-report", help="write the coverage report to this path")
    p.add_argument("--no-coverage", action="store_true",
                   help="disable coverage-guided generation and use one compact extract per document")
    p.add_argument("--trace-report", help="write a readable Markdown trace for this run")
    p.add_argument("--environment", help="JSON file describing the target environment contract")
    p.add_argument("--llm-provider", choices=("anthropic", "openai_compatible"), default="anthropic")
    p.add_argument("--llm-base-url", default="", help="base URL for an OpenAI-compatible LLM provider")
    p.add_argument("--llm-api-key-env", default="", help="environment variable containing that provider's API key")
    p.add_argument("--export-format", choices=("openai_chat", "prime_verifiers"),
                   help="export accepted JSONL to a portable training format")
    p.add_argument("--export-out", help="destination for --export-format")
    args = p.parse_args(argv)

    if bool(args.export_format) != bool(args.export_out):
        p.error("--export-format and --export-out must be used together")
    if args.export_format:
        print(f"exported {export_jsonl(args.docs, args.export_out, args.export_format)} records -> {args.export_out}")
        return
    environment = EnvironmentSpec()
    if args.environment:
        environment = EnvironmentSpec.from_mapping(json.loads(pathlib.Path(args.environment).read_text()))
    cfg = Config(mock=args.mock, log_path=args.log, max_total_rounds=args.budget,
                 extractor_workers=max(1, args.extractor_workers), environment=environment,
                 llm_provider=args.llm_provider, llm_base_url=args.llm_base_url,
                 llm_api_key_env=args.llm_api_key_env)
    if args.max_rounds:
        cfg.max_rounds = args.max_rounds

    root = pathlib.Path(args.docs)
    files = [root] if root.is_file() else sorted(
        f for f in root.rglob("*") if f.suffix in {".txt", ".md"})
    if not files:
        p.error(f"no .txt/.md documents under {root}")

    if cfg.dedup_threshold and len(files) > 1:
        kept = dedup(files, cfg.dedup_threshold)
        if len(kept) < len(files):
            print(f"[dedup] {len(files)} -> {len(kept)} docs "
                  f"({len(files) - len(kept)} near-duplicates skipped)")
        files = kept

    log = TrajectoryLogger(cfg.log_path, verbose=args.verbose)
    if args.build_coverage:
        graph = CoverageGraph(args.coverage_cache)
        for f in files:
            document = f.read_text()
            log.start_doc(f.name)
            source_sha256, _ = graph.get_or_create(cfg, f.name, document, log)
            errors = graph.validate(source_sha256, document)
            if errors:
                raise RuntimeError(f"coverage validation failed for {f.name}: {errors}")
            print(f"[COVERAGE] {f.name} -> {source_sha256[:12]}")
        report = graph.report()
        if args.coverage_report:
            pathlib.Path(args.coverage_report).write_text(report)
            print(f"coverage report -> {args.coverage_report}")
        else:
            print(report, end="")
        log.close()
        if args.trace_report:
            turns = write_trajectory_report(cfg.log_path, args.trace_report, log.run_id)
            print(f"trace report ({turns} events) -> {args.trace_report}")
        return

    accepted = 0
    coverage = None if args.no_coverage else CoverageGraph(args.coverage_cache)
    with open(args.out, "w") as out:
        for f in files:
            if cfg.max_total_rounds and cfg.rounds_run >= cfg.max_total_rounds:
                print(f"[BUDGET] {cfg.rounds_run} rounds used -- stopping")
                break
            document = f.read_text()
            ex = run_doc(cfg, log, f.name, document, coverage=coverage)
            status = "ACCEPT" if ex else "reject "
            print(f"[{status}] {f.name}"
                  + (f"  round {ex['round']} gap {ex['gap']}" if ex else ""))
            if ex:
                record = format_accepted(ex, f.name, document, cfg)
                errors = validate_accepted(record)
                if errors:
                    raise RuntimeError(f"accepted record validation failed: {errors}")
                out.write(json.dumps(record) + "\n")
                accepted += 1
    log.close()
    if coverage and args.coverage_report:
        pathlib.Path(args.coverage_report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.coverage_report).write_text(coverage.report())
        print(f"coverage report -> {args.coverage_report}")
    if args.trace_report:
        turns = write_trajectory_report(cfg.log_path, args.trace_report, log.run_id)
        print(f"trace report ({turns} events) -> {args.trace_report}")
    print(f"\n{accepted}/{len(files)} accepted -> {args.out}  |  {cfg.rounds_run} rounds"
          f"  |  trajectory -> {cfg.log_path}")


if __name__ == "__main__":
    main()
