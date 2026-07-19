"""Convert a permitted local documentation/code corpus into Autodata JSONL.

This prepares *grounding sources*, not training examples.  The challenger turns
them into tasks only after ``main.py`` runs.  It intentionally skips hidden and
dependency/build directories and never reads .env files or arbitrary binaries.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


DEFAULT_SUFFIXES = {".md", ".rst", ".txt", ".html", ".htm", ".py", ".js", ".ts", ".java", ".go", ".rs", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}


def chunks(content: str, max_chars: int) -> list[str]:
    """Split long files at a newline when practical, without dropping text."""
    result = []
    while content:
        if len(content) <= max_chars:
            result.append(content)
            break
        boundary = content.rfind("\n", 0, max_chars)
        boundary = boundary if boundary >= max_chars // 2 else max_chars
        result.append(content[:boundary])
        content = content[boundary:].lstrip("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local, permitted corpus for Autodata")
    parser.add_argument("--input", required=True, help="Directory or file containing source material")
    parser.add_argument("--output", required=True, help="Output sources JSONL")
    parser.add_argument("--license", required=True, help="License/provenance label recorded with every source")
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Optional cap for a small, cost-controlled pilot; zero means no cap")
    parser.add_argument("--suffixes", default=",".join(sorted(DEFAULT_SUFFIXES)), help="Comma-separated file suffix allowlist")
    args = parser.parse_args()
    if args.max_chars < 500:
        raise ValueError("--max-chars must be at least 500")
    root = Path(args.input).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    suffixes = {item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
                for item in args.suffixes.split(",") if item.strip()}
    paths = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    records = []
    for path in paths:
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root.parent).parts):
            continue
        if path.suffix.lower() not in suffixes or path.stat().st_size > 2_000_000:
            continue
        content = path.read_text(errors="replace")
        if path.suffix.lower() in {".html", ".htm"}:
            # Documentation HTML is an input format, not task content.  Keep
            # readable text and discard markup/scripts before the challenger
            # sees it.
            content = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = html.unescape(content)
            content = re.sub(r"\s+", " ", content)
        content = content.strip()
        if not content:
            continue
        try:
            relative = str(path.relative_to(root)) if root.is_dir() else path.name
        except ValueError:
            relative = path.name
        for index, piece in enumerate(chunks(content, args.max_chars), start=1):
            records.append({
                "id": f"{relative}#chunk-{index}", "content": piece,
                "metadata": {"path": str(path), "relative_path": relative, "suffix": path.suffix.lower(),
                             "license": args.license, "chunk": index},
            })
            if args.max_records and len(records) >= args.max_records:
                break
        if args.max_records and len(records) >= args.max_records:
            break
    if not records:
        raise ValueError("no permitted text/code files found; adjust --suffixes or choose another corpus")
    Path(args.output).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    print(f"wrote {len(records)} grounded source chunks to {args.output}")


if __name__ == "__main__":
    main()
