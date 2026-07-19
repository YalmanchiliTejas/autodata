"""Download a bounded, same-site HTML documentation corpus.

The crawler is intentionally narrow: it follows only HTML pages under the
provided documentation root, strips query/fragment variants, and does not fetch
external links, source repositories, binaries, or JavaScript assets.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def canonical(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def allowed(url: str, root: str) -> bool:
    candidate, base = urlparse(url), urlparse(root)
    if candidate.scheme not in {"http", "https"} or candidate.netloc != base.netloc:
        return False
    excluded = ("/_modules/", "/genindex.html", "/py-modindex.html", "/search.html")
    return (candidate.path.startswith(base.path)
            and not any(part in candidate.path for part in excluded)
            and (candidate.path.endswith("/") or candidate.path.endswith(".html")))


def local_path(url: str, root: str, output: Path) -> Path:
    relative = urlparse(url).path.removeprefix(urlparse(root).path).lstrip("/") or "index.html"
    if relative.endswith("/"):
        relative += "index.html"
    target = (output / relative).resolve()
    if output.resolve() not in target.parents and target != output.resolve():
        raise ValueError("unsafe documentation URL path")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download same-site HTML documentation pages")
    parser.add_argument("--url", required=True, help="Documentation root URL ending in /")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args()
    root = canonical(args.url)
    if not root.endswith("/"):
        raise ValueError("--url must end with /")
    if args.max_pages < 1:
        raise ValueError("--max-pages must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pending, visited, manifest = deque([root]), set(), []
    while pending and len(visited) < args.max_pages:
        url = pending.popleft()
        if url in visited:
            continue
        request = Request(url, headers={"User-Agent": "Autodata documentation corpus builder/0.1"})
        try:
            with urlopen(request, timeout=30) as response:  # nosec B310: URL is constrained to initial same-site root
                content_type = response.headers.get_content_type()
                body = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"skipping {url}: {exc}")
            continue
        if content_type != "text/html":
            continue
        visited.add(url)
        target = local_path(url, root, output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        manifest.append({"url": url, "path": str(target)})
        parser_html = Links()
        parser_html.feed(body)
        for href in parser_html.links:
            candidate = canonical(urljoin(url, href))
            if allowed(candidate, root) and candidate not in visited:
                pending.append(candidate)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"downloaded {len(manifest)} documentation pages to {output}")
    if pending:
        print(f"stopped at --max-pages={args.max_pages}; increase it only after reviewing this corpus")


if __name__ == "__main__":
    main()
