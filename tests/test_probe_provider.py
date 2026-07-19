from __future__ import annotations

import ast
from pathlib import Path


def test_provider_probe_syntax():
    ast.parse(Path("probe_provider.py").read_text())
