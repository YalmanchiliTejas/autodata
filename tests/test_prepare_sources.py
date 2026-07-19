from __future__ import annotations

from prepare_sources import chunks


def test_chunks_preserve_content():
    source = "a" * 600 + "\n" + "b" * 600
    parts = chunks(source, 700)
    assert "".join(part + ("\n" if index == 0 else "") for index, part in enumerate(parts)) == source
    assert all(len(part) <= 700 for part in parts)
