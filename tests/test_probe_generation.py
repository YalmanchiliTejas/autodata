from __future__ import annotations

from probe_generation import provider


def test_probe_provider_rejects_unknown_backend():
    try:
        provider({"provider": "unknown"})
    except ValueError as exc:
        assert "unsupported provider" in str(exc)
    else:
        raise AssertionError("unknown provider should fail")
