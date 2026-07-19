from __future__ import annotations

import pytest

from autodata.providers import AnthropicModel, _post_json


def test_anthropic_omits_temperature_by_default(monkeypatch):
    captured = {}

    def post(_url, _headers, body, _timeout, **_options):
        captured.update(body)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "key")
    monkeypatch.setattr("autodata.providers._post_json", post)
    assert AnthropicModel("new-model", api_key_env="TEST_ANTHROPIC_KEY").complete("hello") == "ok"
    assert "temperature" not in captured


def test_anthropic_sends_explicitly_configured_temperature(monkeypatch):
    captured = {}

    def post(_url, _headers, body, _timeout, **_options):
        captured.update(body)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "key")
    monkeypatch.setattr("autodata.providers._post_json", post)
    AnthropicModel("older-model", api_key_env="TEST_ANTHROPIC_KEY", temperature=0.2).complete("hello")
    assert captured["temperature"] == 0.2


class _Response:
    def __init__(self, payload=b'{"ok": true}'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_post_json_retries_a_timeout_then_succeeds(monkeypatch):
    calls = 0

    def open_request(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("read timed out")
        return _Response()

    monkeypatch.setattr("autodata.providers.urlopen", open_request)
    monkeypatch.setattr("autodata.providers.sleep", lambda _seconds: None)
    assert _post_json("https://example.test/v1", {}, {}, 1, max_retries=1) == {"ok": True}
    assert calls == 2


def test_post_json_converts_exhausted_timeouts_to_runtime_error(monkeypatch):
    monkeypatch.setattr("autodata.providers.urlopen",
                        lambda _request, timeout: (_ for _ in ()).throw(TimeoutError("read timed out")))
    monkeypatch.setattr("autodata.providers.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="failed after 2 attempts: TimeoutError"):
        _post_json("https://example.test/v1", {}, {}, 1, max_retries=1)
