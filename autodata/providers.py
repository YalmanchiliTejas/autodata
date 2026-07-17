"""Small provider boundary. Bring your own model SDK or HTTP client."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen


class TextModel(Protocol):
    def complete(self, prompt: str, *, system: str = "") -> str: ...


class CallableModel:
    """Adapts a function of ``(prompt, system) -> text`` to the model protocol."""

    def __init__(self, fn):
        self.fn = fn

    def complete(self, prompt: str, *, system: str = "") -> str:
        return self.fn(prompt, system)


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float) -> dict:
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    with urlopen(request, timeout=timeout) as response:  # nosec B310: user config selects the inference endpoint
        return json.loads(response.read())


@dataclass(slots=True)
class AnthropicModel:
    """Minimal Claude Messages API adapter; requires no third-party SDK."""

    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout: float = 120.0
    base_url: str = "https://api.anthropic.com"

    def complete(self, prompt: str, *, system: str = "") -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing API key in environment variable {self.api_key_env}")
        result = _post_json(self.base_url.rstrip("/") + "/v1/messages", {
            "x-api-key": key, "anthropic-version": "2023-06-01",
        }, {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": prompt}]}, self.timeout)
        return "".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")


@dataclass(slots=True)
class OpenAICompatibleModel:
    """Adapter for vLLM, including Modal-hosted OpenAI-compatible endpoints."""

    base_url: str
    model: str
    api_key_env: str | None = None
    extra_headers_env: dict[str, str] | None = None
    max_tokens: int = 2048
    temperature: float = 0.8
    timeout: float = 120.0

    def complete(self, prompt: str, *, system: str = "") -> str:
        headers: dict[str, str] = {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"missing API key in environment variable {self.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        for header, env_name in (self.extra_headers_env or {}).items():
            value = os.environ.get(env_name)
            if not value:
                raise RuntimeError(f"missing endpoint credential in environment variable {env_name}")
            headers[header] = value
        base = self.base_url.rstrip("/")
        endpoint = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        result = _post_json(endpoint, headers, {"model": self.model, "max_tokens": self.max_tokens,
                                                  "temperature": self.temperature,
                                                  "messages": [{"role": "system", "content": system},
                                                               {"role": "user", "content": prompt}]}, self.timeout)
        return result["choices"][0]["message"]["content"]
