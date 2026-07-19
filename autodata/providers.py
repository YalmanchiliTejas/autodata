"""Small provider boundary. Bring your own model SDK or HTTP client."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import sleep
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


logger = logging.getLogger("autodata")
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


class TextModel(Protocol):
    def complete(self, prompt: str, *, system: str = "") -> str: ...


class CallableModel:
    """Adapts a function of ``(prompt, system) -> text`` to the model protocol."""

    def __init__(self, fn):
        self.fn = fn

    def complete(self, prompt: str, *, system: str = "") -> str:
        return self.fn(prompt, system)


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float, *,
               max_retries: int = 2, retry_backoff: float = 2.0) -> dict:
    """POST model input with bounded retries for transient transport failures."""
    encoded = json.dumps(body).encode()
    endpoint = urlsplit(url)
    endpoint_label = f"{endpoint.netloc}{endpoint.path}"
    for attempt in range(max_retries + 1):
        request = Request(url, data=encoded, headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310: user config selects the inference endpoint
                return json.loads(response.read())
        except HTTPError as exc:
            # Provider messages identify bad model IDs/permissions without ever
            # including request headers or API credentials in a traceback.
            detail = exc.read().decode("utf-8", errors="replace")[:1_000]
            if exc.code not in TRANSIENT_HTTP_STATUS or attempt >= max_retries:
                raise RuntimeError(f"inference request failed with HTTP {exc.code}: {detail}") from exc
            error = f"HTTP {exc.code}"
        except (TimeoutError, URLError, ConnectionError, OSError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"inference request failed after {attempt + 1} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            error = f"{type(exc).__name__}: {exc}"
        wait = retry_backoff * (2 ** attempt)
        logger.warning("stage=inference_request_retry endpoint=%s retry=%s/%s wait_seconds=%.1f error=%s",
                       endpoint_label, attempt + 1, max_retries, wait, error)
        sleep(wait)
    raise RuntimeError("inference request failed without a response")


@dataclass(slots=True)
class AnthropicModel:
    """Minimal Claude Messages API adapter; requires no third-party SDK."""

    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 4096
    # Newer Anthropic models reject the parameter entirely.  Omit it by
    # default so the API selects the model's supported default; callers can
    # still configure a value for models/endpoints that support one.
    temperature: float | None = None
    timeout: float = 120.0
    max_retries: int = 2
    retry_backoff: float = 2.0
    base_url: str = "https://api.anthropic.com"

    def complete(self, prompt: str, *, system: str = "") -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing API key in environment variable {self.api_key_env}")
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        result = _post_json(self.base_url.rstrip("/") + "/v1/messages", {
            "x-api-key": key, "anthropic-version": "2023-06-01",
        }, body, self.timeout, max_retries=self.max_retries, retry_backoff=self.retry_backoff)
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
    timeout: float = 360.0
    max_retries: int = 2
    retry_backoff: float = 2.0

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
                                                               {"role": "user", "content": prompt}]}, self.timeout,
                            max_retries=self.max_retries, retry_backoff=self.retry_backoff)
        return result["choices"][0]["message"]["content"]
