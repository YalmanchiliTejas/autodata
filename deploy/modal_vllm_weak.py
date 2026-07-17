"""Deploy the weak solver as an authenticated Modal OpenAI-compatible vLLM API.

Prerequisites:
  pip install modal
  modal setup
  modal deploy deploy/modal_vllm_weak.py

Set MODEL_NAME/GPU for the small model you want to compare against the stronger
external solver. The deployed URL serves /v1/chat/completions.
"""
from __future__ import annotations

import subprocess
import time
from urllib.request import urlopen

import modal


APP_NAME = "autodata-weak-vllm"
MODEL_NAME = "Qwen/Qwen3.5-4B"
GPU = "L4"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "30"})
)
hf_cache = modal.Volume.from_name("autodata-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("autodata-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=15 * MINUTES,
    timeout=24 * 60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES, requires_proxy_auth=True)
def serve():
    """Compatible with Modal 1.x's public web-server decorator."""
    process = subprocess.Popen([
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
    ])
    deadline = time.monotonic() + 14 * MINUTES
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup")
        try:
            with urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2) as response:  # nosec B310: loopback health check
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy before Modal startup timeout")
