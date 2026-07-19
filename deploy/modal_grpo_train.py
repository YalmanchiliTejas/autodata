"""Run the GRPO pilot on Modal without placing model weights on the laptop.

Upload ``train.jsonl`` and ``heldout.jsonl`` to ``autodata-grpo-data`` first.
For an exact-match-only pilot:

    modal run deploy/modal_grpo_train.py::train_exact \
      --model Qwen/Qwen3.5-4B --train-file /data/train.jsonl \
      --output-name qwen-grpo-pilot

Rubric-judged training additionally needs an ``ANTHROPIC_API_KEY`` Modal secret
named ``autodata-anthropic`` and a judge JSON config in the data volume.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("autodata-grpo-train")
data = modal.Volume.from_name("autodata-grpo-data", create_if_missing=True)
outputs = modal.Volume.from_name("autodata-grpo-output", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(ROOT / "requirements-train.txt", "/tmp/requirements-train.txt", copy=True)
    .uv_pip_install_from_requirements("/tmp/requirements-train.txt")
    .add_local_python_source("autodata")
    .add_local_file(ROOT / "train_grpo.py", "/root/train_grpo.py")
    .add_local_file(ROOT / "evaluate_grpo.py", "/root/evaluate_grpo.py")
)


def _command(model: str, train_file: str, output_name: str, *, judge_config: str | None = None,
             max_steps: int = 100, num_generations: int = 4) -> list[str]:
    command = ["python", "/root/train_grpo.py", "--model", model, "--train", train_file,
               "--output-dir", f"/outputs/{output_name}", "--max-steps", str(max_steps),
               "--num-generations", str(num_generations)]
    if judge_config:
        command.extend(["--judge-config", judge_config])
    return command


@app.function(image=image, gpu="A100-80GB", timeout=12 * 60 * 60,
              volumes={"/data": data, "/outputs": outputs})
def train_exact(model: str, train_file: str = "/data/train.jsonl", output_name: str = "qwen-grpo-pilot",
                max_steps: int = 100, num_generations: int = 4) -> str:
    """Train only deterministic exact-match records; no remote API secret needed."""
    subprocess.run(_command(model, train_file, output_name, max_steps=max_steps,
                            num_generations=num_generations), check=True)
    outputs.commit()
    return f"/outputs/{output_name}"


@app.function(image=image, gpu="A100-80GB", timeout=12 * 60 * 60,
              volumes={"/data": data, "/outputs": outputs},
              secrets=[modal.Secret.from_name("autodata-anthropic", required_keys=["ANTHROPIC_API_KEY"])])
def train_with_judge(model: str, train_file: str = "/data/train.jsonl", output_name: str = "qwen-grpo-pilot",
                     judge_config: str = "/data/judge.json", max_steps: int = 100,
                     num_generations: int = 4) -> str:
    """Train mixed generic records using an independent rubric judge."""
    subprocess.run(_command(model, train_file, output_name, judge_config=judge_config,
                            max_steps=max_steps, num_generations=num_generations), check=True)
    outputs.commit()
    return f"/outputs/{output_name}"


@app.function(image=image, gpu="A100-80GB", timeout=6 * 60 * 60,
              volumes={"/data": data, "/outputs": outputs})
def evaluate(model: str, benchmark_file: str = "/data/heldout.jsonl", output_name: str = "base.json",
             adapter_name: str | None = None, rollouts: int = 4) -> str:
    """Evaluate the base model or an adapter stored in the output volume."""
    command = ["python", "/root/evaluate_grpo.py", "--model", model, "--benchmark", benchmark_file,
               "--output", f"/outputs/{output_name}", "--rollouts", str(rollouts)]
    if adapter_name:
        command.extend(["--adapter", f"/outputs/{adapter_name}"])
    subprocess.run(command, check=True)
    outputs.commit()
    return f"/outputs/{output_name}"
