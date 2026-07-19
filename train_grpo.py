"""Train a LoRA adapter with GRPO on exported Autodata records.

Heavy ML dependencies are imported only when this command is run, keeping the
dataset-construction library lightweight.  Install ``requirements-train.txt``
inside a CUDA environment before launching this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodata.providers import AnthropicModel, OpenAICompatibleModel, TextModel
from autodata.training import _completion_text, exact_match_reward, load_grpo_records


def build_judge(config_path: str) -> TextModel:
    config = json.loads(Path(config_path).read_text())
    kind = config.pop("provider")
    if kind == "anthropic":
        return AnthropicModel(**config)
    if kind == "openai_compatible":
        return OpenAICompatibleModel(**config)
    raise ValueError(f"unsupported judge provider: {kind}")


def rubric_judge_reward_factory(judge: TextModel):
    """Create a GRPO reward callback. The reference answer is never shown."""
    def reward(completions, task, rubric, reward_type, **_: Any):
        rewards: list[float | None] = []
        for completion, visible_task, criteria, kind in zip(completions, task, rubric, reward_type):
            if kind != "rubric_judge":
                rewards.append(None)
                continue
            prompt = (
                "You are a strict independent reward function. Score the solver completion against each "
                "observable rubric criterion using only the solver-visible task and declared environment. "
                "Do not infer a hidden reference answer. Return ONLY JSON {\"score\": number from 0 to 1, "
                "\"reasons\": [short strings]}.\n\n"
                f"TASK: {json.dumps(visible_task, ensure_ascii=False)}\n"
                f"RUBRIC: {json.dumps(criteria, ensure_ascii=False)}\n"
                f"COMPLETION: {_completion_text(completion)}"
            )
            try:
                value = float(json.loads(judge.complete(prompt)).get("score", 0.0))
                rewards.append(max(0.0, min(1.0, value)))
            except (ValueError, TypeError, json.JSONDecodeError):
                # A malformed external judgment must not become a positive reward.
                rewards.append(0.0)
        return rewards
    return reward


def selective_exact_reward(completions, reference, reward_type, **kwargs: Any):
    raw = exact_match_reward(completions, reference, **kwargs)
    return [value if kind == "exact_match" else None for value, kind in zip(raw, reward_type)]


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO LoRA training over accepted Autodata records")
    parser.add_argument("--model", required=True, help="Hugging Face base model ID or local checkpoint")
    parser.add_argument("--train", required=True, help="GRPO training JSONL from export_grpo_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-config", help="Required if any record has reward_type=rubric_judge")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = load_grpo_records(args.train)
    types = {record["reward_type"] for record in records}
    if "external_verifier" in types:
        raise ValueError("external_verifier records require a sandbox reward adapter and cannot use this text-only trainer")
    if "rubric_judge" in types and not args.judge_config:
        raise ValueError("rubric_judge records require --judge-config")
    if args.num_generations < 2:
        raise ValueError("GRPO needs at least two generations per prompt to form a relative advantage")

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt in a CUDA environment before GRPO training") from exc

    reward_funcs = [selective_exact_reward]
    if "rubric_judge" in types:
        reward_funcs.append(rubric_judge_reward_factory(build_judge(args.judge_config)))
    train_dataset = Dataset.from_list(records)
    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(10, args.max_steps // 5),
        report_to="none",
        seed=args.seed,
    )
    peft = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_funcs, args=config,
                          train_dataset=train_dataset, peft_config=peft)
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "autodata-training-manifest.json").write_text(json.dumps({
        "base_model": args.model, "train_record_ids": [record["id"] for record in records],
        "train_group_ids": sorted({record["group_id"] for record in records}),
        "reward_types": sorted(types), "num_generations": args.num_generations,
        "max_steps": args.max_steps, "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
