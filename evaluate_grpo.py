"""Score a frozen base model or a GRPO LoRA adapter on a held-out JSONL set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from autodata.training import exact_match_reward, load_grpo_records
from train_grpo import build_judge, rubric_judge_reward_factory


def score_record(record, completions, judge=None):
    kind = record["reward_type"]
    if kind == "exact_match":
        return exact_match_reward(completions, [record["reference"]] * len(completions))
    if kind == "rubric_judge" and judge:
        callback = rubric_judge_reward_factory(judge)
        values = callback(completions, [record["task"]] * len(completions), [record["rubric"]] * len(completions),
                          [kind] * len(completions))
        return [float(value or 0.0) for value in values]
    raise ValueError(f"record {record['id']} needs a supported reward backend (type={kind})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out evaluation for a GRPO pilot")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", help="LoRA adapter directory; omit for frozen baseline")
    parser.add_argument("--judge-config", help="Required for rubric_judge benchmark records")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    records = load_grpo_records(args.benchmark)
    if any(record["reward_type"] == "external_verifier" for record in records):
        raise ValueError("external_verifier evaluation needs a sandbox adapter; it is intentionally not text-only")
    if any(record["reward_type"] == "rubric_judge" for record in records) and not args.judge_config:
        raise ValueError("rubric_judge benchmark records require --judge-config")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt in a CUDA environment before evaluation") from exc
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    judge = build_judge(args.judge_config) if args.judge_config else None
    device = next(model.parameters()).device
    results = []
    for record in records:
        inputs = tokenizer(record["prompt"], return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                       max_new_tokens=args.max_new_tokens, num_return_sequences=args.rollouts,
                                       pad_token_id=tokenizer.eos_token_id)
        prompt_tokens = inputs["input_ids"].shape[1]
        completions = tokenizer.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)
        scores = score_record(record, completions, judge)
        results.append({"id": record["id"], "group_id": record["group_id"], "scores": scores,
                        "mean_reward": mean(scores), "best_reward": max(scores),
                        "completion_count": len(completions)})
    summary = {
        "model": args.model, "adapter": args.adapter, "benchmark_size": len(results), "rollouts": args.rollouts,
        "mean_reward": mean(item["mean_reward"] for item in results),
        "mean_best_of_rollouts_reward": mean(item["best_reward"] for item in results),
        "solved_rate": mean(float(item["best_reward"] >= 0.999) for item in results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
