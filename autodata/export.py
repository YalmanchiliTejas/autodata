"""Portable exports for supervised learning and RL environment datasets."""
import json


def training_messages(record):
    workflow = record["workflow"]
    example = record.get("learning", {}).get("preferred_example", {})
    response = example.get("response")
    if not response:
        raise ValueError(f"{record.get('workflow_id', 'record')}: no preferred response")
    return [
        {"role": "system", "content": "Use only the supplied environment contract and context."},
        {"role": "user", "content": f"{workflow['user_prompt']}\n\nCONTEXT:\n{workflow['context']}"},
        {"role": "assistant", "content": response},
    ]


def export_record(record, target):
    """Return one generic chat or Prime Verifiers-compatible dataset row."""
    messages = training_messages(record)
    if target == "openai_chat":
        return {"messages": messages, "metadata": {"workflow_id": record["workflow_id"]}}
    if target == "prime_verifiers":
        return {
            "prompt": messages[:-1],
            "answer": messages[-1]["content"],
            "rubric": record["workflow"]["rubric"],
            "metadata": {"workflow_id": record["workflow_id"], "environment": record.get("environment", {})},
        }
    raise ValueError(f"unsupported export target: {target}")


def export_jsonl(input_path, output_path, target):
    written = 0
    with open(input_path) as source, open(output_path, "w") as destination:
        for line in source:
            if not line.strip():
                continue
            destination.write(json.dumps(export_record(json.loads(line), target)) + "\n")
            written += 1
    return written
