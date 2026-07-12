"""Render a JSONL trajectory into a readable per-run Markdown review document."""
import json
from pathlib import Path


def _fence(text):
    return "~~~~text\n" + (text or "(not recorded)") + "\n~~~~\n"


def write_trajectory_report(trajectory_path, report_path, run_id):
    """Write a Markdown report for exactly one run recorded in a JSONL trace."""
    events = []
    with open(trajectory_path) as source:
        for line in source:
            event = json.loads(line)
            if event.get("run_id") == run_id:
                events.append(event)

    lines = ["# Autodata Run Trace", "", f"Run ID: `{run_id}`", ""]
    current_doc = object()
    for event in events:
        doc_id = event.get("doc_id")
        if doc_id != current_doc:
            current_doc = doc_id
            lines.extend([f"## Document: `{doc_id}`", ""])
        if event["kind"] == "turn":
            round_i = event.get("round", "-")
            lines.extend([f"### Round {round_i}: {event['agent']}", ""])
            if event.get("model"):
                lines.extend([f"Model: `{event['model']}`", ""])
            if "sample" in event:
                lines.extend([f"Sample: `{event['sample']}`", ""])
            if "attempt" in event:
                lines.extend([f"Evaluator attempt: `{event['attempt']}`", ""])
            lines.extend(["#### System Instructions", "", _fence(event.get("system_prompt")),
                          "#### Context Passed To Model", "", _fence(event.get("prompt")),
                          "#### Model Response", "", _fence(event.get("response"))])
        else:
            fields = {key: value for key, value in event.items() if key not in {
                "schema_version", "ts", "run_id", "doc_id", "level", "kind", "agent"}}
            lines.extend([f"### {event['level']}: {event['kind']}", "",
                          _fence(json.dumps(fields, indent=2, sort_keys=True))])

    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return len(events)
