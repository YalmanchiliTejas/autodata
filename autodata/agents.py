"""Agentic Self-Instruct subagents (Autodata, Appendix C.1), domain-generic.

Prompts follow the paper's Figures 8 (challenger), 9 (quality verifier), and the
main-agent workflow (Fig. 7), but are written to work across any domain rather than
CS papers specifically. The main agent's control flow lives in loop.py; here each
subagent is one LLM call. Solver scoring (Fig. 7's evaluate_rubric.py) is the
`solve` + `judge` pair: solvers answer, the Kimi-style judge scores per rubric."""
import json
from concurrent.futures import ThreadPoolExecutor

from .llm import complete

# --- Source extraction (cost fix: read the full document ONCE, cheaply) -----
EXTRACTOR_SYS = """You produce a compact extract of a source document so downstream
agents never re-read the full document.

Distill it into a self-contained extract that preserves everything needed to pose a
hard reasoning task: key facts, entities, numbers, definitions, relationships,
constraints, causal links, and any surprising or non-obvious details. Drop
boilerplate, repetition, and filler. Keep it faithful — do not invent. Output plain
text only, no preamble. This is the "compact extract" the pipeline runs on."""

COVERAGE_EXTRACTOR_SYS = """Build a compact coverage map of a source document for
synthetic-workflow generation. Preserve factual content and do not invent details.
Output ONLY JSON with this shape:
{"summary":"...","cards":[{"title":"...","tags":["..."],"facts":["..."],"evidence_quotes":["..."]}]}

Create 6-12 non-overlapping cards when the source permits. Each card must capture a
distinct concept, procedure, constraint set, tradeoff, failure mode, or factual
dependency. Facts must be specific enough to ground a question and rubric. Use short
tags for reasoning capabilities or subject areas so a scheduler can prioritize
concepts that have not yet been used. Each evidence quote MUST be an exact,
contiguous quote from the source, no more than 300 characters, supporting the card's
facts. Do not invent offsets; the caller resolves them."""

# --- Fig. 8: Challenger (domain-generic) -----------------------------------
CHALLENGER_SYS = """You generate an EXECUTABLE AGENT TASK with a grading rubric from a source document.

The output is training data for an agent that must perform a workflow, not answer a
question. Make the objective action-oriented: inspect, diagnose, configure, plan,
transform, verify, or produce an artifact. Do not ask for an explanation, summary,
or essay as the primary task.

Given a source document, produce a single JSON object with keys:
 "type": short phrase for the task type (e.g. "multi-step workflow",
   "constraint-based decision", "failure-mode diagnosis", "plan synthesis",
   "counterfactual outcome").
 "skill_tags": 2-3 reasoning-skill tags (e.g. causal_reasoning, tradeoff_analysis,
   planning, quantitative_estimation).
 "context": agent-visible execution environment and available source-grounded facts;
   do not leak the reference workflow.
 "user_prompt": a natural request from a realistic user of the agent system. State
   the desired outcome and any user-known constraints, but do NOT name hidden
   evaluation criteria, fixtures, ordered actions, target interfaces, or the
   reference workflow. This is the only task request the evaluated agent receives.
 "task": an object with EXACTLY these keys:
   - "objective": one imperative, concrete goal.
   - "inputs": array of {"name": string, "value": string} values the agent receives.
   - "constraints": array of concrete limits, risks, or policies the agent must honor.
   - "required_actions": array of 2-6 ordered action requirements.
   - "deliverables": array of {"name": string, "format": string, "description": string}.
 "reference_workflow": an array of 2-8 objects, each with EXACTLY "step" (integer),
   "action" (string), and "expected_result" (string). It is the ideal execution
   sequence and expected artifacts, grounded in the source document.
 "rubric": a FLAT JSON array of EXACTLY 10-15 criteria. Each item has exactly three
   keys: "criterion" (string), "weight" (integer), "category" ("positive" or
   "negative"). Use 7-10 positive criteria (weight +1..+10) testing specific
   insights/steps a correct solution must contain, and 3-5 negative criteria
   (weight -1..-10) catching specific reasoning errors. Each positive criterion must
   require reasoning beyond the context; each negative must catch a specific
   reasoning error, not a vague style complaint.

Before writing criteria, scratchpad-analyse the critical execution steps, artifacts,
common agent failures, and what distinguishes a reliable workflow from shallow prose.

REFINEMENT: when given previous tasks that failed (grouped TOO EASY = weak too
high, or FAILED ON STRONG = gap too small / strong too low), generate an ENTIRELY
NEW executable task from a different operational angle. Not a rephrasing.

Do not assume any particular subject matter; adapt to whatever the source is about.
Output ONLY the JSON object."""

# --- Fig. 9: Quality Verifier (domain-generic) -----------------------------
VERIFIER_SYS = """You verify whether an agent-task package tests genuine execution reasoning.
You receive a faithful source extract, context, task specification, reference workflow, rubric, and task type.

Check 1 - Workflow leakage: can an agent reproduce the full workflow just by copying
  the context and task fields? If yes -> FAIL. The context may describe the setting,
  but must not pre-state the action sequence or deliverables.
Check 2 - Task quality: is this an executable objective with inputs, constraints,
  actions, and deliverables? Reject recall questions, essay requests, vague advice,
  or a collection of unrelated tasks.
Check 3 - Rubric quality (STRICT, reject if ANY fail): positive criteria (weight>0)
  must be >= 4; total criteria in [10,20], reject if < 10; each positive must
  require reasoning beyond the context (not paraphrasing); each negative must catch
  a specific reasoning ERROR (not vague style). Report exact counts.
Check 4 - Task-type consistency: does the type label match the actual workflow?
Check 5 - Source fidelity: every material factual claim in the reference workflow and
  rubric must be supported by the source extract. Reject invented API behavior,
  numbers, constraints, or causal claims. Do not require exact wording.

Judge only against these checks; do not penalise the package for its subject matter.
Output ONLY JSON: {"check1":"NO_LEAKAGE|LEAKS_WORKFLOW","check2":"GOOD|TOO_EASY|NON_EXECUTABLE",
"check3":"PASS|FAIL","check3_issues":[...],"check4":"CONSISTENT|INCONSISTENT",
"check5":"GROUNDED|UNSUPPORTED","overall":"PASS|FAIL","feedback":"specific issues to fix"}"""

SOLVER_SYS = """You are the evaluated agent in an agent system. Carry out the user's
request using the available context and tools. Return the final response that the user
should receive: state what you did, the outcome, and relevant evidence or limitations.
Do not return an execution plan, hidden evaluation criteria, or a generic essay."""


# Fig. 7: scoring is done by Kimi grading each answer against the rubric.
JUDGE_SYS = """You are the workflow-execution grader. Score each agent execution
result and final response against the rubric.
Grade each positive criterion as met (award its weight) or not (0); each negative
criterion penalises by adding its negative weight when the answer exhibits that error.
Normalise each answer's total to [0,1] as (awarded / sum-of-positive-weights),
clamped to [0,1]. Return ONLY JSON: {"scores":[s1,...],"notes":"..."} with one
score per execution result, in order. Grade purely on the rubric, not against any
reference workflow. Keep notes under 150 words."""


def _parse_json(text):
    # Models wrap JSON in scratchpad prose that itself contains small JSON snippets.
    # Return the LARGEST decodable object (the real payload dwarfs any fragment);
    # only fall back to arrays if no object decodes at all.
    dec = json.JSONDecoder()
    for ch in "{[":
        best = None  # (span, value)
        i = text.find(ch)
        while i != -1:
            try:
                val, end = dec.raw_decode(text, i)
                if best is None or end - i > best[0]:
                    best = (end - i, val)
            except json.JSONDecodeError:
                pass
            i = text.find(ch, i + 1)
        if best:
            return best[1]
    raise ValueError(f"no JSON found in: {text[:200]}")


def validate_example(example):
    """Return schema errors that can be checked without paying for an LLM judge."""
    if not isinstance(example, dict):
        return ["candidate is not a JSON object"]
    required = {"type", "context", "user_prompt", "task", "reference_workflow", "rubric"}
    missing = required - example.keys()
    if missing:
        return [f"missing keys: {sorted(missing)}"]
    errors = []
    for key in {"type", "context", "user_prompt"}:
        if not isinstance(example[key], str) or not example[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if isinstance(example.get("user_prompt"), str) and len(example["user_prompt"].split()) < 12:
        errors.append("user_prompt must be a realistic request of at least 12 words")
    task = example["task"]
    task_keys = {"objective", "inputs", "constraints", "required_actions", "deliverables"}
    if not isinstance(task, dict) or set(task) != task_keys:
        errors.append("task must have objective, inputs, constraints, required_actions, deliverables")
    else:
        if not isinstance(task["objective"], str) or not task["objective"].strip():
            errors.append("task.objective must be non-empty")
        if not isinstance(task["inputs"], list) or any(not isinstance(item, dict) or set(item) != {"name", "value"} for item in task["inputs"]):
            errors.append("task.inputs must contain name/value objects")
        if not isinstance(task["constraints"], list) or not all(isinstance(item, str) and item.strip() for item in task["constraints"]):
            errors.append("task.constraints must be a string array")
        if not isinstance(task["required_actions"], list) or not 2 <= len(task["required_actions"]) <= 6 or not all(isinstance(item, str) and item.strip() for item in task["required_actions"]):
            errors.append("task.required_actions must contain 2-6 actions")
        deliverable_keys = {"name", "format", "description"}
        if not isinstance(task["deliverables"], list) or not task["deliverables"] or any(not isinstance(item, dict) or set(item) != deliverable_keys for item in task["deliverables"]):
            errors.append("task.deliverables must contain name/format/description objects")
    workflow = example["reference_workflow"]
    step_keys = {"step", "action", "expected_result"}
    if not isinstance(workflow, list) or not 2 <= len(workflow) <= 8 or any(not isinstance(item, dict) or set(item) != step_keys or not isinstance(item["step"], int) for item in workflow):
        errors.append("reference_workflow must contain 2-8 numbered action/result steps")
    rubric = example["rubric"]
    if not isinstance(rubric, list) or not 10 <= len(rubric) <= 15:
        return errors + ["rubric must contain 10-15 criteria"]
    positives = negatives = 0
    for i, item in enumerate(rubric):
        if not isinstance(item, dict) or set(item) != {"criterion", "weight", "category"}:
            errors.append(f"rubric[{i}] must have exactly criterion, weight, category")
            continue
        weight, category = item["weight"], item["category"]
        if not isinstance(item["criterion"], str) or not item["criterion"].strip():
            errors.append(f"rubric[{i}].criterion must be non-empty")
        if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= abs(weight) <= 10:
            errors.append(f"rubric[{i}].weight must be an integer in [-10,-1] or [1,10]")
        if category not in {"positive", "negative"}:
            errors.append(f"rubric[{i}].category is invalid")
        elif (category == "positive" and weight <= 0) or (category == "negative" and weight >= 0):
            errors.append(f"rubric[{i}] category and weight sign disagree")
        elif category == "positive":
            positives += 1
        else:
            negatives += 1
    if not 7 <= positives <= 10:
        errors.append("rubric must contain 7-10 positive criteria")
    if not 3 <= negatives <= 5:
        errors.append("rubric must contain 3-5 negative criteria")
    return errors


# Input documents can contain code and tables that tokenize much more densely than
# prose. Keep coverage chunks conservative for the 200k-token API context limit.
CHUNK_CHARS = 350_000
COVERAGE_CHUNK_CHARS = 100_000


def _extract_parts(cfg, system, parts, max_tokens):
    """Run independent extraction chunks concurrently while preserving input order."""
    workers = max(1, min(int(getattr(cfg, "extractor_workers", 1)), len(parts)))
    if workers == 1:
        return [complete(cfg, cfg.extractor, system, part, max_tokens=max_tokens) for part in parts]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(
            lambda part: complete(cfg, cfg.extractor, system, part, max_tokens=max_tokens), parts
        ))


def extract(cfg, document):
    """One cheap pass over the full document -> compact extract everything else uses.
    Docs past CHUNK_CHARS are split so they don't overflow the extractor's context.
    ponytail: fixed char split + concat; fine for docs a handful of chunks long."""
    if len(document) <= CHUNK_CHARS:
        return complete(cfg, cfg.extractor, EXTRACTOR_SYS, document, max_tokens=1500)
    parts = [document[i:i + CHUNK_CHARS] for i in range(0, len(document), CHUNK_CHARS)]
    return "\n\n".join(_extract_parts(cfg, EXTRACTOR_SYS, parts, max_tokens=1500))


def extract_coverage(cfg, document):
    """Map a source into reusable cards, chunking before the API context limit."""
    parts = [(offset, document[offset:offset + COVERAGE_CHUNK_CHARS])
             for offset in range(0, len(document), COVERAGE_CHUNK_CHARS)]
    summaries, cards, raw_parts = [], [], []
    raw_by_part = _extract_parts(
        cfg, COVERAGE_EXTRACTOR_SYS, [part for _, part in parts], cfg.coverage_mapper_max_tokens
    )
    for (offset, part), raw in zip(parts, raw_by_part):
        raw_parts.append(raw)
        try:
            data = _parse_json(raw)
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
            data = {"summary": part[:4000], "cards": [{"title": "source extract",
                    "tags": ["source_grounding"], "facts": [part[:4000]],
                    "evidence_quotes": [part[:300]]}]}
        summaries.append(str(data.get("summary", "")).strip())
        for card in data["cards"]:
            if isinstance(card, dict):
                card = dict(card)
                card["_source_offset"] = offset
                cards.append(card)
    return {"summary": "\n\n".join(summary for summary in summaries if summary), "cards": cards}, "\n\n".join(raw_parts)


def challenge(cfg, source, feedback):
    user = f"SOURCE (extract):\n{source}\n"
    if cfg.rejection_reasons:  # cross-doc yield: steer away from recurring failures
        user += ("\nRECURRING FAILURE MODES across recent docs (avoid these):\n"
                 + "; ".join(cfg.rejection_reasons[-5:]) + "\n")
    if feedback:
        user += f"\nTHIS-DOC-FEEDBACK (previous failed attempts):\n{feedback}\n"
    raw = complete(cfg, cfg.orchestrator, CHALLENGER_SYS, user,
                   max_tokens=cfg.challenger_max_tokens)
    try:
        return _parse_json(raw), raw
    except ValueError:
        return None, raw


def verify(cfg, source, example):
    user = (f"SOURCE EXTRACT:\n{source}\n\nTASK TYPE: {example.get('type')}\n"
            f"CONTEXT:\n{example['context']}\n\nTASK SPEC:\n{json.dumps(example['task'])}\n\n"
            f"REFERENCE WORKFLOW:\n{json.dumps(example['reference_workflow'])}\n\n"
            f"RUBRIC:\n{json.dumps(example['rubric'])}")
    raw = complete(cfg, cfg.orchestrator, VERIFIER_SYS, user, max_tokens=1500)
    try:
        v = _parse_json(raw)
    except ValueError:
        return False, {"feedback": "verifier returned invalid JSON"}, raw
    if not isinstance(v, dict):  # truncated/malformed -> treat as a failed check
        return False, {"feedback": "verifier returned malformed JSON"}, raw
    return v.get("overall") == "PASS", v, raw


def solve(cfg, model, example):
    user = f"USER REQUEST:\n{example['user_prompt']}\n\nAGENT CONTEXT:\n{example['context']}"
    return complete(cfg, model, SOLVER_SYS, user, max_tokens=cfg.solver_max_tokens)


def judge(cfg, example, answers):
    rubric = json.dumps(example["rubric"])
    joined = "\n---\n".join(f"EXECUTION RESULT {i}:\n{a}" for i, a in enumerate(answers))
    user = f"RUBRIC:\n{rubric}\n\nTASK SPEC:\n{json.dumps(example['task'])}\n\n{joined}"
    raw = complete(cfg, cfg.orchestrator, JUDGE_SYS, user, max_tokens=cfg.judge_max_tokens)
    try:  # spec is {"scores":[...]} but models sometimes emit a bare [...] or bad JSON
        data = _parse_json(raw)
    except ValueError:
        return None, raw
    return (data.get("scores") if isinstance(data, dict) else data), raw
