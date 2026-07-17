# Autodata

An implementation scaffold for the [Autodata paper](https://arxiv.org/abs/2606.25996), extended from per-example filtering into **dataset-level quality control**.

It builds grounded synthetic data through a challenger → weak solver → strong solver → judge loop, then continually evaluates the emerging dataset for duplicate tasks, capability gaps, source over-concentration, and unhelpful difficulty distributions. Those findings become feedback for later generations.

## What is generic here?

- `qa`: grounded question, answer, and rubric.
- `math`: problem, worked solution, answer, and verification rule.
- `coding`: programming prompt, reference solution, and deterministic tests.
- `custom`: arbitrary JSON schema and instructions.

Each record retains source provenance, capabilities, difficulty, task payload, and evaluation results. The core pipeline has no model-SDK dependency: pass any object implementing `complete(prompt, system="") -> str` for generator, weak solver, strong solver, and judge.

## Appendix-aligned profiles

Set `profile` on a task specification to use the paper's actual generation and acceptance format:

- `cs_research`: a single research question with non-leaking context, a question type, reasoning-skill tags, and a flat 10–15-item weighted rubric. It enforces the paper's weak/strong gates: weak mean ≤65%, no weak score of zero, strong mean in [60%, 95%), and ≥20-point gap.
- `legal_reasoning`: legal-source extraction style, a new natural client scenario, 15–25 six-field rubric items with capability tags, and a loop-judge `accept`/`improve` decision. Its prompt explicitly rejects the common failure mode of making a rubric easier instead of making it better.
- `scientific_reasoning`: atomic, one-sentence-answer problems with exact four-rollout gates: weak solves ≤1 and strong solves ≥3.
- `generic`: the flexible default for QA, math, coding, and custom JSON data.

### Generic profile contract

`generic` is deliberately strict, not an instruction-light fallback. It treats source documents as **untrusted reference data** rather than executable instructions, requires source-grounded claims, blocks unstated domain expansion, asks for a self-contained and independently gradable task, and requires a `self_audit` declaring grounding, scope, leakage, fairness, and verifiability. The adapter rejects a generic candidate if any audit field is absent or false.

### Rubrics are reward contracts

Every task with a rubric must declare an `environment` with `success_conditions` (and, where relevant, available tools, permitted inputs, execution limits, and evaluator behavior). Before rollouts, the local gate rejects malformed, generic, overlapping, zero-weight, capability-missing, or environment-free rubric criteria. When a `quality_verifier` is configured, it must also return a criterion-by-criterion audit proving each reward is grounded, observable, compatible with the environment, and discriminative; otherwise the candidate is rejected before solver inference. This makes the rubric a checked reward contract rather than synthetic prose.

Use task-level controls to state the boundary explicitly:

```json
{
  "name": "grounded-code-review",
  "kind": "custom",
  "instructions": "Create a code-review task based only on the supplied repository excerpt.",
  "source_policy": "grounded_only",
  "constraints": ["Use Python only", "Do not require external packages"],
  "forbidden_content": ["network access", "personal data", "security exploitation"],
  "environment": {
    "available_tools": [],
    "permitted_inputs": ["repository excerpt"],
    "success_conditions": ["identify the defect", "propose a patch compatible with the excerpt"],
    "evaluator_behavior": "Score only claims demonstrably supported by the excerpt."
  },
  "output_schema": {"required": ["input", "answer", "rubric", "self_audit"]}
}
```

The optional `quality_verifier` model passed to `CandidateEvaluator` receives the separate leakage/structural verifier prompt before expensive solver rollouts. `DatasetBuilder.last_attempts` retains both accepted and rejected rounds with evaluator diagnostics; write this to an audit JSONL in production to preserve the complete trajectory, as the paper does.

## Inputs

Task specifications are a JSON array:

```json
[
  {
    "name": "algorithm-design",
    "kind": "coding",
    "instructions": "Create Python algorithm problems based on the source.",
    "capabilities": ["algorithms", "edge-cases"],
    "target_difficulty": "adaptive"
  },
  {
    "name": "proof-practice",
    "kind": "math",
    "instructions": "Create rigorous proof exercises.",
    "capabilities": ["proof", "algebra"]
  }
]
```

Sources are JSONL, one document per line:

```json
{"id":"doc-1","content":"...grounding material...","metadata":{"license":"..."}}
```

## Use from Python

```python
from autodata import DatasetBuilder
from autodata.evaluation import CandidateEvaluator
from autodata.io import load_sources, load_specs, write_jsonl

# provider.complete(prompt, system="") must return text.
evaluator = CandidateEvaluator(weak_model, strong_model, judge_model)
builder = DatasetBuilder(generator_model, evaluator)
candidates, report = builder.build(load_specs("tasks.json"), load_sources("sources.jsonl"))
write_jsonl("dataset.jsonl", candidates)
print(report.as_dict())
```

Or use the CLI with a small provider factory you own:

```bash
autodata --tasks tasks.json --sources sources.jsonl --output dataset.jsonl \
  --report report.json --provider my_models:build_providers --rounds 4
```

`build_providers()` returns `{"generator": ..., "weak": ..., "strong": ..., "judge": ...}`. Each object needs only `complete(prompt, system="") -> str`, which makes the system compatible with hosted APIs, local inference, or test doubles.

## Dataset-level policy

`DatasetQualityController` rejects near duplicates before acceptance and reports:

- capability coverage against every requested capability;
- source distribution/concentration;
- easy/medium/hard distribution;
- near-duplicate pairs using Jaccard token overlap.

Recommendations feed back into later prompts. In production, replace the lexical duplicate check with embeddings, add domain-specific execution/verification (especially for code and math), audit source licensing, and use held-out human or independent-model evaluation to detect judge overfitting.

## Challenger → rubric → solver flow

The `generator` provider is the paper's **challenger**. For each candidate it first authors the task and reference internally, then emits the task package and its `rubric`. The static rubric gate and optional quality verifier run before weak/strong solver rollouts. The solvers receive only the solver-visible surface (`context`, `question`, `problem`, or `prompt`); answers, solutions, rubrics, tests, and audit metadata are deliberately withheld. The judge receives the full package plus rollouts, decides whether the reward is useful, and supplies a concrete improvement instruction for the next challenger round.

`require_rubric` defaults to `true` for RL-oriented data. Set it to `false` only for a task with an explicit non-rubric verifier, such as deterministic program execution.

### Solver and judge prompts

The solver prompts implement the evaluation discipline described in the paper rather than asking the weak model to “be weak.” Weak and strong receive the **identical system prompt**, exact same solver-visible task, and same environment. This prevents a prompt-induced gap from being mistaken for a capability gap. If a stronger solver should have more inference-time compute, tools, or aggregation—as permitted by the paper—configure that difference in the two provider/harness instances, with every extra capability declared in the environment contract; never encode it by silently changing the system prompt. The rubric scorer sees the task, environment, rubric, and responses—but not the reference answer—and scores each criterion independently and binary before producing normalized per-rollout rewards.

## Dynamic ingestion and utility verification

Avoid selecting a legal, research, or coding prompt based on a filename or domain label. The dynamic path is:

```text
documents → source profile → task/evaluation contract → probe generation
          → rollout calibration → accepted dataset → short RL pilot → held-out comparison
```

- `ingest_path()` loads local text, Markdown, reStructuredText, and HTML sources. Add an extractor adapter for PDFs, office files, repositories, databases, or APIs.
- `SourceProfiler` extracts grounded facts, transferable operations, candidate capabilities, and a proposed environment; unsuitable sources are excluded early.
- `ContractSynthesizer` turns that evidence plus the user objective into a `TaskSpec`, including solver-visible inputs, success conditions, available tools, and reward plan. It does not choose a hard-coded domain profile.
- `AdaptiveUtilityGate` first collects probe-rollout statistics, then accepts tasks whose weak-rollout variance and weak/strong gap are high relative to the observed population. It explicitly rejects flat reward groups.
- `PilotResult` gates promotion on positive held-out improvement over a baseline. In a real training runner, populate it with paired scores from a short training run and an independent held-out evaluator.

This does not guarantee a useful dataset from an LLM-only judgment. The short, held-out RL pilot is the final causal test: if the trained policy does not improve beyond paired evaluation noise, reject or revise the contract/dataset.

## Running with Claude and Modal vLLM

Deploy the weak solver first:

```bash
pip install modal
modal setup
modal deploy deploy/modal_vllm_weak.py
```

Then copy the deployed endpoint URL into [config.example.json](/Users/tejas/Desktop/autodata/config.example.json), create Modal proxy credentials with `modal workspace proxy-tokens create`, set the named API-key environment variables, and run:

```bash
python main.py --config run-config.json
```

The runner uses an OpenAI-compatible endpoint for the small weak model, so the included Modal-hosted vLLM deployment works directly. Modal endpoints serve OpenAI Chat Completions under `/v1` and are authenticated by proxy credentials by default. [Modal endpoints guide](https://modal.com/docs/guide/endpoints) It uses Claude-compatible Messages API roles for challenger, judge, optional quality verifier, and strong solver. The example assigns Sonnet 4 to challenger/judging and Opus 4 to the strong solver; keep model names configurable because API availability changes. Anthropic describes Opus as its most capable model and Sonnet as the high-performance reasoning/efficiency option. [Anthropic model overview](https://docs.anthropic.com/en/docs/welcome)

Evaluation is **weak-first**: the weak model is scored before the quality verifier or strong solver. If its mean score exceeds `weak_screen_max` (or it passes at least two of four scientific rollouts), the candidate is rejected as too easy and the expensive roles are not called.
