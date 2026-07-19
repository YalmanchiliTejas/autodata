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

The `generator` provider is the paper's **challenger**. For each candidate it first authors the task and reference internally, then emits the task package and its `rubric`. The static rubric gate and optional quality verifier run before weak/strong solver rollouts. The solvers receive only the solver-visible surface (`context`, `question`, `problem`, or `prompt`); answers, solutions, rubrics, tests, and audit metadata are deliberately withheld. The judge receives the solver-visible task, rubric, and rollouts—but not the reference solution—marks every rubric criterion true or false for each response, and supplies a concrete improvement instruction. Autodata computes normalized rewards deterministically from those matches and the signed rubric weights. A candidate is never accepted when the loop judge returns `improve`.

`require_rubric` defaults to `true` for RL-oriented data. Set it to `false` only for a task with an explicit non-rubric verifier, such as deterministic program execution.

### Solver and judge prompts

The solver prompts implement the evaluation discipline described in the paper rather than asking the weak model to “be weak.” Weak and strong receive the **identical system prompt**, exact same solver-visible task, and same environment. This prevents a prompt-induced gap from being mistaken for a capability gap. If a stronger solver should have more inference-time compute, tools, or aggregation—as permitted by the paper—configure that difference in the two provider/harness instances, with every extra capability declared in the environment contract; never encode it by silently changing the system prompt. The rubric scorer sees the task, environment, rubric, and responses—but not the reference answer—and marks each criterion independently and binary. Autodata then applies min-max normalization over the rubric's attainable range, so avoiding every negative criterion and satisfying every positive criterion always scores `1.0`.

## Checkpointing and resume

The configured runner writes an atomic checkpoint after every audit event. It contains accepted candidates, rejected attempts, the audit trajectory, and source-local challenger feedback. If a process exits during a model request, the incomplete attempt is retried; attempts with terminal events are not repeated.

```bash
python main.py --config run-config.json --checkpoint outputs/run.checkpoint.json
python main.py --config run-config.json --checkpoint outputs/run.checkpoint.json --resume
```

On resume, the formatted solver-prompt transcript is appended rather than overwritten. Final dataset, audit JSONL, and report files are materialized normally when the resumed build completes.

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

## GRPO pilot: frozen base vs trained adapter

The repository now includes a real **LoRA + GRPO** training path. It is intentionally separate from dataset construction: rollout scores used to decide whether a challenger example is useful are not recycled as rewards. During GRPO, every newly sampled completion is scored from its own reward contract.

There is no benchmark bundled in this repository—the `tests/` directory is only a unit-test suite. `export_grpo_data.py` creates a stable, source-disjoint split from accepted records for a small *in-distribution* pilot. For a general coding claim, evaluate the frozen and trained models on the same untouched external benchmark, such as SWE-bench Verified, after training; do not generate tasks from that benchmark or use its issue text during data construction.

Export only accepted candidates (with a fixed split manifest):

```bash
python export_grpo_data.py \
  --input dataset.jsonl \
  --train-output runs/pilot/train.jsonl \
  --benchmark-output runs/pilot/heldout.jsonl \
  --benchmark-fraction 0.20 \
  --seed qwen-grpo-pilot-1
```

Short-answer QA/math records use a deterministic exact-match reward. Accepted tasks without an exact answer use an independent rubric judge configured through a provider JSON file; the judge receives the task, environment, rubric, and new completion, never the reference answer. Coding records are rejected by the text-only runner because executing challenger-authored tests without a sandbox would be unsafe. Add a sandbox reward adapter for them before training.

Install the optional GPU stack in a CUDA environment, then train a LoRA adapter. GRPO needs multiple completions for each prompt, so keep `--num-generations` at least two (four is the default).

```bash
pip install -r requirements-train.txt

python train_grpo.py \
  --model Qwen/Qwen3.5-4B \
  --train runs/pilot/train.jsonl \
  --output-dir runs/pilot/qwen-grpo-adapter \
  --max-steps 100 \
  --num-generations 4
```

Use exactly the same held-out records, sampling settings, and reward implementation for the frozen base and adapter:

```bash
python evaluate_grpo.py \
  --model Qwen/Qwen3.5-4B \
  --benchmark runs/pilot/heldout.jsonl \
  --output runs/pilot/base.json

python evaluate_grpo.py \
  --model Qwen/Qwen3.5-4B \
  --adapter runs/pilot/qwen-grpo-adapter \
  --benchmark runs/pilot/heldout.jsonl \
  --output runs/pilot/grpo.json

python compare_grpo_runs.py \
  --baseline runs/pilot/base.json \
  --trained runs/pilot/grpo.json
```

The comparison reports the paired mean-reward delta, a bootstrap 95% interval, and per-example wins/losses. Promote a dataset only when the interval is above zero and the result also transfers to a separate external benchmark. Record the base model revision, dataset hashes, seed, reward version, rollout count, and training manifest; the runner writes the final manifest alongside the adapter.

For cloud training, [deploy/modal_grpo_train.py](/Users/tejas/Desktop/autodata/deploy/modal_grpo_train.py) uses an A100-80GB and Modal Volumes, so no model checkpoint is stored on your Mac:

```bash
modal volume put autodata-grpo-data runs/pilot/train.jsonl /data/train.jsonl
modal volume put autodata-grpo-data runs/pilot/heldout.jsonl /data/heldout.jsonl

modal run deploy/modal_grpo_train.py::train_exact \
  --model Qwen/Qwen3.5-4B \
  --train-file /data/train.jsonl \
  --output-name qwen-grpo-pilot

modal run deploy/modal_grpo_train.py::evaluate \
  --model Qwen/Qwen3.5-4B \
  --benchmark-file /data/heldout.jsonl \
  --output-name base.json

modal run deploy/modal_grpo_train.py::evaluate \
  --model Qwen/Qwen3.5-4B \
  --benchmark-file /data/heldout.jsonl \
  --adapter-name qwen-grpo-pilot \
  --output-name grpo.json
```

Download the two JSON reports from `autodata-grpo-output`, then run `compare_grpo_runs.py` locally. For rubric-judged records, add a provider config as `/data/judge.json` and create the Modal secret `autodata-anthropic`; use `train_with_judge` instead. The secret is needed only because an online judge receives fresh rollout completions during training, not for deterministic exact-match rewards.

## Running with Claude and Modal vLLM

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` before running. `main.py` loads that local file automatically; a shell-exported key still takes precedence.

Deploy the weak solver first:

```bash
pip install modal
modal setup
modal deploy deploy/modal_vllm_weak.py
```

Then copy the deployed endpoint URL into [config.example.json](/Users/tejas/Desktop/autodata/config.example.json), set the named Anthropic API-key environment variable, and run:

```bash
python main.py --config run-config.json
```

The runner uses an OpenAI-compatible endpoint for the small weak model, so the included Modal-hosted vLLM deployment works directly. It serves OpenAI Chat Completions under `/v1`. This deployment is intentionally public (`requires_proxy_auth=False`); anyone with its URL can consume its GPU capacity, so stop it when it is not in use. [Modal endpoints guide](https://modal.com/docs/guide/endpoints) It uses Claude-compatible Messages API roles for challenger, judge, optional quality verifier, and strong solver. The example assigns Sonnet 4 to challenger/judging and Opus 4 to the strong solver; keep model names configurable because API availability changes. Anthropic describes Opus as its most capable model and Sonnet as the high-performance reasoning/efficiency option. [Anthropic model overview](https://docs.anthropic.com/en/docs/welcome)

## Context budgeting

Every VLLM deployment has a physical context ceiling because its KV cache must fit on the selected GPU. Autodata's `ContextController` makes the policy above that ceiling explicit: provide `ContextSegment`s for policy/task/environment (`protected=True`) and lower-priority source excerpts, feedback, or old tool observations (`artifact_ref` can point at the full stored material). It uses a conservative token estimate, compacts only unprotected segments at a soft threshold, and raises `ContextOverflow` rather than silently truncating protected task material. Interactive tool harnesses should preserve tool-call metadata and compact older tool outputs first; the included Mini-SWE-Agent weak-model config applies that policy with a 32K context window and a 2K output reserve.

Evaluation is **quality-first, then weak-first**: structural and model-based leakage/reward checks run before any solver. If they pass, the weak model is scored before the strong solver. If its mean score exceeds `weak_screen_max` (or it passes at least two of four scientific rollouts), the candidate is rejected as too easy and the strong solver is not called.
