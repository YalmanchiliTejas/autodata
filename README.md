# Autodata — Agentic Self-Instruct

Open-source implementation of **Agentic Self-Instruct**, the practical method from
*Autodata: An agentic data scientist to create high quality synthetic data*
([arXiv:2606.25996](https://arxiv.org/pdf/2606.25996), Meta FAIR).

This implements the **CS pipeline** (paper Appendix C.1). A main agent orchestrates
subagents to turn grounding papers into **challenging-but-solvable** training
examples. Subagent prompts follow the paper's Figures 7–9:

- **Challenger** (Fig. 8) — writes one `{type, skill_tags, context, user_prompt,
  task, reference_workflow, rubric}`; `user_prompt` is the natural request shown
  to the evaluated agent, while `task` contains the hidden executable objective, inputs,
  constraints, required actions, and deliverables. The rubric is a flat array of 10–15 weighted criteria
  (7–10 positive / 3–5 negative).
- **Quality verifier** (Fig. 9) — gates the example on 4 checks (answer leakage,
  reasoning-vs-recall, rubric well-formedness, type consistency) before any solver
  compute is spent.
- **Weak / Strong solver** — small vs large execution agents receive the generated
  user request and visible context, then attempt the workflow `N` times each. The
  hidden task contract, reference workflow, and rubric are available only to the
  generator and judge.
- **Judge** — rubric grader scoring each execution result and deliverable.

The loop (Fig. 7) accepts only when the verifier passes **and** `weak_avg ≤ 0.65`,
`max_weak ≤ 0.75`, no zero weak scores, `0.60 ≤ strong_avg < 0.95`, and
`gap ≥ 0.20`. On any failure it feeds the failure mode back to the Challenger, which
generates an entirely new executable task, until accepted or the round budget is spent.

Every subagent turn and every accept/reject/feedback event is appended to a
**trajectory log** (JSONL) so each agent's path can be replayed and audited.

## Provider-neutral environments and learning data

Autodata never hard-codes a target application. Supply an environment contract with
its explicit capabilities, tools, and constraints; task generation can use only that
contract and the grounding source. This works for a Gauntlet sandbox, a Prime
environment, or any custom service without putting provider assumptions in prompts.

```json
{
  "provider": "gauntlet",
  "name": "staging-sandbox",
  "description": "A sandboxed application workspace.",
  "capabilities": ["inspect run status", "read project files"],
  "tools": [{"name": "get_run", "description": "Read one run's status"}],
  "constraints": ["Do not mutate production data"]
}
```

Accepted records contain the contract, acceptance metrics, and the highest-scoring
strong rollout as a preferred response. They can therefore feed a continuous loop:
generate → verify → collect rollouts → retain a preferred response → export → train
or evaluate → add failure evidence to the next source corpus.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# real run over a folder of .txt/.md grounding docs
python -m autodata.cli data/ --out accepted.jsonl

# use bounded parallel extraction for large documentation files
python -m autodata.cli data/ --out accepted.jsonl --extractor-workers 4

# bind generation to an explicit environment contract
python -m autodata.cli docs/ --environment environment.json --out accepted.jsonl

# use an OpenAI-compatible inference endpoint (for example, Fireworks)
export FIREWORKS_API_KEY=...
python -m autodata.cli docs/ --llm-provider openai_compatible \
  --llm-base-url https://api.fireworks.ai/inference/v1 \
  --llm-api-key-env FIREWORKS_API_KEY --out accepted.jsonl

# produce training-ready records from accepted data
python -m autodata.cli accepted.jsonl --export-format openai_chat --export-out train.jsonl
python -m autodata.cli accepted.jsonl --export-format prime_verifiers --export-out prime.jsonl

# offline dry run (deterministic mock LLM, no key needed)
python -m autodata.cli data/ --mock
python test_autodata.py
```

Outputs: accepted examples in `accepted.jsonl`, full agent trajectories in
`trajectories.jsonl`.

## Layout

| File | Role |
|------|------|
| `autodata/config.py` | models per role + acceptance thresholds + loop budget |
| `autodata/llm.py` | Anthropic call wrapper (+ offline mock) |
| `autodata/agents.py` | the four subagents and their prompts |
| `autodata/loop.py` | the accept/reject/feedback loop |
| `autodata/trajectory.py` | JSONL trajectory logger |
| `autodata/environment.py` | portable environment contract |
| `autodata/export.py` | OpenAI-chat and Prime Verifiers dataset exports |
| `autodata/cli.py` | run over a folder of documents |
| `test_autodata.py` | offline self-check |

Not implemented (paper mentions, out of scope here): the RL/GRPO training that
consumes `accepted.jsonl`, and the outer meta-optimization loop. Add when you want
to actually train on the generated data.
