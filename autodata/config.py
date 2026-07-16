"""Config for Agentic Self-Instruct (Autodata, arXiv:2606.25996)."""
from dataclasses import dataclass, field
from .environment import EnvironmentSpec


@dataclass
class Config:
    # Model-provider transport. `openai_compatible` supports endpoints such as
    # Fireworks without making a particular inference vendor part of task logic.
    llm_provider: str = "anthropic"
    llm_base_url: str = ""
    llm_api_key_env: str = ""
    # Which Anthropic model plays each role. Strong must out-reason weak (the whole
    # pipeline mines tasks where weak fails and strong succeeds), so generation +
    # judging use the strong model; extraction + the weak solver use the cheap one.
    orchestrator: str = "claude-sonnet-5"              # challenger + judge run here
    strong_solver: str = "claude-sonnet-5"
    weak_solver: str = "claude-haiku-4-5-20251001"
    extractor: str = "claude-haiku-4-5-20251001"       # cheap one-pass distillation

    # Acceptance criteria (CS pipeline, Appendix C.1 / Fig. 7 main-agent prompt).
    weak_avg_max: float = 0.65   # weak_avg must be <= this
    weak_max_max: float = 0.75   # max single weak attempt must be <= this
    weak_no_zeros: bool = True   # reject if any weak attempt scored 0
    strong_min: float = 0.60     # strong_avg must be >= this
    strong_max: float = 0.95     # ...and < this (not trivially easy)
    min_gap: float = 0.20        # strong_avg - weak_avg gap

    # Loop control.
    max_rounds: int = 10       # step budget per grounding doc
    solver_samples: int = 3    # attempts per solver, averaged (variance reduction)
    solver_max_tokens: int = 800   # enough for a grounded user-facing result
    challenger_max_tokens: int = 3000
    judge_max_tokens: int = 500
    request_timeout_seconds: float = 90.0
    request_max_retries: int = 1
    evaluation_retries: int = 2    # retry malformed judge output on the same candidate
    coverage_mapper_max_tokens: int = 1800
    extractor_workers: int = 4       # bounded parallel fan-out for independent document chunks
    coverage_cards_per_round: int = 4
    coverage_escalation_rounds: int = 2  # failures before full-extract fallback

    log_path: str = "trajectories.jsonl"
    mock: bool = False         # offline deterministic LLM, for the self-check

    # Global loop budget: total challenger rounds across ALL docs. 0 = unlimited.
    # max_rounds caps rounds per doc; this caps the whole run.
    max_total_rounds: int = 0
    rounds_run: int = 0        # runtime counter, incremented in loop.run_doc

    # Corpus dedup: skip a doc whose shingle-Jaccard vs a kept doc >= this. 0 = off.
    dedup_threshold: float = 0.8

    # Yield: recent cross-doc rejection reasons, fed back into the challenger.
    rejection_reasons: list = field(default_factory=list)

    # A provider supplies this contract; task generation may use no other tools.
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
