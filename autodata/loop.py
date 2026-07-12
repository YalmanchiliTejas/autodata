"""Agentic Self-Instruct acceptance loop (Autodata, Appendix C.1, Fig. 7), domain-generic.

Per source document: challenger writes an executable task+rubric -> quality verifier
gates it (retry challenger on fail) -> both agents execute it N times -> Kimi-style
judge scores each execution result -> accept only when the verifier passed AND the weak solver
struggles AND the strong solver succeeds-but-not-trivially AND the gap is wide. On
any failure the failure mode is fed back to the challenger for an ENTIRELY new
task, until accepted or the step budget is exhausted."""
from . import agents
from concurrent.futures import ThreadPoolExecutor
import json
import math


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _remember_rejection(cfg, reason):
    # Keep a bounded rolling window of recent failures to feed back to the challenger.
    cfg.rejection_reasons.append(reason)
    del cfg.rejection_reasons[:-50]


def _acceptable(cfg, weak, strong):
    """Fig. 7 acceptance criteria on the solver score lists."""
    wa, sa = _avg(weak), _avg(strong)
    checks = {
        "weak_avg<=": wa <= cfg.weak_avg_max,
        "weak_max<=": (max(weak) if weak else 1.0) <= cfg.weak_max_max,
        "weak_no_zeros": (not cfg.weak_no_zeros) or all(x > 0 for x in weak),
        "strong>=": sa >= cfg.strong_min,
        "strong<": sa < cfg.strong_max,
        "gap>=": (sa - wa) >= cfg.min_gap,
    }
    return all(checks.values()), checks, wa, sa


def _weak_rejection(cfg, scores):
    """Return an irreversible weak-only rejection reason, else None.

    This is evaluated after the complete weak rollout set. Skipping strong
    evaluation for a weak failure preserves the final acceptance set while avoiding
    its expensive calls.
    """
    if any(score == 0 for score in scores) and cfg.weak_no_zeros:
        return "weak_no_zeros"
    if scores and max(scores) > cfg.weak_max_max:
        return "weak_max<="
    if sum(scores) > cfg.solver_samples * cfg.weak_avg_max:
        return "weak_avg<="
    return None


def _valid_scores(scores, expected):
    return (isinstance(scores, list) and len(scores) == expected and
            all(isinstance(score, (int, float)) and not isinstance(score, bool) and
                math.isfinite(score) and 0 <= score <= 1 for score in scores))


def _solve_samples(cfg, model, example):
    """Independent rollouts share no state, so run them concurrently to cut latency."""
    with ThreadPoolExecutor(max_workers=cfg.solver_samples) as executor:
        return list(executor.map(lambda _: agents.solve(cfg, model, example),
                                 range(cfg.solver_samples)))


def _judge_with_retry(cfg, log, round_i, stage, example, answers):
    """Retry evaluator-format failures without changing the candidate task."""
    prompt = json.dumps({"task": example["task"], "rubric": example["rubric"],
                         "execution_results": answers})
    for attempt in range(cfg.evaluation_retries + 1):
        scores, raw = agents.judge(cfg, example, answers)
        log.log(stage, round_i, prompt, raw, {"attempt": attempt,
                "model": cfg.orchestrator, "system_prompt": agents.JUDGE_SYS})
        if _valid_scores(scores, len(answers)):
            return scores
        log.event("evaluation_retry", round=round_i, stage=stage, attempt=attempt)
    return None


def run_doc(cfg, log, doc_id, document, coverage=None):
    """Run the loop on one paper. Returns the accepted example or None."""
    log.start_doc(doc_id)
    source_sha256 = coverage_entry = None
    fallback_source = None
    if coverage:
        source_sha256, coverage_entry = coverage.get_or_create(cfg, doc_id, document, log)
    else:
        source = agents.extract(cfg, document)
        log.log("extractor", 0, document, source,
                {"model": cfg.extractor, "system_prompt": agents.EXTRACTOR_SYS})

    feedback = ""
    coverage_failure_streak = 0
    for r in range(1, cfg.max_rounds + 1):
        if cfg.max_total_rounds and cfg.rounds_run >= cfg.max_total_rounds:
            log.event("budget_exhausted", scope="global", rounds_run=cfg.rounds_run)
            return None
        cfg.rounds_run += 1
        selected_cards = []
        source_mode = "compact_extract"
        if coverage and coverage_failure_streak < cfg.coverage_escalation_rounds:
            selected_cards = coverage.select(source_sha256, cfg.coverage_cards_per_round)
            source = coverage.render(coverage_entry, selected_cards)
            source_mode = "coverage_cards"
            log.event("coverage_selected", round=r, source_sha256=source_sha256,
                      card_ids=[card["id"] for card in selected_cards],
                      tags=sorted({tag for card in selected_cards for tag in card["tags"]}))
        elif coverage:
            if fallback_source is None:
                fallback_source = agents.extract(cfg, document)
                log.log("coverage_fallback_extractor", r, document, fallback_source,
                        {"model": cfg.extractor, "system_prompt": agents.EXTRACTOR_SYS})
            source = fallback_source
            source_mode = "full_extract_fallback"
            log.event("coverage_escalated", round=r, source_sha256=source_sha256,
                      failures=coverage_failure_streak)
        example, raw = agents.challenge(cfg, source, feedback)
        log.log("challenger", r, source, raw,
                {"model": cfg.orchestrator, "system_prompt": agents.CHALLENGER_SYS,
                 "feedback": feedback})

        # Challenger occasionally emits a malformed/truncated object; retry the round
        # instead of crashing (uses the same round budget as any other rejection).
        schema_errors = agents.validate_example(example)
        if schema_errors:
            feedback = f"round {r}: malformed output: {'; '.join(schema_errors)}. Re-emit a complete valid JSON object."
            log.event("reject", round=r, stage="challenger_parse", errors=schema_errors)
            continue

        # Fig. 7 step (3): quality verifier gates before spending solver compute.
        qv_ok, verdict, vraw = agents.verify(cfg, source, example)
        log.log("quality_verifier", r, json.dumps({"source": source, "example": example}), vraw,
                {"model": cfg.orchestrator, "system_prompt": agents.VERIFIER_SYS})
        if not qv_ok:
            feedback = f"round {r}: FAILED QV -> {verdict.get('feedback', verdict)}"
            _remember_rejection(cfg, f"QV: {verdict.get('feedback', '')}")
            log.event("reject", round=r, stage="quality_verifier", verdict=verdict)
            if coverage and verdict.get("check5") == "UNSUPPORTED":
                coverage_failure_streak += 1
            continue

        weak_answers = _solve_samples(cfg, cfg.weak_solver, example)
        for i, wa in enumerate(weak_answers):
            log.log("weak_solver", r, json.dumps({"user_prompt": example["user_prompt"], "context": example["context"]}), wa,
                    {"sample": i, "model": cfg.weak_solver, "system_prompt": agents.SOLVER_SYS})

        weak_scores = _judge_with_retry(cfg, log, r, "weak_screen_judge", example, weak_answers)
        if weak_scores is None:
            feedback = f"round {r}: weak evaluator remained unavailable after retries; new task."
            log.event("evaluation_error", round=r, stage="weak_screen_judge")
            continue
        weak_reason = _weak_rejection(cfg, weak_scores)
        if weak_reason:
            feedback = f"round {r}: weak-only screen failed {weak_reason}; previous task: {example['task']['objective']}"
            _remember_rejection(cfg, f"{example.get('type', '?')}: {weak_reason}")
            log.event("reject", round=r, stage="weak_screen", failed=[weak_reason],
                      weak_avg=round(_avg(weak_scores), 3))
            coverage_failure_streak = 0
            continue

        # The candidate passed the complete weak gate. Strong calls are now needed.
        strong_answers = _solve_samples(cfg, cfg.strong_solver, example)
        for i, sa in enumerate(strong_answers):
            log.log("strong_solver", r, json.dumps({"user_prompt": example["user_prompt"], "context": example["context"]}), sa,
                    {"sample": i, "model": cfg.strong_solver, "system_prompt": agents.SOLVER_SYS})

        strong_scores = _judge_with_retry(cfg, log, r, "strong_judge", example, strong_answers)
        if strong_scores is None:
            feedback = f"round {r}: strong evaluator remained unavailable after retries; new task."
            log.event("evaluation_error", round=r, stage="strong_judge")
            continue

        ok, checks, w, s = _acceptable(cfg, weak_scores, strong_scores)
        metrics = {"weak_avg": round(w, 3), "strong_avg": round(s, 3),
                   "gap": round(s - w, 3)}

        if ok:
            coverage_info = None
            if coverage:
                card_ids = [card["id"] for card in selected_cards]
                if card_ids:
                    coverage.accept(source_sha256, card_ids)
                coverage_info = {"mode": source_mode, "source_sha256": source_sha256,
                                 "card_ids": card_ids,
                                 "tags": sorted({tag for card in selected_cards for tag in card["tags"]}),
                                 "fallback_after_failures": coverage_failure_streak}
            log.event("accept", round=r, **metrics)
            result = {"doc_id": doc_id, "round": r, **example, **metrics}
            if coverage_info:
                result["coverage"] = coverage_info
            return result

        # Rejected: name which criteria failed (TOO EASY vs FAILED ON STRONG).
        failed = [k for k, v in checks.items() if not v]
        feedback = f"round {r}: {metrics} failed {failed}; previous task: {example['task']['objective']}"
        _remember_rejection(cfg, f"{example.get('type', '?')}: failed {failed}")
        log.event("reject", round=r, stage="solvers", failed=failed, **metrics)
        if coverage and {"strong>=", "gap>="} & set(failed):
            coverage_failure_streak += 1
        else:
            coverage_failure_streak = 0

    log.event("budget_exhausted", rounds=cfg.max_rounds)
    return None
