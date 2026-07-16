"""Thin LLM wrapper. Real calls go through the Anthropic SDK; mock mode is a
deterministic offline stand-in so the loop and self-check run without a key.
The key comes from ANTHROPIC_API_KEY (loaded from .env in autodata/__init__.py)."""
import json
import os

_client = None


def _real(cfg, model, system, user, max_tokens):
    if cfg.llm_provider == "openai_compatible":
        return _openai_compatible(cfg, model, system, user, max_tokens)
    if cfg.llm_provider != "anthropic":
        raise ValueError(f"unsupported llm_provider: {cfg.llm_provider}")
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=cfg.request_timeout_seconds,
                                      max_retries=cfg.request_max_retries)
    resp = _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "disabled"},  # off: thinking tokens would truncate JSON output
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _openai_compatible(cfg, model, system, user, max_tokens):
    """Call an OpenAI-compatible endpoint without coupling tasks to that vendor."""
    from openai import OpenAI
    key_name = cfg.llm_api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"{key_name} is required for the openai_compatible provider")
    client = OpenAI(api_key=api_key, base_url=cfg.llm_base_url or None,
                    timeout=cfg.request_timeout_seconds, max_retries=cfg.request_max_retries)
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""


def _mock(cfg, model, system, user, max_tokens):
    # Deterministic pseudo-responses keyed by role, so the loop exercises real
    # control flow (accept/reject/feedback) offline. ponytail: not a model.
    import re
    if "compact coverage map" in system:
        return json.dumps({"summary": "Mock source summary.", "cards": [
            {"title": "mock constraints", "tags": ["causal_reasoning"],
             "facts": ["Mock fact one.", "Mock fact two."], "evidence_quotes": [user[:50]]},
            {"title": "mock tradeoff", "tags": ["design_tradeoff"],
             "facts": ["Mock tradeoff fact."], "evidence_quotes": [user[:50]]},
        ]})
    if "compact extract" in system:  # extractor: one cheap pass over the source
        return "Mock extract of source."
    if "You verify whether" in system:  # quality verifier: always pass in mock
        return json.dumps({"check1": "NO_LEAKAGE", "check2": "GOOD",
                           "check3": "PASS", "check4": "CONSISTENT",
                           "overall": "PASS", "feedback": ""})
    if "FLAT JSON array" in system:  # challenger: emit a full example
        rnd = 2 if "THIS-DOC-FEEDBACK" in user else 1
        rubric = ([{"criterion": f"pos{i}", "weight": 3, "category": "positive"}
                   for i in range(7)]
                  + [{"criterion": f"neg{i}", "weight": -3, "category": "negative"}
                     for i in range(3)])
        return json.dumps({
            "type": "failure mode prediction",
            "skill_tags": ["causal_reasoning", "design_tradeoff"],
            "context": "Mock grounding excerpt.",
            "user_prompt": "Please investigate the documented situation, complete the requested work, and tell me the outcome with evidence I can use.",
            "task": {
                "objective": f"Complete Mock task v{rnd}",
                "inputs": [{"name": "source", "value": "mock"}],
                "constraints": ["Use the documented facts."],
                "required_actions": ["Inspect the source", "Produce the requested artifact"],
                "deliverables": [{"name": "result", "format": "markdown", "description": "User-facing outcome with evidence"}],
            },
            "reference_workflow": [
                {"step": 1, "action": "Inspect the source", "expected_result": "Relevant facts identified"},
                {"step": 2, "action": "Produce the artifact", "expected_result": "Result completed"},
            ],
            "rubric": rubric,
        })
    if "workflow-execution grader" in system:
        m = re.search(r"Mock task v(\d+)", user)
        rnd = int(m.group(1)) if m else 1
        strong = min(0.9, 0.4 + 0.1 * rnd)   # strong clears 0.60 only by round 2
        blocks = re.split(r"EXECUTION RESULT \d+:", user)[1:]
        # The production loop grades weak and strong answer groups separately.
        scores = [strong if "claude-sonnet-5" in block else 0.3 for block in blocks]
        return json.dumps({"scores": scores, "notes": "mock"})
    return f"[{model}] mock answer"  # solver embeds its model so the judge can tell


def complete(cfg, model, system, user, max_tokens=1500):
    fn = _mock if cfg.mock else _real
    return fn(cfg, model, system, user, max_tokens)
