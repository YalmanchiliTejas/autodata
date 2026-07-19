"""Contract-aware extraction of JSON objects from model responses."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable


def extract_json_object(raw: str, required_fields: Iterable[str], *, producer: str,
                        validator: Callable[[dict], None] | None = None) -> dict:
    """Return the first decoded object satisfying the complete response contract.

    Models occasionally echo a JSON Schema or example before their actual
    response. Selecting the first syntactically valid object would then parse
    the schema rather than the requested result, so selection is based on the
    response contract as well as JSON syntax.
    """
    required = tuple(required_fields)
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    objects_seen = 0
    contract_errors: list[str] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        objects_seen += 1
        if not all(field in value for field in required):
            continue
        if validator is not None:
            try:
                validator(value)
            except (ValueError, TypeError, KeyError) as exc:
                contract_errors.append(str(exc))
                continue
        return value

    preview = re.sub(r"\s+", " ", raw).strip()[:800]
    expected = ", ".join(required)
    if contract_errors:
        details = "; ".join(dict.fromkeys(contract_errors))
        raise ValueError(f"{producer} returned matching JSON objects, but none satisfied the value contract: {details}; "
                         f"response preview: {preview!r}")
    if objects_seen:
        raise ValueError(f"{producer} returned JSON objects, but none matched required fields: {expected}; response preview: {preview!r}")
    raise ValueError(f"{producer} must return a JSON object with required fields: {expected}; response preview: {preview!r}")
