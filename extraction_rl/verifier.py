"""Rule-based verifier: the single source of truth for grading AND for the RL reward.

Three gates, in order:
  1. valid JSON   -- the whole output must parse (strict mode).
  2. schema       -- exactly the expected keys, each value of the right type or null.
  3. field match  -- each field compared to ground truth; score = fraction correct.

Partial credit matters for RL: a completion with 5/6 fields right must earn
more than one with 2/6, or the policy gets no gradient signal toward "better".
"""
import json
import math
import re
import unicodedata

from extraction_rl.schema import FIELDS, FIELD_TYPES

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_output(output: str, lenient: bool = False):
    """Return the parsed object or None.

    strict (default, used for the reward): the stripped output must be JSON.
    lenient (diagnostic only): also accepts one ```json fenced block, so we can
    separate "knows the answer but wraps it" from "cannot extract".
    """
    text = output.strip()
    if lenient:
        m = _FENCE.match(text)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _type_ok(field: str, value) -> bool:
    if value is None:
        return True
    expected = FIELD_TYPES[field]
    if expected is float:
        # bool is a subclass of int in Python; reject it explicitly.
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return isinstance(value, expected)


def _norm_str(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def field_correct(field: str, pred, truth) -> bool:
    if truth is None or pred is None:
        return pred is None and truth is None
    if field == "amount":
        return abs(float(pred) - float(truth)) < 0.005
    return _norm_str(pred) == _norm_str(truth)


def verify(output: str, truth: dict, lenient: bool = False) -> dict:
    pred = parse_output(output, lenient=lenient)
    if pred is None:
        return {"valid_json": False, "schema_ok": False, "score": 0.0, "per_field": None, "pred": None}

    if not isinstance(pred, dict) or set(pred) != set(FIELDS) or not all(
        _type_ok(f, pred[f]) for f in FIELDS
    ):
        return {"valid_json": True, "schema_ok": False, "score": 0.0, "per_field": None, "pred": pred}

    per_field = {f: field_correct(f, pred[f], truth[f]) for f in FIELDS}
    score = sum(per_field.values()) / len(FIELDS)
    return {"valid_json": True, "schema_ok": True, "score": score, "per_field": per_field, "pred": pred}


def exact_match(result: dict) -> bool:
    """All fields correct -- the strict, binary headline metric."""
    return result["score"] == 1.0


def penalized_reward(output: str, truth: dict, invent_penalty: float = 1.0) -> float:
    """Reward variant for the v0.3 ablation: verifier score minus a penalty for
    every field that is null in the truth but filled in by the model.

    Motivation (from a real run): with the plain score, inventing a value for a
    missing field costs only 1/6, the same as any other wrong field, and GRPO from
    the base model learned to fill missing fields anyway (invented_on_missing_rate
    went from 0.01 for the untuned model to 1.0, 1.0 and 0.59 across 3 seeds).
    """
    r = verify(output, truth)
    if not r["schema_ok"]:
        return r["score"]
    invented = sum(1 for f in FIELDS if truth[f] is None and r["pred"][f] is not None)
    return r["score"] - invent_penalty * invented / len(FIELDS)
