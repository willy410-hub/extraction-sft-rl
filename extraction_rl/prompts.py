"""Prompt construction shared by baseline evaluation, SFT and GRPO.

Using one function everywhere keeps the train-time and eval-time prompt identical.
"""
import json

from extraction_rl.schema import FIELDS

SYSTEM_PROMPT = (
    "You extract invoice data from messy customer text. "
    "Reply with ONLY a JSON object, no explanation and no code fences, with exactly these keys: "
    + ", ".join(FIELDS)
    + ". Rules: date as YYYY-MM-DD; amount as a number; currency as a 3-letter ISO code "
    "(USD, EUR, EGP, GBP); use null for any field that is not present in the text. "
    "If the text has broken characters from a wrong encoding, return the correctly spelled value."
)


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def target_json(truth: dict) -> str:
    """The canonical answer string used as the SFT target (behavior cloning target)."""
    return json.dumps({f: truth[f] for f in FIELDS}, ensure_ascii=False)
