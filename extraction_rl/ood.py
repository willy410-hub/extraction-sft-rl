"""Held-out FORMAT test set ("test_ood").

The main test split uses new seeds but the SAME templates, label words and date/
amount formats as training. That measures generalization to new values, not to
new formats. This split renders the same kind of ground truth with templates,
label synonyms and date/amount formats that never appear in training, so the gap
between test and test_ood shows how much of a gain is format memorization.

Ground truth, noise slices and the encoding/missing-field rules are identical to
extraction_rl.generator. All data here is synthetic.
"""
import random
import unicodedata

from extraction_rl.generator import (
    NAMES, NON_ASCII_NAMES, NOISE_TYPES, corrupt_encoding, is_recoverable,
)
from extraction_rl.schema import CURRENCIES

OOD_SEEDS = range(40_000, 40_400)

MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

OOD_CANONICAL = {"invoice": "Invoice", "date": "Date", "amount": "Amount",
                 "name": "Customer name", "cid": "Customer ID"}
# Label words that never occur in the training templates.
OOD_RENAMED = {"invoice": "Voucher", "date": "Posted", "amount": "Payable",
               "name": "Account holder", "cid": "Member no."}


def _render_date(iso: str, rng: random.Random) -> str:
    y, m, d = iso.split("-")
    if rng.random() < 0.5:
        return f"{int(d)} {MONTHS_LONG[int(m) - 1]} {y}"   # unseen format
    return f"{y}/{m}/{d}"                                   # unseen format


def _render_amount(amount: float, currency: str) -> str:
    return f"{currency} {amount:,.2f}"                     # code-first: unseen format


def make_ood_example(seed: int) -> dict:
    rng = random.Random(seed)
    noise = NOISE_TYPES[seed % len(NOISE_TYPES)]
    truth = {
        "invoice_id": f"INV-{rng.randint(10000, 99999)}",
        "date": f"{rng.choice([2024, 2025, 2026])}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "amount": round(rng.uniform(10, 9999), 2),
        "currency": rng.choice(CURRENCIES),
        "customer_id": f"C-{rng.randint(1000, 9999)}",
        "customer_name": unicodedata.normalize(
            "NFC", rng.choice(NON_ASCII_NAMES if noise == "encoding" else NAMES)),
    }
    labels = OOD_RENAMED if noise == "renamed_field" else OOD_CANONICAL
    missing = None
    if noise == "missing_field":
        missing = rng.choice(["customer_id", "customer_name"])
        truth[missing] = None

    parts = [(labels["invoice"], truth["invoice_id"]),
             (labels["date"], _render_date(truth["date"], rng)),
             (labels["amount"], _render_amount(truth["amount"], truth["currency"]))]
    if truth["customer_name"] is not None:
        parts.append((labels["name"], truth["customer_name"]))
    if truth["customer_id"] is not None:
        parts.append((labels["cid"], truth["customer_id"]))
    rng.shuffle(parts)

    if rng.random() < 0.5:  # markdown table: unseen template
        text = ("| " + " | ".join(k for k, _ in parts) + " |\n|" + "---|" * len(parts)
                + "\n| " + " | ".join(str(v) for _, v in parts) + " |")
    else:                   # support-ticket style: unseen template
        text = "TICKET #" + str(rng.randint(100, 999)) + " -- customer upload\n" + \
               " / ".join(f"{k} -> {v}" for k, v in parts)

    if noise == "encoding":
        broken = corrupt_encoding(text)
        if not is_recoverable(text, broken):
            raise ValueError(f"seed {seed}: unrecoverable encoding corruption")
        text = broken

    return {"seed": seed, "text": text, "truth": truth, "noise": noise,
            "missing_field": missing, "synthetic": True, "split": "test_ood"}


def make_ood_split(n: int | None = None) -> list[dict]:
    seeds = OOD_SEEDS if n is None else OOD_SEEDS[:n]
    return [make_ood_example(s) for s in seeds]
