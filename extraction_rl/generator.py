"""Synthetic data generator with known ground truth.

The ground truth is chosen FIRST, then rendered into messy text. Because the
answer is known by construction, grading needs no human and no LLM judge.

All data produced here is synthetic. Every example carries `synthetic: True`.

Determinism: an example is a pure function of its integer seed. Train,
validation and test use non-overlapping seed ranges (see SPLITS).
"""
import random
import unicodedata

from extraction_rl.schema import CURRENCIES

NOISE_TYPES = ["clean", "encoding", "missing_field", "renamed_field"]

# Non-overlapping seed ranges per split (leakage control).
SPLITS = {
    "train": range(0, 20_000),
    "val": range(20_000, 21_000),
    "test": range(30_000, 32_000),
}

# Every name here was checked to survive a UTF-8 -> cp1252 mojibake round trip
# (see tests/test_generator.py). Names that lose bytes under errors="replace"
# (e.g. "Łukasz", "José Álvarez") are excluded: they would make the task
# unsolvable rather than hard.
NAMES = [
    "Zoë Müller", "François Dubois", "محمد علي", "Ana Peña", "Søren Kjær",
    "Björn Åström", "Inès Côté", "أحمد سمير", "Chen Wei", "Priya Nair",
    "John Smith", "Maria Rossi",
]

NON_ASCII_NAMES = [n for n in NAMES if any(ord(c) > 127 for c in n)]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "EGP": "EGP "}

CANONICAL_LABELS = {
    "invoice": "Invoice", "date": "Date", "amount": "Amount",
    "currency": "Currency", "name": "Customer name", "cid": "Customer ID",
}
RENAMED_LABELS = [
    {"invoice": "Bill No.", "date": "Issued on", "amount": "Grand total",
     "currency": "Ccy", "name": "Client", "cid": "Acct #"},
    {"invoice": "Ref", "date": "Billing date", "amount": "Total due",
     "currency": "Money", "name": "Bill to", "cid": "Client code"},
    {"invoice": "Doc number", "date": "Dt", "amount": "Sum",
     "currency": "Cur.", "name": "Payer", "cid": "Account"},
]


def _render_date(iso: str, rng: random.Random) -> str:
    y, m, d = iso.split("-")
    style = rng.randrange(4)
    # 05/03/2026 is ambiguous (May 3 or March 5?). Numeric day-first formats
    # are only used when the day > 12, so every rendered date has ONE reading.
    if style in (1, 3) and int(d) <= 12:
        style = rng.choice([0, 2])
    if style == 0:
        return iso
    if style == 1:
        return f"{d}/{m}/{y}"
    if style == 2:
        return f"{MONTHS[int(m) - 1]} {int(d)}, {y}"
    return f"{d}.{m}.{y}"


def _render_amount(amount: float, currency: str, rng: random.Random) -> str:
    style = rng.randrange(3)
    if style == 0:
        return f"{amount:.2f} {currency}"
    if style == 1:
        return f"{CURRENCY_SYMBOLS[currency]}{amount:,.2f}"
    # European style: 1.234,56 EUR
    s = f"{amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {currency}"


def is_recoverable(original: str, broken: str) -> bool:
    """True if the mojibake can be reversed exactly (the task stays solvable)."""
    try:
        return broken.encode("cp1252").decode("utf-8") == original
    except UnicodeError:
        return False


def corrupt_encoding(text: str) -> str:
    """Real-world mojibake: UTF-8 bytes mis-decoded as Windows-1252."""
    return text.encode("utf-8").decode("cp1252", errors="replace")


def make_example(seed: int) -> dict:
    rng = random.Random(seed)
    noise = NOISE_TYPES[seed % len(NOISE_TYPES)]  # balanced slices

    month, day = rng.randint(1, 12), rng.randint(1, 28)
    truth = {
        "invoice_id": f"INV-{rng.randint(10000, 99999)}",
        "date": f"{rng.choice([2024, 2025, 2026])}-{month:02d}-{day:02d}",
        "amount": round(rng.uniform(10, 9999), 2),
        "currency": rng.choice(CURRENCIES),
        "customer_id": f"C-{rng.randint(1000, 9999)}",
        # The encoding slice always uses a non-ASCII name, so the corruption
        # actually touches a field the model must recover.
        "customer_name": unicodedata.normalize(
            "NFC", rng.choice(NON_ASCII_NAMES if noise == "encoding" else NAMES)
        ),
    }

    labels = rng.choice(RENAMED_LABELS) if noise == "renamed_field" else CANONICAL_LABELS

    missing = None
    if noise == "missing_field":
        missing = rng.choice(["customer_id", "customer_name"])
        truth[missing] = None

    # If the currency is rendered as a code, it is also stated explicitly.
    amount_txt = _render_amount(truth["amount"], truth["currency"], rng)
    parts = [
        (labels["invoice"], truth["invoice_id"]),
        (labels["date"], _render_date(truth["date"], rng)),
        (labels["amount"], amount_txt),
    ]
    if truth["customer_name"] is not None:
        parts.append((labels["name"], truth["customer_name"]))
    if truth["customer_id"] is not None:
        parts.append((labels["cid"], truth["customer_id"]))
    rng.shuffle(parts)

    template = rng.randrange(3)
    if template == 0:  # free-text line
        text = ", ".join(f"{k} {v}" for k, v in parts)
    elif template == 1:  # email
        body = "\n".join(f"{k}: {v}" for k, v in parts)
        text = f"Hi team,\nPlease process the following.\n{body}\nThanks"
    else:  # key=value export line
        text = "; ".join(f"{k}={v}" for k, v in parts)

    if noise == "encoding":
        broken = corrupt_encoding(text)
        if not is_recoverable(text, broken):  # guarded by tests; should not happen
            raise ValueError(f"seed {seed}: unrecoverable encoding corruption")
        text = broken  # truth keeps the original, correctly-encoded name

    return {
        "seed": seed,
        "text": text,
        "truth": truth,
        "noise": noise,
        "missing_field": missing,
        "synthetic": True,
    }


def make_split(split: str, n: int | None = None) -> list[dict]:
    seeds = SPLITS[split]
    if n is not None:
        seeds = seeds[:n]
    return [make_example(s) for s in seeds]
