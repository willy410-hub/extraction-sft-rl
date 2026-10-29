"""Score a list of model outputs against generated examples and summarize."""
import numpy as np

from extraction_rl.hacking import hacking_metrics
from extraction_rl.schema import FIELDS
from extraction_rl.stats import bootstrap_ci, slice_table
from extraction_rl.verifier import exact_match, verify


def score_outputs(examples: list[dict], outputs: list[str]) -> list[dict]:
    rows = []
    for ex, out in zip(examples, outputs):
        strict = verify(out, ex["truth"])
        lenient = verify(out, ex["truth"], lenient=True)
        rows.append({
            "seed": ex["seed"], "noise": ex["noise"], "output": out,
            "score": strict["score"], "exact": exact_match(strict),
            "valid_json": strict["valid_json"], "schema_ok": strict["schema_ok"],
            "lenient_score": lenient["score"], "per_field": strict["per_field"],
            "pred": strict["pred"],
        })
    return rows


def summarize(examples: list[dict], rows: list[dict]) -> dict:
    scores = [r["score"] for r in rows]
    exact = [float(r["exact"]) for r in rows]
    slices = [r["noise"] for r in rows]
    m, lo, hi = bootstrap_ci(scores)
    em, elo, ehi = bootstrap_ci(exact)
    ok = [r for r in rows if r["per_field"] is not None]
    per_field = {f: float(np.mean([r["per_field"][f] for r in ok])) if ok else None for f in FIELDS}
    results_for_hacking = [{"schema_ok": r["schema_ok"], "pred": r["pred"]} for r in rows]
    return {
        "n": len(rows),
        "field_score": {"mean": m, "ci95": [lo, hi]},
        "exact_match": {"mean": em, "ci95": [elo, ehi]},
        "valid_json_rate": float(np.mean([r["valid_json"] for r in rows])),
        "schema_ok_rate": float(np.mean([r["schema_ok"] for r in rows])),
        "lenient_field_score": float(np.mean([r["lenient_score"] for r in rows])),
        "per_field_accuracy_when_schema_ok": per_field,
        "slices_field_score": slice_table(scores, slices),
        "slices_exact_match": slice_table(exact, slices),
        "hacking_checks": hacking_metrics(examples, results_for_hacking),
    }
