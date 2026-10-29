"""Paired comparison of two evaluated runs on the SAME test items.

Usage:
  python -m scripts.compare results/baseline results/sft_seed0
"""
import argparse
import json
import os

from extraction_rl.stats import MIN_SLICE_SIZE, paired_bootstrap_diff


def load(run_dir):
    with open(os.path.join(run_dir, "predictions.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compare(a_dir: str, b_dir: str) -> dict:
    a, b = load(a_dir), load(b_dir)
    if [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("runs were not evaluated on the same items in the same order")
    out = {"a": a_dir, "b": b_dir, "overall": {}, "slices": {}}
    for metric in ("score", "exact"):
        out["overall"][metric] = paired_bootstrap_diff([float(r[metric]) for r in a],
                                                       [float(r[metric]) for r in b])
    for sl in sorted({r["noise"] for r in a}):
        idx = [i for i, r in enumerate(a) if r["noise"] == sl]
        if len(idx) < MIN_SLICE_SIZE:
            out["slices"][sl] = {"n": len(idx), "note": "below minimum slice size"}
            continue
        out["slices"][sl] = {m: paired_bootstrap_diff([float(a[i][m]) for i in idx],
                                                      [float(b[i][m]) for i in idx])
                             for m in ("score", "exact")}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = compare(args.a, args.b)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
