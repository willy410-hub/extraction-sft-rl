"""Small, dependency-light statistics for comparing models on the same items.

- bootstrap_ci: 95% percentile bootstrap CI for a mean.
- paired_bootstrap_diff: CI for mean(B - A) when A and B scored the SAME items.
  Pairing matters: per-item difficulty is shared, so resampling item indices
  (not each system separately) gives a tighter, honest interval.
- slice_table: per-slice means, refusing to report a CI below MIN_SLICE_SIZE.
"""
import numpy as np

MIN_SLICE_SIZE = 30
N_BOOT = 2000


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(a, b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same items in the same order")
    d = b - a
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"mean_diff": mean, "ci_low": lo, "ci_high": hi, "n": int(len(d)),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def slice_table(values, slices) -> dict:
    values, slices = np.asarray(values, dtype=float), np.asarray(slices)
    out = {}
    for name in sorted(set(slices.tolist())):
        v = values[slices == name]
        if len(v) < MIN_SLICE_SIZE:
            out[name] = {"n": int(len(v)), "mean": float(v.mean()), "ci": None,
                         "note": f"below minimum slice size {MIN_SLICE_SIZE}"}
        else:
            m, lo, hi = bootstrap_ci(v)
            out[name] = {"n": int(len(v)), "mean": m, "ci": [lo, hi]}
    return out
