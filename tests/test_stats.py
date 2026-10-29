import numpy as np
import pytest

from extraction_rl.stats import MIN_SLICE_SIZE, bootstrap_ci, paired_bootstrap_diff, slice_table


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.random.default_rng(1).random(200)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi
    assert bootstrap_ci(x) == (m, lo, hi)


def test_paired_diff_detects_a_clear_improvement():
    a = np.zeros(100)
    b = np.ones(100) * 0.5
    r = paired_bootstrap_diff(a, b)
    assert r["mean_diff"] == pytest.approx(0.5) and r["ci_excludes_zero"]


def test_paired_diff_no_change_includes_zero():
    a = np.random.default_rng(0).random(200)
    r = paired_bootstrap_diff(a, a.copy())
    assert r["mean_diff"] == 0.0 and not r["ci_excludes_zero"]


def test_paired_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1, 2, 3])


def test_slice_table_refuses_ci_below_min_size():
    vals = [1.0] * (MIN_SLICE_SIZE + 5) + [0.0] * 5
    slices = ["big"] * (MIN_SLICE_SIZE + 5) + ["small"] * 5
    t = slice_table(vals, slices)
    assert t["big"]["ci"] is not None
    assert t["small"]["ci"] is None and "minimum" in t["small"]["note"]
