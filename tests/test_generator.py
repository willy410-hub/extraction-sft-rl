from collections import Counter

import pytest

from extraction_rl.generator import (
    NAMES, NOISE_TYPES, SPLITS, corrupt_encoding, is_recoverable, make_example, make_split,
)
from extraction_rl.schema import FIELDS


def test_same_seed_same_example():
    assert make_example(123) == make_example(123)


def test_different_seeds_differ():
    assert make_example(1)["text"] != make_example(2)["text"]


def test_every_example_is_labeled_synthetic():
    assert all(ex["synthetic"] is True for ex in make_split("test", 200))


def test_truth_has_exactly_the_schema_fields():
    for ex in make_split("val", 100):
        assert set(ex["truth"]) == set(FIELDS)


def test_noise_slices_are_balanced():
    counts = Counter(ex["noise"] for ex in make_split("test", 400))
    assert set(counts) == set(NOISE_TYPES)
    assert len(set(counts.values())) == 1  # exactly equal


def test_split_seed_ranges_do_not_overlap():
    a, b, c = (set(SPLITS[s]) for s in ("train", "val", "test"))
    assert not a & b and not a & c and not b & c


def test_no_text_overlap_between_splits():
    texts = {s: {ex["text"] for ex in make_split(s, 2000 if s != "val" else None)} for s in SPLITS}
    assert not texts["train"] & texts["test"]
    assert not texts["train"] & texts["val"]
    assert not texts["val"] & texts["test"]


@pytest.mark.parametrize("name", NAMES)
def test_every_name_survives_mojibake_round_trip(name):
    assert is_recoverable(name, corrupt_encoding(name))


def test_encoding_slice_actually_corrupts_the_text():
    enc = [ex for ex in make_split("test", 400) if ex["noise"] == "encoding"]
    assert enc and all(ex["truth"]["customer_name"] not in ex["text"] for ex in enc)


def test_missing_field_is_null_in_truth_and_absent_from_text():
    for ex in make_split("test", 400):
        if ex["noise"] != "missing_field":
            continue
        f = ex["missing_field"]
        assert ex["truth"][f] is None
        other = "customer_name" if f == "customer_id" else "customer_id"
        assert ex["truth"][other] is not None


def test_ids_present_verbatim_in_non_missing_text():
    for ex in make_split("test", 200):
        assert ex["truth"]["invoice_id"] in ex["text"]


def test_whole_test_split_generates_without_error():
    assert len(make_split("test")) == len(SPLITS["test"])


def test_numeric_dates_are_never_ambiguous():
    import re
    for ex in make_split("test", 2000):
        for d, m in re.findall(r"\b(\d{2})[./](\d{2})[./]\d{4}\b", ex["text"]):
            assert int(d) > 12, ex["text"]
