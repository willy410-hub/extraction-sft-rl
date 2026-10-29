from collections import Counter

from extraction_rl.generator import SPLITS, make_split
from extraction_rl.ood import OOD_RENAMED, OOD_SEEDS, make_ood_split
from extraction_rl.prompts import target_json
from extraction_rl.verifier import verify


def test_ood_seeds_do_not_overlap_other_splits():
    for r in SPLITS.values():
        assert not set(OOD_SEEDS) & set(r)


def test_ood_label_words_never_appear_in_training_text():
    train_text = " ".join(ex["text"] for ex in make_split("train", 3000))
    for label in OOD_RENAMED.values():
        assert label not in train_text, label


def test_ood_slices_balanced_and_synthetic():
    exs = make_ood_split()
    assert len(set(Counter(e["noise"] for e in exs).values())) == 1
    assert all(e["synthetic"] for e in exs)


def test_ood_canonical_answer_scores_one():
    for ex in make_ood_split(100):
        assert verify(target_json(ex["truth"]), ex["truth"])["score"] == 1.0


def test_ood_encoding_slice_is_corrupted():
    enc = [e for e in make_ood_split() if e["noise"] == "encoding"]
    assert all(e["truth"]["customer_name"] not in e["text"] for e in enc)
