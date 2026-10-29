"""Guards on what the training scripts feed the model (no model is loaded here)."""
import json

from extraction_rl.generator import SPLITS
from extraction_rl.prompts import target_json
from scripts.train_grpo import build_dataset as grpo_dataset
from scripts.train_grpo import verifier_reward
from scripts.train_sft import build_dataset as sft_dataset


def test_sft_targets_are_canonical_json():
    ds = sft_dataset(20, seed=0)
    for row in ds:
        json.loads(row["completion"][0]["content"])  # parses
        assert row["prompt"][0]["role"] == "system"


def test_sft_dataset_is_seed_dependent_but_deterministic():
    a, b, c = sft_dataset(10, 0), sft_dataset(10, 0), sft_dataset(10, 1)
    assert a["completion"] == b["completion"]
    assert a["completion"] != c["completion"]


def test_grpo_noise_filter_keeps_only_that_slice():
    ds = grpo_dataset(30, seed=0, noise_filter="encoding")
    assert len(ds) == 30
    # encoding slice texts contain mojibake markers
    assert all("Ã" in r["prompt"][1]["content"] or "Ø" in r["prompt"][1]["content"] or "Ù" in r["prompt"][1]["content"]
               for r in ds)


def test_training_texts_never_come_from_test_split():
    from extraction_rl.generator import make_split

    test_texts = {e["text"] for e in make_split("test")}
    for ds in (sft_dataset(500, 0), grpo_dataset(500, 0)):
        assert not {r["prompt"][1]["content"] for r in ds} & test_texts


def test_verifier_reward_matches_verifier():
    ds = grpo_dataset(3, seed=0)
    truths = [json.loads(r["truth_json"]) for r in ds]
    perfect = [[{"role": "assistant", "content": target_json(t)}] for t in truths]
    garbage = [[{"role": "assistant", "content": "no idea"}] for _ in truths]
    assert verifier_reward(perfect, ds["truth_json"]) == [1.0, 1.0, 1.0]
    assert verifier_reward(garbage, ds["truth_json"]) == [0.0, 0.0, 0.0]


def test_split_ranges_cover_training_sources():
    assert max(SPLITS["train"]) < min(SPLITS["val"]) < min(SPLITS["test"])
