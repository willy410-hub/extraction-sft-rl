"""Compare the plain-reward and penalized-reward GRPO runs (seed 0): weight difference
and whether any test prediction changed. Writes results/comparisons/penalty_ablation_identity.json."""
import json

from safetensors.torch import load_file

A, B = "checkpoints/grpo_base_seed0", "checkpoints/grpo_base_penalty_seed0"

if __name__ == "__main__":
    wa, wb = load_file(f"{A}/adapter_model.safetensors"), load_file(f"{B}/adapter_model.safetensors")
    pa = [json.loads(x)["output"] for x in open("results/grpo_base_seed0/predictions.jsonl")]
    pb = [json.loads(x)["output"] for x in open("results/grpo_base_penalty_seed0/predictions.jsonl")]
    out = {
        "max_abs_weight_diff": max((wa[k] - wb[k]).abs().max().item() for k in wa),
        "max_abs_weight": max(wa[k].abs().max().item() for k in wa),
        "n_lora_tensors": len(wa),
        "n_test_predictions": len(pa),
        "n_predictions_identical": sum(x == y for x, y in zip(pa, pb)),
    }
    with open("results/comparisons/penalty_ablation_identity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
