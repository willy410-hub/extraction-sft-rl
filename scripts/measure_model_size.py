"""Serialized model size for each benchmark configuration (same function as bench_inference).

Each configuration is measured in its own subprocess so only one model is in memory at a time.
"""
import json
import subprocess
import sys

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIGS = [("base_fp32", "", "none"), ("sft_seed0_fp32", "checkpoints/sft_seed0", "none"),
           ("sft_seed0_int8_dynamic", "checkpoints/sft_seed0", "int8_dynamic"),
           ("sft_seed0_int8_wo", "checkpoints/sft_seed0", "int8_wo")]


def measure_one(adapter: str, quant: str) -> int:
    from scripts.bench_inference import load, model_bytes

    model, _ = load(MODEL, adapter or None, quant)
    return model_bytes(model)


if __name__ == "__main__":
    if len(sys.argv) == 3:  # child process: print one number
        print("BYTES", measure_one(sys.argv[1], sys.argv[2]))
        sys.exit(0)
    out = {}
    for name, adapter, quant in CONFIGS:
        res = subprocess.run([sys.executable, "-m", "scripts.measure_model_size", adapter, quant],
                             capture_output=True, text=True, check=True)
        n = int(next(line for line in res.stdout.splitlines() if line.startswith("BYTES")).split()[1])
        out[name] = {"quant": quant, "serialized_state_dict_bytes": n}
        print(name, out[name])
    with open("results/inference/model_sizes.json", "w") as f:
        json.dump(out, f, indent=2)
