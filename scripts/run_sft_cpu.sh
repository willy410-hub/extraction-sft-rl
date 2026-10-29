#!/usr/bin/env bash
# CPU smoke-scale SFT run used for the committed v0.2 results:
# 3 training seeds x 50 optimizer steps x 8 examples/step (400 synthetic examples each).
set -euo pipefail
for seed in 0 1 2; do
  python -m scripts.train_sft --seed "$seed" --n-train 400 --max-steps 50 \
      --batch-size 2 --grad-accum 4 --out "checkpoints/sft_seed${seed}"
  python -m scripts.eval_model --name "sft_seed${seed}" --adapter "checkpoints/sft_seed${seed}" --n 400
  mkdir -p results/comparisons
  python -m scripts.compare results/baseline "results/sft_seed${seed}" \
      --out "results/comparisons/baseline_vs_sft_seed${seed}.json" > /dev/null
  cp "checkpoints/sft_seed${seed}/train_meta.json" "results/sft_seed${seed}/train_meta.json"
done
echo ALL_SFT_DONE
