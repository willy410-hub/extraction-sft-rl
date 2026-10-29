#!/usr/bin/env bash
# Held-out-format evaluation (test_ood, 200 items = 50 per slice) for baseline and SFT seeds,
# plus a validation-split error analysis of SFT seed 0 used to choose the RL prompt focus.
set -euo pipefail
python -m scripts.eval_model --name baseline_ood --split test_ood --n 200
for seed in 0 1 2; do
  python -m scripts.eval_model --name "sft_seed${seed}_ood" --split test_ood --n 200 \
      --adapter "checkpoints/sft_seed${seed}"
  python -m scripts.compare results/baseline_ood "results/sft_seed${seed}_ood" \
      --out "results/comparisons/baseline_vs_sft_seed${seed}_ood.json" > /dev/null
done
python -m scripts.eval_model --name sft_seed0_val --split val --n 200 --adapter checkpoints/sft_seed0
echo ALL_OOD_DONE
