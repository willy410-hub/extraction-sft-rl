#!/usr/bin/env bash
# CPU smoke-scale GRPO runs used for the committed v0.3 results.
# (1) GRPO from the SFT seed-0 model, focused on the encoding slice (the weakest
#     slice on the VALIDATION split): measures how much learning signal is left.
# (2) GRPO from the untuned base model, 3 seeds x 60 steps x 1 prompt x 4 samples.
set -euo pipefail
mkdir -p results/comparisons
python -m scripts.train_grpo --init-adapter checkpoints/sft_seed0 --seed 0 --max-steps 15 \
    --n-prompts 200 --noise-filter encoding --lr 5e-5 --out checkpoints/grpo_from_sft_seed0
mkdir -p results/grpo_from_sft_seed0
cp checkpoints/grpo_from_sft_seed0/train_meta.json results/grpo_from_sft_seed0/train_meta.json
for seed in 0 1 2; do
  python -m scripts.train_grpo --seed "$seed" --max-steps 60 --n-prompts 500 --lr 5e-5 \
      --out "checkpoints/grpo_base_seed${seed}"
  python -m scripts.eval_model --name "grpo_base_seed${seed}" --adapter "checkpoints/grpo_base_seed${seed}" --n 400
  cp "checkpoints/grpo_base_seed${seed}/train_meta.json" "results/grpo_base_seed${seed}/train_meta.json"
  python -m scripts.compare results/baseline "results/grpo_base_seed${seed}" \
      --out "results/comparisons/baseline_vs_grpo_base_seed${seed}.json" > /dev/null
  python -m scripts.compare "results/grpo_base_seed${seed}" "results/sft_seed${seed}" \
      --out "results/comparisons/grpo_base_vs_sft_seed${seed}.json" > /dev/null
done
echo ALL_GRPO_DONE
