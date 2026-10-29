#!/usr/bin/env bash
# (1) Reward-design ablation (single seed): GRPO from base with an extra penalty for
#     invented values on missing fields, after the plain-score GRPO runs learned to
#     never output null.
# (2) CPU inference benchmark: base fp32, SFT fp32, SFT dynamic int8.
set -euo pipefail
python -m scripts.train_grpo --seed 0 --max-steps 60 --n-prompts 500 --lr 5e-5 --invent-penalty 2.0 \
    --out checkpoints/grpo_base_penalty_seed0
python -m scripts.eval_model --name grpo_base_penalty_seed0 --adapter checkpoints/grpo_base_penalty_seed0 --n 400
cp checkpoints/grpo_base_penalty_seed0/train_meta.json results/grpo_base_penalty_seed0/train_meta.json
python -m scripts.compare results/grpo_base_seed0 results/grpo_base_penalty_seed0 \
    --out results/comparisons/grpo_base_vs_grpo_base_penalty_seed0.json > /dev/null
python -m scripts.check_ablation_identity
python -m scripts.compare results/baseline results/grpo_base_penalty_seed0 \
    --out results/comparisons/baseline_vs_grpo_base_penalty_seed0.json > /dev/null
python -m scripts.bench_inference --name base_fp32 --quant none
python -m scripts.bench_inference --name sft_seed0_fp32 --adapter checkpoints/sft_seed0 --quant none
python -m scripts.bench_inference --name sft_seed0_int8_dynamic --adapter checkpoints/sft_seed0 --quant int8_dynamic
python -m scripts.bench_inference --name sft_seed0_int8_wo --adapter checkpoints/sft_seed0 --quant int8_wo
python -m scripts.measure_model_size
echo ALL_ABL_BENCH_DONE
