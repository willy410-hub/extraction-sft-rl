"""Supervised fine-tuning with LoRA on synthetic (text -> canonical JSON) pairs.

Training on known-correct target answers is behavior cloning: the model imitates
reference outputs. Loss is computed on the completion (the JSON answer) only.

Usage (GPU, e.g. Kaggle T4):
  python -m scripts.train_sft --seed 0 --n-train 4000 --epochs 1 --out checkpoints/sft_seed0
Usage (CPU smoke-scale, what this repo's committed results used -- see README):
  python -m scripts.train_sft --seed 0 --n-train 400 --max-steps 100 --out checkpoints/sft_seed0
"""
import argparse
import json
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from extraction_rl.generator import SPLITS, make_example
from extraction_rl.inference import DEFAULT_MODEL
from extraction_rl.prompts import build_messages, target_json


def build_dataset(n: int, seed: int) -> Dataset:
    """Sample n training seeds (from the train range only), shuffled by `seed`."""
    seeds = list(SPLITS["train"])
    random.Random(seed).shuffle(seeds)
    rows = []
    for s in seeds[:n]:
        ex = make_example(s)
        rows.append({
            "prompt": build_messages(ex["text"]),
            "completion": [{"role": "assistant", "content": target_json(ex["truth"])}],
        })
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--grad-ckpt", action="store_true", help="trade speed for memory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = build_dataset(args.n_train, args.seed)
    cuda = torch.cuda.is_available()

    config = SFTConfig(
        output_dir=args.out,
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=384,
        gradient_checkpointing=args.grad_ckpt,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
        use_cpu=not cuda,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = SFTTrainer(model=args.model, args=config, train_dataset=ds, peft_config=peft_config)
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    trainer.save_model(args.out)

    meta = {"stage": "sft", "seed": args.seed, "n_train": args.n_train, "max_steps": args.max_steps,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "lora_r": args.lora_r, "grad_ckpt": args.grad_ckpt, "train_seconds": elapsed,
            "device": "cuda" if cuda else "cpu", "log_history": trainer.state.log_history}
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {elapsed:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
