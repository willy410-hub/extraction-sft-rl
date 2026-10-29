"""Evaluate a model (base, or base + LoRA adapter) on the synthetic test split.

Writes:
  results/<name>/predictions.jsonl  -- one row per test item (for paired comparison)
  results/<name>/summary.json       -- metrics, slices, hacking checks, throughput

Usage:
  python -m scripts.eval_model --name baseline --n 400
  python -m scripts.eval_model --name sft_seed0 --adapter checkpoints/sft_seed0
"""
import argparse
import json
import os
import platform

import torch

from extraction_rl.evaluate import score_outputs, summarize
from extraction_rl.generator import make_split
from extraction_rl.inference import DEFAULT_MODEL, generate, load_model
from extraction_rl.ood import make_ood_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n", type=int, default=400, help="test items (first n of the test seed range)")
    p.add_argument("--split", default="test", choices=["test", "val", "test_ood"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    examples = make_ood_split(args.n) if args.split == "test_ood" else make_split(args.split, args.n)
    model, tok = load_model(args.model, adapter=args.adapter)
    outputs, speed = generate(model, tok, [e["text"] for e in examples], batch_size=args.batch_size)

    rows = score_outputs(examples, outputs)
    summary = summarize(examples, rows)
    summary.update({
        "name": args.name, "model": args.model, "adapter": args.adapter,
        "decoding": "greedy (deterministic)", "data": f"synthetic {args.split} split",
        "throughput": speed,
        "hardware": {"device": "cuda" if torch.cuda.is_available() else "cpu",
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "cpu_threads": torch.get_num_threads(), "platform": platform.platform()},
    })

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: summary[k] for k in ("n", "field_score", "exact_match", "valid_json_rate",
                                              "schema_ok_rate", "hacking_checks")}, indent=2))


if __name__ == "__main__":
    main()
