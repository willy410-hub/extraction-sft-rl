"""Inference benchmark: quality vs. latency / throughput / memory, with and without quantization.

Quantization modes:
  none          full precision (fp32 on CPU, bf16/fp16 on GPU)
  int8_dynamic  CPU only: torch dynamic int8 quantization of every nn.Linear, i.e.
                int8 weights AND per-tensor int8 activations
                (torch.ao.quantization.quantize_dynamic -- deprecated upstream in
                favor of torchao). Kept as a documented negative result: on this
                model it destroyed output quality (exact match 0.99 -> 0.00).
  int8_wo       int8 weight-only quantization via torchao (Int8WeightOnlyConfig);
                activations stay in full precision
  bnb8 / bnb4   GPU only: bitsandbytes 8-bit / 4-bit (NF4) weight loading

Measures, on the same synthetic test items:
  - quality (field score, exact match) via the verifier
  - latency at batch size 1 (median and p90 per request)
  - throughput (requests/s, generated tokens/s) at several batch sizes
  - memory: parameter+buffer bytes, and peak process RSS (CPU) or peak CUDA memory (GPU)
  - cost per 1,000 requests ONLY if --usd-per-hour is given (no price is assumed)

Usage:
  python -m scripts.bench_inference --name sft_fp32 --adapter checkpoints/sft_seed0 --quant none
  python -m scripts.bench_inference --name sft_int8 --adapter checkpoints/sft_seed0 --quant int8_dynamic
"""
import argparse
import json
import os
import platform
import resource
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extraction_rl.evaluate import score_outputs, summarize
from extraction_rl.generator import make_split
from extraction_rl.inference import DEFAULT_MODEL, generate


def load(model_name, adapter, quant):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    cuda = torch.cuda.is_available()
    kwargs = {}
    if quant in ("bnb8", "bnb4"):
        if not cuda:
            raise SystemExit("bitsandbytes modes need a CUDA GPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quant == "bnb8" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=torch.float16))
        kwargs["device_map"] = "auto"
    if adapter and quant in ("bnb8", "bnb4"):
        # Merge adapters in full precision first, then quantize the merged weights.
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
        from peft import PeftModel

        for path in adapter.split(","):
            base = PeftModel.from_pretrained(base, path).merge_and_unload()
        tmp = "/tmp/merged_for_quant"
        base.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        model = AutoModelForCausalLM.from_pretrained(tmp, **kwargs)
    else:
        dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
        if adapter:
            from peft import PeftModel

            for path in adapter.split(","):
                model = PeftModel.from_pretrained(model, path).merge_and_unload()
        if quant == "int8_dynamic":
            if cuda:
                raise SystemExit("int8_dynamic is a CPU mode")
            from torch.ao.quantization import quantize_dynamic

            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        if quant == "int8_wo":
            from torchao.quantization import Int8WeightOnlyConfig, quantize_

            quantize_(model, Int8WeightOnlyConfig())
        if "device_map" not in kwargs:
            model.to("cuda" if cuda else "cpu")
    return model.eval(), tok


def model_bytes(model) -> int:
    """Serialized state_dict size in bytes. Works for fp32/fp16 and for dynamic-int8
    models, whose packed int8 weights are not visible through .parameters()."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--quant", default="none", choices=["none", "int8_dynamic", "int8_wo", "bnb8", "bnb4"])
    p.add_argument("--n-quality", type=int, default=200)
    p.add_argument("--n-latency", type=int, default=20)
    p.add_argument("--batch-sizes", default="1,4,16")
    p.add_argument("--usd-per-hour", type=float, default=None)
    p.add_argument("--out-dir", default="results/inference")
    args = p.parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model, tok = load(args.model, args.adapter, args.quant)

    examples = make_split("test", args.n_quality)
    texts = [e["text"] for e in examples]

    # warm-up
    generate(model, tok, texts[:2], batch_size=2)

    # latency at batch size 1
    lat = []
    for t in texts[: args.n_latency]:
        t0 = time.perf_counter()
        generate(model, tok, [t], batch_size=1)
        lat.append(time.perf_counter() - t0)

    # throughput per batch size (quality computed from the largest batch run)
    throughput, outputs = {}, None
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        n = min(len(texts), max(bs * 4, args.n_latency))
        outs, st = generate(model, tok, texts[:n], batch_size=bs)
        throughput[str(bs)] = {"n": n, "requests_per_s": st["examples_per_s"], "tokens_per_s": st["tokens_per_s"]}
    outputs, _ = generate(model, tok, texts, batch_size=max(int(x) for x in args.batch_sizes.split(",")))

    rows = score_outputs(examples, outputs)
    s = summarize(examples, rows)
    best_rps = max(v["requests_per_s"] for v in throughput.values())
    res = {
        "name": args.name, "model": args.model, "adapter": args.adapter, "quant": args.quant,
        "data": "synthetic test split",
        "quality": {"n": s["n"], "field_score": s["field_score"], "exact_match": s["exact_match"],
                    "valid_json_rate": s["valid_json_rate"]},
        "latency_batch1_s": {"n": len(lat), "median": statistics.median(lat),
                             "p90": sorted(lat)[int(0.9 * (len(lat) - 1))]},
        "throughput": throughput,
        "memory": {"model_bytes": model_bytes(model),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "cost_per_1k_requests_usd": (args.usd_per_hour / 3600 * 1000 / best_rps) if args.usd_per_hour else None,
        "cost_note": "computed from --usd-per-hour at the best measured throughput" if args.usd_per_hour
                     else "not computed: no hardware price given (see README for the formula)",
        "hardware": {"device": "cuda" if cuda else "cpu",
                     "gpu": torch.cuda.get_device_name(0) if cuda else None,
                     "cpu_threads": torch.get_num_threads(), "platform": platform.platform()},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
