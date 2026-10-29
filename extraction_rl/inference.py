"""Batched greedy generation shared by evaluation and benchmarking."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extraction_rl.prompts import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(model_name: str = DEFAULT_MODEL, adapter: str | None = None, dtype=None):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        # "a,b" applies adapter a, merges it, then adapter b (e.g. SFT then GRPO).
        for path in adapter.split(","):
            model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def prompt_text(tok, text: str) -> str:
    return tok.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tok, texts: list[str], batch_size: int = 16, max_new_tokens: int = 128):
    """Greedy decoding (deterministic). Returns (outputs, stats)."""
    outputs, n_new_tokens = [], 0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        prompts = [prompt_text(tok, t) for t in texts[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        n_new_tokens += int((new != tok.pad_token_id).sum())
        outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
    elapsed = time.perf_counter() - t0
    return outputs, {"seconds": elapsed, "n": len(texts), "new_tokens": n_new_tokens,
                     "examples_per_s": len(texts) / elapsed, "tokens_per_s": n_new_tokens / elapsed}
