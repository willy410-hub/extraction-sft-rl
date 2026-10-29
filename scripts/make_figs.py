"""README figures. Every plotted number is read from the committed files under results/.

    python -m scripts.make_figs        # writes docs/img/*.svg
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figstyle import (AMBER, BG, CARD, GRID, IMG, MUTED, RES, ROSE, SKY, TEAL, TEXT, VIOLET,  # noqa: E402
                      footer, heading)

SLICES = ["clean", "encoding", "missing_field", "renamed_field"]
SLICE_LABELS = ["clean", "encoding", "missing\nfield", "renamed\nfield", "overall"]
SEEDS = [0, 1, 2]


def summary(run):
    return json.loads((RES / run / "summary.json").read_text())


def train_meta(run):
    return json.loads((RES / run / "train_meta.json").read_text())


def exact_by_slice(run):
    s = summary(run)
    return [s["slices_exact_match"][k]["mean"] for k in SLICES] + [s["exact_match"]["mean"]]


def save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.svg")
    plt.close(fig)


# ------------------------------------------------------------------ banner
def banner():
    fig = plt.figure(figsize=(12.8, 3.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(0, 34); ax.axis("off")
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[0, 128, 0, 34], aspect="auto", zorder=0,
              cmap=__import__("matplotlib").colors.LinearSegmentedColormap.from_list("g", [BG, "#10233F", "#0B3B47"]))
    ax.text(5, 23, "extraction-sft-rl", fontsize=44, fontweight="heavy", color=TEXT, va="center")
    ax.text(5.4, 13.6, "Training a 0.5B model to turn messy text into strict JSON", fontsize=14, color="#C7D4E8", va="center")
    ax.text(5.4, 9.6, "SFT (LoRA)  ·  GRPO with a rule-based verifier  ·  reward-hacking checks  ·  inference benchmark",
            fontsize=10.5, color=MUTED, va="center")
    for i, (t, c) in enumerate([("SFT", TEAL), ("GRPO", AMBER), ("verifier", SKY), ("benchmark", VIOLET)]):
        x0 = 5.4 + i * 14
        ax.add_patch(FancyBboxPatch((x0, 2.4), 12.5, 3.2, boxstyle="round,pad=0.02,rounding_size=1.4",
                                    fc="none", ec=c, lw=1.6))
        ax.text(x0 + 6.25, 4.0, t, ha="center", va="center", fontsize=9.5, color=c, fontweight="bold",
                family=["DejaVu Sans Mono"])
    # messy text -> JSON motif
    mono = ["DejaVu Sans Mono"]
    ax.add_patch(FancyBboxPatch((80, 15), 18, 13, boxstyle="round,pad=0.4,rounding_size=1.2", fc=CARD, ec=ROSE, lw=1.4))
    for j, line in enumerate(["Bill No. INV-73465", "Client: BjÃ¶rn Ã…strÃ¶m", "amt=2,205.45 usd", "date 01/04/24 ??"]):
        ax.text(81.2, 25.8 - j * 3, line, fontsize=7.8, color="#C7D4E8", family=mono, va="center")
    ax.annotate("", xy=(104.5, 21.5), xytext=(99.5, 21.5),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2))
    ax.add_patch(FancyBboxPatch((106, 13), 18, 17, boxstyle="round,pad=0.4,rounding_size=1.2", fc=CARD, ec=TEAL, lw=1.4))
    for j, line in enumerate(['{', ' "invoice_id": "INV-73465",', ' "amount": 2205.45,', ' "currency": "USD",',
                              ' "customer_id": null,', ' ...', '}']):
        ax.text(107, 28.4 - j * 2.35, line, fontsize=7.4, color=TEAL if j in (0, 6) else "#C7D4E8", family=mono, va="center")
    save(fig, "banner")


# ------------------------------------------------------------------ slices
def slices():
    base = exact_by_slice("baseline")
    sft = np.array([exact_by_slice(f"sft_seed{s}") for s in SEEDS])
    grpo = np.array([exact_by_slice(f"grpo_base_seed{s}") for s in SEEDS])

    fig = plt.figure(figsize=(10.5, 5.0), dpi=100)
    heading(fig, "SFT fixed every slice. GRPO from the base model broke one.",
            "Exact match (all 6 fields right) on the 400-item test split. Bars = mean of 3 training seeds, dots = each seed.")
    ax = fig.add_axes([0.07, 0.17, 0.9, 0.6])
    x = np.arange(len(SLICE_LABELS)); w = 0.26
    groups = [("untuned base", VIOLET, np.array([base]), -w), ("SFT, 3 seeds", TEAL, sft, 0), ("GRPO from base, 3 seeds", AMBER, grpo, w)]
    for label, col, vals, off in groups:
        m = vals.mean(axis=0)
        ax.bar(x + off, m, width=w - 0.03, color=col, label=label, zorder=2, edgecolor=CARD, linewidth=1)
        if len(vals) > 1:
            for v in vals:
                ax.scatter(x + off, v, s=16, color=TEXT, edgecolor=BG, linewidth=0.8, zorder=3)
    # call out the broken slice
    i = SLICES.index("missing_field")
    ax.annotate("GRPO: 0.00 / 0.00 / 0.30, it filled fields that are not in the text",
                xy=(i + w, grpo[:, i].max() + 0.04), xytext=(i + w, 1.14), fontsize=9, color=ROSE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ROSE, lw=1.1))
    ax.set_xticks(x); ax.set_xticklabels(SLICE_LABELS); ax.set_ylim(0, 1.22); ax.set_ylabel("exact match")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvline(3.5, color=GRID, lw=1); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=9.5)
    footer(fig, "synthetic data  ·  greedy decoding  ·  results/*/summary.json  ·  scripts/make_figs.py")
    save(fig, "slices")


# ------------------------------------------------------------------ reward hacking
def hacking():
    fig = plt.figure(figsize=(10.5, 4.7), dpi=100)
    heading(fig, "The training reward went up while the model learned to invent values",
            "Left: logged GRPO reward per 5-step window. Right: share of missing fields the model filled anyway (not part of the reward).")
    ax = fig.add_axes([0.07, 0.18, 0.4, 0.58])
    styles = ["-", "--", ":"]
    ends = []
    for s, ls in zip(SEEDS, styles):
        h = [r for r in train_meta(f"grpo_base_seed{s}")["log_history"] if "reward" in r]
        steps = [r["step"] for r in h]; rew = [r["reward"] for r in h]
        ax.plot(steps, rew, color=AMBER, lw=2, ls=ls, marker="o", ms=4, zorder=3)
        ends.append((rew[-1], s, steps[-1]))
    last_y = None
    for y, s, x_end in sorted(ends):  # spread end labels so they do not overlap
        y_lab = y if last_y is None else max(y, last_y + 0.045)
        ax.text(x_end + 1.5, y_lab, f"seed {s}", color=TEXT, fontsize=8.5, va="center")
        last_y = y_lab
    ax.set_xlim(0, 68); ax.set_ylim(0.2, 1.02); ax.set_xlabel("training step"); ax.set_ylabel("mean verifier reward")
    ax.set_title("GRPO from base, 3 seeds", fontsize=11, color=MUTED)

    ax2 = fig.add_axes([0.57, 0.18, 0.4, 0.58])
    rows = [("base", summary("baseline"), VIOLET)] + \
           [(f"SFT s{s}", summary(f"sft_seed{s}"), TEAL) for s in SEEDS] + \
           [(f"GRPO s{s}", summary(f"grpo_base_seed{s}"), AMBER) for s in SEEDS]
    vals = [r[1]["hacking_checks"]["invented_on_missing_rate"] for r in rows]
    xs = np.arange(len(rows))
    ax2.bar(xs, vals, color=[r[2] for r in rows], width=0.7, zorder=2, edgecolor=CARD, linewidth=1)
    for xi, v in zip(xs, vals):
        ax2.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=8.5, color=TEXT)
    ax2.set_xticks(xs); ax2.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("invented_on_missing_rate"); ax2.grid(axis="x", visible=False)
    ax2.set_title("lower is better", fontsize=11, color=MUTED)
    footer(fig, "60-step CPU runs, 1 prompt x 4 samples per step  ·  results/grpo_base_seed*/train_meta.json")
    save(fig, "reward_hacking")


# ------------------------------------------------------------------ inference
def inference():
    runs = [("SFT fp32", "sft_seed0_fp32", TEAL, "-"), ("SFT int8 weight-only", "sft_seed0_int8_wo", SKY, "--"),
            ("SFT int8 dynamic", "sft_seed0_int8_dynamic", ROSE, ":")]
    data = {k: json.loads((RES / "inference" / f"{k}.json").read_text()) for _, k, _, _ in runs}
    fig = plt.figure(figsize=(10.5, 4.7), dpi=100)
    heading(fig, "Batching is the free win. Dynamic int8 was faster but broke the output.",
            "CPU, 2 threads, no GPU. Left: requests per second by batch size. Right: exact match on 200 test items.")
    ax = fig.add_axes([0.07, 0.18, 0.48, 0.58])
    bs = ["1", "4", "16"]
    for label, k, col, ls in runs:
        y = [data[k]["throughput"][b]["requests_per_s"] for b in bs]
        ax.plot([1, 4, 16], y, color=col, lw=2, ls=ls, marker="o", ms=6, zorder=3)
        ax.text(17, y[-1], label, color=TEXT, fontsize=8.5, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16]); ax.set_xticklabels(["1", "4", "16"])
    ax.set_xlim(0.8, 60); ax.set_ylim(0, 0.65); ax.set_xlabel("batch size"); ax.set_ylabel("requests / second")

    ax2 = fig.add_axes([0.66, 0.18, 0.31, 0.58])
    em = [data[k]["quality"]["exact_match"]["mean"] for _, k, _, _ in runs]
    ys = np.arange(len(runs))[::-1]
    ax2.barh(ys, em, color=[r[2] for r in runs], height=0.6, zorder=2, edgecolor=CARD, linewidth=1)
    for y, v, (label, _, _, _) in zip(ys, em, runs):
        ax2.text(max(v, 0) + 0.03, y, f"{v:.3f}" + ("  (valid JSON 0%)" if v == 0 else ""), va="center", fontsize=8.5, color=TEXT)
    ax2.set_yticks(ys); ax2.set_yticklabels([r[0].replace("SFT ", "") for r in runs], fontsize=9)
    ax2.set_xlim(0, 1.25); ax2.set_xticks([0, 0.5, 1]); ax2.grid(axis="y", visible=False); ax2.set_xlabel("exact match")
    footer(fig, "SFT seed 0  ·  results/inference/*.json  ·  timings on a shared machine are rough")
    save(fig, "inference")


if __name__ == "__main__":
    banner(); slices(); hacking(); inference()
    print("wrote", sorted(p.name for p in IMG.glob("*.svg")))
