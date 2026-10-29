"""Shared look for the README figures: dark card, text converted to paths so the
figures render the same on GitHub (which cannot load custom fonts)."""
from pathlib import Path

import matplotlib as mpl

BG, CARD, GRID = "#0B1220", "#111B2E", "#22304A"
TEXT, MUTED = "#E6EDF7", "#8FA3C0"
TEAL, AMBER, ROSE, VIOLET, SKY = "#2DD4BF", "#F5A524", "#FB7185", "#A78BFA", "#38BDF8"

mpl.rcParams.update({
    "svg.fonttype": "path",
    "font.family": ["Inter", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": TEXT, "font.size": 10.5,
})

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
IMG = ROOT / "docs" / "img"


def heading(fig, title, sub=None):
    fig.text(0.02, 0.965, title, fontsize=16, fontweight="bold", va="top")
    if sub:
        fig.text(0.02, 0.895, sub, fontsize=10, color=MUTED, va="top")


def footer(fig, text):
    fig.text(0.02, 0.012, text, fontsize=8.5, color=MUTED, va="bottom", family=["DejaVu Sans Mono"])
