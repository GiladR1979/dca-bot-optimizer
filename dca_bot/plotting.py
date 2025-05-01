"""
Plot helpers – capped at 3000 px width and **fixed 4000 px height**
for both single‑panel and triple‑panel outputs.

(Uses headless Agg backend; down‑samples to at most 3000 points so
files render in <2 s even on large data sets.)
"""

from datetime import datetime
import math
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------- #
MAX_W_PX = 3000  # horizontal cap
H_PX = 1688  # **requested fixed height**
DPI = 100
MAX_PTS = MAX_W_PX


# ------------------------------------------------------------------ #
def _downsample(ts: List, val: List) -> Tuple[List, List]:
    """Return strided ts/val so len ≤ MAX_PTS."""
    n = len(ts)
    if n <= MAX_PTS:
        return ts, val
    step = math.ceil(n / MAX_PTS)
    return ts[::step], val[::step]


# ------------------------------------------------------------------ #
def _fig_size():
    """Return (width_inch, height_inch) for the fixed pixel caps."""
    return MAX_W_PX / DPI, H_PX / DPI


# ------------------------------------------------------------------ #
def equity_curve(equity, deals, title, path):
    ts = [datetime.fromtimestamp(t) for t, _ in equity]
    val = [v for _, v in equity]
    ts, val = _downsample(ts, val)

    lookup = dict(equity)

    fig = plt.figure(figsize=_fig_size(), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, val, linewidth=0.8)

    for _, e, _, _ in deals:
        y = lookup.get(e)
        if y is not None:
            ax.scatter(datetime.fromtimestamp(e), y,
                       marker="v", s=12, color="red")

    ax.set_title(title)
    ax.set_ylabel("USD")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ #
def panel(items: List[Tuple], path: str):
    n = len(items)
    fig_w, fig_h = _fig_size()
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True, dpi=DPI)

    if n == 1:
        axes = [axes]

    for ax, (equity, deals, label) in zip(axes, items):
        ts = [datetime.fromtimestamp(t) for t, _ in equity]
        val = [v for _, v in equity]
        ts, val = _downsample(ts, val)

        ax.plot(ts, val, linewidth=0.8)
        lookup = dict(equity)
        for _, e, _, _ in deals:
            y = lookup.get(e)
            if y is not None:
                ax.scatter(datetime.fromtimestamp(e), y,
                           marker="v", s=12, color="red")
        ax.set_title(label)
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    if n > 1:
        axes[n // 2].set_ylabel("USD")

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
