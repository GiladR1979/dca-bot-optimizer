"""
Plot helpers – fast & size-capped (≤ 3000 px width).

Changes versus the original:
• dpi is fixed at 100  → width_px = dpi * width_inch ≤ 3000.
• Equity arrays are down-sampled so we never plot >3000 points.
"""

from datetime import datetime
import math
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")                # headless, slightly faster
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
MAX_PX    = 3000                     # hard pixel-width cap
DPI       = 100                      # 100 dpi  ⇒  width_inch = 30
MAX_PTS   = MAX_PX                   # no more points than horizontal pixels


# ------------------------------------------------------------------ #
def _downsample(ts: List, val: List) -> Tuple[List, List]:
    """Return strided ts/val so len ≤ MAX_PTS."""
    n = len(ts)
    if n <= MAX_PTS:
        return ts, val
    step = math.ceil(n / MAX_PTS)
    return ts[::step], val[::step]


# ------------------------------------------------------------------ #
def equity_curve(equity, deals, title, path):
    """
    Save a single-panel equity curve PNG with ▼ sell markers,
    limited to ~3000 px wide and at most 3000 data points.
    """
    ts = [datetime.fromtimestamp(t) for t, _ in equity]
    val = [v for _, v in equity]
    ts, val = _downsample(ts, val)         # ↓ speed-boost

    lookup = dict(equity)

    # figure inches so width_px = 3000
    fig_w = MAX_PX / DPI
    fig = plt.figure(figsize=(fig_w, 4), dpi=DPI)
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
    """
    Draw N stacked equity panels, each capped at 3000 px width
    and down-sampled to ≤3000 points.
    """
    n = len(items)
    fig_w = MAX_PX / DPI
    fig_h = 4 * n
    fig, axes = plt.subplots(
        n, 1, figsize=(fig_w, fig_h), sharex=True, dpi=DPI
    )

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
