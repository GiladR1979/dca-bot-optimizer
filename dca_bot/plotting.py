"""
Plot helpers: single equity curve + flexible N-panel comparison.
"""

from datetime import datetime
import matplotlib.pyplot as plt


def equity_curve(equity, deals, title, path):
    """
    Save a single-panel equity curve PNG with ▼ sell markers.
    """
    ts = [datetime.fromtimestamp(t) for t, _ in equity]
    val = [v for _, v in equity]
    lookup = dict(equity)

    plt.figure(figsize=(10, 4))
    plt.plot(ts, val, label="Equity")
    for _, e, _, _ in deals:
        y = lookup.get(e)
        if y is not None:
            plt.scatter(datetime.fromtimestamp(e), y,
                        marker="v", color="red")
    plt.title(title)
    plt.ylabel("USD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def panel(items, path):
    """
    Draw N stacked panels in a single figure.

    Parameters
    ----------
    items : list of (equity, deals, label)
        equity : list[(epoch, equity_value)]
        deals  : list of trades (entry_ts, exit_ts, ...)
        label  : str – panel title
    path  : str – PNG output path
    """
    n = len(items)
    fig, axes = plt.subplots(n, 1,
                             figsize=(10, 4 * n),
                             sharex=True)

    # axes is not iterable when n == 1
    if n == 1:
        axes = [axes]

    for ax, (equity, deals, label) in zip(axes, items):
        ts = [datetime.fromtimestamp(t) for t, _ in equity]
        val = [v for _, v in equity]
        ax.plot(ts, val)
        lookup = dict(equity)
        for _, e, _, _ in deals:
            y = lookup.get(e)
            if y is not None:
                ax.scatter(datetime.fromtimestamp(e), y,
                           marker="v", color="red")
        ax.set_title(label)
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    if n > 1:
        axes[n // 2].set_ylabel("USD")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
