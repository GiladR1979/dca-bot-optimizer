import numpy as np
import argparse

def calc_metrics(deals, equity):
    total_pl = sum(d[2] for d in deals)

    # --- Annualized return based only on realized balance up to the last closed deal ---
    initial_balance = equity[0][1]  # starting account value (e.g., 1000 USD)

    if deals:
        first_open = deals[0][0]          # timestamp of the first deal open
        last_close = deals[-1][1]         # timestamp of the last deal close

        # Find the equity value at (or just before) the last deal close
        bal_end = next(val for t, val in reversed(equity) if t <= last_close)

        years = (last_close - first_open) / (365 * 24 * 3600)
    else:
        # Fallback when no deals are present
        bal_end = equity[-1][1]
        years = (equity[-1][0] - equity[0][0]) / (365 * 24 * 3600)

    roi_pct = (bal_end - initial_balance) / initial_balance * 100
    annual_pct = ((1 + roi_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    annual_usd = initial_balance * annual_pct / 100

    # Other metrics calculations (unchanged)
    num_deals = len(deals)
    wins = [d[2] for d in deals if d[2] > 0]
    losses = [d[2] for d in deals if d[2] <= 0]
    win_rate = len(wins) / num_deals * 100 if num_deals > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    max_drawdown = 0  # placeholder for max drawdown calculation

    return {
        "total_pl": total_pl,
        "roi_pct": roi_pct,
        "annual_pct": annual_pct,
        "annual_usd": annual_usd,
        "num_deals": num_deals,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
    }

parser = argparse.ArgumentParser(description="Run one DCA back‑test")
parser.add_argument("--symbol", required=True, help="Trading pair, e.g. SOLUSDT")
parser.add_argument("--csv", required=True, help="Path to 1‑minute OHLC CSV")
parser.add_argument("--fast-ema", type=int, default=None,
                    help="Fast EMA length for trend filter (optional)")
parser.add_argument("--slow-ema", type=int, default=None,
                    help="Slow EMA length for trend filter (optional)")
# add any other existing parameters here …

args = parser.parse_args()

# -------- EMA sanity checks -----------
if (args.fast_ema is None) ^ (args.slow_ema is None):
    parser.error("Both --fast-ema and --slow-ema must be given together")
if args.fast_ema and args.slow_ema and args.fast_ema >= args.slow_ema:
    parser.error("--fast-ema must be smaller than --slow-ema")
