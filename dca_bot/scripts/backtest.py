import numpy as np

def calc_metrics(deals, equity):
    closed = [d for d in deals if d[1] is not None]  # realised (closed) deals only
    total_pl = sum(d[2] for d in closed)

    # --- Annualized return based only on realized balance up to the last closed deal ---
    initial_balance = equity[0][1]  # starting account value (e.g., 1000 USD)

    if closed:
        first_open = closed[0][0]
        last_close = closed[-1][1]
        # Equity value at or just before the last closed deal
        bal_end = next(val for t, val in reversed(equity) if t <= last_close)
        years = (last_close - first_open) / (365 * 24 * 3600)
    else:
        # Fallback when nothing closed yet
        bal_end = equity[-1][1]
        years = (equity[-1][0] - equity[0][0]) / (365 * 24 * 3600)

    # Ensure 'years' is never zero (or extremely small) to avoid overflow/NaN
    years = max(years, 1 / 365)  # floor at one day

    roi_pct = (bal_end - initial_balance) / initial_balance * 100
    annual_pct = roi_pct / years if years > 0 else 0  # linear APR (no compounding)
    annual_usd = initial_balance * annual_pct / 100

    wins = [d[2] for d in closed if d[2] > 0]
    losses = [d[2] for d in closed if d[2] <= 0]
    num_deals = len(closed)
    win_rate = len(wins) / num_deals * 100 if num_deals > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Max draw‑down (includes last open position)
    peak = equity[0][1]
    max_drawdown = 0.0
    for _, bal in equity:
        if bal > peak:
            peak = bal
        else:
            dd = (peak - bal) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

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
