"""
Performance-metric helper – robust on Py 3.8+, protects against:
  • open deals  (exit_ts None / NaN)
  • negative ROI ⇒ complex annual_pct
"""

import math
from typing import List, Tuple, Optional

import numpy as np

DealRow   = Tuple[Optional[int], Optional[int], float, float]
EquityRow = Tuple[int, float]


# ------------------------- helpers ---------------------------------- #
def _is_nan(x) -> bool:
    try:
        return np.isnan(x)
    except TypeError:
        return False


def calc_metrics(deals: List[DealRow], equity: List[EquityRow]) -> dict:
    """Return summary dict with daily‑compounded APY **and** legacy annual_pct key."""

    # protect empty back‑test
    if not equity:
        return {
            "deals":            0,
            "total_pl":         0,
            "roi_pct":          0,
            "annual_pct":       0,
            "apy_pct":          0,
            "annual_usd":       0,
            "avg_deal_min":     0,
            "max_drawdown_pct": 0,
            "longest_drawdown_min": 0,
        }

    closed = [
        (int(s), int(e), p, f)
        for s, e, p, f in deals
        if (s is not None and e is not None and not _is_nan(s) and not _is_nan(e))
    ]

    total_pl = equity[-1][1] - equity[0][1]
    initial_balance = equity[0][1]
    final_balance   = equity[-1][1]

    days_span = max((equity[-1][0] - equity[0][0]) / (24 * 3600), 1)
    roi_pct   = (final_balance / initial_balance - 1) * 100

    # === Daily‑compounded APY ===
    apy_pct   = ((final_balance / initial_balance) ** (365 / days_span) - 1) * 100
    annual_pct = apy_pct                            # legacy alias
    annual_usd = initial_balance * annual_pct / 100

    # ---------------- peak‑to‑valley max draw‑down ---------------------
    peak   = equity[0][1]
    max_dd = 0
    longest_min = 0
    current_len = 0

    for ts, bal in equity:
        if bal > peak:
            peak = bal
            current_len = 0
        else:
            dd = (peak - bal) / peak * 100
            if dd > max_dd:
                max_dd = dd
            current_len += 1
            longest_min = max(longest_min, current_len)

    avg_deal = (
        sum((e - s) / 60 for s, e, _, _ in closed) / len(closed)
        if closed else 0
    )

    return {
        "deals": len(deals),
        "total_pl": round(total_pl, 2),
        "roi_pct": round(roi_pct, 2),
        "annual_pct": round(annual_pct, 2),  # APY daily‑compounded (alias)
        "apy_pct": round(annual_pct, 2),
        "annual_usd": round(annual_usd, 2),
        "avg_deal_min": round(avg_deal, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "longest_drawdown_min": round(longest_min, 2),
    }
