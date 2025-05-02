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


def _annualised_pct(roi_pct: float, years: float) -> float:
    """
    Safer alternative to (1+roi)**(1/years)-1.

    • Positive ROI → geometric formula (identical result).
    • Zero ROI     → 0.
    • Negative ROI → linear annualisation (roi / years).
      This avoids complex numbers and roughly matches IRR tools.
    """
    if years <= 0:
        return 0.0
    if roi_pct == 0:
        return 0.0

    roi_frac = roi_pct / 100.0
    if 1.0 + roi_frac <= 0:              # would yield complex root
        return (roi_frac / years) * 100  # linear fallback
    # geometric version
    return (math.exp(math.log1p(roi_frac) / years) - 1.0) * 100.0


# ------------------------- main API --------------------------------- #
def calc_metrics(deals: List[DealRow],
                 equity: List[EquityRow]) -> dict:
    if not equity:
        return {}

    # ---------------- filter closed deals ---------------------------
    closed = [
        (int(s), int(e), p, f)
        for s, e, p, f in deals
        if (s is not None and e is not None and not _is_nan(s) and not _is_nan(e))
    ]

    total_pl = sum(d[2] for d in closed)
    roi_pct  = total_pl / 1000 * 100        # base capital = 1 000 USD

    # ----- time span for annualisation (closed deals only) ---------------
    if closed:
        first_open  = closed[0][0]
        last_close  = closed[-1][1]
        years = (last_close - first_open) / (365 * 24 * 3600)
    else:
        # fall back to full equity window when nothing has closed yet
        years = (equity[-1][0] - equity[0][0]) / (365 * 24 * 3600)

    years = max(years, 1 / 365)                 # floor at one day
    annual_pct = roi_pct / years                # linear APR (no compounding)
    annual_usd = 1000 * annual_pct / 100        # still on the constant $1 000 base

    # ---------------- peak‑to‑valley max draw‑down ---------------------
    peak          = equity[0][1]
    max_dd_pct    = 0.0
    longest       = 0        # longest underwater period (seconds)
    in_drawdown   = False
    dd_start_time = 0

    for ts, bal in equity:
        if bal > peak:
            # New all‑time high resets draw‑down
            peak = bal
            if in_drawdown:
                in_drawdown = False
                longest = max(longest, ts - dd_start_time)
        else:
            # Below peak → draw‑down
            if not in_drawdown:
                in_drawdown = True
                dd_start_time = ts
            dd = (peak - bal) / peak * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

    # If we finish still underwater, extend longest DD to the end
    if in_drawdown:
        longest = max(longest, equity[-1][0] - dd_start_time)

    longest_min = longest / 60

    # ---------------- average deal duration -------------------------
    avg_deal = (
        sum((e - s) / 60 for s, e, _, _ in closed) / len(closed)
        if closed else 0
    )

    return {
        "deals": len(deals),
        "total_pl": round(total_pl, 2),
        "roi_pct": round(roi_pct, 2),
        "annual_pct": round(annual_pct, 2),
        "annual_usd": round(annual_usd, 2),
        "avg_deal_min": round(avg_deal, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "longest_drawdown_min": round(longest_min, 2),
    }
