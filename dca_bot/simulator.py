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

    start, end = equity[0][0], equity[-1][0]
    years = (end - start) / (365 * 24 * 3600)
    annual_pct = roi_pct / years if years > 0 else 0  # linear, no reinvest
    annual_usd = 1000 * annual_pct / 100  # on the same $1 000 base

    # ---------------- drawdowns -------------------------------------
    peak = equity[0][1]
    peak_time = start
    max_dd = longest = 0
    for t, val in equity:
        if val > peak:
            peak = val
            peak_time = t
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
        if val < peak:
            dur = t - peak_time
            if dur > longest:
                longest = dur
    max_dd_pct  = max_dd / peak * 100 if peak else 0
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
