"""
Numba-accelerated fixed-interval DCA strategy (Python 3.9 compatible).

Key fix vs. previous revision
-----------------------------
* Inside `_run_loop_nb` we now use `numba.typed.List` to collect rows
  instead of `np.vstack`, avoiding the dimension-mismatch TypingError.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numba as nb
from numba.typed import List as NbList
import pandas as pd


@dataclass
class DCAJITStrategy:
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = True
    trailing_pct: float = 0.1
    max_safety: int = 50
    usd_per_order: float = 1000 / 51.0

    # ------------------------------------------------------------------ #
    def backtest(self, df: pd.DataFrame
                 ) -> Tuple[List[Tuple[int, int, float, float]],
                            List[Tuple[int, float]]]:

        close = df["close"].to_numpy(np.float64)
        ts    = df.index.view("int64") // 1_000_000_000  # to seconds

        deals_rows, equity_rows = _run_loop_nb(
            ts,
            close,
            self.spacing_pct,
            self.tp_pct,
            self.trailing,
            self.trailing_pct,
            self.max_safety,
            self.usd_per_order,
        )

        # ---------- convert typed lists back to Python lists ------------
        deals  = [(int(r[0]), int(r[1]), float(r[2]), 0.0) for r in deals_rows]
        equity = [(int(r[0]), float(r[1]))               for r in equity_rows]
        return deals, equity


# =========================  JIT core  ================================= #

@nb.njit(cache=True)
def _run_loop_nb(ts, px,
                 spacing_pct, tp_pct,
                 trailing, trailing_pct,
                 max_safety, usd_per_order):

    n = len(px)

    # typed lists to avoid np.vstack inside JIT
    deals  = NbList.empty_list(nb.float64[:] )
    equity = NbList.empty_list(nb.float64[:] )

    in_trade = False
    entry_ts = 0
    qty      = 0.0
    avg      = 0.0
    next_buy = 0.0
    tp_price = 0.0
    trail_top = 0.0
    safety_cnt = 0
    cash = 0.0

    for i in range(n):
        t = ts[i]
        p = px[i]

        # --------------- open first order ------------------------------
        if not in_trade:
            qty   = usd_per_order / p
            cash -= usd_per_order
            avg   = p
            next_buy = p * (1 - spacing_pct/100)
            tp_price = p * (1 + tp_pct/100)
            trail_top = 0.0
            safety_cnt = 0
            entry_ts = t
            in_trade = True

        # --------------- DCA buys --------------------------------------
        else:
            if safety_cnt < max_safety and p <= next_buy:
                buy_qty = usd_per_order / p
                qty += buy_qty
                cash -= usd_per_order
                avg = (avg*(safety_cnt+1) + p) / (safety_cnt+2)
                safety_cnt += 1
                next_buy = p * (1 - spacing_pct/100)
                tp_price = avg * (1 + tp_pct/100)
                trail_top = 0.0

        # --------------- TP / trailing ---------------------------------
        if in_trade and p >= tp_price:
            exit_now = False
            if trailing:
                if p > trail_top:
                    trail_top = p
                if trail_top > 0 and p <= trail_top*(1-trailing_pct/100):
                    exit_now = True
            else:
                exit_now = True

            if exit_now:
                cash += qty * p
                pl = cash                          # profit in USD this deal
                row = np.array((entry_ts, t, pl), dtype=np.float64)
                deals.append(row)

                # reset
                qty = avg = 0.0
                in_trade = False

        # --------------- equity curve ----------------------------------
        eq_row = np.array((t, cash + qty * p), dtype=np.float64)
        equity.append(eq_row)

    return deals, equity
