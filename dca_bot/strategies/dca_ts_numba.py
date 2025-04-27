"""
Numba‑accelerated fixed‑interval DCA strategy
------------------------------------------------
• correct per‑deal profit (was cumulative cash)
• proper qty‑weighted average price
• fee handling (default 0.1 %) identical to the pure‑Python version
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
    fee_rate: float = 0.001  # exchange taker fee (0.1 %)

    # ------------------------------------------------------------------ #
    def backtest(
        self, df: pd.DataFrame
    ) -> Tuple[List[Tuple[int, int, float, float]], List[Tuple[int, float]]]:
        """Run the strategy on a DataFrame of 1‑minute candles."""

        close = df["close"].to_numpy(np.float64)
        ts = df.index.view("int64") // 1_000_000_000  # epoch seconds

        deals_rows, equity_rows = _run_loop_nb(
            ts,
            close,
            self.spacing_pct,
            self.tp_pct,
            self.trailing,
            self.trailing_pct,
            self.max_safety,
            self.usd_per_order,
            self.fee_rate,
        )

        # ---------- convert typed lists back to Python lists ------------
        deals = [
            (int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_rows
        ]
        equity = [(int(r[0]), float(r[1])) for r in equity_rows]
        return deals, equity


# =========================  JIT core  ================================= #


@nb.njit(cache=True)
def _run_loop_nb(
    ts: np.ndarray,
    px: np.ndarray,
    spacing_pct: float,
    tp_pct: float,
    trailing: bool,
    trailing_pct: float,
    max_safety: int,
    usd_per_order: float,
    fee_rate: float,
):
    """Core loop – fully JIT‑compiled by Numba."""

    n = len(px)

    # typed lists to avoid np.vstack inside JIT
    deals = NbList.empty_list(nb.float64[:])
    equity = NbList.empty_list(nb.float64[:])

    in_trade = False

    entry_ts = 0
    qty = 0.0
    avg = 0.0
    next_buy = 0.0
    tp_price = 0.0
    trail_top = 0.0
    safety_cnt = 0
    cash = 0.0
    cost = 0.0  # USD spent (incl. fees) in current deal

    for i in range(n):
        t = ts[i]
        p = px[i]

        # --------------- open first order ------------------------------
        if not in_trade:
            fee = usd_per_order * fee_rate
            buy_qty = usd_per_order / p
            qty = buy_qty
            cash -= usd_per_order + fee
            cost = usd_per_order + fee
            avg = p
            next_buy = p * (1 - spacing_pct / 100)
            tp_price = p * (1 + tp_pct / 100)
            trail_top = 0.0
            safety_cnt = 0
            entry_ts = t
            in_trade = True

        # --------------- DCA buys --------------------------------------
        else:
            if safety_cnt < max_safety and p <= next_buy:
                fee = usd_per_order * fee_rate
                buy_qty = usd_per_order / p
                # qty‑weighted average price
                avg = (avg * qty + p * buy_qty) / (qty + buy_qty)

                qty += buy_qty
                cash -= usd_per_order + fee
                cost += usd_per_order + fee
                safety_cnt += 1

                next_buy = p * (1 - spacing_pct / 100)
                tp_price = avg * (1 + tp_pct / 100)
                trail_top = 0.0

        # --------------- TP / trailing ---------------------------------
        exit_now = False
        if in_trade and p >= tp_price:
            if trailing:
                if p > trail_top:
                    trail_top = p
                if trail_top > 0 and p <= trail_top * (1 - trailing_pct / 100):
                    exit_now = True
            else:
                exit_now = True

        if exit_now and in_trade:
            proceeds = qty * p
            fee = proceeds * fee_rate
            cash += proceeds - fee
            profit = (proceeds - fee) - cost  # P/L this deal

            row = np.array((entry_ts, t, profit, fee), dtype=np.float64)
            deals.append(row)

            # reset state for the next deal
            qty = 0.0
            avg = 0.0
            cost = 0.0
            in_trade = False
            trail_top = 0.0

        # --------------- equity curve ----------------------------------
        eq_row = np.array((t, cash + qty * p), dtype=np.float64)
        equity.append(eq_row)

    return deals, equity
