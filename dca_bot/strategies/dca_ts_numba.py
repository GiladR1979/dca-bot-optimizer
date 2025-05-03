"""
Numba-accelerated DCA strategy – SuperTrend filter + geometric ladder
Base  = 6.5109 USDT,  Factor = 1.04,  Safety = 50  (51 total orders)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numba as nb
import numpy as np
import pandas as pd
import pandas_ta as pta
from numba.typed import List as NbList


# =====================  Trend-filter helper  ======================= #
def _build_entry_signal(df: pd.DataFrame, tf: str = "1D") -> np.ndarray:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    st = pta.supertrend(ohlc["high"], ohlc["low"], ohlc["close"],
                        length=10, multiplier=3)
    dir_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
    risk_tf = (st[dir_col] == 1).astype(np.uint8)

    return (
        risk_tf.reindex(df.index, method="ffill")
               .fillna(0)
               .to_numpy(np.uint8)
    )


# =========================  Strategy class  ======================== #
@dataclass
class DCAJITStrategy:
    # ----- DCA ladder -----------------------------------------------
    base_order:   float = 6.5109      # USDT first order
    mult:         float = 1.04
    max_safety:   int   = 50          # 50 safety + 1 base = 51 orders
    fee_rate:     float = 0.001

    # ----- TP / trailing --------------------------------------------
    spacing_pct:  float = 1.0         # 1 % gap between ladder steps
    tp_pct:       float = 0.6
    trailing:     bool  = True
    trailing_pct: float = 0.1

    # ----- Account ---------------------------------------------------
    initial_balance: float = 1000.0

    # ----- Mode switches --------------------------------------------
    reopen_sec: int = -1              # -1 ⇒ use indicator
    use_sig:    int = 1               # 1 ⇒ use indicator, 0 ⇒ timer

    # ----------------------------------------------------------------
    def backtest(self, df: pd.DataFrame
                 ) -> Tuple[List[Tuple], List[Tuple]]:
        px  = df["close"].to_numpy(np.float64)
        ts  = df.index.view("int64") // 1_000_000_000
        sig = _build_entry_signal(df)

        deals_np, equity_np = _run_loop_nb(
            ts, px, sig,
            self.spacing_pct, self.tp_pct, self.trailing, self.trailing_pct,
            self.max_safety,
            self.base_order, self.mult,    # NEW
            self.fee_rate, self.initial_balance,
            self.reopen_sec, self.use_sig,
        )

        deals  = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in equity_np]
        return deals, equity


# ======================  Numba core loop  ========================== #
@nb.njit(cache=True)
def _run_loop_nb(
    ts: np.ndarray, px: np.ndarray, sig: np.ndarray,
    spacing_pct: float, tp_pct: float, trailing: bool, trailing_pct: float,
    max_safety: int,
    base_order: float, mult: float,          # NEW
    fee_rate: float,
    init_bal: float,
    reopen_sec: int, use_sig: int,
):
    n = len(px)
    deals   = NbList.empty_list(nb.float64[:])
    equity  = NbList.empty_list(nb.float64[:])

    in_trade = False
    qty = avg = next_buy = tp_px = trail_top = 0.0
    safety_cnt = 0
    cash = init_bal
    cost = 0.0
    entry = last_close = -1e18

    for i in range(n):
        t = ts[i]
        p = px[i]

        # -------- open conditions ----------------------------------
        risk_on = sig[i] == 1
        open_sig  = risk_on if use_sig == 1 else 0
        open_time = (risk_on and (t >= last_close + reopen_sec)) if use_sig == 0 else 0

        if (not in_trade) and (open_sig or open_time):
            usd = base_order
            fee = usd * fee_rate
            qty = usd / p
            cash -= usd + fee
            cost = usd + fee

            avg = p
            next_buy = p * (1 - spacing_pct / 100)
            tp_px = p * (1 + tp_pct / 100)
            trail_top = 0.0
            safety_cnt = 0
            entry = t
            in_trade = True

        # -------- safety buys --------------------------------------
        elif in_trade and safety_cnt < max_safety and p <= next_buy:
            safety_cnt += 1
            usd = base_order * (mult ** safety_cnt)
            fee = usd * fee_rate
            buy_qty = usd / p

            cash -= usd + fee
            cost += usd + fee
            avg = (avg * qty + p * buy_qty) / (qty + buy_qty)
            qty += buy_qty

            next_buy = p * (1 - spacing_pct / 100)
            tp_px = avg * (1 + tp_pct / 100)
            trail_top = 0.0

        # -------- take-profit / trailing ---------------------------
        exit_now = False
        if in_trade and p >= tp_px:
            if trailing:
                trail_top = max(trail_top, p)
                if p <= trail_top * (1 - trailing_pct / 100):
                    exit_now = True
            else:
                exit_now = True

        if in_trade and exit_now:
            proceeds = qty * p
            fee = proceeds * fee_rate
            cash += proceeds - fee
            profit = (proceeds - fee) - cost

            deals.append(np.array((entry, t, profit, fee), dtype=np.float64))

            qty = 0.0
            in_trade = False
            last_close = t

        # -------- equity snapshot ----------------------------------
        equity.append(np.array((t, cash + qty * p), dtype=np.float64))

    return deals, equity
