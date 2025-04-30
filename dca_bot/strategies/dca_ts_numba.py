"""
Numba-accelerated DCA strategy
• Entry = Bollinger %B(20, 2) crosses up 0  AND  RSI-7 < 30
  both calculated on **3-minute candles**,
  forward-filled onto 1-minute resolution.
Only the indicator prep is Python; the heavy loop stays JIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numba as nb
import numpy as np
import pandas as pd
import ta
from numba.typed import List as NbList


# =====================  Python-side helpers  ======================== #
def _build_entry_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Return uint8 array (0/1) for “entry fires” per 1-minute row.
    Indicators use 3-minute closes.
    """
    close_3m = df["close"].resample("3min").last().dropna()

    ma = close_3m.rolling(20, min_periods=20).mean()
    sd = close_3m.rolling(20, min_periods=20).std()
    lo = ma - 2.0 * sd
    hi = ma + 2.0 * sd
    bbp3 = (close_3m - lo) / (hi - lo)

    rsi3 = ta.momentum.RSIIndicator(close_3m, window=7).rsi()

    bbp1 = bbp3.reindex(df.index, method="ffill").to_numpy(np.float64)
    rsi1 = rsi3.reindex(df.index, method="ffill").to_numpy(np.float64)

    sig = np.zeros(len(df), dtype=np.uint8)
    sig[1:] = (bbp1[:-1] < 0) & (bbp1[1:] >= 0) & (rsi1[:-1] < 30)
    return sig


# =========================  Strategy class  ========================= #
@dataclass
class DCAJITStrategy:
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = True
    trailing_pct: float = 0.1
    max_safety: int = 50
    usd_per_order: float = 1000 / 51.0
    fee_rate: float = 0.001  # taker fee 0.1 %

    # ------------------------------------------------------------------
    def backtest(
        self, df: pd.DataFrame
    ) -> Tuple[List[Tuple[int, int, float, float]], List[Tuple[int, float]]]:
        close = df["close"].to_numpy(np.float64)
        ts = df.index.view("int64") // 1_000_000_000  # epoch seconds
        sig = _build_entry_signal(df)

        deals_rows, equity_rows = _run_loop_nb(
            ts,
            close,
            sig,
            self.spacing_pct,
            self.tp_pct,
            self.trailing,
            self.trailing_pct,
            self.max_safety,
            self.usd_per_order,
            self.fee_rate,
        )

        # convert typed lists back to Python lists
        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_rows]
        equity = [(int(r[0]), float(r[1])) for r in equity_rows]
        return deals, equity


# ======================  Numba JIT core loop  ======================= #
@nb.njit(cache=True)
def _run_loop_nb(
    ts: np.ndarray,
    px: np.ndarray,
    sig: np.ndarray,  # 0/1 entry signal
    spacing_pct: float,
    tp_pct: float,
    trailing: bool,
    trailing_pct: float,
    max_safety: int,
    usd_per_order: float,
    fee_rate: float,
):
    n = len(px)
    deals = NbList.empty_list(nb.float64[:])
    equity = NbList.empty_list(nb.float64[:])

    in_trade = False
    entry_ts = 0
    qty = avg = next_buy = tp_price = trail_top = 0.0
    safety_cnt = 0
    cash = 0.0
    cost = 0.0

    for i in range(n):
        t = ts[i]
        p = px[i]

        # ------------- open first order -------------------------------
        if (not in_trade) and sig[i]:
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

        # ------------- DCA buys --------------------------------------
        elif in_trade and safety_cnt < max_safety and p <= next_buy:
            fee = usd_per_order * fee_rate
            buy_qty = usd_per_order / p
            avg = (avg * qty + p * buy_qty) / (qty + buy_qty)

            qty += buy_qty
            cash -= usd_per_order + fee
            cost += usd_per_order + fee
            safety_cnt += 1

            next_buy = p * (1 - spacing_pct / 100)
            tp_price = avg * (1 + tp_pct / 100)
            trail_top = 0.0

        # ------------- TP / trailing stop -----------------------------
        exit_now = False
        if in_trade and p >= tp_price:
            if trailing:
                if p > trail_top:
                    trail_top = p
                if p <= trail_top * (1 - trailing_pct / 100):
                    exit_now = True
            else:
                exit_now = True

        if in_trade and exit_now:
            proceeds = qty * p
            fee = proceeds * fee_rate
            cash += proceeds - fee
            profit = (proceeds - fee) - cost

            row = np.array((entry_ts, t, profit, fee), dtype=np.float64)
            deals.append(row)
            qty = avg = cost = 0.0
            in_trade = False

        # ------------- equity snapshot -------------------------------
        equity.append(np.array((t, cash + qty * p), dtype=np.float64))

    return deals, equity
