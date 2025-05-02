"""
Numba-accelerated DCA strategy
• “Smart” mode (default):  entry when %B(20,2) crosses up 0 AND RSI-7<30
• “Stupid” mode:           ignore indicators, reopen N seconds after
  the previous deal closed  (set use_sig=0, reopen_sec=N)

Indicators are built on 3-minute closes, then forward-filled onto the
native 1-minute frame.  The heavy loop remains fully JIT-compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numba as nb
import numpy as np
import pandas as pd
import ta
from numba.typed import List as NbList


# =====================  Python-side helpers  ======================= #
def _build_entry_signal(df: pd.DataFrame) -> np.ndarray:
    """Daily SuperTrend → 1-minute risk-on array (uint8)."""
    # --- 1-day OHLC via resample ------------------------------------
    hi_d = df["high"].resample("1h").max()
    lo_d = df["low"] .resample("1h").min()
    cl_d = df["close"].resample("1h").last()

    atr = ta.volatility.AverageTrueRange(hi_d, lo_d, cl_d, window=10).average_true_range()
    hl2 = (hi_d + lo_d) / 2
    upper = hl2 + 6 * atr
    lower = hl2 - 6 * atr

    # -------- convert to NumPy → position-safe & faster -------------
    up   = upper.to_numpy(np.float64)
    lo   = lower.to_numpy(np.float64)
    close= cl_d.to_numpy(np.float64)

    st   = np.full(len(close), np.nan, dtype=np.float64)
    dir  = np.ones(len(close), dtype=np.uint8)      # 1 = bullish

    for i in range(1, len(close)):
        if dir[i-1]:                                # currently bullish
            st[i]  = max(lo[i],  st[i-1] if not np.isnan(st[i-1]) else lo[i])
            dir[i] = close[i] > st[i]
        else:                                       # currently bearish
            st[i]  = min(up[i], st[i-1] if not np.isnan(st[i-1]) else up[i])
            dir[i] = close[i] > st[i]

    # -------- forward-fill onto 1-minute frame ----------------------
    dir_1m = (
        pd.Series(dir, index=cl_d.index)
          .reindex(df.index, method="ffill")
          .fillna(0)
          .to_numpy(np.uint8)
    )
    return dir_1m


# =========================  Strategy class  ======================== #
@dataclass
class DCAJITStrategy:
    # user-tunable parameters
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = True
    trailing_pct: float = 0.1
    max_safety: int = 50
    fee_rate: float = 0.001
    initial_balance: float = 1000.0
    usd_per_order: float | None = None

    # NEW: smart vs. stupid switch
    reopen_sec: int = -1          # -1 ⇒ use indicator signal
    use_sig:    int = 1           # 1 ⇒ use signal, 0 ⇒ ignore signal

    # -----------------------------------------------------------------
    def __post_init__(self):
        if self.usd_per_order is None:
            self.usd_per_order = self.initial_balance / 51.0

    # -----------------------------------------------------------------
    def backtest(
        self, df: pd.DataFrame
    ) -> Tuple[List[Tuple[int, int, float, float]], List[Tuple[int, float]]]:

        close = df["close"].to_numpy(np.float64)
        ts    = df.index.view("int64") // 1_000_000_000   # epoch seconds
        sig   = _build_entry_signal(df)

        deals_rows, equity_rows = _run_loop_nb(
            ts, close, sig,
            self.spacing_pct, self.tp_pct, self.trailing,
            self.trailing_pct, self.max_safety,
            self.usd_per_order, self.fee_rate,
            self.initial_balance,
            self.reopen_sec, self.use_sig,
        )

        deals   = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_rows]
        equity  = [(int(r[0]), float(r[1])) for r in equity_rows]
        return deals, equity


# ======================  Numba JIT core loop  ====================== #
@nb.njit(cache=True)
def _run_loop_nb(
    ts: np.ndarray,
    px: np.ndarray,
    sig: np.ndarray,
    spacing_pct: float,
    tp_pct: float,
    trailing: bool,
    trailing_pct: float,
    max_safety: int,
    usd_per_order: float,
    fee_rate: float,
    init_bal: float,
    reopen_sec: int,
    use_sig: int,
):
    n = len(px)
    deals   = NbList.empty_list(nb.float64[:])
    equity  = NbList.empty_list(nb.float64[:])

    in_trade = False
    entry_ts = 0
    qty = avg = next_buy = tp_price = trail_top = 0.0
    safety_cnt = 0
    cash  = init_bal
    cost  = 0.0
    last_close = -1e18          # very old timestamp

    for i in range(n):
        t = ts[i]
        p = px[i]

        # ---------- open logic ---------------------------------------------
        risk_on = sig[i] == 1  # 1 = allowed, 0 = blocked

        if use_sig == 1:  # smart mode
            open_sig = risk_on  # ← only when signal is 1
            open_time = False

        else:  # use_sig == 0  (timer mode)
            open_sig = False
            open_time = risk_on and (t >= last_close + reopen_sec)
            # ← timer AND signal must agree

        if (not in_trade) and (open_sig or open_time):
            fee = usd_per_order * fee_rate
            buy_qty = usd_per_order / p
            qty  = buy_qty
            cash -= usd_per_order + fee
            cost = usd_per_order + fee
            avg  = p
            next_buy = p * (1 - spacing_pct / 100)
            tp_price = p * (1 + tp_pct / 100)
            trail_top = 0.0
            safety_cnt = 0
            entry_ts = t
            in_trade = True

        # ---------- safety buys --------------------------------------
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

        # ---------- take-profit / trailing ---------------------------
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
            last_close = t

        # ---------- equity snapshot ---------------------------------
        equity.append(np.array((t, cash + qty * p), dtype=np.float64))

    return deals, equity
