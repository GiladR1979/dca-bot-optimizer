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
    """Return uint8 array (0/1) per 1-minute row for the smart trigger."""
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

        # ---------- open logic ---------------------------------------
        open_sig  = (sig[i] == 1) if use_sig == 1 else False
        open_time = (t >= last_close + reopen_sec) if use_sig == 0 else False

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
