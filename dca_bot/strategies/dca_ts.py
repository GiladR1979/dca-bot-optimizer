"""
Pure-Python dual-side DCA strategy with *flip-pending freeze*.

`exit_on_flip` (bool, default **True**)
---------------------------------------
* **True** – original behaviour: close the running position the moment
  the 8 h SuperTrend reverses.
* **False** – keep the ladder alive, freeze new entries until the deal
  exits ≥ 0, then allow trading in the new trend direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as pta


# ------------------------------------------------------------------ #
def _entry_signal(df: pd.DataFrame, tf: str = "8h") -> pd.Series:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    st = pta.supertrend(ohlc["high"], ohlc["low"], ohlc["close"], length=10, multiplier=3)
    dir_col = next(c for c in st.columns if c.startswith("SUPERTd"))
    bull_tf = (st[dir_col] == 1).astype(int)
    return bull_tf.reindex(df.index, method="ffill").fillna(0).astype(int)


# ------------------------------------------------------------------ #
@dataclass
class DCATrailingStrategy:
    # sizing
    base_order: float = 16.6078
    mult: float = 1.0
    max_safety: int = 50
    fee_rate: float = 0.001
    compound: bool = True
    risk_pct: float = 0.0166078

    # grid
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = False
    trailing_pct: float = 0.1

    # session
    initial_balance: float = 1_000.0
    reopen_sec: int = 60
    exit_on_flip: bool = True
    use_sig: bool = True

    # -------------------------------------------------------------- #
    def run(self, df: pd.DataFrame) -> Tuple[List[Tuple[int, int, float, float]],
                                             List[Tuple[int, float]]]:
        ts = df.index.astype("int64") // 10 ** 9
        px = df["close"].to_numpy(float)
        bull = _entry_signal(df).to_numpy(int)

        cash = self.initial_balance
        qty = avg = ladder0 = 0.0
        in_trade = False
        side = 0
        safety_cnt = 0
        next_order = trail_ext = 0.0
        cash_start = 0.0
        entry_ts = 0

        last_close = -1_000_000
        flip_pending = False

        deals: List[Tuple[int, int, float, float]] = []
        equity: List[Tuple[int, float]] = []
        snapshot_step = 30 * 60  # sec

        for i, p in enumerate(px):
            t = int(ts[i])
            trend_bull = bull[i] == 1

            # -------- open trade -----------------------------------
            if not in_trade:
                if flip_pending:
                    if i % snapshot_step == 0:
                        equity.append((t, cash))
                    continue

                reopen_ok = (self.reopen_sec == -1) or (t >= last_close + self.reopen_sec)
                open_long = trend_bull and reopen_ok
                open_short = (not trend_bull) and reopen_ok

                if open_long or open_short:
                    side = 1 if open_long else -1
                    usd = cash * self.risk_pct if self.compound else self.base_order
                    fee = usd * self.fee_rate
                    qty_change = side * usd / p

                    cash_start = cash
                    entry_ts = t

                    if side == 1:
                        cash -= usd + fee
                    else:
                        cash += usd - fee

                    qty = qty_change
                    avg = p
                    ladder0 = usd
                    safety_cnt = 0
                    next_order = p * (1 - self.spacing_pct / 100) if side == 1 else p * (1 + self.spacing_pct / 100)
                    trail_ext = p
                    in_trade = True

            if not in_trade:
                if i % snapshot_step == 0:
                    equity.append((t, cash))
                continue

            # -------- safety order ---------------------------------
            if safety_cnt < self.max_safety:
                if (side == 1 and p <= next_order) or (side == -1 and p >= next_order):
                    usd = ladder0 * (self.mult ** (safety_cnt + 1))
                    qty_change = side * usd / p
                    fee = usd * self.fee_rate

                    if side == 1:
                        cash -= usd + fee
                    else:
                        cash += usd - fee

                    qty += qty_change
                    avg = (avg * (abs(qty) - abs(qty_change)) + abs(qty_change) * p) / abs(qty)
                    safety_cnt += 1
                    next_order = p * (1 - self.spacing_pct / 100) if side == 1 else p * (1 + self.spacing_pct / 100)

            # -------- TP / trailing --------------------------------
            tp_target = avg * (1 + self.tp_pct / 100) if side == 1 else avg * (1 - self.tp_pct / 100)
            exit_now = False
            if (side == 1 and p >= tp_target) or (side == -1 and p <= tp_target):
                if self.trailing:
                    if side == 1:
                        if p > trail_ext:
                            trail_ext = p
                        if p <= trail_ext * (1 - self.trailing_pct / 100):
                            exit_now = True
                    else:
                        if p < trail_ext:
                            trail_ext = p
                        if p >= trail_ext * (1 + self.trailing_pct / 100):
                            exit_now = True
                else:
                    exit_now = True

            # -------- flip-pending freeze --------------------------
            trend_flip = (side == 1 and not trend_bull) or (side == -1 and trend_bull)
            if trend_flip:
                if self.exit_on_flip:
                    exit_now = True
                else:
                    flip_pending = True

            # -------- close deal ----------------------------------
            if exit_now:
                if side == 1:
                    proceeds = abs(qty) * p
                    fee = proceeds * self.fee_rate
                    cash += proceeds - fee
                else:
                    cost = abs(qty) * p
                    fee = cost * self.fee_rate
                    cash -= cost + fee

                profit = cash - cash_start
                deals.append((entry_ts, t, profit, fee))

                qty = avg = 0.0
                in_trade = False
                safety_cnt = 0
                next_order = trail_ext = 0.0
                last_close = t
                flip_pending = False

            # -------- equity snapshot -----------------------------
            if i % snapshot_step == 0:
                equity.append((t, cash + qty * p))

        if not equity or equity[-1][0] != ts[-1]:
            equity.append((int(ts[-1]), cash + qty * px[-1]))

        return deals, equity

    # legacy name
    def backtest(self, df: pd.DataFrame):
        return self.run(df)