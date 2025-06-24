"""
Numba-accelerated dual-side DCA strategy (spot) *with flip-pending freeze*.

* `exit_on_flip = True`  (default) — original behaviour: close the running
  deal immediately when the 8 h SuperTrend changes side.
* `exit_on_flip = False` — **freeze** new entries on a flip; keep the
  current ladder alive until it exits at TP / breakeven, then allow
  trading in the new direction.  No loss is realised at the flip point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numba as nb
import numpy as np
import pandas as pd
import pandas_ta as pta
from numba.typed import List as NbList


# --------------------------------------------------------------------- #
#  helper – slow SuperTrend direction                                   #
# --------------------------------------------------------------------- #
def _entry_signal(df: pd.DataFrame, tf: str = "15min") -> np.ndarray:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    st = pta.supertrend(ohlc["high"], ohlc["low"], ohlc["close"], length=10, multiplier=3)
    dir_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
    bull_tf = (st[dir_col] == 1).astype(np.uint8)
    return bull_tf.reindex(df.index, method="ffill").fillna(0).to_numpy(np.uint8)


# --------------------------------------------------------------------- #
#  OO wrapper so the rest of the codebase stays unchanged               #
# --------------------------------------------------------------------- #
@dataclass
class DCAJITStrategy:
    # --- sizing ------------------------------------------------------ #
    base_order: float = 16.6078
    mult: float = 1.0
    max_safety: int = 50
    fee_rate: float = 0.001
    compound: bool = True
    risk_pct: float = 0.0166078  # fraction of `cash` for the *initial* order

    # --- geometry ---------------------------------------------------- #
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = False
    trailing_pct: float = 0.1

    # --- session params --------------------------------------------- #
    initial_balance: float = 1_000.0
    reopen_sec: int = 60
    exit_on_flip: bool = True
    use_sig: bool = True

    # ---------------------------------------------------------------- #
    def run(self, df: pd.DataFrame) -> Tuple[List[Tuple[int, int, float, float]],
                                             List[Tuple[int, float]]]:
        ts = df.index.astype("int64") // 10 ** 9
        px = df["close"].to_numpy(np.float64)
        bull = _entry_signal(df)

        deals_np, eq_np = _loop(
            ts, px, bull,
            self.spacing_pct, self.tp_pct,
            int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec, int(self.compound), self.risk_pct,
            int(self.exit_on_flip),
        )

        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in eq_np]
        return deals, equity

    # ---------------------------------------------------------- #
    # public alias – keeps old call-sites working                #
    def backtest(self, df: pd.DataFrame):
        """Legacy wrapper; same as run()."""
        return self.run(df)


# --------------------------------------------------------------------- #
#  core engine (Numba)                                                  #
# --------------------------------------------------------------------- #
def _loop(
        ts: np.ndarray, px: np.ndarray, bull: np.ndarray,
        spacing_pct: float, tp_pct: float,
        trailing_int: int, trailing_pct: float,
        max_safety: int, base_order: float, mult: float,
        fee_rate: float, init_cash: float,
        reopen_sec: int, compound_int: int, risk_pct: float,
        exit_on_flip_int: int,
):
    n = len(px)
    deals = NbList.empty_list(nb.float64[:])  # entry_ts, exit_ts, profit, fee
    equity = NbList.empty_list(nb.float64[:])  # ts, nav

    # -------- persistent state -------------------------------------- #
    cash = init_cash
    qty = 0.0
    avg = 0.0
    ladder0 = 0.0

    in_trade = False
    side = 0  # +1 long, −1 short
    safety_cnt = 0
    next_order = 0.0
    trail_ext = 0.0
    cash_start = 0.0
    entry_ts = 0

    last_close = -1_000_000  # far past
    flip_pending = 0  # 1 while waiting for current deal to finish

    snapshot_step = 30 * 60  # sec

    for i in range(n):
        t = int(ts[i])
        p = px[i]
        trend_bull = bull[i] == 1

        # ---------- open -------------------------------------------- #
        if in_trade is False:
            if flip_pending == 1:
                # do nothing until existing (opposite) trade finishes
                if i % snapshot_step == 0:
                    equity.append(np.array((t, cash), dtype=np.float64))
                continue

            reopen_ok = (reopen_sec == -1) or (t >= last_close + reopen_sec)
            open_long = trend_bull and reopen_ok
            open_short = (trend_bull == False) and reopen_ok

            if open_long or open_short:
                side = 1 if open_long else -1
                usd = cash * risk_pct if compound_int == 1 else base_order
                fee = usd * fee_rate
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
                next_order = p * (1 - spacing_pct / 100) if side == 1 else p * (1 + spacing_pct / 100)
                trail_ext = p
                in_trade = True

        # ------------- skip snapshots when still flat --------------- #
        if in_trade is False:
            if i % snapshot_step == 0:
                equity.append(np.array((t, cash), dtype=np.float64))
            continue

        # ---------- safety order ------------------------------------ #
        if safety_cnt < max_safety:
            order_hit = (side == 1 and p <= next_order) or (side == -1 and p >= next_order)
            if order_hit:
                usd = ladder0 * (mult ** (safety_cnt + 1))
                qty_change = side * usd / p
                fee = usd * fee_rate

                if side == 1:
                    cash -= usd + fee
                else:
                    cash += usd - fee

                qty += qty_change
                avg = (avg * (abs(qty) - abs(qty_change)) + abs(qty_change) * p) / abs(qty)
                safety_cnt += 1
                next_order = p * (1 - spacing_pct / 100) if side == 1 else p * (1 + spacing_pct / 100)

        # ---------- take-profit / trailing -------------------------- #
        tp_target = avg * (1 + tp_pct / 100) if side == 1 else avg * (1 - tp_pct / 100)
        exit_now = False
        if (side == 1 and p >= tp_target) or (side == -1 and p <= tp_target):
            if trailing_int == 1:
                if side == 1:
                    if p > trail_ext:
                        trail_ext = p
                    if p <= trail_ext * (1 - trailing_pct / 100):
                        exit_now = True
                else:
                    if p < trail_ext:
                        trail_ext = p
                    if p >= trail_ext * (1 + trailing_pct / 100):
                        exit_now = True
            else:
                exit_now = True

        # ---------- slow trend flip --------------------------------- #
        trend_flip = (side == 1 and trend_bull == 0) or (side == -1 and trend_bull == 1)
        if trend_flip:
            if exit_on_flip_int == 1:
                exit_now = True
            else:
                flip_pending = 1  # freeze future entries

        # ---------- close ------------------------------------------- #
        if exit_now:
            if side == 1:
                proceeds = abs(qty) * p
                fee = proceeds * fee_rate
                cash += proceeds - fee
            else:
                cost = abs(qty) * p
                fee = cost * fee_rate
                cash -= cost + fee

            profit = cash - cash_start
            deals.append(np.array((entry_ts, t, profit, fee), dtype=np.float64))

            # reset
            qty = 0.0
            avg = 0.0
            in_trade = False
            safety_cnt = 0
            next_order = 0.0
            trail_ext = 0.0
            last_close = t
            flip_pending = 0  # allow new trades

        # ---------- equity ------------------------------------------ #
        if i % snapshot_step == 0:
            equity.append(np.array((t, cash + qty * p), dtype=np.float64))

    # final snapshot
    if len(equity) == 0 or equity[-1][0] != ts[-1]:
        equity.append(np.array((ts[-1], cash + qty * px[-1]), dtype=np.float64))

    return deals, equity
