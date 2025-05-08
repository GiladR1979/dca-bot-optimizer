
"""
Numba‑accelerated dual‑side DCA strategy (spot).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numba as nb
import numpy as np
import pandas as pd
import pandas_ta as pta
from numba.typed import List as NbList


def _entry_signal(df: pd.DataFrame, tf: str = "8h") -> np.ndarray:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    st = pta.supertrend(ohlc['high'], ohlc['low'], ohlc['close'], length=10, multiplier=3)
    dir_col = [c for c in st.columns if c.startswith('SUPERTd')][0]
    bull_tf = (st[dir_col] == 1).astype(np.uint8)
    return bull_tf.reindex(df.index, method='ffill').fillna(0).to_numpy(np.uint8)


@dataclass
class DCAJITStrategy:
    base_order: float = 16.6078
    mult: float = 1.0
    max_safety: int = 50
    fee_rate: float = 0.001
    compound: bool = True
    risk_pct: float = 0.0166078
    spacing_pct: float = 1.0
    tp_pct: float = 0.6
    trailing: bool = True
    trailing_pct: float = 0.1
    initial_balance: float = 1000.0
    use_sig: int = 1  # compatibility placeholder
    reopen_sec: int = -1

    def backtest(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        px = df['close'].to_numpy(np.float64)
        ts = df.index.view('int64') // 1_000_000_000
        bull = _entry_signal(df)

        deals_np, eq_np = _loop(
            ts, px, bull,
            self.spacing_pct, self.tp_pct, int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec,
            int(self.compound), self.risk_pct
        )

        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in eq_np]
        return deals, equity


# ------------------ numba core ------------------
@nb.njit(cache=True)
def _loop(
    ts: np.ndarray, px: np.ndarray, bull: np.ndarray,
    spacing_pct: float, tp_pct: float, trailing_int: int, trailing_pct: float,
    max_safety: int, base_order: float, mult: float,
    fee_rate: float, init_cash: float,
    reopen_sec: int, compound_int: int, risk_pct: float
):
    n = len(px)
    deals = NbList.empty_list(nb.float64[:])
    equity = NbList.empty_list(nb.float64[:])

    cash = init_cash
    qty = 0.0
    avg = 0.0
    side = 0     # 0 idle, +1 long, −1 short
    in_trade = False
    ladder0 = base_order
    safety_cnt = 0
    next_order = 0.0
    trail_ext = 0.0
    cash_start = 0.0
    entry_ts = -1
    last_close = -1e18

    for i in range(n):
        t = ts[i]
        p = px[i]
        eq = cash + qty * p
        equity.append(np.array((t, eq), dtype=np.float64))

        trend_bull = bull[i] == 1

        # ------------- open trade -------------
        if not in_trade:
            open_long = trend_bull and (reopen_sec == -1 or t >= last_close + reopen_sec)
            open_short = (not trend_bull) and (reopen_sec == -1 or t >= last_close + reopen_sec)
            if not (open_long or open_short):
                continue

            side = 1 if open_long else -1
            usd = cash * risk_pct if compound_int == 1 else base_order
            fee = usd * fee_rate
            qty_change = side * usd / p

            cash_start = cash

            if side == 1:
                cash -= usd + fee
            else:
                cash += usd - fee

            qty += qty_change
            avg = p
            ladder0 = usd
            safety_cnt = 0
            next_order = p * (1 - spacing_pct / 100) if side == 1 else p * (1 + spacing_pct / 100)
            trail_ext = p
            entry_ts = t
            in_trade = True
            continue

        # ------------- safety orders -------------
        need_safety = (side == 1 and p <= next_order) or (side == -1 and p >= next_order)
        if in_trade and need_safety and safety_cnt < max_safety:
            safety_cnt += 1
            usd = ladder0 * (mult ** safety_cnt)
            fee = usd * fee_rate
            qty_change = side * usd / p

            if side == 1:
                cash -= usd + fee
            else:
                cash += usd - fee

            qty_old = qty
            qty += qty_change
            avg = (avg * abs(qty_old) + p * abs(qty_change)) / abs(qty)
            next_order = p * (1 - spacing_pct / 100) if side == 1 else p * (1 + spacing_pct / 100)
            trail_ext = p

        # ------------- TP / trailing -------------
        tp_target = avg * (1 + tp_pct / 100) if side == 1 else avg * (1 - tp_pct / 100)
        tp_hit = (side == 1 and p >= tp_target) or (side == -1 and p <= tp_target)
        exit_now = False
        if tp_hit:
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

        # trend flip
        trend_flip = (side == 1 and not trend_bull) or (side == -1 and trend_bull)
        if trend_flip:
            exit_now = True

        # ------------- close -------------
        if exit_now:
            if side == 1:
                proceeds = abs(qty) * p
                fee = proceeds * fee_rate
                cash += proceeds - fee
            else:
                buy_cost = abs(qty) * p
                fee = buy_cost * fee_rate
                cash -= buy_cost + fee

            profit = cash - cash_start
            deals.append(np.array((entry_ts, t, profit, fee), dtype=np.float64))

            qty = 0.0
            avg = 0.0
            in_trade = False
            side = 0
            last_close = t

    return deals, equity
