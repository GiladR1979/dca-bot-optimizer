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


def _bb_percent(df: pd.DataFrame) -> np.ndarray:
    ohlc_5m = df.resample("5min").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    bb = pta.bbands(ohlc_5m['close'], length=20, std=2)
    dir_col = [c for c in bb.columns if c.startswith('BBP_')][0]
    bb_per = bb[dir_col].reindex(df.index, method='ffill').fillna(0.5)
    return bb_per.to_numpy(np.float64)


@dataclass
class DCAJITStrategy:
    base_order: float = 16.6078
    mult: float = 1.5
    max_safety: int = 8
    fee_rate: float = 0.001
    compound: bool = True
    risk_pct: float = 0.013085
    spacing_pct: float = 0.3
    tp_pct: float = 0.6
    trailing: bool = True
    trailing_pct: float = 0.1
    initial_balance: float = 1000.0
    use_sig: int = 1  # compatibility placeholder
    reopen_sec: int = -1
    long_only: bool = True
    exit_on_flip: bool = True  # Use BB >=1 as flip for exit
    max_hold_days: int = 30  # New: Max days to hold a deal before forced exit
    stop_loss_pct: float = 20.0  # New: % below initial base price to SL exit (e.g., 20 = -20% from base)

    def backtest(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        px = df['close'].to_numpy(np.float64)
        ts = df.index.view('int64') // 1_000_000_000
        bb_percent = _bb_percent(df)

        deals_np, eq_np = _loop(
            ts, px, bb_percent,
            self.spacing_pct, self.tp_pct, int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec,
            int(self.compound), self.risk_pct,
            int(self.long_only),
            int(self.exit_on_flip),
            self.max_hold_days, self.stop_loss_pct  # Pass new params
        )

        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in eq_np]
        return deals, equity


# ------------------ numba core ------------------
@nb.njit(cache=True)
def _loop(
    ts: np.ndarray, px: np.ndarray, bb_percent: np.ndarray,
    spacing_pct: float, tp_pct: float, trailing_int: int, trailing_pct: float,
    max_safety: int, base_order: float, mult: float,
    fee_rate: float, init_cash: float,
    reopen_sec: int, compound_int: int, risk_pct: float,
    long_only_int: int,
    exit_on_flip_int: int,
    max_hold_days: int,  # New
    stop_loss_pct: float  # New
):
    n = len(px)
    deals = NbList.empty_list(nb.float64[:])
    equity = NbList.empty_list(nb.float64[:])

    cash = init_cash
    qty = 0.0
    avg = 0.0
    base_price = 0.0  # New: Track initial base order price for SL
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

        bbp = bb_percent[i]

        # ------------- open trade -------------
        if not in_trade:
            if long_only_int == 1:
                # Long-only mode
                open_long = (0 <= bbp <= 0.5) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = False
            else:
                open_long = (0 <= bbp <= 0.5) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = (bbp > 1) and (reopen_sec == -1 or t >= last_close + reopen_sec)  # Example for short, but not used
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
            base_price = p  # New: Set base_price to initial entry price
            ladder0 = usd
            safety_cnt = 0
            next_order = p * (1 - spacing_pct / 100) if side == 1 else p * (1 + spacing_pct / 100)
            trail_ext = p
            entry_ts = t
            in_trade = True
            continue

        # ------------- safety orders -------------
        need_safety = (side == 1 and bbp < 0 and p <= next_order) or (side == -1 and bbp > 1 and p >= next_order)
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

        # trend flip using BB
        trend_flip = (side == 1 and bbp >= 1) or (side == -1 and bbp <= 0)
        if exit_on_flip_int and trend_flip:
            exit_now = True

        # New: Max hold time exit (in seconds, assuming ts is unix)
        if in_trade and (t - entry_ts) > (max_hold_days * 86400):
            exit_now = True

        # New: Stop-loss exit (now from base_price)
        sl_target = base_price * (1 - stop_loss_pct / 100) if side == 1 else base_price * (1 + stop_loss_pct / 100)
        sl_hit = (side == 1 and p <= sl_target) or (side == -1 and p >= sl_target)
        if sl_hit:
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
            base_price = 0.0  # Reset
            in_trade = False
            side = 0
            last_close = t

    return deals, equity
