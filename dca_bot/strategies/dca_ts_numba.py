# dca_ts_numba.py
"""
Numba‑accelerated dual‑side DCA strategy (spot).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List as NbList


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def _bb_percent(df: pd.DataFrame) -> np.ndarray:
    ohlc_3m = df.resample("3min").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    close = ohlc_3m['close']
    mean = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower = mean - 2 * std
    upper = mean + 2 * std
    bbp = (close - lower) / (upper - lower)
    bb_per = bbp.reindex(df.index, method='ffill').fillna(0.5)
    return bb_per.to_numpy(np.float64)


def _supertrend(df: pd.DataFrame) -> np.ndarray:
    ohlc_30m = df.resample("30min").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    atr = _atr(ohlc_30m.high, ohlc_30m.low, ohlc_30m.close, window=10)
    hl2 = (ohlc_30m.high + ohlc_30m.low) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    st = pd.Series(np.nan, index=ohlc_30m.index)
    bull = pd.Series(True, index=ohlc_30m.index)
    for i in range(1, len(ohlc_30m)):
        if bull.iat[i - 1]:
            st.iat[i] = max(lower.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else lower.iat[i])
            bull.iat[i] = ohlc_30m.close.iat[i] > st.iat[i]
        else:
            st.iat[i] = min(upper.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else upper.iat[i])
            bull.iat[i] = ohlc_30m.close.iat[i] > st.iat[i]
    return bull.reindex(df.index, method='ffill').fillna(False).to_numpy(np.bool_)


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
    long_only: bool = False
    exit_on_flip: bool = True

    def backtest(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        px = df['close'].to_numpy(np.float64)
        ts = df.index.view('int64') // 1_000_000_000
        bb_percent = _bb_percent(df)
        bull = _supertrend(df)

        deals_np, eq_np = _loop(
            ts, px, bb_percent, bull,
            self.spacing_pct, self.tp_pct, int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec,
            int(self.compound), self.risk_pct,
            int(self.long_only),
            int(self.exit_on_flip),
            60  # cooldown_sec for safety orders
        )

        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in eq_np]
        return deals, equity


# ------------------ numba core ------------------
@nb.njit(cache=True)
def _loop(
    ts: np.ndarray, px: np.ndarray, bb_percent: np.ndarray, bull: np.ndarray,
    spacing_pct: float, tp_pct: float, trailing_int: int, trailing_pct: float,
    max_safety: int, base_order: float, mult: float,
    fee_rate: float, init_cash: float,
    reopen_sec: int, compound_int: int, risk_pct: float,
    long_only_int: int,
    exit_on_flip_int: int,
    cooldown_sec: int
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
    prev_bbp = 0.5

    for i in range(n):
        t = ts[i]
        p = px[i]
        eq = cash + qty * p
        equity.append(np.array((t, eq), dtype=np.float64))

        bbp = bb_percent[i]
        is_bull = bull[i]

        # ------------- open trade -------------
        if not in_trade:
            if long_only_int == 1:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = False
            else:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = (not is_bull) and (prev_bbp >= 1 > bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
            if not (open_long or open_short):
                prev_bbp = bbp
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
            prev_bbp = bbp
            continue

        # ------------- safety orders -------------
        need_safety = (side == 1 and p <= next_order) or (side == -1 and p >= next_order)
        bb_condition = (side == 1 and bbp < 0.1) or (side == -1 and bbp > 0.9)
        if in_trade and need_safety and bb_condition and safety_cnt < max_safety and (t - entry_ts >= cooldown_sec):
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
            entry_ts = t

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

        # trend flip using Supertrend
        trend_flip = (side == 1 and not is_bull) or (side == -1 and is_bull)
        if exit_on_flip_int and trend_flip:
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

        prev_bbp = bbp

    return deals, equity
