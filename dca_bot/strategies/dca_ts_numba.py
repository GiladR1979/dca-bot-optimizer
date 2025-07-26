# dca_ts_numba.py (modified)

"""
Numba‑accelerated dual‑side DCA strategy (spot) with GPU Monte Carlo support.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List as NbList
from numba import cuda
import cupy as cp


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def _bb_percent(df: pd.DataFrame, tf: str = "3min") -> np.ndarray:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    close = ohlc['close']
    mean = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower = mean - 2 * std
    upper = mean + 2 * std
    bbp = (close - lower) / (upper - lower)
    bb_per = bbp.reindex(df.index, method='ffill').fillna(0.5)
    return bb_per.to_numpy(np.float64)


def _supertrend(df: pd.DataFrame, tf: str = "30min") -> np.ndarray:
    ohlc = df.resample(tf).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    atr = _atr(ohlc.high, ohlc.low, ohlc.close, window=10)
    hl2 = (ohlc.high + ohlc.low) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    st = pd.Series(np.nan, index=ohlc.index)
    bull = pd.Series(True, index=ohlc.index)
    for i in range(1, len(ohlc)):
        if bull.iat[i - 1]:
            st.iat[i] = max(lower.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else lower.iat[i])
            bull.iat[i] = ohlc.close.iat[i] > st.iat[i]
        else:
            st.iat[i] = min(upper.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else upper.iat[i])
            bull.iat[i] = ohlc.close.iat[i] > st.iat[i]
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
    bb_tf: str = "3min"
    use_bb_safety: bool = True
    supertrend_tf: str = "30min"

    def backtest(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        px = df['close'].to_numpy(np.float64)
        ts = df.index.view('int64') // 1_000_000_000
        bb_percent = _bb_percent(df, self.bb_tf)
        bull = _supertrend(df, self.supertrend_tf)

        deals_np, eq_np = _loop(
            ts, px, bb_percent, bull,
            self.spacing_pct, self.tp_pct, int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec,
            int(self.compound), self.risk_pct,
            int(self.long_only),
            int(self.exit_on_flip),
            60,  # cooldown_sec for safety orders
            int(self.use_bb_safety)
        )

        deals = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in deals_np]
        equity = [(int(r[0]), float(r[1])) for r in eq_np]
        return deals, equity

    def backtest_gpu_monte_carlo(self, df: pd.DataFrame, num_sims: int = 100, noise_std: float = 0.001) -> np.ndarray:
        """Run Monte Carlo simulations in parallel on GPU."""
        # Convert to GPU arrays
        px = cp.asarray(df['close'].values)
        ts = cp.asarray(df.index.view('int64') // 1_000_000_000)
        bb_percent = cp.asarray(_bb_percent(df, self.bb_tf))
        bull = cp.asarray(_supertrend(df, self.supertrend_tf))

        # Generate noise on GPU
        noise = cp.random.normal(0, noise_std, (num_sims, len(px)))
        px_sims = px * (1 + noise)

        # Launch GPU kernel
        threads_per_block = 128
        blocks = (num_sims + threads_per_block - 1) // threads_per_block
        results = cp.zeros((num_sims, 5))  # ratio, max_dd, avg_deal, num_deals, longest_dd_min

        _loop_gpu[blocks, threads_per_block](
            ts, px_sims, bb_percent, bull,
            self.spacing_pct, self.tp_pct, int(self.trailing), self.trailing_pct,
            self.max_safety, self.base_order, self.mult,
            self.fee_rate, self.initial_balance,
            self.reopen_sec, int(self.compound), self.risk_pct,
            int(self.long_only), int(self.exit_on_flip),
            60, int(self.use_bb_safety), results
        )

        return cp.asnumpy(results)


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
        cooldown_sec: int,
        use_bb_safety_int: int
):
    n = len(px)
    peak = init_cash
    max_dd = 0.0
    current_len = 0
    max_len = 0
    sum_dur = 0.0
    num_deals = 0

    deals = NbList.empty_list(nb.float64[:])
    equity = NbList.empty_list(nb.float64[:])

    cash = init_cash
    qty = 0.0
    avg = 0.0
    side = 0  # 0 idle, +1 long, −1 short
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

        if eq > peak:
            peak = eq
            current_len = 0
        else:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
            current_len += 1
            if current_len > max_len:
                max_len = current_len

        bbp = bb_percent[i]
        is_bull = bull[i]

        # ------------- open trade -------------
        if not in_trade:
            if long_only_int == 1:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = False
            else:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = (not is_bull) and (prev_bbp >= 1 > bbp) and (
                            reopen_sec == -1 or t >= last_close + reopen_sec)
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
        bb_condition = True if use_bb_safety_int == 0 else ((side == 1 and bbp < 0.1) or (side == -1 and bbp > 0.9))
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

            dur_min = (t - entry_ts) / 60.0
            sum_dur += dur_min
            num_deals += 1

            qty = 0.0
            avg = 0.0
            in_trade = False
            side = 0
            last_close = t

        prev_bbp = bbp

    return deals, equity

    final_eq = cash + qty * px[n - 1]
    ratio = final_eq / init_cash
    avg_deal = sum_dur / num_deals if num_deals > 0 else 0.0

    results[sim_idx, 0] = ratio
    results[sim_idx, 1] = max_dd
    results[sim_idx, 2] = avg_deal
    results[sim_idx, 3] = num_deals
    results[sim_idx, 4] = max_len


# ------------------ GPU Monte Carlo kernel ------------------
@cuda.jit
def _loop_gpu(
        ts, px_sims, bb_percent, bull,
        spacing_pct, tp_pct, trailing_int, trailing_pct,
        max_safety, base_order, mult,
        fee_rate, init_cash, reopen_sec,
        compound_int, risk_pct, long_only_int, exit_on_flip_int,
        cooldown_sec, use_bb_safety_int, results
):
    sim_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if sim_idx >= px_sims.shape[0]:
        return

    n = len(ts)
    cash = init_cash
    qty = 0.0
    avg = 0.0
    side = 0
    in_trade = False
    ladder0 = base_order
    safety_cnt = 0
    next_order = 0.0
    trail_ext = 0.0
    cash_start = 0.0
    entry_ts = -1
    last_close = -1e18
    prev_bbp = 0.5
    px = px_sims[sim_idx]

    peak = init_cash
    max_dd = 0.0
    current_len = 0
    max_len = 0
    sum_dur = 0.0
    num_deals = 0

    for i in range(n):
        t = ts[i]
        p = px[i]
        bbp = bb_percent[i]
        is_bull = bull[i]

        eq = cash + qty * p

        if eq > peak:
            peak = eq
            current_len = 0
        else:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
            current_len += 1
            if current_len > max_len:
                max_len = current_len

        if not in_trade:
            if long_only_int == 1:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = False
            else:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = (not is_bull) and (prev_bbp >= 1 > bbp) and (
                            reopen_sec == -1 or t >= last_close + reopen_sec)
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
        bb_condition = True if use_bb_safety_int == 0 else ((side == 1 and bbp < 0.1) or (side == -1 and bbp > 0.9))
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

            dur_min = (t - entry_ts) / 60.0
            sum_dur += dur_min
            num_deals += 1

            qty = 0.0
            avg = 0.0
            in_trade = False
            side = 0
            last_close = t

        prev_bbp = bbp

    final_eq = cash + qty * px[n - 1]
    ratio = final_eq / init_cash
    avg_deal = sum_dur / num_deals if num_deals > 0 else 0.0

    results[sim_idx, 0] = ratio
    results[sim_idx, 1] = max_dd
    results[sim_idx, 2] = avg_deal
    results[sim_idx, 3] = num_deals
    results[sim_idx, 4] = max_len


# ------------------ GPU Grid kernel ------------------
@cuda.jit
def _grid_gpu(
        ts, px, bb_arrays, bull_arrays,
        params, results
):
    param_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if param_idx >= params.shape[0]:
        return

    spacing_pct = params[param_idx, 0]
    tp_pct = params[param_idx, 1]
    trailing_int = int(params[param_idx, 2])
    trailing_pct = params[param_idx, 3]
    exit_on_flip_int = int(params[param_idx, 4])
    bb_idx = int(params[param_idx, 5])
    st_idx = int(params[param_idx, 6])

    n = len(ts)
    cash = 1000.0  # fixed initial_balance
    qty = 0.0
    avg = 0.0
    side = 0
    in_trade = False
    ladder0 = 16.6078  # fixed base_order
    safety_cnt = 0
    next_order = 0.0
    trail_ext = 0.0
    cash_start = 0.0
    entry_ts = -1
    last_close = -1e18
    prev_bbp = 0.5

    peak = cash
    max_dd = 0.0
    current_len = 0
    max_len = 0
    sum_dur = 0.0
    num_deals = 0

    # fixed other params for grid
    max_safety = 8
    mult = 1.5
    fee_rate = 0.001
    reopen_sec = -1
    compound_int = 1
    risk_pct = 0.013085
    long_only_int = 0
    cooldown_sec = 60
    use_bb_safety_int = 1

    bb_percent = bb_arrays[bb_idx]
    bull = bull_arrays[st_idx]

    for i in range(n):
        t = ts[i]
        p = px[i]
        bbp = bb_percent[i]
        is_bull = bull[i]

        eq = cash + qty * p

        if eq > peak:
            peak = eq
            current_len = 0
        else:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
            current_len += 1
            if current_len > max_len:
                max_len = current_len

        if not in_trade:
            if long_only_int == 1:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = False
            else:
                open_long = is_bull and (prev_bbp <= 0 < bbp) and (reopen_sec == -1 or t >= last_close + reopen_sec)
                open_short = (not is_bull) and (prev_bbp >= 1 > bbp) and (
                            reopen_sec == -1 or t >= last_close + reopen_sec)
            if not (open_long or open_short):
                prev_bbp = bbp
                continue

            side = 1 if open_long else -1
            usd = cash * risk_pct if compound_int == 1 else ladder0
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

        need_safety = (side == 1 and p <= next_order) or (side == -1 and p >= next_order)
        bb_condition = True if use_bb_safety_int == 0 else ((side == 1 and bbp < 0.1) or (side == -1 and bbp > 0.9))
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

        trend_flip = (side == 1 and not is_bull) or (side == -1 and is_bull)
        if exit_on_flip_int and trend_flip:
            exit_now = True

        if exit_now:
            if side == 1:
                proceeds = abs(qty) * p
                fee = proceeds * fee_rate
                cash += proceeds - fee
            else:
                buy_cost = abs(qty) * p
                fee = buy_cost * fee_rate
                cash -= buy_cost + fee

            dur_min = (t - entry_ts) / 60.0
            sum_dur += dur_min
            num_deals += 1

            qty = 0.0
            avg = 0.0
            in_trade = False
            side = 0
            last_close = t

        prev_bbp = bbp

    final_eq = cash + qty * px[n - 1]
    ratio = final_eq / 1000.0  # fixed
    avg_deal = sum_dur / num_deals if num_deals > 0 else 0.0

    results[param_idx, 0] = ratio
    results[param_idx, 1] = max_dd
    results[param_idx, 2] = avg_deal
    results[param_idx, 3] = num_deals
    results[param_idx, 4] = max_len
