"""
Optuna helper — Bayesian optimisation with multi-CPU + Numba acceleration.
"""

import logging
import os

import numpy as np
import optuna

try:
    from numba import njit
except ImportError:  # NumPy fallback if Numba missing
    njit = lambda *a, **k: (lambda f: f)

from .simulator import calc_metrics         # only for ROI/DD math


# ------------------------------------------------------------------ #
#  Numba-JIT back-test (vectorised outer loop)                       #
# ------------------------------------------------------------------ #
@njit(fastmath=True, cache=True)
def dca_loop(prices: np.ndarray,
             spacing: float, tp: float,
             trailing: int, trailing_gap: float):
    """
    Very small, numba-friendly DCA engine.

    Returns: annual_% , max_dd_% , avg_deal_min
    """
    cash = 1000.0
    qty = 0.0
    avg = 0.0
    peak_eq = cash
    max_dd = 0.0
    deals = 0
    deal_len_sum = 0
    deal_open_idx = -1

    n = prices.shape[0]
    for i in range(n):
        p = prices[i]

        # open first deal immediately
        if qty == 0.0:
            buy_cost = cash / 51.0          # base $19.61
            qty += buy_cost / p
            cash -= buy_cost
            avg = p
            dca_level = avg * (1 - spacing / 100)
            tp_price = avg * (1 + tp / 100)
            trail_armed = False
            trail_high = 0.0
            deal_open_idx = i

        # DCA buy
        if p <= dca_level and qty < (51 / 50) * (cash / p):
            buy_cost = cash / 50.0
            qty += buy_cost / p
            cash -= buy_cost
            avg = (avg * (qty - buy_cost / p) + buy_cost) / qty
            dca_level = avg * (1 - spacing / 100)
            tp_price = avg * (1 + tp / 100)
            trail_armed = False
            trail_high = 0.0

        # update trailing
        if trailing and qty > 0:
            if p >= tp_price and not trail_armed:
                trail_armed = True
                trail_high = p
            if trail_armed:
                if p > trail_high:
                    trail_high = p
                stop = trail_high * (1 - trailing_gap / 100)
                if p <= stop:
                    cash += qty * p          # sell all
                    qty = 0
                    deals += 1
                    deal_len_sum += (i - deal_open_idx)
                    continue  # next candle

        # take-profit without trailing
        if not trailing and p >= tp_price and qty > 0:
            cash += qty * p
            qty = 0
            deals += 1
            deal_len_sum += (i - deal_open_idx)

        # equity / DD
        eq = cash + qty * p
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq
        if dd > max_dd:
            max_dd = dd

    # final equity (ignore open deal)
    final_eq = cash
    roi_pct = (final_eq - 1000.0) / 10.0     # % vs 1 000
    days = n / 1440.0
    annual_pct = ((1 + roi_pct / 100) ** (365 / days) - 1) * 100 if days else 0
    avg_deal_min = deal_len_sum / deals if deals else n
    return annual_pct, max_dd * 100, avg_deal_min


# ------------------------------------------------------------------ #
#  Optuna wrapper                                                    #
# ------------------------------------------------------------------ #
def make_objective(prices):
    """Factory so we capture the NumPy array once, not per trial."""
    def _objective(trial):
        spacing   = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
        tp        = trial.suggest_float("tp_pct",     0.5, 3.0, step=0.1)
        trailing  = trial.suggest_categorical("trailing", [True, False])
        trail_pct = trial.suggest_float("trailing_pct", 0.1, 0.5, step=0.1)

        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        annual, dd, avg_len = dca_loop(
            prices, spacing, tp, int(trailing), trail_pct)

        m = {
            "annual_pct": annual,
            "max_drawdown_pct": dd,
            "avg_deal_min": avg_len
        }
        trial.set_user_attr("metrics", m)
        trial.set_user_attr("params", {
            "spacing_pct": spacing,
            "tp_pct": tp,
            "trailing": trailing,
            "trailing_pct": trail_pct
        })
        return -annual   # minimise
    return _objective


def run_optuna(df, *, n_trials=200, n_jobs=None,
               storage=None, seed=42):
    """
    df : DataFrame with 'close' column
    n_jobs : 0/None ⇒ all logical cores
    storage: sqlite:///…  for persistence & multi-proc locking
    """
    if n_jobs in (None, 0):
        n_jobs = os.cpu_count()

    prices = df["close"].values.astype(np.float64)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
        storage=storage,
        load_if_exists=bool(storage),
    )

    logging.info("Optuna | trials=%d | jobs=%d", n_trials, n_jobs)
    study.optimize(make_objective(prices),
                   n_trials=n_trials,
                   n_jobs=n_jobs,
                   show_progress_bar=True)
    return study
