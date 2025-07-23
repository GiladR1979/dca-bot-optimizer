"""
Full‑engine Optuna optimiser – *study names are per‑symbol*.

This prevents cross‑contamination when you optimise multiple coins into the
same SQLite file.  The only change is that each study name gets the `symbol`
appended (e.g. `dca_best_BTCUSDT`).  Everything else is identical.
"""
from __future__ import annotations
import logging
import os
from typing import Optional, Dict, Tuple

import optuna
import pandas as pd

# ------------------------------------------------------------------ #
#  duplicate‑trial guard (Optuna 2.x)                                #
# ------------------------------------------------------------------ #
_seen_params: set[tuple] = set()

def _param_sig(spacing: float, tp: float, trailing: bool, trail_pct: float) -> tuple:
    """Rounded signature so small FP noise does not count as new."""
    return (
        round(spacing, 3),
        round(tp, 3),
        bool(trailing),
        round(trail_pct, 3),
    )
import sqlalchemy
import sqlalchemy.pool

from .strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy
from .simulator import calc_metrics

# ------------------------------------------------------------------ #
#  one full back‑test                                                #
# ------------------------------------------------------------------ #

def _evaluate(
    df: pd.DataFrame,
    spacing: float,
    tp: float,
    trailing: bool,
    trail_pct: float,
    *,
    use_sig: int,
    reopen_sec: int,
    long_only: bool = False,
    exit_on_flip: bool = True,
) -> Dict[str, float]:
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
        use_sig=use_sig,
        reopen_sec=reopen_sec,
        long_only=long_only,
        exit_on_flip=exit_on_flip,
    )
    deals, eq = bot.backtest(df)
    return calc_metrics(deals, eq)


# ------------------------------------------------------------------ #
#  objective factory                                                 #
# ------------------------------------------------------------------ #

def make_objective(df_full: pd.DataFrame, metric_key: str, *, use_sig: int, reopen_sec: int, long_only: bool = False, exit_on_flip: bool = True):
    """Return an Optuna objective that optimises a single metric."""

    head = (
        df_full.resample("30min").first().iloc[: max(300, len(df_full) // 10)]
    )

    # ------------------------------------------------------------------ #
    def _objective(trial: optuna.Trial):
        spacing = trial.suggest_float("spacing_pct", 0.3, 9.0, step=0.1)
        tp = trial.suggest_float("tp_pct", 0.5, 3.0, step=0.1)
        trailing = trial.suggest_categorical("trailing", [True, False])
        trail_pct = 0.1

        # ---- skip exact‑duplicate parameter sets --------------------
        sig = _param_sig(spacing, tp, trailing, trail_pct)
        if sig in _seen_params:
            raise optuna.TrialPruned()
        _seen_params.add(sig)

        # invalidate combos where trailing SL is larger than TP
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- fast head‑run for early pruning --------------------
        m_head = _evaluate(head, spacing, tp, trailing, trail_pct, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, exit_on_flip=exit_on_flip)
        trial.report(m_head[metric_key], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---------- full back‑test ------------------------------------
        m_full = _evaluate(df_full, spacing, tp, trailing, trail_pct, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, exit_on_flip=exit_on_flip)
        trial.set_user_attr("metrics", m_full)
        trial.set_user_attr(
            "params",
            {
                "spacing_pct": spacing,
                "tp_pct": tp,
                "trailing": trailing,
                "trailing_pct": trail_pct,
            },
        )
        return m_full[metric_key]

    return _objective


# ------------------------------------------------------------------ #
#  register existing trials in the duplicate cache                   #
# ------------------------------------------------------------------ #
def _register_trials(study: optuna.study.Study):
    """Push signatures of all COMPLETE trials into _seen_params."""
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        sig = _param_sig(
            t.params.get("spacing_pct"),
            t.params.get("tp_pct"),
            t.params.get("trailing"),
            t.params.get("trailing_pct"),
        )
        _seen_params.add(sig)


# ------------------------------------------------------------------ #
#  create / run a study                                              #
# ------------------------------------------------------------------ #

def _new_study(base_name: str, direction: str, storage: Optional[str], symbol: str, window_id: str = ""):
    """Create (or reopen) an Optuna study whose name is unique per symbol and window."""

    full_name = f"{base_name}_{symbol}"
    if window_id:
        full_name += f"_{window_id}"
    sampler = optuna.samplers.TPESampler(seed=42)  # no duplicates
    pruner = optuna.pruners.NopPruner()

    if storage:
        engine_kw = {
            "connect_args": {"timeout": 60, "check_same_thread": False},
            "poolclass": sqlalchemy.pool.NullPool,
        }
        storage_obj = optuna.storages.RDBStorage(url=storage, engine_kwargs=engine_kw)
    else:
        storage_obj = None

    study = optuna.create_study(
        study_name=full_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage_obj,
        load_if_exists=True,
    )
    # make sure duplicates already in DB are remembered
    _register_trials(study)

    return study


# ------------------------------------------------------------------ #
#  high‑level helper                                                 #
# ------------------------------------------------------------------ #

def run_best_study(
    df: pd.DataFrame,
    symbol: str,
    n_trials: int,
    n_jobs: int,
    storage: Optional[str],
    use_sig: int = 1,
    reopen_sec: int = 60,
    long_only: bool = False,
    exit_on_flip: bool = True,
    window_id: str = "",
):
    study_best = _new_study("dca_best", "maximize", storage, symbol, window_id)
    study_best.optimize(
        make_objective(df, "annual_pct", use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, exit_on_flip=exit_on_flip),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study_best
