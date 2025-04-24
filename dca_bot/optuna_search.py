"""
Optuna helper – full Python DCATrailingStrategy, early pruning,
multi-CPU, and SQLite lock timeout.  Works on Optuna < 3.4.

The quick-test slice is resampled to **30-minute** candles so
millions of 1-minute bars no longer bottleneck the pruning pass.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Tuple

import optuna
import pandas as pd
import sqlalchemy
import sqlalchemy.pool

from .strategies.dca_ts import DCATrailingStrategy
from .simulator import calc_metrics


# ------------------------------------------------------------------ #
#  Helper to run one full strategy                                   #
# ------------------------------------------------------------------ #
def _evaluate(df: pd.DataFrame,
              spacing: float, tp: float,
              trailing: bool, trail_pct: float
              ) -> Tuple[float, float, float]:
    """Run canonical strategy and return (annual%, dd%, avg_len)."""
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
    )
    deals, eq = bot.backtest(df)
    m = calc_metrics(deals, eq)
    return m["annual_pct"], m["max_drawdown_pct"], m["avg_deal_min"]


# ------------------------------------------------------------------ #
#  Objective with early pruning                                      #
# ------------------------------------------------------------------ #
def make_objective(df_full: pd.DataFrame):
    """
    * Quick head-run uses 30-minute bars (1/6th of the data volume).
    * MedianPruner can therefore discard ~60 % hopeless trials fast.
    """
    head = (df_full.resample("30min").first()
                     .iloc[: max(300, len(df_full) // 10)])

    def _objective(trial: optuna.Trial):
        # --- sample parameters ------------------------------------
        spacing   = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
        tp        = trial.suggest_float("tp_pct",     0.5, 3.0, step=0.1)
        trailing  = trial.suggest_categorical("trailing", [True, False])
        trail_pct = trial.suggest_float("trailing_pct", 0.1, 0.5, step=0.1)

        # enforce effective TP ≥ 0.5 %
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # --- quick 30-min head run --------------------------------
        annual_head, *_ = _evaluate(head,
                                    spacing, tp, trailing, trail_pct)
        trial.report(-annual_head, step=0)     # minimise
        if trial.should_prune():
            raise optuna.TrialPruned()

        # --- full back-test ---------------------------------------
        annual, dd, avg = _evaluate(df_full,
                                    spacing, tp, trailing, trail_pct)

        trial.set_user_attr("metrics", {
            "annual_pct": annual,
            "max_drawdown_pct": dd,
            "avg_deal_min": avg
        })
        trial.set_user_attr("params", {
            "spacing_pct": spacing,
            "tp_pct": tp,
            "trailing": trailing,
            "trailing_pct": trail_pct
        })
        return -annual                                    # minimise

    return _objective


# ------------------------------------------------------------------ #
#  Create + run Optuna study                                         #
# ------------------------------------------------------------------ #
def run_optuna(df: pd.DataFrame, *,
               n_trials: int = 200,
               n_jobs: Optional[int] = None,
               storage: Optional[str] = None,
               seed: int = 42):

    if n_jobs in (None, 0):
        n_jobs = os.cpu_count()

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=10,     # start pruning sooner
        n_warmup_steps=0,
        interval_steps=1,
    )

    # ---------- SQLite storage with 60 s lock timeout -------------
    if storage:
        engine_kw = {
            "connect_args": {"timeout": 60, "check_same_thread": False},
            "poolclass": sqlalchemy.pool.NullPool,
        }
        storage_obj = optuna.storages.RDBStorage(
            url=storage,
            engine_kwargs=engine_kw,
        )
    else:
        storage_obj = None  # in-memory (fastest, non-resumable)

    logging.info("Optuna | trials=%d | jobs=%d | storage=%s",
                 n_trials, n_jobs, storage or "memory")

    study = optuna.create_study(
        study_name="dca_full",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_obj,
        load_if_exists=bool(storage_obj),
    )
    study.optimize(make_objective(df),
                   n_trials=n_trials,
                   n_jobs=n_jobs,
                   show_progress_bar=True)
    return study
