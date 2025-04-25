"""
Full-engine Optuna optimiser with:

– early pruning on a 30-minute resample
– SQLite lock-timeout patch
– study-seeding helper so we can reuse completed trials
"""

from __future__ import annotations
import logging, os
from typing import Optional, Tuple, Dict

import optuna
import pandas as pd
import sqlalchemy
import sqlalchemy.pool

from .strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy
from .simulator import calc_metrics


# ------------------------------------------------------------------ #
#  one full back-test                                               #
# ------------------------------------------------------------------ #
def _evaluate(df: pd.DataFrame,
              spacing: float, tp: float,
              trailing: bool, trail_pct: float
              ) -> Dict[str, float]:
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
    )
    deals, eq = bot.backtest(df)
    return calc_metrics(deals, eq)


# ------------------------------------------------------------------ #
#  objective factory                                                #
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
#  objective factory – returns a function Optuna can optimise        #
# ------------------------------------------------------------------ #
def make_objective(df_full: pd.DataFrame, metric_key: str):
    """
    Build a single-metric Optuna objective.

    Parameters
    ----------
    df_full : pd.DataFrame
        The full 1-minute price history.
    metric_key : str
        Which metric to optimise:
        • "annual_pct"         – maximise (profit)
        • "max_drawdown_pct"   – minimise (risk)
        • "avg_deal_min"       – minimise (speed)

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective that Optuna will call.
    """
    # quick 30-minute resample head for pruning
    head = (
        df_full
        .resample("30min").first()
        .iloc[: max(300, len(df_full) // 10)]
    )

    def _objective(trial: optuna.Trial):
        spacing   = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
        tp        = trial.suggest_float("tp_pct",     0.5, 3.0, step=0.1)
        trailing  = trial.suggest_categorical("trailing", [True, False])
        trail_pct = trial.suggest_float("trailing_pct", 0.1, 0.1, step=0.1)

        # invalidate combos where trailing SL is larger than TP
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- fast head-run for early pruning ----------------
        m_head = _evaluate(head, spacing, tp, trailing, trail_pct)
        trial.report(m_head[metric_key], step=0)     # **no negation**

        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---------- full back-test --------------------------------
        m = _evaluate(df_full, spacing, tp, trailing, trail_pct)

        # keep full metrics & params for later inspection
        trial.set_user_attr("metrics", m)
        trial.set_user_attr("params", {
            "spacing_pct": spacing,
            "tp_pct": tp,
            "trailing": trailing,
            "trailing_pct": trail_pct,
        })

        # return the metric to optimise (sign already correct)
        return m["annual_pct"] if metric_key == "annual_pct" else m[metric_key]

    return _objective


# ------------------------------------------------------------------ #
#  create / run a study                                              #
# ------------------------------------------------------------------ #
def _new_study(study_name: str,
               direction: str,
               storage: Optional[str]):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10)

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
        storage_obj = None

    return optuna.create_study(study_name=study_name,
                               direction=direction,
                               sampler=sampler,
                               pruner=pruner,
                               storage=storage_obj,
                               load_if_exists=True)


# ------------------------------------------------------------------ #
#  seed a study with finished trials                                 #
# ------------------------------------------------------------------ #
def seed_from(source: optuna.study.Study,
              dest: optuna.study.Study,
              metric_key: str):
    for t in source.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        m  = t.user_attrs["metrics"]
        p  = t.user_attrs["params"]
        val = -m["annual_pct"] if metric_key == "annual_pct" else m[metric_key]

        cloned = optuna.trial.create_trial(
            params=p,
            distributions=source.best_trial.distributions,
            value=val,
            user_attrs=t.user_attrs,
            state=optuna.trial.TrialState.COMPLETE,
        )
        dest.add_trial(cloned)


# ------------------------------------------------------------------ #
#  high-level helper                                                 #
# ------------------------------------------------------------------ #
def run_three_studies(df: pd.DataFrame,
                      n_trials_each: int,
                      n_jobs: int,
                      storage: Optional[str]):

    # ---------- BEST (annual %) -----------------------------------
    study_best = _new_study("dca_best", "maximize", storage)
    study_best.optimize(make_objective(df, "annual_pct"),
                        n_trials=n_trials_each,
                        n_jobs=n_jobs,
                        show_progress_bar=True)

    # ---------- SAFE (min DD) -------------------------------------
    study_safe = _new_study("dca_safe", "minimize", storage)
    seed_from(study_best, study_safe, "max_drawdown_pct")
    study_safe.optimize(make_objective(df, "max_drawdown_pct"),
                        n_trials=n_trials_each,
                        n_jobs=n_jobs,
                        show_progress_bar=True)

    # ---------- FAST (min avg time) -------------------------------
    study_fast = _new_study("dca_fast", "minimize", storage)
    seed_from(study_best, study_fast, "avg_deal_min")
    study_fast.optimize(make_objective(df, "avg_deal_min"),
                        n_trials=n_trials_each,
                        n_jobs=n_jobs,
                        show_progress_bar=True)

    return study_best, study_safe, study_fast