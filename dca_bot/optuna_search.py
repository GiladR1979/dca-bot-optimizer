"""
Optuna helper: Bayesian search with optional multi-CPU execution.
"""

import logging
import os
import optuna

from .strategies.dca_ts import DCATrailingStrategy
from .simulator import calc_metrics


def _objective(trial, df):
    """Return negative annual % so Optuna can minimise the loss."""
    spacing  = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
    tp       = trial.suggest_float("tp_pct",     0.5, 5.0, step=0.1)
    trailing = trial.suggest_categorical("trailing", [True, False])
    trailing_pct = trial.suggest_float("trailing_pct", 0.1, 0.1, step=0.05)

    # rule: effective TP ≥ 0.5 %
    if trailing and tp - trailing_pct < 0.5 - 1e-9:
        raise optuna.TrialPruned()

    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trailing_pct,
    )
    deals, eq = bot.backtest(df)
    annual = calc_metrics(deals, eq)["annual_pct"]
    trial.set_user_attr("metrics", calc_metrics(deals, eq))
    trial.set_user_attr("params",  {
        "spacing_pct": spacing,
        "tp_pct": tp,
        "trailing": trailing,
        "trailing_pct": trailing_pct,
    })
    return -annual


def run_optuna(df, *, n_trials=200, n_jobs=None,
               study_name="dca_optuna", storage=None, seed=42):
    """
    n_jobs:   int | None
        #worker processes; None = os.cpu_count() (all cores)
    """
    if n_jobs in (None, 0):
        n_jobs = os.cpu_count()

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    logging.info("Optuna: %d trials | %d parallel jobs", n_trials, n_jobs)
    study.optimize(lambda tr: _objective(tr, df),
                   n_trials=n_trials,
                   n_jobs=n_jobs,            # ← multi-CPU here
                   show_progress_bar=True)
    return study