"""
Full‑engine Optuna optimiser – *study names are now per‑symbol*.

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
    fast_ema: Optional[int] = None,
    slow_ema: Optional[int] = None,
) -> Dict[str, float]:
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
        use_sig=use_sig,
        reopen_sec=reopen_sec,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
    )
    deals, eq = bot.backtest(df)
    return calc_metrics(deals, eq)


# ------------------------------------------------------------------ #
#  objective factory                                                 #
# ------------------------------------------------------------------ #

def make_objective(
    df_full: pd.DataFrame,
    metric_key: str,
    *,
    use_sig: int,
    reopen_sec: int,
    fast_ema: Optional[int],
    slow_ema: Optional[int],
):
    """Return an Optuna objective that optimises a single metric."""

    head = (
        df_full.resample("30min").first().iloc[: max(300, len(df_full) // 10)]
    )

    # ------------------------------------------------------------------ #
    def _objective(trial: optuna.Trial):
        spacing = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
        tp = trial.suggest_float("tp_pct", 0.5, 3.0, step=0.1)
        trailing = trial.suggest_categorical("trailing", [True, False])
        trail_pct = trial.suggest_float("trailing_pct", 0.1, 0.1, step=0.1)

        # ---- skip exact‑duplicate parameter sets --------------------
        sig = _param_sig(spacing, tp, trailing, trail_pct)
        if sig in _seen_params:
            raise optuna.TrialPruned()
        _seen_params.add(sig)

        # invalidate combos where trailing SL is larger than TP
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- fast head‑run for early pruning --------------------
        m_head = _evaluate(
            head, spacing, tp, trailing, trail_pct,
            use_sig=use_sig, reopen_sec=reopen_sec,
            fast_ema=fast_ema, slow_ema=slow_ema,
        )
        trial.report(m_head[metric_key], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---------- full back‑test ------------------------------------
        m_full = _evaluate(
            df_full, spacing, tp, trailing, trail_pct,
            use_sig=use_sig, reopen_sec=reopen_sec,
            fast_ema=fast_ema, slow_ema=slow_ema,
        )
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

def _new_study(base_name: str, direction: str, storage: Optional[str], symbol: str):
    """Create (or reopen) an Optuna study whose name is unique per symbol."""

    full_name = f"{base_name}_{symbol}"
    sampler = optuna.samplers.RandomSampler(seed=42)  # no duplicates
    # Disable early‑stopping of “bad” trials for now
    #pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
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
#  seed a study with finished trials                                 #
# ------------------------------------------------------------------ #

def seed_from(source: optuna.study.Study, dest: optuna.study.Study, metric_key: str):
    """Clone completed trials from *source* to *dest*, skipping NaNs."""

    import math

    for t in source.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        m = t.user_attrs.get("metrics", {})
        if metric_key not in m:
            continue
        val = m[metric_key]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue

        cloned = optuna.trial.create_trial(
            params=t.user_attrs["params"],
            distributions=source.best_trial.distributions,
            value=val,
            user_attrs=t.user_attrs,
            state=optuna.trial.TrialState.COMPLETE,
        )
        dest.add_trial(cloned)

        # register the cloned parameters so later samplers won't repeat them
        sig = _param_sig(
            cloned.params["spacing_pct"],
            cloned.params["tp_pct"],
            cloned.params["trailing"],
            cloned.params["trailing_pct"],
        )
        _seen_params.add(sig)


# ------------------------------------------------------------------ #
#  high‑level helper                                                 #
# ------------------------------------------------------------------ #

def run_three_studies(
    df: pd.DataFrame,
    symbol: str,
    n_trials_each: int,
    n_jobs: int,
    storage: Optional[str],
    use_sig: int = 1,
    reopen_sec: int = 60,
    fast_ema: Optional[int] = None,
    slow_ema: Optional[int] = None,
):
    """Run BEST, SAFE and FAST Optuna studies for *symbol*."""

    # ---------- BEST (annual %) --------------------------------------
    study_best = _new_study("dca_best", "maximize", storage, symbol)
    study_best.optimize(
        make_objective(
            df, "annual_pct",
            use_sig=use_sig, reopen_sec=reopen_sec,
            fast_ema=fast_ema, slow_ema=slow_ema,
        ),
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---------- SAFE (min DD) ---------------------------------------
    study_safe = _new_study("dca_safe", "minimize", storage, symbol)
    seed_from(study_best, study_safe, "max_drawdown_pct")
    study_safe.optimize(
        make_objective(
            df, "max_drawdown_pct",
            use_sig=use_sig, reopen_sec=reopen_sec,
            fast_ema=fast_ema, slow_ema=slow_ema,
        ),
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---------- FAST (max number of deals) --------------------------
    study_fast = _new_study("dca_fast", "maximize", storage, symbol)
    seed_from(study_best, study_fast, "deals")          # copy existing trials
    study_fast.optimize(
        make_objective(
            df, "deals",
            use_sig=use_sig, reopen_sec=reopen_sec,
            fast_ema=fast_ema, slow_ema=slow_ema,
        ),                    # optimise the “deals” metric
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    return study_best, study_safe, study_fast
