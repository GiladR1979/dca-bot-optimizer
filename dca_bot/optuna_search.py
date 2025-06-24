"""
optuna_search.py  –  full-engine Optuna optimiser (BEST / SAFE / FAST)

• Study names are suffixed with <SYMBOL> to prevent cross-pair contamination
  inside a shared SQLite file (e.g.  dca_best_BTCUSDT).

• NEW flag  --exit-on-flip {0|1}
  0 = keep the running ladder alive and *freeze* new entries until it exits ≥ 0
  1 = (default) close the deal immediately on every 8 h SuperTrend reversal
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Dict, Tuple

import optuna
import pandas as pd
import sqlalchemy
import sqlalchemy.pool

from .strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy
from .simulator import calc_metrics
from .loader import load_binance

# ------------------------------------------------------------------ #
#  duplicate-trial guard                                             #
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


# ------------------------------------------------------------------ #
#  one full back-test                                                #
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
    exit_on_flip: int,
) -> Dict[str, float]:
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
        use_sig=use_sig,
        reopen_sec=reopen_sec,
        exit_on_flip=bool(exit_on_flip),
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
    exit_on_flip: int,
):
    """
    Build an Optuna objective that

    1.   runs a *head* sub-sample for early pruning,
    2.   runs the full back-test if the head passes,
    3.   stores metrics & params in `user_attrs`,
    4.   respects duplicate-trial cache.
    """

    head = df_full.resample("30min").first().iloc[: max(300, len(df_full) // 10)]

    def _objective(trial: optuna.Trial):
        spacing = trial.suggest_float("spacing_pct", 0.3, 2.0, step=0.1)
        tp = trial.suggest_float("tp_pct", 0.5, 3.0, step=0.1)
        trailing = trial.suggest_categorical("trailing", [True, False])
        trail_pct = trial.suggest_float("trailing_pct", 0.1, 0.5, step=0.1)

        # ---- skip exact-duplicate parameter sets -----------------
        sig = _param_sig(spacing, tp, trailing, trail_pct)
        if sig in _seen_params:
            raise optuna.TrialPruned()
        _seen_params.add(sig)

        # ---- discard combos where trailing stop > TP -------------
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- fast head-run ---------------------------------
        m_head = _evaluate(
            head,
            spacing,
            tp,
            trailing,
            trail_pct,
            use_sig=use_sig,
            reopen_sec=reopen_sec,
            exit_on_flip=exit_on_flip,
        )
        trial.report(m_head[metric_key], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---------- full back-test --------------------------------
        m_full = _evaluate(
            df_full,
            spacing,
            tp,
            trailing,
            trail_pct,
            use_sig=use_sig,
            reopen_sec=reopen_sec,
            exit_on_flip=exit_on_flip,
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
#  duplicate cache initialisation                                    #
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
#  create / open a study                                             #
# ------------------------------------------------------------------ #
def _new_study(base_name: str, direction: str, storage: Optional[str], symbol: str):
    """Create (or reopen) an Optuna study whose name is unique per symbol."""
    full_name = f"{base_name}_{symbol}"
    sampler = optuna.samplers.RandomSampler(seed=42)
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
    _register_trials(study)
    return study


# ------------------------------------------------------------------ #
#  seed helper                                                       #
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

        _seen_params.add(
            _param_sig(
                cloned.params["spacing_pct"],
                cloned.params["tp_pct"],
                cloned.params["trailing"],
                cloned.params["trailing_pct"],
            )
        )


# ------------------------------------------------------------------ #
#  high-level orchestrator                                           #
# ------------------------------------------------------------------ #
def run_three_studies(
    df: pd.DataFrame,
    *,
    symbol: str,
    n_trials_each: int,
    n_jobs: int,
    storage: Optional[str],
    use_sig: int = 1,
    reopen_sec: int = 60,
    exit_on_flip: int = 1,
):
    """Run BEST, SAFE and FAST Optuna studies for *symbol*."""
    # ---------- BEST : maximise annual % ---------------------------
    best = _new_study("dca_best", "maximize", storage, symbol)
    best.optimize(
        make_objective(df, "annual_pct", use_sig=use_sig, reopen_sec=reopen_sec,
                       exit_on_flip=exit_on_flip),
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---------- SAFE : minimise max DD ----------------------------
    safe = _new_study("dca_safe", "minimize", storage, symbol)
    seed_from(best, safe, "max_drawdown_pct")
    safe.optimize(
        make_objective(df, "max_drawdown_pct", use_sig=use_sig, reopen_sec=reopen_sec,
                       exit_on_flip=exit_on_flip),
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---------- FAST : maximise deal count ------------------------
    fast = _new_study("dca_fast", "maximize", storage, symbol)
    seed_from(best, fast, "deals")
    fast.optimize(
        make_objective(df, "deals", use_sig=use_sig, reopen_sec=reopen_sec,
                       exit_on_flip=exit_on_flip),
        n_trials=n_trials_each,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return best, safe, fast


# ------------------------------------------------------------------ #
#  CLI wrapper                                                       #
# ------------------------------------------------------------------ #
def _cli() -> None:
    import argparse
    import json
    import sys

    RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(RES, exist_ok=True)

    pa = argparse.ArgumentParser(description="BEST / SAFE / FAST optimiser")
    pa.add_argument("symbol")
    pa.add_argument("start")
    pa.add_argument("end")

    pa.add_argument("--trials", type=int, default=200)
    pa.add_argument("--jobs", type=int, default=0)
    pa.add_argument("--storage", default="sqlite:///dca_search.sqlite")

    pa.add_argument("--use-sig", type=int, choices=[0, 1], default=1)
    pa.add_argument("--reopen-sec", type=int, default=60)
    pa.add_argument("--exit-on-flip", type=int, choices=[0, 1], default=1)

    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    if args.storage.lower() == "none":
        args.storage = None
    if args.jobs == 0:
        args.jobs = os.cpu_count()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        print("No candles returned – check inputs", file=sys.stderr)
        sys.exit(1)

    best, safe, fast = run_three_studies(
        df,
        symbol=args.symbol,
        n_trials_each=args.trials,
        n_jobs=args.jobs,
        storage=args.storage,
        use_sig=args.use_sig,
        reopen_sec=args.reopen_sec,
        exit_on_flip=args.exit_on_flip,
    )

    def _pack(study):
        t = study.best_trial
        return dict(params=t.user_attrs["params"], metrics=t.user_attrs["metrics"])

    summary = {"best": _pack(best), "safe": _pack(safe), "fast": _pack(fast)}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_search_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    _cli()
