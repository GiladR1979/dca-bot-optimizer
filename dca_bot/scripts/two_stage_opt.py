
"""two_stage_opt.py  (v8)

• Stage 1: optimise max_safety, mult, spacing_pct
  – spacing_pct ∈ 0.2 % … 10 % (down‑rounded to 0.1 %)
  – base_order is solved so full ladder = initial_balance
  – base_order must be ≥ 6 USD
  – duplicate combos pruned

• Stage 2: optimise tp_pct, trailing starting from Stage‑1 winner
  – duplicate combos pruned
  – bookkeeping keys removed before back‑test

Outputs: results/<SYMBOL>_{default,best}.png + two_stage.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Dict, Tuple

import optuna
import pandas as pd
import sqlalchemy
import sqlalchemy.pool

from ..loader import load_binance
from ..strategies.dca_ts_numba import DCAJITStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve

# ───────────────────────── constants ─────────────────────────
MIN_FIRST_ORDER_USD = 6.0
MAX_SPACING_CAP = 10.0  # %
_EXTRA_KEYS = {  # drop before feeding strategy
    "reserved_budget",
    "first_safety_order",
    "ladder_ratio",
    "first_so_ratio",
}

# duplicate tracking sets
_SEEN_STAGE1: set[Tuple[int, float, float]] = set()
_SEEN_STAGE2: set[Tuple[float, bool]] = set()

# ───────────────────── helper functions ─────────────────────
def reserved_budget(base: float, mult: float, n: int) -> float:
    return base * (n + 1) if mult == 1 else base * (1 - mult ** (n + 1)) / (1 - mult)

def calc_base(balance: float, mult: float, n: int) -> float:
    if mult == 1:
        return balance / (n + 1)
    ladder_sum = (1 - mult ** (n + 1)) / (1 - mult)
    return balance / ladder_sum

def _evaluate(df: pd.DataFrame, params: Dict, use_sig: int, reopen_sec: int):
    strat = DCAJITStrategy(**params, use_sig=use_sig, reopen_sec=reopen_sec)
    deals, eq = strat.backtest(df)
    return deals, eq, calc_metrics(deals, eq)

def _study(name: str, storage: str | None):
    return optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        storage=None
        if storage is None
        else optuna.storages.RDBStorage(
            url=storage,
            engine_kwargs={
                "connect_args": {"timeout": 60, "check_same_thread": False},
                "poolclass": sqlalchemy.pool.NullPool,
            },
        ),
        load_if_exists=True,
    )

# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    pa = argparse.ArgumentParser("Two‑stage optimiser v8")
    pa.add_argument("symbol")
    pa.add_argument("start")
    pa.add_argument("end")
    pa.add_argument("--trials1", type=int, default=200)
    pa.add_argument("--trials2", type=int, default=200)
    pa.add_argument("--jobs", type=int, default=0)
    pa.add_argument("--storage", default="sqlite:///dca.sqlite")
    pa.add_argument("--initial-balance", type=float, default=1000.0)
    pa.add_argument("--use-sig", type=int, choices=[0, 1], default=1)
    pa.add_argument("--reopen-sec", type=int, default=60)
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # candles
    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        raise SystemExit("No candles for given range.")

    # reference default params
    default_mult, default_n = 1.0, 50
    default_base = calc_base(args.initial_balance, default_mult, default_n)
    default_params = dict(
        base_order=default_base,
        mult=default_mult,
        max_safety=default_n,
        spacing_pct=1.0,
        tp_pct=0.6,
        trailing=True,
        trailing_pct=0.1,
        compound=False,
        risk_pct=0.0,
        fee_rate=0.001,
        initial_balance=args.initial_balance,
    )

    # ───────────────────── Stage 1 ─────────────────────
    study1 = _study(f"{args.symbol}_stage1", args.storage)

    def objective1(trial: optuna.Trial):
        n_safety = trial.suggest_int("max_safety", 3, 50, step=1)
        mult = trial.suggest_float("mult", 1.0, 2.0, step=0.05)

        s_max_cov = (1 - 0.01 ** (1 / n_safety)) * 100
        s_max = math.floor(min(s_max_cov, MAX_SPACING_CAP) * 10) / 10
        spacing = trial.suggest_float("spacing_pct", 0.2, s_max, step=0.1)

        # duplicate pruning
        key = (n_safety, mult, spacing)
        if key in _SEEN_STAGE1:
            raise optuna.TrialPruned()
        _SEEN_STAGE1.add(key)

        base = calc_base(args.initial_balance, mult, n_safety)
        if base < MIN_FIRST_ORDER_USD:
            raise optuna.TrialPruned()

        params = default_params | {
            "base_order": base,
            "mult": mult,
            "max_safety": n_safety,
            "spacing_pct": spacing,
        }

        deals, eq, metrics = _evaluate(df, params, args.use_sig, args.reopen_sec)
        trial.set_user_attr(
            "params",
            params
            | {
                "reserved_budget": reserved_budget(base, mult, n_safety),
                "first_safety_order": base * mult,
                "ladder_ratio": mult,
                "first_so_ratio": mult,
            },
        )
        trial.set_user_attr("metrics", metrics)
        return metrics["annual_pct"]

    study1.optimize(
        objective1,
        n_trials=args.trials1,
        n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
        show_progress_bar=not args.verbose,
    )

    best1_params_full = study1.best_trial.user_attrs["params"]
    logging.info("Stage‑1 best: %s", best1_params_full)

    # ───────────────────── Stage 2 ─────────────────────
    study2 = _study(f"{args.symbol}_stage2", args.storage)

    def objective2(trial: optuna.Trial):
        params_full = best1_params_full | {
            "tp_pct": trial.suggest_float("tp_pct", 0.1, 1.0, step=0.1),
            "trailing": trial.suggest_categorical("trailing", [True, False]),
        }

        # duplicate pruning
        key = (params_full["tp_pct"], params_full["trailing"])
        if key in _SEEN_STAGE2:
            raise optuna.TrialPruned()
        _SEEN_STAGE2.add(key)

        params_clean = {k: v for k, v in params_full.items() if k not in _EXTRA_KEYS}

        deals, eq, metrics = _evaluate(df, params_clean, args.use_sig, args.reopen_sec)
        trial.set_user_attr("params", params_full)
        trial.set_user_attr("metrics", metrics)
        return metrics["annual_pct"]

    study2.optimize(
        objective2,
        n_trials=args.trials2,
        n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
        show_progress_bar=not args.verbose,
    )

    best2_params_full = study2.best_trial.user_attrs["params"]
    best2_metrics = study2.best_trial.user_attrs["metrics"]
    logging.info("Stage‑2 best: %s", best2_params_full)

    # ───────────────────── outputs ─────────────────────
    res_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(res_dir, exist_ok=True)

    def run_plot(label: str, p_full: Dict):
        p_clean = {k: v for k, v in p_full.items() if k not in _EXTRA_KEYS}
        deals, eq, met = _evaluate(df, p_clean, args.use_sig, args.reopen_sec)
        png = os.path.join(res_dir, f"{args.symbol}_{label}.png")
        equity_curve(eq, deals, label, png)
        return met, os.path.basename(png)

    def_met, def_png = run_plot("default", default_params)
    best_met, best_png = run_plot("best", best2_params_full)

    summary = {
        "default": {"params": default_params, "metrics": def_met, "png": def_png},
        "best": {"params": best2_params_full, "metrics": best_met, "png": best_png},
        "stage1_trials": len(study1.trials),
        "stage2_trials": len(study2.trials),
    }

    json_path = os.path.join(res_dir, f"{args.symbol}_two_stage.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Written → {json_path}")

if __name__ == "__main__":
    main()
