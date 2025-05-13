
"""one_stage_opt.py – single-stage optimiser with min-orders limit

Usage example:
    python -m dca_bot.scripts.one_stage_opt SOLUSDT 2022-01-01 2025-05-09 \
           --trials 2500 --startup 400 --min-orders 12 --jobs 4 -v

Key points
----------
* Random warm‑up trials (n_startup_trials, default 400)
* Multivariate TPE sampler
* min-orders flag sets lower bound for max_safety
* Drawdown-penalised objective (annual_pct − dd_penalty × drawdown)
* TP range 0.5‑5.0 %, spacing clamp 0.2‑10 %
* ≥ $6 first order rule, duplicate pruning
"""

from __future__ import annotations
import argparse, json, logging, math, os
from typing import Tuple
import optuna, sqlalchemy
import sqlalchemy.pool

from ..loader import load_binance
from ..strategies.dca_ts_numba import DCAJITStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve

MIN_FIRST_ORDER_USD = 6.0
MAX_SPACING_CAP = 10.0
_EXTRA_KEYS = {"reserved_budget","first_safety_order","ladder_ratio","first_so_ratio"}
_SEEN: set[Tuple[int,float,float,float,bool]] = set()

def reserved_budget(base, mult, n):
    return base*(n+1) if mult==1 else base*(1-mult**(n+1))/(1-mult)

def calc_base(balance, mult, n):
    if mult == 1:
        return balance / (n + 1)
    return balance / ((1 - mult**(n+1)) / (1 - mult))

def _evaluate(df, params, use_sig, reopen):
    st = DCAJITStrategy(**params, use_sig=use_sig, reopen_sec=reopen)
    deals, eq = st.backtest(df)
    return deals, eq, calc_metrics(deals, eq)

def _study(name, storage, startup):
    sampler = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=startup, multivariate=True, warn_independent_sampling=False)
    return optuna.create_study(
        study_name=name, direction="maximize",
        sampler=sampler, pruner=optuna.pruners.NopPruner(),
        storage=None if storage is None else optuna.storages.RDBStorage(
            url=storage,
            engine_kwargs={"connect_args":{"timeout":60,"check_same_thread":False},
                           "poolclass": sqlalchemy.pool.NullPool}),
        load_if_exists=True)

def main():
    pa = argparse.ArgumentParser("one‑stage optimiser with min-orders")
    pa.add_argument("symbol"); pa.add_argument("start"); pa.add_argument("end")
    pa.add_argument("--trials", type=int, default=2500)
    pa.add_argument("--startup", type=int, default=400)
    pa.add_argument("--min-orders", type=int, default=3)
    pa.add_argument("--jobs", type=int, default=0)
    pa.add_argument("--storage", default="sqlite:///dca.sqlite")
    pa.add_argument("--initial-balance", type=float, default=1000.0)
    pa.add_argument("--use-sig", type=int, choices=[0,1], default=1)
    pa.add_argument("--reopen-sec", type=int, default=60)
    pa.add_argument("--dd-penalty", type=float, default=0.3)
    pa.add_argument("-v","--verbose", action="store_true")
    args = pa.parse_args()

    if args.storage.lower() == "none":
        args.storage = None
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        raise SystemExit("No candles found")

    default_params = dict(
        base_order = calc_base(args.initial_balance,1.0,50),
        mult = 1.0,
        max_safety = 50,
        spacing_pct = 1.0,
        tp_pct = 0.6,
        trailing = True,
        trailing_pct = 0.1,
        compound = False,
        risk_pct = 0.0,
        fee_rate = 0.001,
        initial_balance = args.initial_balance,
    )

    study = _study(f"{args.symbol}_one_stage", args.storage, args.startup)

    def objective(trial: optuna.Trial):
        n = trial.suggest_int("max_safety", args.min_orders, 50, step=1)
        mult = trial.suggest_float("mult", 1.0, 2.0, step=0.05)
        s_max_cov = (1 - 0.01 ** (1 / n)) * 100
        s_max = math.floor(min(s_max_cov, MAX_SPACING_CAP) * 10) / 10
        spacing = trial.suggest_float("spacing_pct", 0.3, s_max, step=0.1)
        tp = trial.suggest_float("tp_pct", 0.5, 5.0, step=0.1)
        trailing = trial.suggest_categorical("trailing", [True, False])

        key = (n, mult, spacing, tp, trailing)
        if key in _SEEN:
            raise optuna.TrialPruned()
        _SEEN.add(key)

        base = calc_base(args.initial_balance, mult, n)
        if base < MIN_FIRST_ORDER_USD:
            raise optuna.TrialPruned()

        params_full = default_params | {
            "base_order": base,
            "mult": mult,
            "max_safety": n,
            "spacing_pct": spacing,
            "tp_pct": tp,
            "trailing": trailing,
        }
        params_clean = {k:v for k,v in params_full.items() if k not in _EXTRA_KEYS}
        _,_,metrics = _evaluate(df, params_clean, args.use_sig, args.reopen_sec)
        score = metrics["annual_pct"] - args.dd_penalty * metrics["max_drawdown_pct"]
        trial.set_user_attr("params", params_full | {
            "reserved_budget": reserved_budget(base,mult,n),
            "first_safety_order": base*mult,
            "ladder_ratio": mult,
            "first_so_ratio": mult,
        })
        trial.set_user_attr("metrics", metrics)
        return score

    study.optimize(objective, n_trials=args.trials,
                   n_jobs=(os.cpu_count() if args.jobs==0 else args.jobs),
                   show_progress_bar=not args.verbose)

    best = study.best_trial.user_attrs["params"]
    best_metrics = study.best_trial.user_attrs["metrics"]

    res_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    def plot(label, p_full):
        p_clean = {k:v for k,v in p_full.items() if k not in _EXTRA_KEYS}
        _,eq,_ = _evaluate(df, p_clean, args.use_sig, args.reopen_sec)
        fname = f"{args.symbol}_{label}.png"
        equity_curve(eq, [], label, os.path.join(res_dir, fname))
        return fname

    def_png = plot("default", default_params)
    best_png = plot("best", best)

    summary = {
        "default": {"params": default_params, "png": def_png},
        "best": {"params": best, "metrics": best_metrics, "png": best_png},
        "trials": len(study.trials),
        "min_orders": args.min_orders,
    }
    out = os.path.join(res_dir, f"{args.symbol}_one_stage.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Written →", out)

if __name__ == "__main__":
    import math, os, json
    main()
