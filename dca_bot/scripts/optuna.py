"""
CLI – full-engine Optuna optimiser with early pruning & SQLite lock fix.
"""

import argparse, json, logging, os, optuna, sys
from ..loader import load_binance
from ..optuna_search import run_optuna
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)


def run_set(params, df, label, base):
    deals, eq = DCATrailingStrategy(**params).backtest(df)
    met = calc_metrics(deals, eq)
    png = os.path.join(RES, f"{base}_{label}.png")
    equity_curve(eq, deals, label, png)
    return met, png, (eq, deals, label)


def main():
    pa = argparse.ArgumentParser(description="Optuna optimiser (full engine)")
    pa.add_argument("symbol"); pa.add_argument("start"); pa.add_argument("end")
    pa.add_argument("--trials", type=int, default=200)
    pa.add_argument("--jobs", type=int, default=0)
    pa.add_argument("--storage", default="sqlite:///dca.sqlite",
                    help="Optuna storage URI or 'none' for in-memory")
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()
    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("optuna-cli")

    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        sys.exit("No candles returned. Check symbol or date range.")

    default_params = dict(spacing_pct=1, tp_pct=0.6,
                          trailing=True, trailing_pct=0.1)
    met_def, png_def, item_def = run_set(default_params, df,
                                         "default", args.symbol)

    study = run_optuna(df,
                       n_trials=args.trials,
                       n_jobs=args.jobs,
                       storage=args.storage)

    best = safe = fast = None
    best_m = safe_m = fast_m = None
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        m = t.user_attrs["metrics"]; p = t.user_attrs["params"]
        if not best_m or m["annual_pct"] > best_m["annual_pct"]:
            best, best_m = p, m
        if not safe_m or m["max_drawdown_pct"] < safe_m["max_drawdown_pct"]:
            safe, safe_m = p, m
        if not fast_m or m["avg_deal_min"] < fast_m["avg_deal_min"]:
            fast, fast_m = p, m

    log.info("best %s", best_m)
    log.info("safe %s", safe_m)
    log.info("fast %s", fast_m)

    met_best, png_best, item_best = run_set(best, df, "best", args.symbol)
    met_safe, png_safe, item_safe = run_set(safe, df, "safe", args.symbol)
    met_fast, png_fast, item_fast = run_set(fast, df, "fast", args.symbol)

    trip_png = os.path.join(RES, f"{args.symbol}_optuna_triple.png")
    panel([item_best, item_safe, item_fast], trip_png)

    summary = {
        "default": {"params": default_params, "metrics": met_def, "png": png_def},
        "best":    {"params": best,  "metrics": met_best,  "png": png_best},
        "safe":    {"params": safe,  "metrics": met_safe,  "png": png_safe},
        "fast":    {"params": fast,  "metrics": met_fast,  "png": png_fast},
        "triple":  trip_png
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(
            RES, f"{args.symbol}_optuna_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
