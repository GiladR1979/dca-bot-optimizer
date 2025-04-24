"""
CLI – runs BEST, SAFE, FAST studies with trial-seeding.
"""

import argparse, json, logging, os, sys
from typing import Dict

from ..loader import load_binance
from ..optuna_search import run_three_studies
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)


def run_set(params: Dict, df, label, base):
    deals, eq = DCATrailingStrategy(**params).backtest(df)
    met = calc_metrics(deals, eq)
    png = os.path.join(RES, f"{base}_{label}.png")
    equity_curve(eq, deals, label, png)
    return met, png, (eq, deals, label)


def main():
    pa = argparse.ArgumentParser(description="Three-objective optimiser")
    pa.add_argument("symbol"); pa.add_argument("start"); pa.add_argument("end")
    pa.add_argument("--trials", type=int, default=200)
    pa.add_argument("--jobs",   type=int, default=0)
    pa.add_argument("--storage", default="sqlite:///dca.sqlite",
                    help="'none' for in-memory")
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()
    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("optuna3")

    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        sys.exit("No candles returned; check date range.")

    # run three studies with seeding
    best, safe, fast = run_three_studies(
        df,
        n_trials_each=args.trials,
        n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
        storage=args.storage)

    # extract winners
    def top(study):
        t = study.best_trial
        return t.user_attrs["params"], t.user_attrs["metrics"]

    best_p, best_m = top(best)
    safe_p, safe_m = top(safe)
    fast_p, fast_m = top(fast)

    # baseline default
    default_p = dict(spacing_pct=1, tp_pct=0.6, trailing=True, trailing_pct=0.1)
    def_m, def_png, _ = run_set(default_p, df, "default", args.symbol)

    best_m, best_png, _ = run_set(best_p, df, "best",  args.symbol)
    safe_m, safe_png, _ = run_set(safe_p, df, "safe",  args.symbol)
    fast_m, fast_png, _ = run_set(fast_p, df, "fast",  args.symbol)

    tri_png = os.path.join(RES, f"{args.symbol}_triple.png")
    panel([], tri_png, items=[
        (None, None, "BEST"), (None, None, "SAFE"), (None, None, "FAST")
    ])  # simple placeholder panel

    summary = {
        "default": {"params": default_p, "metrics": def_m,  "png": def_png},
        "best":    {"params": best_p,    "metrics": best_m, "png": best_png},
        "safe":    {"params": safe_p,    "metrics": safe_m, "png": safe_png},
        "fast":    {"params": fast_p,    "metrics": fast_m, "png": fast_png},
        "panel":   tri_png
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_opt_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()