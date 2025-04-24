"""
CLI – runs BEST, SAFE and FAST Optuna studies in one shot.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Tuple, List

from ..loader import load_binance
from ..optuna_search import run_three_studies
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel

# -------------------------------------------------------------------- constants
RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)


# -------------------------------------------------------------------- helpers
def run_set(params: Dict, df, label: str, base: str) -> Tuple[Dict, str, Tuple]:
    """
    Back-test one parameter set, return:
        • metrics dict
        • path to PNG equity-curve
        • tuple (eq, deals, label) for multi-plot panels
    """
    deals, eq = DCATrailingStrategy(**params).backtest(df)
    met = calc_metrics(deals, eq)

    png = os.path.join(RES, f"{base}_{label}.png")
    equity_curve(eq, deals, label, png)

    return met, png, (eq, deals, label)


# -------------------------------------------------------------------- main CLI
def main() -> None:
    pa = argparse.ArgumentParser(description="Three-objective optimiser")
    pa.add_argument("symbol")
    pa.add_argument("start")
    pa.add_argument("end")
    pa.add_argument("--trials",  type=int, default=200,
                    help="number of trials for *each* study")
    pa.add_argument("--jobs",    type=int, default=0,
                    help="0 = all CPU cores")
    pa.add_argument("--storage", default="sqlite:///dca.sqlite",
                    help="'none' for in-memory Optuna studies")
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("optuna3")

    # ------------------------------------------------ load candles
    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        sys.exit("No candles returned – check date range.")

    # ------------------------------------------------ run three studies
    best_st, safe_st, fast_st = run_three_studies(
        df,
        n_trials_each=args.trials,
        n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
        storage=args.storage,
    )

    def _pick(study):
        t = study.best_trial
        return t.user_attrs["params"], t.user_attrs["metrics"]

    best_p, best_m = _pick(best_st)
    safe_p, safe_m = _pick(safe_st)
    fast_p, fast_m = _pick(fast_st)

    # ------------------------------------------------ baseline default
    default_p = dict(spacing_pct=1,
                     tp_pct=0.6,
                     trailing=True,
                     trailing_pct=0.1)

    def_m,  def_png,  item_def  = run_set(default_p, df, "default", args.symbol)
    best_m, best_png, item_best = run_set(best_p,    df, "best",    args.symbol)
    safe_m, safe_png, item_safe = run_set(safe_p,    df, "safe",    args.symbol)
    fast_m, fast_png, item_fast = run_set(fast_p,    df, "fast",    args.symbol)

    # ------------------------------------------------ triple comparison panel
    tri_png = os.path.join(RES, f"{args.symbol}_triple.png")
    panel([item_best, item_safe, item_fast], tri_png)

    # ------------------------------------------------ summary file
    summary = {
        "default": {"params": default_p, "metrics": def_m,  "png": def_png},
        "best":    {"params": best_p,    "metrics": best_m, "png": best_png},
        "safe":    {"params": safe_p,    "metrics": safe_m, "png": safe_png},
        "fast":    {"params": fast_p,    "metrics": fast_m, "png": fast_png},
        "panel":   tri_png,
    }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_opt_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
