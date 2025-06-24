import argparse
import json
import logging
import os

from ..loader import load_binance
from ..optimiser import grid_search
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel   # ← import the new helper

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)


def run_set(params, df, label, base, exit_on_flip):
    deals, eq = DCATrailingStrategy(**params, exit_on_flip=bool(exit_on_flip)).backtest(df)
    met = calc_metrics(deals, eq)
    png = os.path.join(RES, f"{base}_{label}.png")
    equity_curve(eq, deals, label, png)
    return met, png, (eq, deals, label)


def main():
    pa = argparse.ArgumentParser(description="Grid optimise DCA bot")
    pa.add_argument("symbol")
    pa.add_argument("start")
    pa.add_argument("end")
    pa.add_argument("--spacings", default="0.5,1,1.5,2")
    pa.add_argument("--tps", default="0.5,0.6,1")
    pa.add_argument("--trailing-pct", type=float, default=0.1)
    pa.add_argument("--exit-on-flip", type=int, choices=[0, 1], default=1)
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    df = load_binance(args.symbol, args.start, args.end, "1m")

    # --- default run -------------------------------------------------
    default_params = {"spacing_pct": 1,
                      "tp_pct": 0.6,
                      "trailing": True,
                      "trailing_pct": 0.1}

    met_def, png_def, item_def = run_set(
        default_params, df, "default", args.symbol, args.exit_on_flip)

    # --- grid search -------------------------------------------------
    grid = {
        "spacing_pct": [float(x) for x in args.spacings.split(",")],
        "tp_pct":      [float(x) for x in args.tps.split(",")],
        "trailing":    [True, False],
        "trailing_pct": [args.trailing_pct],
    }

    res = grid_search(df, grid)
    best_params, best_met = res["best"]
    safe_params, safe_met = res["safe"]
    fast_params, fast_met = res["fast"]

    met_best, png_best, item_best = run_set(
        best_params, df, "best", args.symbol)
    met_safe, png_safe, item_safe = run_set(
        safe_params, df, "safe", args.symbol)
    met_fast, png_fast, item_fast = run_set(
        fast_params, df, "fast", args.symbol)

    # --- quad plot ---------------------------------------------------
    quad_png = os.path.join(RES, f"{args.symbol}_quad.png")
    panel([item_def, item_best, item_safe, item_fast], quad_png)

    summary = {
        "default": {"params": default_params, "metrics": met_def, "png": png_def},
        "best":    {"params": best_params,    "metrics": met_best, "png": png_best},
        "safe":    {"params": safe_params,    "metrics": met_safe, "png": png_safe},
        "fast":    {"params": fast_params,    "metrics": met_fast, "png": png_fast},
        "quad":    quad_png
    }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(
            RES, f"{args.symbol}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
