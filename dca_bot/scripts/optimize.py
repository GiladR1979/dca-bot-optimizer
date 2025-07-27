import argparse
import json
import logging
import os

from ..loader import load_binance
from ..optimiser import grid_search_advanced
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)


def run_set(params, df, label, base, use_sig, reopen_sec, long_only, exit_on_flip, use_bb_safety):
    # Merge the command line args with the params
    full_params = {
        **params,
        'use_sig': use_sig,
        'reopen_sec': reopen_sec,
        'long_only': long_only,
        'exit_on_flip': exit_on_flip,
        'use_bb_safety': use_bb_safety
    }

    deals, eq = DCATrailingStrategy(**full_params).backtest(df)
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

    # Add the missing command line arguments
    pa.add_argument("--use-sig", type=int, choices=[0, 1], default=1,
                    help="1 = use trigger (default); 0 = ignore trigger")
    pa.add_argument("--reopen-sec", type=int, default=60,
                    help="Delay before reopening when --use-sig 0 (default 60 s)")
    pa.add_argument("--long-only", type=int, choices=[0, 1], default=0,
                    help="1 = long positions only, 0 = both long and short (default)")
    pa.add_argument("--no-flip-exit", type=int, choices=[0, 1], default=0,
                    help="1 = disable Supertrend flip exits (exit only on TP/trailing), 0 = keep flip exits (default)")
    pa.add_argument("--no-bb-safety", action="store_true",
                    help="Disable BB condition for safety orders, use constant spacing only")
    pa.add_argument("--supertrend-tf", type=str, default="30min",
                    help="Supertrend timeframe (e.g., 30min, 1h, default: 30min)")
    pa.add_argument("--bb-tf", type=str, default="3min",
                    help="Bollinger Bands timeframe (e.g., 3min, 5min, default: 3min)")

    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    df = load_binance(args.symbol, args.start, args.end, "1m")

    # Convert flags to expected format
    exit_on_flip = args.no_flip_exit == 0
    use_bb_safety = not args.no_bb_safety

    # --- default run -------------------------------------------------
    default_params = {
        "spacing_pct": 1,
        "tp_pct": 0.6,
        "trailing": True,
        "trailing_pct": 0.1,
        "bb_tf": args.bb_tf,
        "supertrend_tf": args.supertrend_tf
    }

    met_def, png_def, item_def = run_set(
        default_params, df, "default", args.symbol,
        args.use_sig, args.reopen_sec, bool(args.long_only),
        exit_on_flip, use_bb_safety
    )

    # --- grid search -------------------------------------------------
    grid = {
        "spacing_pct": [float(x) for x in args.spacings.split(",")],
        "tp_pct": [float(x) for x in args.tps.split(",")],
        "trailing": [True, False],
        "trailing_pct": [args.trailing_pct],
        "bb_tf": [args.bb_tf],  # Use the command line value
        "supertrend_tf": [args.supertrend_tf]  # Use the command line value
    }

    # Use an updated grid search that passes command line args
    res = grid_search_advanced(
        df, grid,
        use_sig=args.use_sig,
        reopen_sec=args.reopen_sec,
        long_only=bool(args.long_only),
        exit_on_flip=exit_on_flip,
        use_bb_safety=use_bb_safety
    )

    best_params, best_met = res["best"]
    safe_params, safe_met = res["safe"]
    fast_params, fast_met = res["fast"]

    met_best, png_best, item_best = run_set(
        best_params, df, "best", args.symbol,
        args.use_sig, args.reopen_sec, bool(args.long_only),
        exit_on_flip, use_bb_safety
    )
    met_safe, png_safe, item_safe = run_set(
        safe_params, df, "safe", args.symbol,
        args.use_sig, args.reopen_sec, bool(args.long_only),
        exit_on_flip, use_bb_safety
    )
    met_fast, png_fast, item_fast = run_set(
        fast_params, df, "fast", args.symbol,
        args.use_sig, args.reopen_sec, bool(args.long_only),
        exit_on_flip, use_bb_safety
    )

    # --- quad plot ---------------------------------------------------
    quad_png = os.path.join(RES, f"{args.symbol}_quad.png")
    panel([item_def, item_best, item_safe, item_fast], quad_png)

    summary = {
        "default": {"params": default_params, "metrics": met_def, "png": png_def},
        "best": {"params": best_params, "metrics": met_best, "png": png_best},
        "safe": {"params": safe_params, "metrics": met_safe, "png": png_safe},
        "fast": {"params": fast_params, "metrics": met_fast, "png": png_fast},
        "quad": quad_png,
        # Include command line args in summary for reference
        "cmd_args": {
            "use_sig": args.use_sig,
            "reopen_sec": args.reopen_sec,
            "long_only": args.long_only,
            "exit_on_flip": exit_on_flip,
            "use_bb_safety": use_bb_safety,
            "supertrend_tf": args.supertrend_tf,
            "bb_tf": args.bb_tf
        }
    }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(
            RES, f"{args.symbol}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
