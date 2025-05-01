"""
CLI – runs BEST, SAFE and FAST Optuna studies in one shot, then
re-tests the winning parameter sets.

New flags
---------
--use-sig    1 (default) = wait for Bollinger+RSI trigger
             0           = ignore trigger

--reopen-sec N   Seconds to wait after a deal closes when --use-sig is 0
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Tuple

from ..loader import load_binance
from ..optuna_search import run_three_studies
from ..strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy
from ..simulator import calc_metrics
from ..plotting import equity_curve, panel

# -------------------------------------------------------------------- constants
RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)

# -------------------------------------------------------------------- helpers
def run_set(
    params: Dict,
    df,
    label: str,
    base: str,
    use_sig: int,
    reopen_sec: int,
    fast_ema: int | None,
    slow_ema: int | None,
) -> Tuple[Dict, str, Tuple]:
    """Back-test one parameter set and return (metrics, PNG path, panel item)."""
    bot = DCATrailingStrategy(
        **params,
        use_sig=use_sig,
        reopen_sec=reopen_sec,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
    )
    deals, eq = bot.backtest(df)
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

    pa.add_argument("--trials", type=int, default=200,
                    help="number of trials for *each* study")
    pa.add_argument("--jobs", type=int, default=0,
                    help="0 = all CPU cores")
    pa.add_argument("--storage", default="sqlite:///dca.sqlite",
                    help="'none' for in-memory Optuna studies")

    # NEW flags -------------------------------------------------------
    pa.add_argument("--use-sig", type=int, choices=[0, 1], default=1,
                    help="1 = use Bollinger/RSI trigger (default); "
                         "0 = ignore trigger")
    pa.add_argument("--reopen-sec", type=int, default=60,
                    help="Delay before reopening when --use-sig 0 "
                         "(default 60 s)")
    pa.add_argument("--fast-ema", type=int, default=None,
                    help="Fast EMA length for trend filter (optional)")
    pa.add_argument("--slow-ema", type=int, default=None,
                    help="Slow EMA length for trend filter (optional)")

    pa.add_argument("-v", "--verbose", action="store_true")
    pa.add_argument("--spacing-pct", type=float, default=None)

    pa.add_argument("--tp-pct",      type=float, default=None)
    pa.add_argument("--trailing",    action="store_true")
    pa.add_argument("--trailing-pct",type=float, default=None)
    args = pa.parse_args()

    # ---------------- EMA sanity checks -----------------
    if (args.fast_ema is None) ^ (args.slow_ema is None):
        pa.error("Both --fast-ema and --slow-ema must be given together")
    if args.fast_ema and args.slow_ema and args.fast_ema >= args.slow_ema:
        pa.error("--fast-ema must be smaller than --slow-ema")

    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )

    # ------------------------------------------------ load candles
    df = load_binance(args.symbol, args.start, args.end, "1m")
    if df.empty:
        sys.exit("No candles returned – check date range.")

    # ------------------------------------------------ run three studies
    best_st, safe_st, fast_st = run_three_studies(
        df,
        symbol=args.symbol,
        n_trials_each=args.trials,
        n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
        storage=args.storage,
        use_sig=args.use_sig,
        reopen_sec=args.reopen_sec,
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        spacing_fix=args.spacing_pct,
        tp_fix=args.tp_pct,
        trailing_fix=args.trailing,
        trailpct_fix=args.trailing_pct,
    )

    def _pick(study):
        t = study.best_trial
        return t.user_attrs["params"], t.user_attrs["metrics"]

    best_p, _ = _pick(best_st)
    safe_p, _ = _pick(safe_st)
    fast_p, _ = _pick(fast_st)

    # ------------------------------------------------ baseline default
    default_p = dict(
        spacing_pct=1,
        tp_pct=0.6,
        trailing=True,
        trailing_pct=0.1,
    )

    def_m, def_png, item_def = run_set(
        default_p, df, "default", args.symbol,
        args.use_sig, args.reopen_sec,
        args.fast_ema, args.slow_ema,
    )
    best_m, best_png, item_best = run_set(
        best_p, df, "best", args.symbol,
        args.use_sig, args.reopen_sec,
        args.fast_ema, args.slow_ema,
    )
    safe_m, safe_png, item_safe = run_set(
        safe_p, df, "safe", args.symbol,
        args.use_sig, args.reopen_sec,
        args.fast_ema, args.slow_ema,
    )
    fast_m, fast_png, item_fast = run_set(
        fast_p, df, "fast", args.symbol,
        args.use_sig, args.reopen_sec,
        args.fast_ema, args.slow_ema,
    )

    # ------------------------------------------------ triple comparison panel
    tri_png = os.path.join(RES, f"{args.symbol}_triple.png")
    panel([item_best, item_safe, item_fast], tri_png)

    # ------------------------------------------------ summary file
    summary = {
        "default": {"params": default_p, "metrics": def_m, "png": def_png},
        "best":    {"params": best_p,    "metrics": best_m, "png": best_png},
        "safe":    {"params": safe_p,    "metrics": safe_m, "png": safe_png},
        "fast":    {"params": fast_p,    "metrics": fast_m, "png": fast_png},
        "panel": tri_png,
    }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_opt_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
