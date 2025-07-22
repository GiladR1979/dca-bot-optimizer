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
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd

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
    long_only: bool = False,
) -> Tuple[Dict, str, Tuple]:
    """Back-test one parameter set and return (metrics, PNG path, panel item)."""
    bot = DCATrailingStrategy(**params, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, slippage_pct=0.001)
    deals, eq = bot.backtest(df)
    met = calc_metrics(deals, eq)

    png = os.path.join(RES, f"{base}_{label}.png")
    equity_curve(eq, deals, label, png)

    return met, png, (eq, deals, label)

def monte_carlo_backtest(
    df: pd.DataFrame,
    params: Dict,
    use_sig: int,
    reopen_sec: int,
    long_only: bool = False,
    num_sims: int = 100,
    noise_std: float = 0.001,  # 0.1% std dev noise
) -> Dict:
    """Run Monte Carlo simulations with price perturbations."""
    all_met = []
    for _ in range(num_sims):
        df_pert = df.copy()
        # Multiplicative noise for realistic volatility simulation
        df_pert['close'] *= (1 + np.random.normal(0, noise_std, len(df_pert)))
        bot = DCATrailingStrategy(**params, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, slippage_pct=0.001)
        deals, eq = bot.backtest(df_pert)
        met = calc_metrics(deals, eq)
        all_met.append(met)

    # Aggregate key metrics
    agg = {
        'avg_apy_pct': np.mean([m['apy_pct'] for m in all_met]),
        'std_apy_pct': np.std([m['apy_pct'] for m in all_met]),
        'worst_drawdown_pct': np.max([m['max_drawdown_pct'] for m in all_met]),
        'avg_deals': np.mean([m['deals'] for m in all_met]),
    }
    return agg

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
    pa.add_argument("--long-only", type=int, choices=[0, 1], default=0,
                    help="1 = long positions only, 0 = both long and short (default)")

    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

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
    df = df.sort_index()  # Ensure sorted by time

    # ------------------------------------------------ Walk-Forward Optimization
    window_results = []
    overall_summary = {'best': [], 'safe': [], 'fast': []}
    current_start = df.index.min()
    while current_start + relativedelta(months=12) <= df.index.max():
        end_win = current_start + relativedelta(months=12)
        delta = end_win - current_start
        train_days = int(0.7 * delta.days)
        train_end = current_start + pd.Timedelta(days=train_days)
        train_df = df.loc[current_start:train_end]
        test_df = df.loc[train_end + pd.Timedelta(seconds=1):end_win]  # Out-of-sample

        logging.info(f"Processing window: {current_start} to {end_win} (train: {current_start} to {train_end}, test: {train_end} to {end_win})")

        # Run optimization on train
        best_st, safe_st, fast_st = run_three_studies(
            train_df,
            symbol=args.symbol,
            n_trials_each=args.trials,
            n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
            storage=args.storage,
            use_sig=args.use_sig,
            reopen_sec=args.reopen_sec,
            long_only=bool(args.long_only),
        )

        def _pick(study):
            t = study.best_trial
            return t.user_attrs["params"], t.user_attrs["metrics"]

        best_p, _ = _pick(best_st)
        safe_p, _ = _pick(safe_st)
        fast_p, _ = _pick(fast_st)

        # Validate with Monte Carlo on testF
        best_mc = monte_carlo_backtest(test_df, best_p, args.use_sig, args.reopen_sec, bool(args.long_only))
        safe_mc = monte_carlo_backtest(test_df, safe_p, args.use_sig, args.reopen_sec, bool(args.long_only))
        fast_mc = monte_carlo_backtest(test_df, fast_p, args.use_sig, args.reopen_sec, bool(args.long_only))

        window_summary = {
            'window_start': str(current_start),
            'window_end': str(end_win),
            'best': {'params': best_p, 'mc_metrics': best_mc},
            'safe': {'params': safe_p, 'mc_metrics': safe_mc},
            'fast': {'params': fast_p, 'mc_metrics': fast_mc},
        }
        window_results.append(window_summary)

        # Aggregate for overall
        overall_summary['best'].append(best_mc['avg_apy_pct'])
        overall_summary['safe'].append(safe_mc['avg_apy_pct'])
        overall_summary['fast'].append(fast_mc['avg_apy_pct'])

        current_start += relativedelta(months=6)

    # ------------------------------------------------ Overall aggregates
    overall = {
        'avg_best_apy': np.mean(overall_summary['best']),
        'avg_safe_apy': np.mean(overall_summary['safe']),
        'avg_fast_apy': np.mean(overall_summary['fast']),
        'windows': window_results,
    }

    print(json.dumps(overall, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_wfo_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # ------------------------------------------------ baseline default (full data for comparison)
    default_p = dict(
        spacing_pct=1,
        tp_pct=0.6,
        trailing=True,
        trailing_pct=0.1,
    )
    default_mc = monte_carlo_backtest(df, default_p, args.use_sig, args.reopen_sec, bool(args.long_only))
    print(f"Default MC on full data: {json.dumps(default_mc, indent=2)}")


if __name__ == "__main__":
    main()