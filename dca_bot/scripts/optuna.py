import argparse
import json
import logging
import os
import sys
from typing import Dict, Tuple
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd
import cupy as cp

from ..loader import load_binance
from ..optuna_search import run_best_study
from ..strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy, _bb_percent, _supertrend, _loop_gpu
from ..simulator import calc_metrics
from ..plotting import equity_curve

# -------------------------------------------------------------------- constants
RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RES, exist_ok=True)

# -------------------------------------------------------------------- helpers
def parse_tf_to_min(tf: str) -> int:
    if tf.endswith('min'):
        return int(tf[:-3])
    elif tf.endswith('h'):
        return int(tf[:-1]) * 60
    elif tf.endswith('d'):
        return int(tf[:-1]) * 1440
    elif tf.endswith('w'):
        return int(tf[:-1]) * 1440 * 7
    raise ValueError(f"Unknown timeframe: {tf}")

def run_set(
    params: Dict,
    df,
    label: str,
    base: str,
    use_sig: int,
    reopen_sec: int,
    long_only: bool = False,
    exit_on_flip: bool = True,
    use_bb_safety: bool = True,
) -> Tuple[Dict, str, Tuple]:
    """Back-test one parameter set and return (metrics, PNG path, panel item)."""
    bot = DCATrailingStrategy(**params, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety)
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
    exit_on_flip: bool = True,
    use_bb_safety: bool = True,
    num_sims: int = 50,  # Reduced from 100
    noise_std: float = 0.001,  # 0.1% std dev noise
) -> Dict:
    """Run Monte Carlo simulations with price perturbations on GPU, in batches to avoid OOM."""
    bot = DCATrailingStrategy(**params, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety)
    batch_size = 20  # Process 20 sims per batch to reduce memory usage
    all_results = []

    px = cp.asarray(df['close'].values, dtype=cp.float32)  # Use float32 to halve memory
    ts = cp.asarray(df.index.view('int64') // 1_000_000_000)
    bb_percent = cp.asarray(_bb_percent(df, params['bb_tf']))
    bull = cp.asarray(_supertrend(df, params['supertrend_tf']))

    for i in range(0, num_sims, batch_size):
        current_batch = min(batch_size, num_sims - i)
        noise = cp.random.normal(0, noise_std, (current_batch, len(px)), dtype=cp.float32)
        px_sims = px * (1 + noise)
        results_batch = cp.zeros((current_batch, 5))  # ratio, max_dd, avg_deal, num_deals, longest_dd_min
        threads_per_block = 128
        blocks = (current_batch + threads_per_block - 1) // threads_per_block
        _loop_gpu[blocks, threads_per_block](
            ts, px_sims, bb_percent, bull,
            params['spacing_pct'], params['tp_pct'], int(params['trailing']), params['trailing_pct'],
            bot.max_safety, bot.base_order, bot.mult,
            bot.fee_rate, bot.initial_balance,
            bot.reopen_sec, int(bot.compound), bot.risk_pct,
            int(bot.long_only), int(exit_on_flip),
            60, int(use_bb_safety), results_batch
        )
        all_results.append(cp.asnumpy(results_batch))
        del noise, px_sims, results_batch  # Free memory

    results = np.concatenate(all_results, axis=0)
    ratios = results[:, 0]
    max_dds = results[:, 1]
    avg_deal_mins = results[:, 2]
    num_dealss = results[:, 3]
    longest_dds = results[:, 4]

    days_span = max((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)
    apys = (ratios ** (365 / days_span) - 1) * 100

    calmar_ratios = [apy / max_dd if max_dd > 0 else apy for apy, max_dd in zip(apys, max_dds)]

    agg = {
        'avg_apy_pct': float(np.mean(apys)),
        'std_apy_pct': float(np.std(apys)),
        'avg_drawdown_pct': float(np.mean(max_dds)),
        'worst_drawdown_pct': float(np.max(max_dds)),
        'avg_deals': float(np.mean(num_dealss)),
        'avg_deal_min': float(np.mean(avg_deal_mins)),
        'avg_calmar_ratio': float(np.mean(calmar_ratios)),
    }
    cp.get_default_memory_pool().free_all_blocks()

    return agg

# -------------------------------------------------------------------- main CLI
def main() -> None:
    pa = argparse.ArgumentParser(description="Best-objective optimiser")
    pa.add_argument("symbol")
    pa.add_argument("start")
    pa.add_argument("end")

    pa.add_argument("--trials", type=int, default=200,
                    help="number of trials for the study")
    pa.add_argument("--jobs", type=int, default=0,
                    help="0 = all CPU cores")
    pa.add_argument("--storage", default="sqlite:///dca.sqlite",
                    help="'none' for in-memory Optuna studies")
    pa.add_argument("--window-months", type=int, default=12,
                    help="Size of each walk-forward window in months (e.g., 6 for half-year slices)")
    pa.add_argument("--full-dataset", action="store_true",
                    help="Disable walk-forward windows and optimize on the full dataset instead (default: False, windows enabled)")

    # NEW flags -------------------------------------------------------
    pa.add_argument("--use-sig", type=int, choices=[0, 1], default=1,
                    help="1 = use trigger (default); "
                         "0 = ignore trigger")
    pa.add_argument("--reopen-sec", type=int, default=60,
                    help="Delay before reopening when --use-sig 0 "
                         "(default 60 s)")
    pa.add_argument("--long-only", type=int, choices=[0, 1], default=0,
                    help="1 = long positions only, 0 = both long and short (default)")
    pa.add_argument("--no-flip-exit", type=int, choices=[0, 1], default=0,
                    help="1 = disable Supertrend flip exits (exit only on TP/trailing), 0 = keep flip exits (default)")
    pa.add_argument("--no-graph", action="store_true",
                    help="Skip generating and saving the equity curve PNG (speeds up execution)")
    pa.add_argument("--no-bb-safety", action="store_true",
                    help="Disable BB condition for safety orders, use constant spacing only")
    pa.add_argument("--supertrend-tf", type=str, default="30min",
                    help="Supertrend timeframe (e.g., 30min, 1h, default: 30min)")
    pa.add_argument("--interval", type=str, default="1m",
                    help="Candle timeframe for download (e.g., 1s, 1m, 5m, default: 1m)")
    pa.add_argument("--use-gpu", action="store_true",
                    help="Use GPU for optimization by evaluating a grid in parallel")

    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    if args.storage.lower() == "none":
        args.storage = None

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )

    # ------------------------------------------------ load candles
    df = load_binance(args.symbol, args.start, args.end, args.interval)
    if df.empty:
        sys.exit("No candles returned – check date range.")
    df = df.sort_index()  # Ensure sorted by time

    # Define exit_on_flip based on flag (1 = no flip exit = False)
    exit_on_flip = args.no_flip_exit == 0
    use_bb_safety = not args.no_bb_safety
    supertrend_tf = args.supertrend_tf

    if args.full_dataset:
        # ------------------------------------------------ Full dataset optimization (no windows)
        logging.info(f"Optimizing on full dataset: {df.index.min()} to {df.index.max()}")
        best_st = run_best_study(
            df,
            symbol=args.symbol,
            n_trials=args.trials,
            n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
            storage=args.storage,
            use_sig=args.use_sig,
            reopen_sec=args.reopen_sec,
            long_only=bool(args.long_only),
            exit_on_flip=exit_on_flip,
            use_bb_safety=use_bb_safety,
            window_id="",  # No window ID for full
            supertrend_tf=supertrend_tf,
            use_gpu=args.use_gpu,
        )

        def _pick(study):
            trials = study.best_trials
            if not trials:
                raise ValueError("No best trials found in the study.")
            # Select the trial with the highest annual_pct (APY)
            best_t = max(trials, key=lambda t: t.user_attrs["metrics"]["annual_pct"])
            return best_t.user_attrs["params"], best_t.user_attrs["metrics"]

        best_p, _ = _pick(best_st)

        # MC on full (no split)
        best_mc = monte_carlo_backtest(df, best_p, args.use_sig, args.reopen_sec, bool(args.long_only), exit_on_flip, use_bb_safety=use_bb_safety)

        window_results = [{
            'window_start': str(df.index.min()),
            'window_end': str(df.index.max()),
            'best': {'params': best_p, 'mc_metrics': best_mc},
        }]
        overall_summary = {'best': [best_mc['avg_calmar_ratio']]}
    else:
        # ------------------------------------------------ Walk-Forward Optimization
        window_results = []
        overall_summary = {'best': []}
        current_start = df.index.min()
        window_size = args.window_months
        slide_months = window_size // 2  # Slide by half the window size for overlap
        while current_start + relativedelta(months=window_size) <= df.index.max():
            end_win = current_start + relativedelta(months=window_size)
            delta = end_win - current_start
            train_days = int(0.7 * delta.days)
            train_end = current_start + pd.Timedelta(days=train_days)
            train_df = df.loc[current_start:train_end]
            test_df = df.loc[train_end + pd.Timedelta(seconds=1):end_win]  # Out-of-sample

            window_id = current_start.strftime('%Y-%m-%d')
            logging.info(f"Processing window: {current_start} to {end_win} (train: {current_start} to {train_end}, test: {train_end} to {end_win})")

            # Run optimization on train
            best_st = run_best_study(
                train_df,
                symbol=args.symbol,
                n_trials=args.trials,
                n_jobs=(os.cpu_count() if args.jobs == 0 else args.jobs),
                storage=args.storage,
                use_sig=args.use_sig,
                reopen_sec=args.reopen_sec,
                long_only=bool(args.long_only),
                exit_on_flip=exit_on_flip,
                use_bb_safety=use_bb_safety,
                window_id=window_id,
                supertrend_tf=supertrend_tf,
                use_gpu=args.use_gpu,
            )

            def _pick(study):
                trials = study.best_trials
                if not trials:
                    raise ValueError("No best trials found in the study.")
                # Select the trial with the highest annual_pct (APY)
                best_t = max(trials, key=lambda t: t.user_attrs["metrics"]["annual_pct"])
                return best_t.user_attrs["params"], best_t.user_attrs["metrics"]

            best_p, _ = _pick(best_st)

            # Validate with Monte Carlo on test
            best_mc = monte_carlo_backtest(test_df, best_p, args.use_sig, args.reopen_sec, bool(args.long_only), exit_on_flip, use_bb_safety=use_bb_safety)

            window_summary = {
                'window_start': str(current_start),
                'window_end': str(end_win),
                'best': {'params': best_p, 'mc_metrics': best_mc},
            }
            window_results.append(window_summary)

            # Aggregate for overall
            overall_summary['best'].append(best_mc['avg_calmar_ratio'])

            current_start += relativedelta(months=slide_months)

    # ------------------------------------------------ Overall aggregates
    overall = {
        'avg_best_calmar': float(np.mean(overall_summary['best'])),
        'windows': window_results,
    }

    # ------------------------------------------------ Compute optimal overall parameters
    if window_results:
        spacings = [w['best']['params']['spacing_pct'] for w in window_results]
        tps = [w['best']['params']['tp_pct'] for w in window_results]
        trailings = [w['best']['params']['trailing'] for w in window_results]
        trail_pcts = [w['best']['params']['trailing_pct'] for w in window_results]
        bb_mins_list = [parse_tf_to_min(w['best']['params']['bb_tf']) for w in window_results]
        st_mins_list = [parse_tf_to_min(w['best']['params']['supertrend_tf']) for w in window_results]

        avg_spacing = np.mean(spacings)
        avg_tp = np.mean(tps)
        majority_trailing = bool(np.sum(trailings) > len(trailings) / 2)  # Explicitly cast to Python bool
        avg_trail_pct = np.mean(trail_pcts)
        avg_bb_min = np.mean(bb_mins_list)
        avg_st_min = np.mean(st_mins_list)

        # Round to nearest 0.1 (matching search step)
        rounded_spacing = round(avg_spacing / 0.1) * 0.1
        rounded_tp = round(avg_tp / 0.1) * 0.1
        rounded_trail_pct = round(avg_trail_pct / 0.1) * 0.1

        possible_bb = ['3min', '5min', '15min', '30min', '1h', '4h']
        possible_st = ['15min', '30min', '1h', '4h', '8h', '1d', '1w']

        closest_bb_tf = min(possible_bb, key=lambda tf: abs(parse_tf_to_min(tf) - avg_bb_min))
        closest_st_tf = min(possible_st, key=lambda tf: abs(parse_tf_to_min(tf) - avg_st_min))

        optimal_params = {
            'spacing_pct': rounded_spacing,
            'tp_pct': rounded_tp,
            'trailing': majority_trailing,
            'trailing_pct': rounded_trail_pct,
            'exit_on_flip': True,
            'bb_tf': closest_bb_tf,
            'supertrend_tf': closest_st_tf,
        }
        print(f"Optimal overall parameters: {json.dumps(optimal_params, indent=2)}")

        # Generate and save "best" graph with optimal params on full data (if not skipped)
        if not args.no_graph:
            optimal_met, optimal_png, _ = run_set(
                optimal_params, df, "optimal", args.symbol, args.use_sig, args.reopen_sec, bool(args.long_only), exit_on_flip, use_bb_safety=use_bb_safety
            )
        else:
            bot = DCATrailingStrategy(**optimal_params, use_sig=args.use_sig, reopen_sec=args.reopen_sec, long_only=bool(args.long_only), use_bb_safety=use_bb_safety)
            deals, eq = bot.backtest(df)
            optimal_met = calc_metrics(deals, eq)
            optimal_png = None

        overall['optimal_params'] = optimal_params
        overall['optimal_png'] = optimal_png
        overall['optimal_mc_metrics'] = monte_carlo_backtest(df, optimal_params, args.use_sig, args.reopen_sec, bool(args.long_only), exit_on_flip, use_bb_safety=use_bb_safety)
    else:
        print("No windows processed – cannot compute optimal parameters.")

    print(json.dumps(overall, indent=2))
    with open(os.path.join(RES, f"{args.symbol}_wfo_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # ------------------------------------------------ baseline default (full data for comparison)
    default_p = dict(
        spacing_pct=0.3,
        tp_pct=0.6,
        trailing=True,
        trailing_pct=0.1,
        bb_tf='3min',
        supertrend_tf=args.supertrend_tf,
    )
    default_mc = monte_carlo_backtest(df, default_p, args.use_sig, args.reopen_sec, bool(args.long_only), exit_on_flip, use_bb_safety=use_bb_safety)
    print(f"Default MC on full data: {json.dumps(default_mc, indent=2)}")


if __name__ == "__main__":
    main()
