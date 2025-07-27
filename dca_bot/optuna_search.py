"""
Full‑engine Optuna optimiser – *study names are per‑symbol*.
"""
from __future__ import annotations
import logging
import os
from typing import Optional, Dict, Tuple
import math
from tqdm import tqdm
import json  # Added import to fix NameError
from collections import defaultdict

import optuna
import pandas as pd
import numpy as np
import cupy as cp
from optuna.trial import TrialState
from numba import cuda
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False  # Suppress low occupancy warnings

PARAM_RANGES = {
    "spacing_pct": {"low": 0.3, "high": 9.0, "step": 0.1},
    "tp_pct": {"low": 0.5, "high": 5.0, "step": 0.1},
    "trailing": [False, True],
    "trailing_pct": {"low": 0.1, "high": 0.5, "step": 0.1},
    "exit_on_flip": [True],
    "bb_tf": ['3min', '5min', '15min', '30min', '1h', '4h', '8h'],
    "supertrend_tf": ['5min', '15min', '30min', '1h', '4h', '8h', '1d', 'W'],
}

# ------------------------------------------------------------------ #
#  duplicate‑trial guard (Optuna 2.x)                                #
# ------------------------------------------------------------------ #
_seen_params: set[tuple] = set()

def _param_sig(spacing: float, tp: float, trailing: bool, trail_pct: float, exit_on_flip: bool, bb_tf: str, supertrend_tf: str) -> tuple:
    """Rounded signature so small FP noise does not count as new."""
    return (
        round(spacing, 3),
        round(tp, 3),
        bool(trailing),
        round(trail_pct, 3),
        bool(exit_on_flip),
        bb_tf,
        supertrend_tf,
    )

import sqlalchemy
import sqlalchemy.pool

from .strategies.dca_ts_numba import DCAJITStrategy as DCATrailingStrategy, _bb_percent, _supertrend, _grid_gpu
from .simulator import calc_metrics

# ------------------------------------------------------------------ #
#  one full back‑test                                                #
# ------------------------------------------------------------------ #

def _evaluate(
    df: pd.DataFrame,
    spacing: float,
    tp: float,
    trailing: bool,
    trail_pct: float,
    exit_on_flip: bool,
    bb_tf: str,
    supertrend_tf: str,
    *,
    use_sig: int,
    reopen_sec: int,
    long_only: bool = False,
    use_bb_safety: bool = True,
) -> Dict[str, float]:
    bot = DCATrailingStrategy(
        spacing_pct=spacing,
        tp_pct=tp,
        trailing=trailing,
        trailing_pct=trail_pct,
        use_sig=use_sig,
        reopen_sec=reopen_sec,
        long_only=long_only,
        exit_on_flip=exit_on_flip,
        use_bb_safety=use_bb_safety,
        supertrend_tf=supertrend_tf,
    )
    deals, eq = bot.backtest(df)
    return calc_metrics(deals, eq)

# ------------------------------------------------------------------ #
#  objective factory                                                 #
# ------------------------------------------------------------------ #

def make_objective(df_full: pd.DataFrame, *, use_sig: int, reopen_sec: int, long_only: bool = False, use_bb_safety: bool = True):
    """Return an Optuna objective that optimizes multiple metrics."""

    head = (
        df_full.resample("30min").first().iloc[: max(300, len(df_full) // 10)]
    )

    # ------------------------------------------------------------------ #
    def _objective(trial: optuna.Trial):
        spacing = trial.suggest_float("spacing_pct", **PARAM_RANGES["spacing_pct"])
        tp = trial.suggest_float("tp_pct", **PARAM_RANGES["tp_pct"])
        trailing = trial.suggest_categorical("trailing", PARAM_RANGES["trailing"])
        trail_pct = trial.suggest_float("trailing_pct", **PARAM_RANGES["trailing_pct"])
        exit_on_flip = trial.suggest_categorical("exit_on_flip", PARAM_RANGES["exit_on_flip"])
        bb_tf = trial.suggest_categorical("bb_tf", PARAM_RANGES["bb_tf"])
        supertrend_tf = trial.suggest_categorical("supertrend_tf", PARAM_RANGES["supertrend_tf"])

        # ---- skip exact‑duplicate parameter sets --------------------
        sig = _param_sig(spacing, tp, trailing, trail_pct, exit_on_flip, bb_tf, supertrend_tf)
        if sig in _seen_params:
            raise optuna.TrialPruned()
        _seen_params.add(sig)

        # invalidate combos where trailing SL is larger than TP
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- full back‑test ------------------------------------
        # (Removed head-run pruning as Trial.report is not supported in MOO)
        m_full = _evaluate(df_full, spacing, tp, trailing, trail_pct, exit_on_flip, bb_tf, supertrend_tf, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety)
        full_apy = m_full['annual_pct']
        full_dd = m_full['max_drawdown_pct']

        trial.set_user_attr("metrics", m_full)
        trial.set_user_attr(
            "params",
            {
                "spacing_pct": spacing,
                "tp_pct": tp,
                "trailing": trailing,
                "trailing_pct": trail_pct,
                "exit_on_flip": exit_on_flip,
                "bb_tf": bb_tf,
                "supertrend_tf": supertrend_tf,
            },
        )
        return full_apy, full_dd  # Multi-objective: max APY, min DD

    return _objective

# ------------------------------------------------------------------ #
#  register existing trials in the duplicate cache                   #
# ------------------------------------------------------------------ #
def _register_trials(study: optuna.study.Study):
    """Push signatures of all COMPLETE trials into _seen_params."""
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Skip if missing required params
        if 'spacing_pct' not in t.params or 'tp_pct' not in t.params or 'trailing' not in t.params:
            continue
        sig = _param_sig(
            t.params.get("spacing_pct"),
            t.params.get("tp_pct"),
            t.params.get("trailing"),
            t.params.get("trailing_pct") or 0.1,  # Default if missing
            t.params.get("exit_on_flip"),
            t.params.get("bb_tf"),
            t.params.get("supertrend_tf"),
        )
        _seen_params.add(sig)

# ------------------------------------------------------------------ #
#  create / run a study                                              #
# ------------------------------------------------------------------ #

def _new_study(base_name: str, directions: list, storage: Optional[str], symbol: str, window_id: str = ""):
    """Create (or reopen) an Optuna study whose name is unique per symbol and window."""

    full_name = f"{base_name}_{symbol}"
    if window_id:
        full_name += f"_{window_id}"
    sampler = optuna.samplers.NSGAIISampler(seed=42)  # Multi-objective sampler
    pruner = optuna.pruners.NopPruner()

    if storage:
        engine_kw = {
            "connect_args": {"timeout": 60, "check_same_thread": False},
            "poolclass": sqlalchemy.pool.NullPool,
        }
        storage_obj = optuna.storages.RDBStorage(url=storage, engine_kwargs=engine_kw)
    else:
        storage_obj = None

    study = optuna.create_study(
        study_name=full_name,
        directions=directions,  # ['maximize', 'minimize']
        sampler=sampler,
        pruner=pruner,
        storage=storage_obj,
        load_if_exists=True,
    )
    # make sure duplicates already in DB are remembered
    _register_trials(study)

    return study

# ------------------------------------------------------------------ #
#  high‑level helper                                                 #
# ------------------------------------------------------------------ #

def run_best_study(
    df: pd.DataFrame,
    symbol: str,
    n_trials: int = 100,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    use_sig: int = 1,
    reopen_sec: int = 60,
    long_only: bool = False,
    exit_on_flip: bool = True,
    use_bb_safety: bool = True,
    window_id: str = "",
    supertrend_tf: str = "30min",
    use_gpu: bool = False,
    max_dd_threshold: float = 20.0,
):
    study_best = _new_study("dca_best", ["maximize", "minimize"], storage, symbol, window_id)
    if use_gpu:
        import itertools
        # Generate grids with exact rounding to avoid FP precision issues
        spacing_list = [round(PARAM_RANGES["spacing_pct"]["low"] + PARAM_RANGES["spacing_pct"]["step"] * i, 1) for i in range(int((PARAM_RANGES["spacing_pct"]["high"] - PARAM_RANGES["spacing_pct"]["low"]) / PARAM_RANGES["spacing_pct"]["step"]) + 1)]
        tp_list = [round(PARAM_RANGES["tp_pct"]["low"] + PARAM_RANGES["tp_pct"]["step"] * i, 1) for i in range(int((PARAM_RANGES["tp_pct"]["high"] - PARAM_RANGES["tp_pct"]["low"]) / PARAM_RANGES["tp_pct"]["step"]) + 1)]
        trail_pct_list = [round(PARAM_RANGES["trailing_pct"]["low"] + PARAM_RANGES["trailing_pct"]["step"] * i, 1) for i in range(int((PARAM_RANGES["trailing_pct"]["high"] - PARAM_RANGES["trailing_pct"]["low"]) / PARAM_RANGES["trailing_pct"]["step"]) + 1)]
        grid = {
            "spacing_pct": spacing_list,
            "tp_pct": tp_list,
            "trailing": PARAM_RANGES["trailing"],
            "trailing_pct": trail_pct_list,
            "exit_on_flip": PARAM_RANGES["exit_on_flip"],
            "bb_tf": PARAM_RANGES["bb_tf"],
            "supertrend_tf": PARAM_RANGES["supertrend_tf"],
        }

        params_list = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), combo))
            if p["trailing"] and p["tp_pct"] - p["trailing_pct"] < 0.5:
                continue
            if not p["trailing"] and p["tp_pct"] < 0.5:
                continue
            params_list.append(p)

        # GPU batch evaluate
        ts = df.index.view('int64') // 1_000_000_000
        px = df['close'].values
        bb_tfs = grid['bb_tf']
        st_tfs = grid['supertrend_tf']
        bb_arrays = [_bb_percent(df, tf) for tf in bb_tfs]
        st_arrays = [_supertrend(df, tf) for tf in st_tfs]
        bb_cp = cp.asarray(bb_arrays)
        bull_cp = cp.asarray(st_arrays)
        px_cp = cp.asarray(px)
        ts_cp = cp.asarray(ts)

        # Batch processing with pruning
        batch_size = 10000  # Adjust based on GPU memory
        index = 0
        b = 0  # Batch counter
        eval_list = []  # List to store (param, apy, dd) for all evaluated
        spacing_apys = defaultdict(list)
        spacing_dds = defaultdict(list)
        tp_apys = defaultdict(list)
        tp_dds = defaultdict(list)
        trailing_apys = defaultdict(list)
        trailing_dds = defaultdict(list)
        trailing_pct_apys = defaultdict(list)
        trailing_pct_dds = defaultdict(list)
        bb_tf_apys = defaultdict(list)
        bb_tf_dds = defaultdict(list)
        supertrend_tf_apys = defaultdict(list)
        supertrend_tf_dds = defaultdict(list)
        pareto_candidates = []
        with tqdm(total=len(params_list), desc=f"Evaluating grid (window {window_id})") as pbar:
            while index < len(params_list):
                end = min(index + batch_size, len(params_list))
                batch = params_list[index:end]

                if not batch:
                    break

                params_cp_batch = cp.array([
                    [
                        p['spacing_pct'],
                        p['tp_pct'],
                        1 if p['trailing'] else 0,
                        p['trailing_pct'],
                        1 if p['exit_on_flip'] else 0,
                        bb_tfs.index(p['bb_tf']),
                        st_tfs.index(p['supertrend_tf'])
                    ] for p in batch
                ])

                outputs_batch = cp.zeros((len(batch), 5))
                threads = 256  # Increased for better occupancy
                blocks = math.ceil(len(batch) / threads)
                _grid_gpu[blocks, threads](ts_cp, px_cp, bb_cp, bull_cp, params_cp_batch, outputs_batch)
                outputs_np_batch = cp.asnumpy(outputs_batch)

                for idx, p in enumerate(batch):
                    row = outputs_np_batch[idx]
                    ratio = row[0]
                    days_span = max((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)
                    exp = 365 / days_span
                    if ratio >= 0:
                        apy = (ratio ** exp - 1) * 100
                    else:
                        apy = float(np.real((complex(ratio) ** exp - 1))) * 100

                    max_dd = row[1]

                    eval_list.append((p, apy, max_dd))

                    # Append to groups
                    spacing_apys[p['spacing_pct']].append(apy)
                    spacing_dds[p['spacing_pct']].append(max_dd)
                    tp_apys[p['tp_pct']].append(apy)
                    tp_dds[p['tp_pct']].append(max_dd)
                    trailing_apys[p['trailing']].append(apy)
                    trailing_dds[p['trailing']].append(max_dd)
                    trailing_pct_apys[p['trailing_pct']].append(apy)
                    trailing_pct_dds[p['trailing_pct']].append(max_dd)
                    bb_tf_apys[p['bb_tf']].append(apy)
                    bb_tf_dds[p['bb_tf']].append(max_dd)
                    supertrend_tf_apys[p['supertrend_tf']].append(apy)
                    supertrend_tf_dds[p['supertrend_tf']].append(max_dd)

                    # Update Pareto candidates
                    is_dominated = False
                    new_pareto = []
                    for cand in pareto_candidates:
                        if (cand['apy'] >= apy and cand['dd'] <= max_dd) and (cand['apy'] > apy or cand['dd'] < max_dd):
                            is_dominated = True
                        if not (apy >= cand['apy'] and max_dd <= cand['dd']) or not (apy > cand['apy'] or max_dd < cand['dd']):
                            new_pareto.append(cand)
                    if not is_dominated:
                        new_pareto.append({'params': p, 'apy': apy, 'dd': max_dd})
                    pareto_candidates = new_pareto

                    m = {
                        "deals": row[3],
                        "total_pl": round((ratio - 1) * 1000, 2),
                        "roi_pct": round((ratio - 1) * 100, 2),
                        "annual_pct": round(apy, 2),
                        "apy_pct": round(apy, 2),
                        "annual_usd": round(1000 * apy / 100, 2),
                        "avg_deal_min": round(row[2], 2),
                        "max_drawdown_pct": round(max_dd, 2),
                        "longest_drawdown_min": round(row[4], 2),
                    }
                    trial = optuna.create_trial(
                        values=[apy, max_dd],  # List for multi-objective
                        params={
                            'spacing_pct': round(p['spacing_pct'], 1),
                            'tp_pct': round(p['tp_pct'], 1),
                            'trailing': p['trailing'],
                            'trailing_pct': round(p['trailing_pct'], 2),
                            'exit_on_flip': p['exit_on_flip'],
                            'bb_tf': p['bb_tf'],
                            'supertrend_tf': p['supertrend_tf'],
                        },
                        distributions={
                            'spacing_pct': optuna.distributions.FloatDistribution(**PARAM_RANGES["spacing_pct"]),
                            'tp_pct': optuna.distributions.FloatDistribution(**PARAM_RANGES["tp_pct"]),
                            'trailing': optuna.distributions.CategoricalDistribution(PARAM_RANGES["trailing"]),
                            'trailing_pct': optuna.distributions.FloatDistribution(**PARAM_RANGES["trailing_pct"]),
                            'exit_on_flip': optuna.distributions.CategoricalDistribution(PARAM_RANGES["exit_on_flip"]),
                            'bb_tf': optuna.distributions.CategoricalDistribution(PARAM_RANGES["bb_tf"]),
                            'supertrend_tf': optuna.distributions.CategoricalDistribution(PARAM_RANGES["supertrend_tf"]),
                        },
                        state=TrialState.COMPLETE,
                        user_attrs={'params': p, 'metrics': m}
                    )
                    study_best.add_trial(trial)

                pbar.update(len(batch))
                index = end
                b += 1

                # Pruning after every 5 batches
                if b % 10 == 0 and b > 0:
                    if eval_list:
                        max_apy = max([item[1] for item in eval_list])
                        min_dd = min([item[2] for item in eval_list])
                        apy_threshold = 0.5 * max_apy
                        dd_threshold = 2 * min_dd

                        # Find bad values for each category
                        bad_spacing = [k for k, v in spacing_apys.items() if np.mean(v) < apy_threshold or np.mean(spacing_dds[k]) > dd_threshold]
                        bad_tp = [k for k, v in tp_apys.items() if np.mean(v) < apy_threshold or np.mean(tp_dds[k]) > dd_threshold]
                        bad_trailing = [k for k, v in trailing_apys.items() if np.mean(v) < apy_threshold or np.mean(trailing_dds[k]) > dd_threshold]
                        bad_trailing_pct = [k for k, v in trailing_pct_apys.items() if np.mean(v) < apy_threshold or np.mean(trailing_pct_dds[k]) > dd_threshold]
                        bad_bb = [k for k, v in bb_tf_apys.items() if np.mean(v) < apy_threshold or np.mean(bb_tf_dds[k]) > dd_threshold]
                        bad_st = [k for k, v in supertrend_tf_apys.items() if np.mean(v) < apy_threshold or np.mean(supertrend_tf_dds[k]) > dd_threshold]

                        # Prune remaining params_list
                        remaining = params_list[index:]
                        filtered = [p for p in remaining if (
                            p['spacing_pct'] not in bad_spacing and
                            p['tp_pct'] not in bad_tp and
                            p['trailing'] not in bad_trailing and
                            p['trailing_pct'] not in bad_trailing_pct and
                            p['bb_tf'] not in bad_bb and
                            p['supertrend_tf'] not in bad_st
                        )]
                        pruned_count = len(remaining) - len(filtered)
                        params_list = params_list[:index] + filtered  # Update params_list
                        logging.info(f"Pruned {pruned_count} hopeless params after batch {b}")

        # After loop: Handle Pareto selection
        pareto_candidates.sort(key=lambda x: (-x['apy'], x['dd']))
        logging.info(f"Pareto front size: {len(pareto_candidates)}")
        selected = next((cand for cand in pareto_candidates if cand['dd'] <= max_dd_threshold), pareto_candidates[0] if pareto_candidates else None)
        if selected:
            logging.info(f"Selected from Pareto: APY={selected['apy']:.2f}%, DD={selected['dd']:.2f}% with params: {json.dumps(selected['params'])}")

    study_best.optimize(
        make_objective(df, use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study_best
