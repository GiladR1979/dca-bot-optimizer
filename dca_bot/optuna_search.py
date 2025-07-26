# optuna_search.py (modified with fix)

"""
Full‑engine Optuna optimiser – *study names are per‑symbol*.
"""
from __future__ import annotations
import logging
import os
from typing import Optional, Dict, Tuple
import math
from tqdm import tqdm
import json

import optuna
import pandas as pd
import numpy as np
import cupy as cp
from optuna.trial import TrialState
from numba import cuda

# ------------------------------------------------------------------ #
#  duplicate‑trial guard (Optuna 2.x)                                #
# ------------------------------------------------------------------ #
_seen_params: set[tuple] = set()


def _param_sig(spacing: float, tp: float, trailing: bool, trail_pct: float, exit_on_flip: bool, bb_tf: str,
               supertrend_tf: str) -> tuple:
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

def make_objective(df_full: pd.DataFrame, metric_key: str, *, use_sig: int, reopen_sec: int, long_only: bool = False,
                   use_bb_safety: bool = True):
    """Return an Optuna objective that optimises a single metric."""

    head = (
        df_full.resample("30min").first().iloc[: max(300, len(df_full) // 10)]
    )

    # ------------------------------------------------------------------ #
    def _objective(trial: optuna.Trial):
        spacing = trial.suggest_float("spacing_pct", 0.3, 9.0, step=0.1)
        tp = trial.suggest_float("tp_pct", 0.5, 5.0, step=0.1)
        trailing = trial.suggest_categorical("trailing", [False, True])
        trail_pct = 0.1
        exit_on_flip = trial.suggest_categorical("exit_on_flip", [False, True])
        bb_tf = trial.suggest_categorical("bb_tf", ['3min', '5min', '15min', '30min', '1h', '4h'])
        supertrend_tf = trial.suggest_categorical("supertrend_tf", ['15min', '30min', '1h', '4h', '8h', '1d', '1w'])

        # ---- skip exact‑duplicate parameter sets --------------------
        sig = _param_sig(spacing, tp, trailing, trail_pct, exit_on_flip, bb_tf, supertrend_tf)
        if sig in _seen_params:
            raise optuna.TrialPruned()
        _seen_params.add(sig)

        # invalidate combos where trailing SL is larger than TP
        if trailing and tp - trail_pct < 0.5 - 1e-9:
            raise optuna.TrialPruned()

        # ---------- fast head‑run for early pruning --------------------
        m_head = _evaluate(head, spacing, tp, trailing, trail_pct, exit_on_flip, bb_tf, supertrend_tf, use_sig=use_sig,
                           reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety)
        trial.report(m_head[metric_key], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---------- full back‑test ------------------------------------
        m_full = _evaluate(df_full, spacing, tp, trailing, trail_pct, exit_on_flip, bb_tf, supertrend_tf,
                           use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only, use_bb_safety=use_bb_safety)
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
        return m_full[metric_key]

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

def _new_study(base_name: str, direction: str, storage: Optional[str], symbol: str, window_id: str = ""):
    """Create (or reopen) an Optuna study whose name is unique per symbol and window."""

    full_name = f"{base_name}_{symbol}"
    if window_id:
        full_name += f"_{window_id}"
    sampler = optuna.samplers.TPESampler(seed=42)  # no duplicates
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
        direction=direction,
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
        n_trials: int,
        n_jobs: int,
        storage: Optional[str],
        use_sig: int = 1,
        reopen_sec: int = 60,
        long_only: bool = False,
        exit_on_flip: bool = True,
        use_bb_safety: bool = True,
        window_id: str = "",
        supertrend_tf: str = "30min",
        use_gpu: bool = False,
):
    study_best = _new_study("dca_best", "maximize", storage, symbol, window_id)
    if use_gpu:
        import itertools
        # Generate grids with exact rounding to avoid FP precision issues
        spacing_list = [round(0.3 + 0.1 * i, 1) for i in range(int((9.0 - 0.3) / 0.1) + 1)]
        tp_list = [round(0.5 + 0.1 * i, 1) for i in range(int((5.0 - 0.5) / 0.1) + 1)]
        grid = {
            "spacing_pct": spacing_list,
            "tp_pct": tp_list,
            "trailing": [True, False],
            "trailing_pct": [0.1],
            "exit_on_flip": [True, False],
            "bb_tf": ['3min', '5min', '15min', '30min', '1h', '4h'],
            "supertrend_tf": ['15min', '30min', '1h', '4h', '8h', '1d', '1w'],
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

        # Batch processing
        batch_size = 10000  # Adjust based on GPU memory
        num_batches = math.ceil(len(params_list) / batch_size)

        best_apy = -float('inf')
        best_p = None

        with tqdm(total=len(params_list), desc=f"Evaluating grid (window {window_id})") as pbar:
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, len(params_list))
                batch = params_list[start:end]

                if not batch:
                    continue

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
                threads = 128
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

                    # Update best
                    if apy > best_apy:
                        best_apy = apy
                        best_p = p
                        tqdm.write(f"Updated best APY: {best_apy:.2f}% with params: {json.dumps(best_p, indent=None)}")

                    m = {
                        "deals": row[3],
                        "total_pl": round((ratio - 1) * 1000, 2),
                        "roi_pct": round((ratio - 1) * 100, 2),
                        "annual_pct": round(apy, 2),
                        "apy_pct": round(apy, 2),
                        "annual_usd": round(1000 * apy / 100, 2),
                        "avg_deal_min": round(row[2], 2),
                        "max_drawdown_pct": round(row[1], 2),
                        "longest_drawdown_min": round(row[4], 2),
                    }
                    trial = optuna.create_trial(
                        value=m['annual_pct'],
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
                            'spacing_pct': optuna.distributions.FloatDistribution(0.3, 9.0, step=0.1),
                            'tp_pct': optuna.distributions.FloatDistribution(0.5, 5.0, step=0.1),
                            'trailing': optuna.distributions.CategoricalDistribution([False, True]),
                            'trailing_pct': optuna.distributions.FloatDistribution(0.05, 0.3, step=0.05),
                            'exit_on_flip': optuna.distributions.CategoricalDistribution([False, True]),
                            'bb_tf': optuna.distributions.CategoricalDistribution(
                                ['3min', '5min', '15min', '30min', '1h', '4h']),
                            'supertrend_tf': optuna.distributions.CategoricalDistribution(
                                ['15min', '30min', '1h', '4h', '8h', '1d', '1w']),
                        },
                        state=TrialState.COMPLETE,
                        user_attrs={'params': p, 'metrics': m}
                    )
                    study_best.add_trial(trial)

                pbar.update(len(batch))

    study_best.optimize(
        make_objective(df, "annual_pct", use_sig=use_sig, reopen_sec=reopen_sec, long_only=long_only,
                       use_bb_safety=use_bb_safety),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study_best
