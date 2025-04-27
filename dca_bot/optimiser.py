
import itertools
import logging
from .strategies.dca_ts import DCATrailingStrategy
from .simulator import calc_metrics

def grid_search(df, grid):
    log = logging.getLogger("opt")
    best = safe = fast = None
    best_m = safe_m = fast_m = None
    for combo in itertools.product(*grid.values()):
        params=dict(zip(grid.keys(),combo))
        
        if params["trailing"] and params["tp_pct"] - params["trailing_pct"] < 0.5:
            continue        # invalid: final target would be below 0.5 %
        if not params["trailing"] and params["tp_pct"] < 0.5:
            continue        # (covers the off‑trailing rule, though grid already does)

        log.info("Testing %s", params)
        
        deals,eq=DCATrailingStrategy(**params).backtest(df)
        m=calc_metrics(deals,eq)
        if not best_m or m['annual_pct']>best_m['annual_pct']:
            best,best_m=params,m
        if not safe_m or m['max_drawdown_pct']<safe_m['max_drawdown_pct']:
            safe,safe_m=params,m
        if not fast_m or m['avg_deal_min'] < fast_m['avg_deal_min']:
            fast, fast_m = params, m
    return {'best': (best, best_m),
            'safe': (safe, safe_m),
            'fast': (fast, fast_m)}
