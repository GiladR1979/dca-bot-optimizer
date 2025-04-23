"""
Binance downloader with local caching.

* CSV per (symbol, interval) lives in ./data/
* load_binance() transparently appends missing candles and
  returns the full DataFrame for [start, end].
"""

import os
import time
import logging
from datetime import datetime

import pandas as pd
import requests


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
log = logging.getLogger("loader")


def _klines(symbol, start_ms, end_ms, interval="1m"):
    url = "https://api.binance.com/api/v3/klines"
    data, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        data.extend(batch)
        cur = batch[-1][0] + 1
        log.info("Batch %4d | %4d candles | up to %s",
                 len(data) // 1000,
                 len(batch),
                 datetime.utcfromtimestamp(batch[-1][0] / 1000))
        # be polite
        time.sleep(0.03)
    return data


def _to_df(raw):
    cols = ["ts", "open", "high", "low", "close", "vol",
            "ct", "qav", "n", "tb", "tbq", "ig"]
    df = pd.DataFrame(raw, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df[["open", "high", "low", "close", "vol"]].astype(float)


def _cache_path(symbol, interval):
    fname = f"{symbol}_{interval}.csv"
    return os.path.join(DATA_DIR, fname)


def load_binance(symbol, start, end, interval="1m"):
    """
    Return a DataFrame of candles for [start, end] (inclusive),
    downloading only what isn't cached yet.
    """
    cache_file = _cache_path(symbol, interval)
    start_dt = pd.to_datetime(start, utc=True).tz_localize(None)
    end_dt   = pd.to_datetime(end,   utc=True).tz_localize(None)

    # ------------------------------------------------ cache hit?
    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, parse_dates=["ts"], index_col="ts")
        have_start = cached.index.min()
        have_end   = cached.index.max()
    else:
        cached = pd.DataFrame()
        have_start = have_end = None

    need_front = start_dt < have_start if have_start is not None else False
    need_back  = end_dt   > have_end   if have_end   is not None else False

    # ------------------------------------------------ download gaps
    frames = [cached]
    if need_front:
        log.info("Downloading front gap: %s → %s", start_dt, have_start)
        rs = _klines(symbol,
                     int(start_dt.timestamp()*1000),
                     int((have_start - pd.Timedelta("1ms")).timestamp()*1000)
                     if have_start else int(end_dt.timestamp()*1000),
                     interval)
        if rs:
            frames.append(_to_df(rs))
    if need_back:
        log.info("Downloading back gap : %s → %s", have_end, end_dt)
        rs = _klines(symbol,
                     int((have_end + pd.Timedelta("1ms")).timestamp()*1000)
                     if have_end else int(start_dt.timestamp()*1000),
                     int(end_dt.timestamp()*1000),
                     interval)
        if rs:
            frames.append(_to_df(rs))

    # ------------------------------------------------ merge + dedupe
    full = pd.concat(frames).sort_index().loc[start_dt:end_dt]
    full = full[~full.index.duplicated(keep="last")]

    # ------------------------------------------------ save cache
    full.to_csv(cache_file)
    log.info("Cache updated → %s (rows: %d)", cache_file, len(full))
    return full
