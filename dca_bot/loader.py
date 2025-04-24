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


import requests
from requests.adapters import HTTPAdapter, Retry

def _klines(symbol: str,
            start_ms: int,
            end_ms: int,
            interval: str = "1m"):
    """
    Fetch up to 1000 klines in one call with automatic retries and
    exponential back‑off.  Raises after 5 failed attempts.
    """
    url = "https://api.binance.com/api/v3/klines"

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=1.5,                # 1.5 s, 3 s, 4.5 s …
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
        ),
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    data, cur = [], start_ms
    while cur < end_ms:
        try:
            r = session.get(url, params=params | {"startTime": cur}, timeout=30)
            r.raise_for_status()
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as err:
            # manual back‑off for rare SSL / DNS stalls
            for attempt in range(1, 6):
                wait = 2 ** attempt            # 2, 4, 8, 16, 32 s
                print(f"Timeout – retry {attempt}/5 in {wait}s …")
                time.sleep(wait)
                try:
                    r = session.get(url, params=params | {"startTime": cur}, timeout=30)
                    r.raise_for_status()
                    break
                except Exception:
                    continue
            else:
                raise err
        batch = r.json()
        if not batch:
            break
        data.extend(batch)
        cur = batch[-1][0] + 1
        log.info("Batch %4d | %4d candles | up to %s",
                 len(data) // 1000,
                 len(batch),
                 datetime.utcfromtimestamp(batch[-1][0] / 1000))
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
        try:
            cached = pd.read_csv(cache_file, parse_dates=["ts"], index_col="ts")
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            log.warning("Ignoring malformed cache %s", cache_file)
            cached = pd.DataFrame()
        have_start = cached.index.min() if not cached.empty else None
        have_end   = cached.index.max() if not cached.empty else None
    else:
        cached = pd.DataFrame()
        have_start = have_end = None

    if have_start is None and have_end is None:
        # no cache at all → one single pass from start → end
        need_front = True
        need_back  = False          # << stop the second duplicate loop
    else:
        need_front = start_dt < have_start
        need_back  = end_dt   > have_end

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
    if len(full):
        full.to_csv(cache_file)
        log.info("Cache updated → %s (rows: %d)", cache_file, len(full))
    else:
        log.warning("Download returned 0 rows – cache not written.")
    return full
