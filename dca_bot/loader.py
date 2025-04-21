
import pandas as pd, requests
import logging

def _klines(symbol,start_ms,end_ms,interval='1m'):
    url='https://api.binance.com/api/v3/klines'
    log = logging.getLogger("loader")
    data=[]; cur=start_ms; batch_no=0
    while cur<end_ms:
        r=requests.get(url,params={'symbol':symbol,'interval':interval,
                                   'startTime':cur,'endTime':end_ms,'limit':1000},timeout=15)
        r.raise_for_status()
        batch=r.json()
        if not batch: break
        data.extend(batch)
        batch_no += 1
        last_ts = batch[-1][0]
        log.info("Batch %-4d | %4d candles | up to %s", batch_no, len(batch), pd.to_datetime(last_ts, unit="ms"))
        cur=last_ts+1
    return data

def load_binance(symbol,start,end,interval='1m'):
    ss=int(pd.Timestamp(start).timestamp()*1000)
    ee=int(pd.Timestamp(end).timestamp()*1000)
    raw=_klines(symbol,ss,ee,interval)
    cols=['ts','open','high','low','close','vol','ct','qav','n','tb','tbq','ig']
    df=pd.DataFrame(raw,columns=cols)
    df['ts']=pd.to_datetime(df['ts'],unit='ms')
    df.set_index('ts',inplace=True)
    return df[['open','high','low','close','vol']].astype(float)
