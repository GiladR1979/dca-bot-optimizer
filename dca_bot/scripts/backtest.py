
import argparse, json, logging, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from ..loader import load_binance
from ..strategies.dca_ts import DCATrailingStrategy
from ..simulator import calc_metrics

RESULTS_DIR=os.path.join(os.path.dirname(__file__),'..','..','results')
os.makedirs(RESULTS_DIR,exist_ok=True)

def plot_equity(equity,deals,symbol,start,end):
    ts=[datetime.fromtimestamp(t) for t,_ in equity]
    val=[v for _,v in equity]
    lu=dict(equity)
    plt.figure(figsize=(10,4))
    plt.plot(ts,val)
    for _,e,_,_ in deals:
        y=lu.get(e)
        if y: plt.scatter(datetime.fromtimestamp(e),y,marker='v',color='red')
    plt.title(f'Equity {symbol}'); plt.ylabel('USD'); plt.grid(True); plt.tight_layout()
    fname=f'equity_{symbol}_{start}_{end}.png'.replace(':','-')
    p=os.path.join(RESULTS_DIR,fname); plt.savefig(p,dpi=120); plt.close(); return p

def main():
    pa=argparse.ArgumentParser(description='Single back‑test')
    pa.add_argument('symbol'); pa.add_argument('start'); pa.add_argument('end')
    pa.add_argument('--spacing-pct',type=float,default=1)
    pa.add_argument('--tp',type=float,default=0.6)
    pa.add_argument('--trailing',action='store_true',default=True)
    pa.add_argument('--trailing-pct',type=float,default=0.1)
    pa.add_argument('--plot',action='store_true')
    pa.add_argument('-v','--verbose',action='store_true')
    args=pa.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(message)s',datefmt='%H:%M:%S')
    df=load_binance(args.symbol,args.start,args.end,'1m')
    strat=DCATrailingStrategy(spacing_pct=args.spacing_pct,tp_pct=args.tp,
                              trailing=args.trailing,trailing_pct=args.trailing_pct)
    deals,eq=strat.backtest(df)
    print(json.dumps(calc_metrics(deals,eq),indent=2))
    if args.plot:
        plot_equity(eq,deals,args.symbol,args.start,args.end)

if __name__=='__main__':
    main()
