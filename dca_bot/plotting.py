
import matplotlib.pyplot as plt
from datetime import datetime
import os

def equity_curve(equity,deals,title,path):
    ts=[datetime.fromtimestamp(t) for t,_ in equity]
    val=[v for _,v in equity]
    lookup=dict(equity)
    plt.figure(figsize=(10,4))
    plt.plot(ts,val,label='Equity')
    for _,e,_,_ in deals:
        y=lookup.get(e); 
        if y: plt.scatter(datetime.fromtimestamp(e),y,marker='v',color='red')
    plt.title(title); plt.ylabel('USD'); plt.grid(True); plt.tight_layout()
    plt.savefig(path,dpi=120); plt.close()

def triple_panel(items, path):
    fig,axes=plt.subplots(3,1,figsize=(10,12),sharex=True)
    for ax,(equity,deals,label) in zip(axes,items):
        ts=[datetime.fromtimestamp(t) for t,_ in equity]
        val=[v for _,v in equity]
        ax.plot(ts,val)
        lu=dict(equity)
        for _,e,_,_ in deals:
            y=lu.get(e)
            if y: ax.scatter(datetime.fromtimestamp(e),y,marker='v',color='red')
        ax.set_title(label); ax.grid(True)
    axes[-1].set_xlabel('Date'); axes[1].set_ylabel('USD')
    plt.tight_layout(); plt.savefig(path,dpi=120); plt.close()
