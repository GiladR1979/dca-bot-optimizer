
def calc_metrics(deals, equity):
    if not equity: return {}
    total_pl=sum(d[2] for d in deals)
    roi_pct=total_pl/1000*100
    start,end=equity[0][0],equity[-1][0]
    years=(end-start)/(365*24*3600)
    annual_pct=((1+roi_pct/100)**(1/years)-1)*100 if years>0 else 0
    annual_usd=1000*annual_pct/100
    peak=equity[0][1]; peak_time=start
    max_dd=longest=0
    for t,val in equity:
        if val>peak:
            peak=val; peak_time=t
        dd=peak-val
        if dd>max_dd: max_dd=dd
        if val<peak:
            dur=t-peak_time
            if dur>longest: longest=dur
    max_dd_pct=max_dd/peak*100 if peak else 0
    longest_min=longest/60
    avg_deal=sum((e-s)/60 for s,e,_,_ in deals)/len(deals) if deals else 0
    return {'deals':len(deals),'total_pl':round(total_pl,2),'roi_pct':round(roi_pct,2),
            'annual_pct':round(annual_pct,2),'annual_usd':round(annual_usd,2),
            'avg_deal_min':round(avg_deal,2),'max_drawdown_pct':round(max_dd_pct,2),
            'longest_drawdown_min':round(longest_min,2)}
