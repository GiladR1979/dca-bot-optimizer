
class DCATrailingStrategy:
    """Price‑spacing DCA with optional trailing stop."""
    def __init__(self, spacing_pct=1.0, tp_pct=0.6, trailing=True,
                 trailing_pct=0.1, max_dca=50, fee_rate=0.001,
                 initial_balance=1000.0):
        self.spacing_pct=spacing_pct
        self.tp_pct=tp_pct
        self.trailing=trailing
        self.trailing_pct=trailing_pct
        self.max_dca=max_dca
        self.fee_rate=fee_rate
        self.initial_balance=initial_balance
        self.order_usd=initial_balance/51

    def backtest(self, df, cooldown_sec=60):
        cash=self.initial_balance
        qty=0.0
        total_profit=0.0
        deals=[]
        equity=[]
        state='idle'
        avg_price=0.0
        next_buy=0.0
        dca_count=0
        trailing_high=0.0
        deal_entry=None
        last_dca_ts=0
        cost=0.0

        for ts,row in df.iterrows():
            price=row.close
            epoch=int(ts.timestamp())
            equity.append((epoch,cash+qty*price))

            if state=='idle':
                usd=self.order_usd; fee=usd*self.fee_rate
                qty_buy=usd/price
                cash-=usd+fee; qty+=qty_buy; cost=usd+fee
                avg_price=price; dca_count=0
                next_buy=price*(1-self.spacing_pct/100)
                deal_entry=epoch; last_dca_ts=epoch; trailing_high=0.0
                state='active'; continue

            if (state=='active' and dca_count<self.max_dca and
                price<=next_buy and epoch-last_dca_ts>=cooldown_sec):
                usd=self.order_usd; fee=usd*self.fee_rate
                qty_buy=usd/price
                cash-=usd+fee; qty+=qty_buy; cost+=usd+fee
                dca_count+=1; last_dca_ts=epoch
                avg_price = ((avg_price*(dca_count))+price)/(dca_count+1)
                next_buy=price*(1-self.spacing_pct/100)

            if price>=avg_price*(1+self.tp_pct/100):
                if self.trailing:
                    trailing_high=max(trailing_high,price)
                    if price<=trailing_high*(1-self.trailing_pct/100):
                        proceeds=qty*price; fee=proceeds*self.fee_rate
                        cash+=proceeds-fee
                        profit=(proceeds-fee)-cost
                        total_profit+=profit
                        deals.append((deal_entry,epoch,profit,fee))
                        cash=self.initial_balance+total_profit
                        qty=0.0; state='idle'
                else:
                    proceeds=qty*price; fee=proceeds*self.fee_rate
                    cash+=proceeds-fee
                    profit=(proceeds-fee)-cost
                    total_profit+=profit
                    deals.append((deal_entry,epoch,profit,fee))
                    cash=self.initial_balance+total_profit
                    qty=0.0; state='idle'

        if state=='idle':
            equity.append((int(df.index[-1].timestamp()),cash))
        return deals,equity
