
"""
Dual‑side (long + short) DCA strategy for spot accounts.
Positions flip automatically on every 8‑hour SuperTrend cross.
Profit is measured as the net change in cash for each deal.
"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange


class DCATrailingStrategy:
def __init__(
    self,
    base_order: float = 16.6078,
    mult: float = 1.0,
    max_safety: int = 50,
    compound: bool = True,
    risk_pct: float = 0.0166078,
    spacing_pct: float = 1.0,
    tp_pct: float = 0.6,
    trailing: bool = True,
    trailing_pct: float = 0.1,
    fee_rate: float = 0.001,
    initial_balance: float = 1000.0,
    reopen_sec: Optional[int] = None,
    use_sig: bool = True,
    **_ignored,
):
        self.base_order = base_order
        self.mult = mult
        self.max_safety = max_safety
        self.compound = compound
        self.risk_pct = risk_pct
        self.spacing_pct = spacing_pct
        self.tp_pct = tp_pct
        self.trailing = trailing
        self.trailing_pct = trailing_pct
        self.fee_rate = fee_rate
        self.initial_balance = initial_balance
        self.reopen_sec = reopen_sec

    # ---------------------------------------------------------------
    @staticmethod
    def _supertrend_signal(df: pd.DataFrame) -> pd.Series:
        tf = "8h"
        hlc = df.resample(tf).agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna()

        atr = AverageTrueRange(hlc["high"], hlc["low"], hlc["close"], window=10).average_true_range()
        hl2 = (hlc["high"] + hlc["low"]) / 2
        upper = hl2 + 3 * atr
        lower = hl2 - 3 * atr

        st = pd.Series(np.nan, index=hlc.index)
        bull = pd.Series(True, index=hlc.index)

        for i in range(1, len(hlc)):
            if bull.iat[i - 1]:
                st.iat[i] = max(lower.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else lower.iat[i])
                bull.iat[i] = hlc['close'].iat[i] > st.iat[i]
            else:
                st.iat[i] = min(upper.iat[i], st.iat[i - 1] if not np.isnan(st.iat[i - 1]) else upper.iat[i])
                bull.iat[i] = hlc['close'].iat[i] > st.iat[i]

        return bull.reindex(df.index, method="ffill").fillna(False)

    # ---------------------------------------------------------------
    def backtest(self, df: pd.DataFrame, cooldown_sec: int = 60) -> Tuple[List[Tuple], List[Tuple]]:
        df = df.copy()
        df["bull"] = self._supertrend_signal(df)

        cash = self.initial_balance
        qty = 0.0
        avg_price = 0.0

        state = "idle"
        side = 0
        ladder0 = self.base_order
        dca_count = 0
        next_order = 0.0
        trailing_extreme = 0.0
        cash_start = 0.0
        deal_entry_ts = 0
        last_close_ts = -1

        deals: List[Tuple] = []
        equity: List[Tuple] = []

        for ts, row in df.iterrows():
            epoch = int(ts.timestamp())
            price = row.close
            equity.append((epoch, cash + qty * price))

            bull = bool(row.bull)

            if state == "idle":
                open_long = bull and self._can_reopen(epoch, last_close_ts)
                open_short = (not bull) and self._can_reopen(epoch, last_close_ts)
                if not (open_long or open_short):
                    continue

                side = 1 if open_long else -1
                usd = self._base_usd(cash)
                fee = usd * self.fee_rate
                qty_change = side * usd / price

                cash_start = cash  # snapshot

                if side == 1:
                    cash -= usd + fee
                else:
                    cash += usd - fee

                qty += qty_change
                avg_price = price
                ladder0 = usd
                dca_count = 0
                next_order = price * (1 - self.spacing_pct / 100) if side == 1 else price * (1 + self.spacing_pct / 100)
                trailing_extreme = price
                deal_entry_ts = epoch
                state = "active"
                continue

            # ---------- safety orders ----------
            need_safety = (side == 1 and price <= next_order) or (side == -1 and price >= next_order)
            if state == "active" and need_safety and dca_count < self.max_safety and epoch - deal_entry_ts >= cooldown_sec:
                dca_count += 1
                usd = ladder0 * (self.mult ** dca_count)
                fee = usd * self.fee_rate
                qty_change = side * usd / price

                if side == 1:
                    cash -= usd + fee
                else:
                    cash += usd - fee

                # update average price
                qty_old = qty
                qty += qty_change
                avg_price = (avg_price * abs(qty_old) + price * abs(qty_change)) / abs(qty)

                next_order = price * (1 - self.spacing_pct / 100) if side == 1 else price * (1 + self.spacing_pct / 100)
                trailing_extreme = price
                deal_entry_ts = epoch

            # ---------- TP / trailing ----------
            tp_target = avg_price * (1 + self.tp_pct / 100) if side == 1 else avg_price * (1 - self.tp_pct / 100)
            tp_hit = (side == 1 and price >= tp_target) or (side == -1 and price <= tp_target)

            exit_now = False
            if tp_hit:
                if self.trailing:
                    if side == 1:
                        trailing_extreme = max(trailing_extreme, price)
                        if price <= trailing_extreme * (1 - self.trailing_pct / 100):
                            exit_now = True
                    else:
                        trailing_extreme = min(trailing_extreme, price)
                        if price >= trailing_extreme * (1 + self.trailing_pct / 100):
                            exit_now = True
                else:
                    exit_now = True

            trend_flip = (side == 1 and not bull) or (side == -1 and bull)
            if trend_flip:
                exit_now = True

            # ---------- close ----------
            if state == "active" and exit_now:
                if side == 1:
                    proceeds = abs(qty) * price
                    fee = proceeds * self.fee_rate
                    cash += proceeds - fee
                else:
                    buy_cost = abs(qty) * price
                    fee = buy_cost * self.fee_rate
                    cash -= buy_cost + fee

                profit = cash - cash_start
                deals.append((deal_entry_ts, epoch, round(profit, 4), round(fee, 4)))

                qty = 0.0
                avg_price = 0.0
                state = "idle"
                side = 0
                last_close_ts = epoch

        # final equity snapshot
        if equity and equity[-1][0] != int(df.index[-1].timestamp()):
            equity.append((int(df.index[-1].timestamp()), cash + qty * df['close'].iat[-1]))

        return deals, equity

    # ---------- helpers ----------
    def _can_reopen(self, now: int, last_close: int) -> bool:
        return self.reopen_sec is None or now >= last_close + self.reopen_sec

    def _base_usd(self, cash: float) -> float:
        return cash * self.risk_pct if self.compound else self.base_order
