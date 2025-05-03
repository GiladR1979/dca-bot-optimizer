"""
Pure-Python DCA strategy  (trailing TP, SuperTrend filter)

• Trend filter = Daily SuperTrend (ATR-10 × 3).  No BB/RSI logic.
• Buying ladder = geometric:
      base_order = 6.5109 USDT
      multiplier = 1.04
      max_safety = 50        → 51 total orders ≈ 999 USDT
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange


# ======================================================================
class DCATrailingStrategy:
    def __init__(
        self,
        # ---- ladder ---------------------------------------------------
        base_order: float = 6.5109,     # first order in USDT
        mult:       float = 1.04,       # geometric factor
        max_safety: int   = 50,         # safety orders (base+50 = 51)

        # ---- trade parameters ----------------------------------------
        spacing_pct: float = 1.0,       # price gap for each next buy
        tp_pct:      float = 0.6,
        trailing:    bool  = True,
        trailing_pct: float = 0.1,

        # ---- fees / account ------------------------------------------
        fee_rate: float = 0.001,
        initial_balance: float = 1000.0,

        # ---- reopen logic --------------------------------------------
        reopen_sec: Optional[int] = None,   # None = obey indicator only
    ):
        self.base_order   = base_order
        self.mult         = mult
        self.max_safety   = max_safety

        self.spacing_pct  = spacing_pct
        self.tp_pct       = tp_pct
        self.trailing     = trailing
        self.trailing_pct = trailing_pct

        self.fee_rate         = fee_rate
        self.initial_balance  = initial_balance
        self.reopen_sec       = reopen_sec

    # ------------------------------------------------------------------
    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add a daily SuperTrend and expose it as Boolean entry_sig."""
        daily_close = df["close"].resample("1D").last().dropna()
        high_d      = df["high"].resample("1D").max().loc[daily_close.index]
        low_d       = df["low"] .resample("1D").min().loc[daily_close.index]

        atr = AverageTrueRange(high_d, low_d, daily_close,
                               window=10).average_true_range()
        hl2   = (high_d + low_d) / 2
        upper = hl2 + 3 * atr
        lower = hl2 - 3 * atr

        st   = pd.Series(np.nan, index=daily_close.index)
        bull = pd.Series(True,   index=daily_close.index)

        for i in range(1, len(daily_close)):
            if bull.iat[i - 1]:
                st.iat[i]  = max(lower.iat[i], st.iat[i - 1]
                                  if not np.isnan(st.iat[i - 1]) else lower.iat[i])
                bull.iat[i] = daily_close.iat[i] > st.iat[i]
            else:
                st.iat[i]  = min(upper.iat[i], st.iat[i - 1]
                                  if not np.isnan(st.iat[i - 1]) else upper.iat[i])
                bull.iat[i] = daily_close.iat[i] > st.iat[i]

        df = df.copy()
        df["entry_sig"] = bull.reindex(df.index, method="ffill").fillna(False)
        return df

    # ------------------------------------------------------------------
    def backtest(
        self, df: pd.DataFrame, cooldown_sec: int = 60
    ) -> Tuple[List[Tuple], List[Tuple]]:

        df = self._add_indicators(df)

        cash   = self.initial_balance
        qty    = 0.0
        deals, equity = [], []

        state = "idle"
        avg_price = next_buy = trailing_high = 0.0
        dca_count = 0
        cost = 0.0
        deal_entry = last_dca_ts = 0
        last_close = -1               # epoch of previous exit

        for ts, row in df.iterrows():
            price = row.close
            epoch = int(ts.timestamp())
            equity.append((epoch, cash + qty * price))

            # ---------- open first order ------------------------------
            want_open = (
                row.entry_sig
                if self.reopen_sec is None
                else (epoch >= last_close + self.reopen_sec) and row.entry_sig
            )
            if state == "idle" and want_open:
                usd = self.base_order
                fee = usd * self.fee_rate
                qty = usd / price
                cash -= usd + fee
                cost = usd + fee

                avg_price = price
                dca_count = 0
                next_buy  = price * (1 - self.spacing_pct / 100)
                deal_entry = last_dca_ts = epoch
                trailing_high = 0.0
                state = "active"
                continue

            # ---------- safety buys -----------------------------------
            if (state == "active"
                and dca_count < self.max_safety
                and price <= next_buy
                and epoch - last_dca_ts >= cooldown_sec):
                dca_count += 1
                usd = self.base_order * (self.mult ** dca_count)
                fee = usd * self.fee_rate
                qty_buy = usd / price

                cash -= usd + fee
                cost += usd + fee
                qty  += qty_buy

                avg_price = ((avg_price * (qty - qty_buy)) + price * qty_buy) / qty
                last_dca_ts = epoch
                next_buy = price * (1 - self.spacing_pct / 100)
                trailing_high = 0.0

            # ---------- take-profit / trailing -------------------------
            if state == "active" and price >= avg_price * (1 + self.tp_pct / 100):
                exit_now = False
                if self.trailing:
                    trailing_high = max(trailing_high, price)
                    if price <= trailing_high * (1 - self.trailing_pct / 100):
                        exit_now = True
                else:
                    exit_now = True

                if exit_now:
                    proceeds = qty * price
                    fee = proceeds * self.fee_rate
                    cash += proceeds - fee
                    profit = (proceeds - fee) - cost
                    deals.append((deal_entry, epoch, profit, fee))

                    # reset for next cycle
                    qty = 0.0
                    state = "idle"
                    last_close = epoch
                    dca_count = 0
                    cost = 0.0

        # final equity snapshot
        if equity and equity[-1][0] != int(df.index[-1].timestamp()):
            equity.append((int(df.index[-1].timestamp()), cash + qty * df.iloc[-1].close))

        return deals, equity
