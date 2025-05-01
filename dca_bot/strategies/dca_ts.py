"""
Pure-Python DCA strategy (trailing TP)
• Entry = Bollinger %B(20, 2) crosses up 0  AND  RSI-7 < 30
  both on **3-minute candles**, f-filled to 1-min.
• Optional `reopen_sec` lets you ignore the signal and reopen a deal
  N seconds after the previous one closed.
"""

from typing import List, Tuple, Optional

import pandas as pd
import ta


class DCATrailingStrategy:
    def __init__(
        self,
        spacing_pct: float = 1.0,
        tp_pct: float = 0.6,
        trailing: bool = True,
        trailing_pct: float = 0.1,
        max_dca: int = 50,
        fast_ema: Optional[int] = None,
        slow_ema: Optional[int] = None,
        fee_rate: float = 0.001,
        initial_balance: float = 1000.0,
        reopen_sec: Optional[int] = None,          # ← NEW in signature
    ):
        self.spacing_pct = spacing_pct
        self.tp_pct = tp_pct
        self.trailing = trailing
        self.trailing_pct = trailing_pct
        self.max_dca = max_dca
        self.fee_rate = fee_rate
        self.initial_balance = initial_balance
        self.order_usd = initial_balance / 51      # 1 base + 50 safety slots
        self.reopen_sec = reopen_sec               # ← store it!
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    # ------------------------------------------------------------------
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bbp3, rsi3 and entry_sig columns (3-minute maths)."""
        close_3m = df["close"].resample("3min").last().dropna()

        ma  = close_3m.rolling(20, min_periods=20).mean()
        sd  = close_3m.rolling(20, min_periods=20).std()
        lo  = ma - 2 * sd
        hi  = ma + 2 * sd
        bbp3 = (close_3m - lo) / (hi - lo)

        rsi3 = ta.momentum.RSIIndicator(close_3m, window=7).rsi()

        df = df.copy()
        df["bbp3"] = bbp3.reindex(df.index, method="ffill")
        df["rsi3"] = rsi3.reindex(df.index, method="ffill")

        df["entry_sig"] = (
            (df["bbp3"].shift(1) < 0) & (df["bbp3"] >= 0) & (df["rsi3"].shift(1) < 30)
        ).fillna(False)

        # ------------ EMA trend filter -----------------
        if self.fast_ema and self.slow_ema:
            ema_fast = close_3m.ewm(span=self.fast_ema, adjust=False).mean()
            ema_slow = close_3m.ewm(span=self.slow_ema, adjust=False).mean()
            up = (ema_fast > ema_slow).reindex(df.index, method="ffill").fillna(False)
            df["uptrend"] = up.astype(bool)
        else:
            df["uptrend"] = True

        return df

    # ------------------------------------------------------------------
    def backtest(
        self, df: pd.DataFrame, cooldown_sec: int = 60
    ) -> Tuple[List[Tuple], List[Tuple]]:

        df = self._add_indicators(df)

        cash = self.initial_balance
        qty = 0.0
        total_profit = 0.0
        deals, equity = [], []

        state = "idle"
        avg_price = next_buy = trailing_high = 0.0
        dca_count = 0
        deal_entry = last_dca_ts = cost = 0

        last_close = -1  # epoch of previous exit
        for ts, row in df.iterrows():
            price = row.close
            epoch = int(ts.timestamp())
            equity.append((epoch, cash + qty * price))

            base_open = (
                row.entry_sig
                if self.reopen_sec is None
                else (epoch >= last_close + self.reopen_sec)
            )
            want_open = row.uptrend and base_open
            # >>> TEST: raise if we’re about to open while trend is DOWN
            if state == "idle" and base_open and not row.uptrend:
                raise RuntimeError(f"Trend filter failed at {ts} price={price}")

            if state == "idle" and want_open:
                usd = self.order_usd
                fee = usd * self.fee_rate
                qty_buy = usd / price

                cash -= usd + fee
                qty += qty_buy
                cost = usd + fee

                avg_price = price
                dca_count = 0
                next_buy = price * (1 - self.spacing_pct / 100)
                deal_entry = last_dca_ts = epoch
                trailing_high = 0.0
                state = "active"
                continue

            # ---------- DCA safety buys ---------------------------------
            if (
                state == "active"
                and dca_count < self.max_dca
                and price <= next_buy
                and epoch - last_dca_ts >= cooldown_sec
            ):
                usd = self.order_usd
                fee = usd * self.fee_rate
                qty_buy = usd / price

                cash -= usd + fee
                qty += qty_buy
                cost += usd + fee

                dca_count += 1
                last_dca_ts = epoch
                avg_price = (avg_price * qty + price * qty_buy) / (qty + qty_buy)
                next_buy = price * (1 - self.spacing_pct / 100)

            # ---------- take-profit / trailing stop ---------------------
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

                    total_profit += profit
                    deals.append((deal_entry, epoch, profit, fee))

                    cash = self.initial_balance + total_profit
                    qty = 0.0
                    state = "idle"
                    last_close = epoch

        # final equity snapshot
        if equity and equity[-1][0] != int(df.index[-1].timestamp()):
            equity.append(
                (int(df.index[-1].timestamp()), cash + qty * df.iloc[-1].close)
            )

        return deals, equity
