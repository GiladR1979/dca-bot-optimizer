
# DCA‑Bot Optimizer 📈

Back‑test and grid‑optimize a Dollar‑Cost‑Averaging (DCA) spot‑strategy
for any Binance symbol.

## Features

* **Base + safety orders** – 1 base buy and up to 50 safety buys  
* **Trailing take‑profit** (optional)  
* **Grid search** over spacing %, TP %, *with / without* trailing  
* Automatically **skips invalid combos**  
  * if trailing is **off** → `tp_pct ≥ 0.5 %`  
  * if trailing is **on** → `tp_pct − trailing_pct ≥ 0.5 %`  
* Metrics: ROI %, annualised %, max draw‑down %, average deal time, …  
* Equity PNGs with red ▼ markers at every sell  
* Picks **best** (max annual %) and **safe** (min draw‑down %) configs  
* Generates a 3‑panel comparison plot *(default | best | safe)*

---

## Quick start

```bash
git clone https://github.com/<your‑user>/<repo>.git
cd <repo>

python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas requests matplotlib
```

### Single back‑test

```bash
python -m dca_bot.scripts.backtest  SYMBOL  START  END         --spacing-pct 1  --tp 0.6  --trailing        --plot  -v
```

* `SYMBOL` – Binance pair (`BTCUSDT`, `1INCHUSDT`, …)  
* `START / END` – `YYYY‑MM‑DD` (UTC)

### Grid optimizer

```bash
python -m dca_bot.scripts.optimize  SYMBOL  START  END  -v
```

Optional knobs:

| Flag | Default | Meaning |
|------|---------|---------|
| `--spacings` | `0.5,1,1.5,2` | list of spacing % values |
| `--tps`      | `0.5,0.6,1`   | list of TP % values (≥ 0.5) |
| `--trailing-pct` | `0.1` | trailing gap % |

Outputs (`results/`):

```
equity_<symbol>_default.png
equity_<symbol>_best.png
equity_<symbol>_safe.png
<symbol>_triple.png      # 3‑panel figure
<symbol>_summary.json    # params & metrics
```

---

## Algorithm

1. **Base order**: \$1 000 / 51 ≈ \$19.61 at the first candle  
2. **Safety orders**: each price drop of *spacing_pct* triggers a fixed‑size buy  
3. **Take‑profit**  
   * trailing **off** → sell at first TP touch  
   * trailing **on** → arm stop at `highest*(1 − trailing_pct)`  
4. Realised P/L rolled into the cash balance  
5. Final open deal is ignored for metrics & plots

---

## Repository layout

```
dca_bot/
├─ loader.py           # Binance downloader (per‑batch logs)
├─ strategies/
│   └─ dca_ts.py       # DCA + trailing engine
├─ simulator.py        # ROI, draw‑down, etc.
├─ optimiser.py        # grid search + validity guard
├─ plotting.py         # equity curves & 3‑panel figure
└─ scripts/
   ├─ backtest.py      # single run
   └─ optimize.py      # grid optimizer CLI
results/               # generated PNGs + JSON summary
```

---

## License

MIT – free to use, fork, and modify.  
Pull requests and bug reports are welcome!
