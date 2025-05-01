# DCA-Bot Optimizer

> **Back-test & parameter-search engine for 3Commas-style “Deal-Start / DCA / Trailing-TP” bots.**  
> • Indicator-based *smart* entries (Bollinger %B + RSI) **or** fixed-delay *stupid* re-entries  
> • Fast Numba core (--jobs 0 = all CPU cores)  
> • Optuna triple search: **BEST** (profit), **SAFE** (drawdown), **FAST** (deal frequency)  
> • PNG equity curves, JSON summaries, auto-panel comparison  
> • Works on Windows / Linux / macOS, Python 3.8+

---

## 1  Features

| Module | What it does |
|--------|--------------|
| `loader.py` | Downloads / caches Binance klines (1‑min) |
| `strategies/` | `dca_ts.py` = pure‑Python back‑tester<br>`dca_ts_numba.py` = Numba‑JIT back‑tester (20–50× faster) |
| `simulator.py` | ROI, APR, drawdown, trade metrics (handles open deals) |
| `plotting.py` | Down‑sampled 3 000 × 4 000 px equity + panel PNGs |
| `optuna_search.py` | Three Optuna studies with duplicate‑guard & DB storage |
| `scripts/optuna.py` | CLI wrapper: runs search → four showcase back‑tests |
| `run_coins.ps1` | Example PowerShell batch to optimise many symbols |

### Entry modes

* **Smart / signal** (default)  
  *Bollinger %B (20, 2) crosses up 0 **and** previous 3‑min RSI‑7 < 30*
* **Stupid / delay**  
  `--use-sig 0 --reopen-sec N` → open a new deal **N seconds** after the last closed.

---

## 2  Installation

```bash
git clone https://github.com/GiladR1979/dca-bot-optimizer.git
cd dca-bot-optimizer
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Dependencies:** `numpy`, `pandas`, `numba`, `ta`, `matplotlib`, `optuna`, `requests`

---

## 3  Quick start

```bash
# optimise SOLUSDT on 1‑min candles, 2021‑01‑01 → 2025‑03‑01
python -m dca_bot.scripts.optuna SOLUSDT 2021-01-01 2025-03-01 \
       --trials 400  --jobs 0            \  # 400 trials, all CPU cores
       --storage sqlite:///dca.sqlite    \  # resume across runs
       --use-sig 1                       \  # indicator mode
       -v                                   # verbose logs
```

Outputs → `results/`

```
SOLUSDT_best.png     SOLUSDT_safe.png     SOLUSDT_fast.png
SOLUSDT_default.png  SOLUSDT_triple.png   SOLUSDT_opt_summary.json
```

---

## 4  Command‑line flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--trials` | `200` | trials **per** study |
| `--jobs` | `0` | Optuna workers (0 = all cores) |
| `--storage` | `sqlite:///dca.sqlite` | RDB url (`none` → in‑memory) |
| `--use-sig` | `1` | `1` = use indicator, `0` = ignore |
| `--reopen-sec` | `60` | delay when `--use-sig 0` |
| `-v` | off | verbose logging |

---

## 5  Batch run

Edit `run_coins.ps1` and then:

```powershell
.
un_coins.ps1
```

---

## 6  Metrics

| Key | Meaning |
|-----|---------|
| `total_pl` | realised profit on \$1 000 base |
| `annual_pct` | **APR** (linear) = ROI / years |
| `max_drawdown_pct` | worst % below \$1 000 |
| `avg_deal_min` | average closed‑deal duration |

---

## 7  Contributing

* Fork → feature branch → PR against **`dev`**  
* Run `black` + `flake8` before push  
* Include test or before/after screenshot

---

## 8  License

MIT © 2025 Gilad R.
