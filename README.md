# SPYer 🎯

**Real-time SPY intraday options signal monitor with multi-indicator quality scoring.**

SPYer watches SPY price action and alerts you when conditions line up for a potential 0–2 DTE options trade — then tells you *how good* the signal is with a quality score from 0–100 and a letter grade (A+ → F).

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

---

## What It Does

- Monitors SPY intraday price every 60 seconds
- Detects CALL signals (price breaks above VWAP with volume surge) and PUT signals (below VWAP)
- **Scores every signal 0–100** across 6 indicators so you know if it's worth acting on
- Shows recommended 0–2 DTE option strikes
- Desktop notifications + sound alerts
- Configurable volume threshold and minimum grade filter

## Signal Quality Scoring

Every signal is evaluated across six indicators, each weighted for a total score of 0–100:

| Indicator | Weight | What It Measures |
|-----------|--------|-----------------|
| **VWAP Distance** | 25 | How far price has moved past VWAP |
| **Volume Surge** | 20 | Current volume vs 20-bar average |
| **EMA Trend** | 20 | EMA-9 / EMA-21 alignment with signal direction |
| **RSI Momentum** | 15 | RSI in the sweet spot (not overbought/oversold) |
| **Range Position** | 10 | Where price sits within today's high/low range |
| **Candle Momentum** | 10 | Consecutive bars moving in signal direction |

### Grade Scale

| Grade | Score | Meaning |
|-------|-------|---------|
| A+ | 85–100 | Strong conviction — all indicators aligned |
| A | 75–84 | High quality signal |
| B | 65–74 | Solid setup, minor concerns |
| C | 50–64 | Marginal — proceed with caution |
| D | 35–49 | Weak signal, most indicators not confirming |
| F | 0–34 | Noise — skip it |

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Internet connection (Yahoo Finance data)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/SPYer.git
cd SPYer
pip install -r requirements.txt
```

### Run

```bash
python spyer.py
```

### Windows One-Click Setup

If you're on Windows and prefer not to use the terminal:

1. Double-click **`INSTALL.bat`** — it handles Python, venv, and dependencies automatically
2. After install, double-click **`run_app.bat`** to launch (created by the installer)
3. You can move the entire folder anywhere afterward — it's fully portable

> **Advanced:** If you prefer PowerShell, you can run `install_windows.ps1` instead.

---

## Usage

1. Launch the app
2. Click **▶ Start Monitoring** to begin auto-scanning every 60 seconds
3. Or click **🔍 Scan Now** for an immediate check
4. When a signal fires, you'll see:
   - The signal type (CALL / PUT)
   - A quality score and letter grade
   - A per-indicator breakdown showing where the score came from
   - Recommended 0–2 DTE option strikes

### Settings

- **Call / Put Signals** — toggle which directions you want alerts for
- **Min Grade** — only alert on signals at or above this grade (default: C)
- **Vol threshold** — volume ratio required to trigger (default: 150% = 1.5x average)

---

## How Signals Work

### CALL Signal
Triggers when **all** of these are true:
- SPY price > VWAP
- Current volume > your volume threshold × average volume

Then the signal is scored across all 6 indicators. If the grade meets your minimum, you get an alert.

### PUT Signal
Same logic but with price < VWAP.

### Cooldown
5-minute cooldown between alerts to prevent spam. Manual scans bypass this.

---

## Configuration

You can tweak these in the code:

| Setting | Location | Default |
|---------|----------|---------|
| Scan interval | `monitor_timer.start(60000)` | 60 seconds |
| Signal cooldown | `self.signal_cooldown` | 300 seconds |
| VWAP weight | `weights["vwap"]` in `evaluate_signal()` | 25 |
| Volume weight | `weights["volume"]` | 20 |

---

## Data Source

Uses **Yahoo Finance** via `yfinance` — free but ~15–20 minute delayed data. Good for testing and learning.

For real-time trading, you can swap in:
- [Tradier](https://tradier.com) (free real-time with account)
- [Polygon.io](https://polygon.io) ($99/mo)
- Any broker API

---

## Project Structure

```
SPYer/
├── spyer.py                  # Main application
├── requirements.txt          # Python dependencies
├── INSTALL.bat               # Windows double-click installer
├── install_windows.ps1       # Windows PowerShell installer (alternative)
├── LICENSE                   # MIT license
└── README.md                 # This file
```

---

## Roadmap

- [ ] RSI divergence signals
- [ ] Moving average crossover alerts
- [ ] Trade journal / win-loss tracking
- [ ] Built-in charting
- [ ] Real-time data source integration
- [ ] Auto-trade via broker API (Tradier, IBKR)

---

## Disclaimer

This app provides **alerts only**, not financial advice. Always verify signals before trading. 0DTE options are extremely risky — use proper position sizing and risk management.

---

## License

MIT — see [LICENSE](LICENSE).
