# SPYderScalp v5.1

<p align="center">
  <img src="sf.jpg" width="200" alt="SPYderScalp">
</p>

**Built by: SkyzFallin**

**Real-time SPY intraday options signal monitor with multi-indicator quality scoring, multi-timeframe confirmation, prediction tracking, inline options value scanner, DTE recommendations, economic calendar awareness, and hold-time estimates.**

SPYderScalp watches SPY price action and alerts you when conditions line up for a potential 0-2 DTE options trade -- then tells you *how good* the signal is with a quality score from 0-100 and a letter grade (A+ to F), explains *why* it gave that rating in plain English, recommends *which DTE to trade* based on time of day and conditions, tells you *how long to hold* based on momentum and upcoming economic events, tracks *prediction accuracy* over time with persistent history, and scans the options chain for the *best value contracts* right in the main window.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

---

## Features

- **Side-by-side layout** -- chart on the left, signals on the right, options scanner at the bottom
- **Live candlestick chart** with VWAP, EMA 9/21, Bollinger Bands, VWAP bands, RSI subplot, and MACD subplot
- **15-candle default view** -- large, readable candles on the 1-minute timeframe
- **Multi-timeframe chart** -- switch between 1m, 5m, 15m, 1h, and 1d views
- **Auto support/resistance levels** -- detected from price action, drawn on the chart
- **Manual price lines** -- add custom levels to the chart at any price
- **Signal quality scoring** (0-100) across 12 weighted indicators
- **Always-on evaluation** -- shows signal direction, grade, and breakdown even when not all confirmation gates are met (labeled "watching"), so the UI is never blank
- **Multi-timeframe confirmation** -- cross-references 1m signals against 5m and 15m trends for higher confidence
- **RSI divergence detection** -- spots bullish/bearish divergences for early reversal signals
- **Bollinger Band squeeze detection** -- identifies low-volatility compression before breakouts
- **Candle pattern recognition** -- detects engulfing, hammer, doji, and other patterns
- **Plain-English explanations** -- every signal tells you exactly why it got that grade
- **DTE recommender** -- suggests 0, 1, or 2 DTE based on time of day, signal strength, volume, RSI, and event proximity
- **Hold-time recommendations** -- how long to hold based on DTE, momentum, events, and time of day
- **Prediction tracking with persistence** -- logs every signal, checks the outcome with countdown timers, rates accuracy over time, and saves history across sessions
- **Open/Close swing forecast** -- predicts today's direction using 3 months of historical daily data with day-of-week breakdown, last 5 sessions, and volatility stats
- **Inline options value scanner** -- scans for the best value contracts and displays the top results in a table at the bottom of the main window, auto-refreshes every 5 minutes
- **Economic calendar** -- built-in calendar of CPI, FOMC, NFP, PPI, GDP, Retail Sales, and Jobless Claims for 2025-2026 with countdown timers, populated immediately on startup
- **Event-aware scoring** -- signals near major releases get automatically downgraded
- **Intraday volatility zones** -- warns during market open surge, lunch lull, power hour
- **Eastern Time aware** -- all market logic uses ET regardless of your local timezone
- **Market hours auto-stop** -- monitoring pauses automatically at market close, resumes at open
- **Tabbed interface** -- Signal breakdown, Hold reasons, History, Calendar, and Log (stays on your selected tab)
- **Status indicator** -- live dot shows monitoring/stopped/scanning state
- Recommended 0-2 DTE option strikes
- Desktop notifications + sound alerts
- Configurable volume threshold and minimum grade filter
- Platform-aware data directory with crash logging
- Works on **Windows, macOS, and Linux**

---

## Signal Quality Scoring

Every signal is evaluated across 12 indicators, each weighted for a total score of 0-100:

| Indicator | Weight | What It Measures |
|-----------|--------|-----------------|
| **VWAP Distance** | 15 | How far price has broken past VWAP |
| **Volume Surge** | 14 | Current volume vs 20-bar average |
| **EMA Trend** | 13 | EMA-9 / EMA-21 alignment with signal direction |
| **Event Timing** | 12 | Penalty when major economic release is imminent |
| **RSI Momentum** | 8 | RSI in the sweet spot (not overbought/oversold) |
| **MACD Confirmation** | 7 | MACD line/signal crossover and histogram direction |
| **Candle Momentum** | 6 | Consecutive bars moving in signal direction |
| **RSI Divergence** | 6 | Bullish/bearish divergence between price and RSI |
| **Bollinger Squeeze** | 5 | Low-volatility compression signaling breakout |
| **Candle Patterns** | 5 | Engulfing, hammer, doji, and other reversal/continuation patterns |
| **Range Position** | 5 | Where price sits within today's high/low range |
| **Trend Alignment** | 4 | Higher-timeframe trend consistency |

Signals are then cross-referenced against 5-minute and 15-minute timeframes for **multi-timeframe confirmation**. When all timeframes agree, the signal score gets a boost; conflicting timeframes reduce confidence.

### Signal Modes

| Mode | What It Means |
|------|--------------|
| **CALL/PUT Signal** | All confirmation gates passed -- full alert with notification |
| **CALL/PUT (watching)** | Direction detected but not all gates met (e.g. low volume) -- evaluation shown, no alert |
| **Waiting for signal** | Price at VWAP or direction unchecked -- no evaluation possible |

### Grade Scale

| Grade | Score | Meaning |
|-------|-------|---------|
| A+ | 85-100 | Strong conviction -- all indicators aligned |
| A | 75-84 | High quality signal |
| B | 65-74 | Solid setup, minor concerns |
| C | 50-64 | Marginal -- proceed with caution |
| D | 35-49 | Weak signal, most indicators not confirming |
| F | 0-34 | Noise -- skip it |

---

## Prediction Tracking

Every evaluated signal is automatically logged in the **History** tab with:
- Signal type, grade, score, and price at signal time
- **Countdown timer** showing time remaining until outcome check
- Outcome determination after hold period expires (WIN / LOSS / FLAT)
- Running accuracy stats (win rate, average return, grade breakdown)
- **Persistent across sessions** -- history saves to disk and loads on next launch
- **CSV export** for offline analysis

---

## Open/Close Swing Forecast

On startup, SPYderScalp fetches 3 months of daily SPY data and displays a detailed forecast:

- **20-day overview** -- avg open-to-close, range, volatility, up/down ratio, best/worst day
- **Day-of-week breakdown** -- historical average and green rate for each weekday
- **Last 5 sessions** -- date, direction, open-to-close %, and range for recent context
- **Signal analysis** -- day-of-week bias, recent momentum, 20-day trend, mean reversion, streak detection, range contraction, gap analysis
- **Direction prediction** with confidence percentage

---

## Inline Options Value Scanner

The bottom panel scans SPY option chains and displays opportunities in a sortable table:

- **Auto-scans on launch** and refreshes every 5 minutes
- **"Scan Options" button** for immediate refresh
- Shows: Type, Strike, DTE, Bid, Ask, Mid, Volume, Score, and Signals
- Top 5 visible without scrolling, full results accessible by scrolling
- Scores color-coded by strength (teal = strong, amber = moderate, grey = weak)
- Checks for: intrinsic edge, Black-Scholes model discount, IV discount, spread value, volume spikes, liquidity, penny contracts

---

## DTE Recommender

SPYderScalp tells you whether to trade 0, 1, or 2 DTE options based on current conditions:

| Factor | 0DTE Preferred | 1DTE Preferred | 2DTE Preferred |
|--------|---------------|----------------|----------------|
| **Time of day** | Before ~1 PM ET | After 1 PM ET | Late power hour |
| **Signal grade** | A+ / A (high conviction) | B / C (needs room) | D / F (weak, needs time) |
| **Volume** | High (2x+ avg) | Normal | Low (weak follow-through) |
| **RSI** | Normal range | Any | Extreme (reversal risk) |
| **Events** | No events near | Event in 30-60 min | Major event imminent |
| **Day of week** | Mon-Thu morning | Friday afternoon | -- |

---

## Hold-Time Recommendations

Every signal includes a suggested hold duration and exit time (in Eastern Time):

| Factor | Effect |
|--------|--------|
| **DTE** | 0DTE: ~15 min scalp, 1DTE: ~30 min, 2DTE: ~45 min |
| **Signal strength** | A+ extends hold time, C/D shortens it |
| **Volume** | Very high volume = move may exhaust quickly |
| **RSI extremes** | Overbought calls / oversold puts = reversal risk |
| **Event proximity** | Exits before upcoming CPI/FOMC/NFP releases |
| **Time of day** | Caps hold time near market close (ET) |

---

## Chart Features

The live candlestick chart includes:
- **15 large candles** by default on 1m (readable wicks and bodies)
- **VWAP line** (gold) with optional VWAP standard deviation bands
- **EMA 9** (blue) and **EMA 21** (purple) overlays
- **Bollinger Bands** (toggleable)
- **Auto support/resistance levels** from price clustering (toggleable)
- **Manual price lines** -- type any price and add it to the chart
- **RSI subplot** with overbought/oversold zones
- **MACD subplot** with signal line and histogram
- **Volume bars** in the background
- **Multi-timeframe switching** -- 1m, 5m, 15m, 1h, 1d chart views

---

## Economic Calendar

Built-in calendar of every major US economic event for 2025-2026. Populated immediately on startup:

- **FOMC Decisions** -- all 8 meetings per year
- **CPI / PPI Reports** -- monthly releases
- **Non-Farm Payrolls (NFP)** -- monthly jobs reports
- **GDP Reports** -- quarterly (advance, second estimate, final)
- **Retail Sales** -- monthly consumer spending data
- **Initial Jobless Claims** -- every Thursday at 8:30 AM ET

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Internet connection (Yahoo Finance data)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/SPYderScalp.git
cd SPYderScalp
pip install -r requirements.txt
```

### Run

```bash
python spyer.py
```

### Windows One-Click Setup

Just double-click **`SPYderScalp.bat`** -- it handles everything:
- Checks for Python 3.9+ (won't silently fail on older versions)
- First run: creates venv, installs deps from requirements.txt, launches (~1 min)
- After that: launches in seconds
- Fully portable -- move the folder anywhere

### macOS One-Click Setup

Double-click **`SPYderScalp.command`** -- same idea as the Windows launcher:
- Checks for Python 3.9+, creates venv, installs deps from requirements.txt
- After that: launches in seconds and closes the Terminal window automatically
- Crash output logged to `crash.log` for debugging

> **Note:** On first run macOS may ask you to allow the script. Right-click -> Open if double-click is blocked by Gatekeeper.

Or manually:

```bash
git clone https://github.com/YOUR_USERNAME/SPYderScalp.git
cd SPYderScalp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python spyer.py
```

---

## Usage

1. Launch the app
2. Monitoring starts automatically during market hours (9:30 AM - 4:00 PM ET)
3. Or click **Scan Now** for an immediate check anytime
4. The signal area always shows the current evaluation:
   - Signal type (CALL / PUT) with quality score and letter grade
   - "watching" label if not all gates are met, full alert when confirmed
   - Multi-timeframe confirmation status
   - DTE recommendation and hold-time with exit time in ET
5. Click **Scan Options** to refresh the inline value scanner
6. Use tabs on the right panel: Signal, Hold, History, Calendar, Log
7. History tab shows prediction tracking with countdown timers

### Top Bar Controls

- **Start / Stop** -- control monitoring (auto-starts on launch during market hours)
- **Scan Now** -- trigger an immediate signal check
- **Scan Options** -- refresh the inline options value scanner
- **Calls / Puts** -- toggle which directions you want alerts for
- **Min** -- minimum grade to trigger full alerts (default: C)
- **Vol** -- volume ratio required for full confirmation (default: 150% = 1.5x average)

### Chart Controls

- **S/R** checkbox -- toggle auto-detected support/resistance levels
- **BB** checkbox -- toggle Bollinger Bands overlay
- **VWAP±** checkbox -- toggle VWAP standard deviation bands
- **Add Line** -- add a manual price line at any level
- **Timeframe buttons** -- switch chart between 1m, 5m, 15m, 1h, 1d

---

## Data Storage

App data is stored in platform-appropriate locations:
- **Windows:** `%APPDATA%\SPYderScalp\`
- **macOS:** `~/Library/Application Support/SPYderScalp/`
- **Linux:** `~/.local/share/SPYderScalp/`

Files stored: `settings.json`, `prediction_history.json`, `crash.log`

On first run, settings are automatically migrated from the old `~/.spyderscalp_settings.json` location if it exists.

---

## Timezone Handling

All market-sensitive logic runs in **US Eastern Time** regardless of your local timezone. Uses `zoneinfo` (Python 3.9+) with `pytz` fallback. Hold time exit times, market close caps, event countdowns, and intraday zones all reference ET market hours. Monitoring auto-stops at market close and resumes at open.

---

## Project Structure

```
SPYderScalp/
  spyer.py                  # Main app (signals, chart, calendar, scanner, UI)
  sf.jpg                    # App logo
  SPYderScalp.bat           # Windows smart launcher (double-click to run)
  SPYderScalp.command        # macOS smart launcher (double-click to run)
  install_windows.ps1       # Windows PowerShell installer (alternative)
  requirements.txt          # Python dependencies
  .gitignore                # Git ignore rules
  LICENSE                   # MIT license
  README.md                 # This file
```

---

## Roadmap

- [x] Multi-indicator signal scoring (12 weighted indicators)
- [x] Multi-timeframe confirmation (5m + 15m cross-reference)
- [x] Always-on signal evaluation (watching mode)
- [x] Candlestick chart with RSI & MACD
- [x] Multi-timeframe chart views (1m, 5m, 15m, 1h, 1d)
- [x] Large candle default view (15 bars)
- [x] Auto support/resistance levels
- [x] Bollinger Bands & VWAP bands overlays
- [x] RSI divergence detection
- [x] Bollinger Band squeeze detection
- [x] Candle pattern recognition
- [x] Economic calendar (2025-2026)
- [x] Hold-time recommendations
- [x] Plain-English signal explanations
- [x] Side-by-side layout
- [x] Eastern Time awareness + market hours auto-stop
- [x] DTE recommender (0/1/2 DTE)
- [x] Inline options value scanner
- [x] Prediction tracking with persistence and countdown timers
- [x] Open/Close swing forecast with detailed stats
- [x] CSV export of prediction history
- [x] Platform-aware data storage with crash logging
- [ ] Real-time data source integration
- [ ] Auto-trade via broker API (Tradier, IBKR)

---

## Disclaimer

This app provides **alerts only**, not financial advice. Always verify signals before trading. 0DTE options are extremely risky -- use proper position sizing and risk management.

---

## License

MIT -- see [LICENSE](LICENSE).
