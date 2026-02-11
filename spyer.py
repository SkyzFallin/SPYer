"""
SPYderScalp - SPY Intraday Options Signal App
Cross-Platform (Windows & macOS)

Multi-indicator signal quality scoring with economic calendar
awareness, hold-time recommendations, and plain-English explanations.
"""

import sys
import platform
import json
import math
import traceback
import os
import csv
import gc
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import yfinance as yf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout, QCheckBox,
    QProgressBar, QFrame, QComboBox, QSpinBox, QSplitter, QScrollArea,
    QTabWidget, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QToolTip, QStatusBar,
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QCursor

import matplotlib
matplotlib.use("Qt5Agg")
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

try:
    from plyer import notification as desktop_notification
except ImportError:
    desktop_notification = None

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        import pytz
        ET = pytz.timezone("America/New_York")
    except ImportError:
        ET = None

# Platform-specific sound
if platform.system() == "Darwin":
    pass  # os already imported
elif platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None


# ---------------------------------------------------------------------------
# Centralized Theme
# ---------------------------------------------------------------------------

THEME = {
    "bg": "#1e1e1e", "bg_dark": "#1a1a2e", "bg_panel": "#1e1e36",
    "border": "#333", "border_light": "#444",
    "text": "#ccc", "text_dim": "#888", "text_muted": "#666", "text_label": "#aaa",
    "green": "#26a69a", "red": "#ef5350", "yellow": "#ffab00",
    "blue": "#42a5f5", "purple": "#ab47bc", "pink": "#e040fb", "orange": "#e65100",
    "grade_ap": "#00c853", "grade_a": "#2e7d32", "grade_b": "#558b2f",
    "grade_c": "#f9a825", "grade_d": "#e65100", "grade_f": "#c62828",
    "btn_start": "background:#333;color:#5eb8a2;border:1px solid #555;padding:4px 12px;border-radius:3px;",
    "btn_stop": "background:#333;color:#cf7b78;border:1px solid #555;padding:4px 12px;border-radius:3px;",
    "btn_scan": "background:#333;color:#7ba7c9;border:1px solid #555;padding:4px 12px;border-radius:3px;",
    "btn_scanner": "background:#333;color:#a48bbf;border:1px solid #555;padding:4px 12px;border-radius:3px;",
    "btn_small": "background:#333;color:#999;border:1px solid #555;padding:2px 8px;border-radius:2px;",
    "btn_active_tf": "background:#444;color:#bbb;border:1px solid #666;border-radius:2px;padding:1px 3px;",
    "btn_inactive_tf": "background:#2a2a2a;color:#666;border:1px solid #444;border-radius:2px;padding:1px 3px;",
    "tab_content": "background:#1a1a2e; color:#ccc; border:1px solid #333;",
}


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def _app_data_dir():
    """Platform-appropriate directory for app data (settings, history, logs)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "SPYderScalp"
    d.mkdir(parents=True, exist_ok=True)
    return d

APP_DATA_DIR = _app_data_dir()
SETTINGS_FILE = APP_DATA_DIR / "settings.json"
HISTORY_FILE = APP_DATA_DIR / "prediction_history.json"
CRASH_LOG = APP_DATA_DIR / "crash.log"


def load_settings():
    defaults = {
        "min_grade": "C", "vol_threshold": 150, "show_calls": True,
        "show_puts": True, "chart_tf": "1m", "show_auto_sr": True,
        "show_bb": True, "show_vwap_bands": False, "alert_sound": True,
        "splitter_sizes": [820, 540],
    }
    try:
        # Migrate old settings file from home directory if it exists
        old_file = Path.home() / ".spyderscalp_settings.json"
        if old_file.exists() and not SETTINGS_FILE.exists():
            import shutil
            shutil.move(str(old_file), str(SETTINGS_FILE))

        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
    except Exception:
        pass
    return defaults


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def _load_prediction_history():
    """Load prediction history from disk."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            # Restore datetime objects from ISO strings
            for pred in data:
                if isinstance(pred.get("timestamp"), str):
                    pred["timestamp"] = datetime.fromisoformat(pred["timestamp"])
                if isinstance(pred.get("check_time"), str):
                    pred["check_time"] = datetime.fromisoformat(pred["check_time"])
            # Cap at 500 entries on load
            return data[-500:] if len(data) > 500 else data
    except Exception:
        pass
    return []


def _save_prediction_history(history):
    """Save prediction history to disk."""
    try:
        # Cap at 500 entries before saving
        to_save = history[-500:] if len(history) > 500 else history
        # Convert datetime objects to ISO strings for JSON
        serializable = []
        for pred in to_save:
            p = dict(pred)
            if isinstance(p.get("timestamp"), datetime):
                p["timestamp"] = p["timestamp"].isoformat()
            if isinstance(p.get("check_time"), datetime):
                p["check_time"] = p["check_time"].isoformat()
            serializable.append(p)
        with open(HISTORY_FILE, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rate-limited yfinance wrapper with retry/backoff
# ---------------------------------------------------------------------------

_yf_last_call = datetime.min
_yf_backoff_until = datetime.min


def yf_download_safe(ticker, **kwargs):
    """Wrapper around yf.download with rate limiting and retry."""
    global _yf_last_call, _yf_backoff_until
    import time as _time

    now = datetime.now()
    if now < _yf_backoff_until:
        return None

    elapsed = (now - _yf_last_call).total_seconds()
    if elapsed < 0.5:
        _time.sleep(0.5 - elapsed)

    kwargs.setdefault("progress", False)

    for attempt in range(3):
        try:
            _yf_last_call = datetime.now()
            df = yf.download(ticker, **kwargs)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = (df.columns.droplevel("Ticker")
                              if "Ticker" in df.columns.names
                              else df.columns.get_level_values(0))
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(set(df.columns)):
                return None
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            return df if len(df) >= 2 else None
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "too many" in err_str:
                backoff = 5 * (attempt + 1)
                _yf_backoff_until = datetime.now() + timedelta(seconds=backoff)
                _time.sleep(backoff)
            elif attempt < 2:
                _time.sleep(1)
            else:
                return None
    return None


# ---------------------------------------------------------------------------
# Time helpers - all market logic uses Eastern Time
# ---------------------------------------------------------------------------

def now_et():
    """Return current time in US Eastern, market-hours aware."""
    if ET is not None:
        return datetime.now(ET)
    # Fallback: assume local IS Eastern (rare but safe default)
    return datetime.now()


def market_is_open(et_now=None):
    """Check if US equity market is currently open."""
    t = et_now or now_et()
    if t.weekday() >= 5:  # Sat/Sun
        return False
    mkt_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
    mkt_close = t.replace(hour=16, minute=0, second=0, microsecond=0)
    return mkt_open <= t <= mkt_close


# ---------------------------------------------------------------------------
# Signal quality helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) as pd.Series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Compute Bollinger Bands. Returns (upper, middle, lower, bandwidth, pct_b)."""
    middle = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    bandwidth = ((upper - lower) / middle * 100).fillna(0)
    pct_b = ((series - lower) / (upper - lower)).fillna(0.5)
    return upper, middle, lower, bandwidth, pct_b


def compute_vwap_bands(df, std_mult=1.0):
    """Compute VWAP with standard deviation bands."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cumvol = df["Volume"].cumsum()
    cumvol_tp = (tp * df["Volume"]).cumsum()
    vwap = cumvol_tp / cumvol
    vwap_diff = tp - vwap
    vwap_diff_sq = (vwap_diff ** 2 * df["Volume"]).cumsum()
    vwap_std = np.sqrt(vwap_diff_sq / cumvol)
    return vwap, vwap + vwap_std, vwap - vwap_std, vwap + vwap_std * 2, vwap - vwap_std * 2


def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20):
    """
    Detect RSI divergence. Returns (type, strength) where type is
    'bullish', 'bearish', or None.
    """
    if len(close) < lookback or len(rsi) < lookback:
        return None, 0
    price_vals = close.iloc[-lookback:].values
    rsi_vals = rsi.iloc[-lookback:].values
    if np.any(np.isnan(rsi_vals)):
        return None, 0

    half = lookback // 2
    # Bullish: price lower low + RSI higher low
    price_low1, price_low2 = np.min(price_vals[:half]), np.min(price_vals[half:])
    rsi_low1, rsi_low2 = np.min(rsi_vals[:half]), np.min(rsi_vals[half:])
    if price_low2 < price_low1 and rsi_low2 > rsi_low1:
        strength = (rsi_low2 - rsi_low1) / max(abs(rsi_low1), 1)
        return "bullish", min(strength, 1.0)

    # Bearish: price higher high + RSI lower high
    price_high1, price_high2 = np.max(price_vals[:half]), np.max(price_vals[half:])
    rsi_high1, rsi_high2 = np.max(rsi_vals[:half]), np.max(rsi_vals[half:])
    if price_high2 > price_high1 and rsi_high2 < rsi_high1:
        strength = (rsi_high1 - rsi_high2) / max(abs(rsi_high1), 1)
        return "bearish", min(strength, 1.0)

    return None, 0


def detect_candle_patterns(df, lookback=3):
    """Detect candlestick patterns. Returns list of (name, bias, strength)."""
    patterns = []
    if len(df) < lookback + 1:
        return patterns

    data = df.iloc[-(lookback + 1):]
    opens, closes = data["Open"].values, data["Close"].values
    highs, lows = data["High"].values, data["Low"].values

    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body = abs(c - o)
    total_range = h - l if h != l else 0.001
    body_ratio = body / total_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    po, pc = opens[-2], closes[-2]
    pbody = abs(pc - po)
    ptotal_range = highs[-2] - lows[-2] if highs[-2] != lows[-2] else 0.001

    # Hammer
    if lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.35:
        patterns.append(("Hammer", "bullish", 0.7))
    # Shooting Star
    if upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.35:
        patterns.append(("Shooting Star", "bearish", 0.7))
    # Doji
    if body_ratio < 0.1 and total_range > 0:
        patterns.append(("Doji", "neutral", 0.5))
    # Bullish Engulfing
    if c > o and pc < po and c > po and o < pc:
        patterns.append(("Bullish Engulfing", "bullish", 0.85))
    # Bearish Engulfing
    if c < o and pc > po and c < po and o > pc:
        patterns.append(("Bearish Engulfing", "bearish", 0.85))

    # 3-candle patterns
    if len(data) >= 3:
        o2, c2 = opens[-3], closes[-3]
        body2 = abs(c2 - o2)
        # Morning Star
        if c2 < o2 and body2 > ptotal_range * 0.3 and pbody < body2 * 0.3 and c > o and body > body2 * 0.5:
            patterns.append(("Morning Star", "bullish", 0.9))
        # Evening Star
        if c2 > o2 and body2 > ptotal_range * 0.3 and pbody < body2 * 0.3 and c < o and body > body2 * 0.5:
            patterns.append(("Evening Star", "bearish", 0.9))
        # Three White Soldiers / Three Black Crows
        if all(closes[-(i+1)] > opens[-(i+1)] for i in range(3)) and closes[-1] > closes[-2] > closes[-3]:
            patterns.append(("Three White Soldiers", "bullish", 0.8))
        if all(closes[-(i+1)] < opens[-(i+1)] for i in range(3)) and closes[-1] < closes[-2] < closes[-3]:
            patterns.append(("Three Black Crows", "bearish", 0.8))

    return patterns


def detect_bb_squeeze(bandwidth: pd.Series, lookback: int = 20, squeeze_pctile: float = 20):
    """Detect Bollinger Band squeeze. Returns (is_squeeze, intensity 0-1)."""
    if len(bandwidth) < lookback:
        return False, 0
    recent_bw = bandwidth.iloc[-lookback:]
    current_bw = float(bandwidth.iloc[-1])
    if np.isnan(current_bw):
        return False, 0
    percentile = (recent_bw < current_bw).sum() / len(recent_bw) * 100
    is_squeeze = percentile < squeeze_pctile
    intensity = max(0, 1.0 - percentile / squeeze_pctile) if is_squeeze else 0
    return is_squeeze, intensity


# ---------------------------------------------------------------------------
# Threaded data fetchers (eliminate UI freezes)
# ---------------------------------------------------------------------------

class DataFetchWorker(QThread):
    """Fetch market data in a background thread."""
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, strategies=None):
        super().__init__()
        self.strategies = strategies or [
            {"period": "1d", "interval": "1m", "label": "live"},
            {"period": "5d", "interval": "5m", "label": "delayed"},
            {"period": "5d", "interval": "1d", "label": "daily"},
        ]

    def run(self):
        for strat in self.strategies:
            df = yf_download_safe("SPY", period=strat["period"], interval=strat["interval"])
            if df is not None and len(df) >= (15 if strat["label"] != "daily" else 2):
                self.finished.emit(df, strat["label"])
                return
        self.error.emit("Could not fetch data from any source")


class MTFDataFetchWorker(QThread):
    """Fetch 5m and 15m data for multi-timeframe analysis."""
    finished = pyqtSignal(object, object)

    def run(self):
        df_5m = yf_download_safe("SPY", period="5d", interval="5m")
        if df_5m is not None and len(df_5m) < 10:
            df_5m = None
        df_15m = yf_download_safe("SPY", period="5d", interval="15m")
        if df_15m is not None and len(df_15m) < 10:
            df_15m = None
        self.finished.emit(df_5m, df_15m)


class SwingFetchWorker(QThread):
    """Fetch historical daily data for swing prediction in background."""
    finished = pyqtSignal(object)

    def run(self):
        df = yf_download_safe("SPY", period="3mo", interval="1d")
        self.finished.emit(df)


# ---------------------------------------------------------------------------
# Economic calendar & event awareness
# ---------------------------------------------------------------------------

ECONOMIC_CALENDAR = [
    # 2025
    ("2025-01-10", "08:30", "Non-Farm Payrolls (Dec)", "high"),
    ("2025-01-14", "08:30", "PPI Report (Dec)", "high"),
    ("2025-01-15", "08:30", "CPI Report (Dec)", "high"),
    ("2025-01-16", "08:30", "Retail Sales (Dec)", "medium"),
    ("2025-01-29", "14:00", "FOMC Decision", "high"),
    ("2025-01-30", "08:30", "GDP Q4 Advance", "high"),
    ("2025-02-07", "08:30", "Non-Farm Payrolls (Jan)", "high"),
    ("2025-02-12", "08:30", "CPI Report (Jan)", "high"),
    ("2025-02-13", "08:30", "PPI Report (Jan)", "high"),
    ("2025-02-14", "08:30", "Retail Sales (Jan)", "medium"),
    ("2025-02-27", "08:30", "GDP Q4 Second Estimate", "medium"),
    ("2025-03-07", "08:30", "Non-Farm Payrolls (Feb)", "high"),
    ("2025-03-12", "08:30", "CPI Report (Feb)", "high"),
    ("2025-03-13", "08:30", "PPI Report (Feb)", "high"),
    ("2025-03-17", "08:30", "Retail Sales (Feb)", "medium"),
    ("2025-03-19", "14:00", "FOMC Decision", "high"),
    ("2025-03-27", "08:30", "GDP Q4 Final", "medium"),
    ("2025-04-04", "08:30", "Non-Farm Payrolls (Mar)", "high"),
    ("2025-04-10", "08:30", "CPI Report (Mar)", "high"),
    ("2025-04-11", "08:30", "PPI Report (Mar)", "high"),
    ("2025-04-16", "08:30", "Retail Sales (Mar)", "medium"),
    ("2025-04-30", "08:30", "GDP Q1 Advance", "high"),
    ("2025-05-02", "08:30", "Non-Farm Payrolls (Apr)", "high"),
    ("2025-05-07", "14:00", "FOMC Decision", "high"),
    ("2025-05-13", "08:30", "CPI Report (Apr)", "high"),
    ("2025-05-14", "08:30", "PPI Report (Apr)", "high"),
    ("2025-05-15", "08:30", "Retail Sales (Apr)", "medium"),
    ("2025-05-29", "08:30", "GDP Q1 Second Estimate", "medium"),
    ("2025-06-06", "08:30", "Non-Farm Payrolls (May)", "high"),
    ("2025-06-11", "08:30", "CPI Report (May)", "high"),
    ("2025-06-12", "08:30", "PPI Report (May)", "high"),
    ("2025-06-17", "08:30", "Retail Sales (May)", "medium"),
    ("2025-06-18", "14:00", "FOMC Decision", "high"),
    ("2025-06-26", "08:30", "GDP Q1 Final", "medium"),
    ("2025-07-03", "08:30", "Non-Farm Payrolls (Jun)", "high"),
    ("2025-07-11", "08:30", "CPI Report (Jun)", "high"),
    ("2025-07-15", "08:30", "PPI Report (Jun)", "high"),
    ("2025-07-16", "08:30", "Retail Sales (Jun)", "medium"),
    ("2025-07-30", "14:00", "FOMC Decision", "high"),
    ("2025-07-30", "08:30", "GDP Q2 Advance", "high"),
    ("2025-08-01", "08:30", "Non-Farm Payrolls (Jul)", "high"),
    ("2025-08-12", "08:30", "CPI Report (Jul)", "high"),
    ("2025-08-13", "08:30", "PPI Report (Jul)", "high"),
    ("2025-08-15", "08:30", "Retail Sales (Jul)", "medium"),
    ("2025-08-28", "08:30", "GDP Q2 Second Estimate", "medium"),
    ("2025-09-05", "08:30", "Non-Farm Payrolls (Aug)", "high"),
    ("2025-09-10", "08:30", "CPI Report (Aug)", "high"),
    ("2025-09-11", "08:30", "PPI Report (Aug)", "high"),
    ("2025-09-16", "08:30", "Retail Sales (Aug)", "medium"),
    ("2025-09-17", "14:00", "FOMC Decision", "high"),
    ("2025-09-25", "08:30", "GDP Q2 Final", "medium"),
    ("2025-10-03", "08:30", "Non-Farm Payrolls (Sep)", "high"),
    ("2025-10-14", "08:30", "CPI Report (Sep)", "high"),
    ("2025-10-15", "08:30", "PPI Report (Sep)", "high"),
    ("2025-10-16", "08:30", "Retail Sales (Sep)", "medium"),
    ("2025-10-29", "08:30", "GDP Q3 Advance", "high"),
    ("2025-11-05", "14:00", "FOMC Decision", "high"),
    ("2025-11-07", "08:30", "Non-Farm Payrolls (Oct)", "high"),
    ("2025-11-12", "08:30", "CPI Report (Oct)", "high"),
    ("2025-11-13", "08:30", "PPI Report (Oct)", "high"),
    ("2025-11-14", "08:30", "Retail Sales (Oct)", "medium"),
    ("2025-11-26", "08:30", "GDP Q3 Second Estimate", "medium"),
    ("2025-12-05", "08:30", "Non-Farm Payrolls (Nov)", "high"),
    ("2025-12-10", "08:30", "CPI Report (Nov)", "high"),
    ("2025-12-11", "08:30", "PPI Report (Nov)", "high"),
    ("2025-12-16", "08:30", "Retail Sales (Nov)", "medium"),
    ("2025-12-17", "14:00", "FOMC Decision", "high"),
    ("2025-12-23", "08:30", "GDP Q3 Final", "medium"),
    # 2026
    ("2026-01-09", "08:30", "Non-Farm Payrolls (Dec 2025)", "high"),
    ("2026-01-13", "08:30", "CPI Report (Dec 2025)", "high"),
    ("2026-01-14", "08:30", "PPI Report (Dec 2025)", "high"),
    ("2026-01-15", "08:30", "Retail Sales (Dec 2025)", "medium"),
    ("2026-01-28", "14:00", "FOMC Decision", "high"),
    ("2026-01-29", "08:30", "GDP Q4 2025 Advance", "high"),
    ("2026-02-06", "08:30", "Non-Farm Payrolls (Jan)", "high"),
    ("2026-02-11", "08:30", "CPI Report (Jan)", "high"),
    ("2026-02-12", "08:30", "PPI Report (Jan)", "high"),
    ("2026-02-13", "08:30", "Retail Sales (Jan)", "medium"),
    ("2026-02-26", "08:30", "GDP Q4 2025 Second Estimate", "medium"),
    ("2026-03-06", "08:30", "Non-Farm Payrolls (Feb)", "high"),
    ("2026-03-11", "08:30", "CPI Report (Feb)", "high"),
    ("2026-03-12", "08:30", "PPI Report (Feb)", "high"),
    ("2026-03-16", "08:30", "Retail Sales (Feb)", "medium"),
    ("2026-03-18", "14:00", "FOMC Decision", "high"),
    ("2026-03-26", "08:30", "GDP Q4 2025 Final", "medium"),
    ("2026-04-03", "08:30", "Non-Farm Payrolls (Mar)", "high"),
    ("2026-04-14", "08:30", "CPI Report (Mar)", "high"),
    ("2026-04-15", "08:30", "PPI Report (Mar)", "high"),
    ("2026-04-16", "08:30", "Retail Sales (Mar)", "medium"),
    ("2026-04-29", "08:30", "GDP Q1 Advance", "high"),
    ("2026-05-01", "08:30", "Non-Farm Payrolls (Apr)", "high"),
    ("2026-05-06", "14:00", "FOMC Decision", "high"),
    ("2026-05-12", "08:30", "CPI Report (Apr)", "high"),
    ("2026-05-13", "08:30", "PPI Report (Apr)", "high"),
    ("2026-05-15", "08:30", "Retail Sales (Apr)", "medium"),
    ("2026-05-28", "08:30", "GDP Q1 Second Estimate", "medium"),
    ("2026-06-05", "08:30", "Non-Farm Payrolls (May)", "high"),
    ("2026-06-10", "08:30", "CPI Report (May)", "high"),
    ("2026-06-11", "08:30", "PPI Report (May)", "high"),
    ("2026-06-15", "08:30", "Retail Sales (May)", "medium"),
    ("2026-06-17", "14:00", "FOMC Decision", "high"),
    ("2026-06-25", "08:30", "GDP Q1 Final", "medium"),
    ("2026-07-02", "08:30", "Non-Farm Payrolls (Jun)", "high"),
    ("2026-07-14", "08:30", "CPI Report (Jun)", "high"),
    ("2026-07-15", "08:30", "PPI Report (Jun)", "high"),
    ("2026-07-16", "08:30", "Retail Sales (Jun)", "medium"),
    ("2026-07-29", "14:00", "FOMC Decision", "high"),
    ("2026-07-30", "08:30", "GDP Q2 Advance", "high"),
    ("2026-08-07", "08:30", "Non-Farm Payrolls (Jul)", "high"),
    ("2026-08-12", "08:30", "CPI Report (Jul)", "high"),
    ("2026-08-13", "08:30", "PPI Report (Jul)", "high"),
    ("2026-08-14", "08:30", "Retail Sales (Jul)", "medium"),
    ("2026-08-27", "08:30", "GDP Q2 Second Estimate", "medium"),
    ("2026-09-04", "08:30", "Non-Farm Payrolls (Aug)", "high"),
    ("2026-09-15", "08:30", "CPI Report (Aug)", "high"),
    ("2026-09-16", "14:00", "FOMC Decision", "high"),
    ("2026-09-16", "08:30", "PPI Report (Aug)", "high"),
    ("2026-09-16", "08:30", "Retail Sales (Aug)", "medium"),
    ("2026-09-24", "08:30", "GDP Q2 Final", "medium"),
    ("2026-10-02", "08:30", "Non-Farm Payrolls (Sep)", "high"),
    ("2026-10-13", "08:30", "CPI Report (Sep)", "high"),
    ("2026-10-14", "08:30", "PPI Report (Sep)", "high"),
    ("2026-10-15", "08:30", "Retail Sales (Sep)", "medium"),
    ("2026-10-28", "08:30", "GDP Q3 Advance", "high"),
    ("2026-11-04", "14:00", "FOMC Decision", "high"),
    ("2026-11-06", "08:30", "Non-Farm Payrolls (Oct)", "high"),
    ("2026-11-12", "08:30", "CPI Report (Oct)", "high"),
    ("2026-11-13", "08:30", "PPI Report (Oct)", "high"),
    ("2026-11-16", "08:30", "Retail Sales (Oct)", "medium"),
    ("2026-11-25", "08:30", "GDP Q3 Second Estimate", "medium"),
    ("2026-12-04", "08:30", "Non-Farm Payrolls (Nov)", "high"),
    ("2026-12-09", "08:30", "CPI Report (Nov)", "high"),
    ("2026-12-10", "08:30", "PPI Report (Nov)", "high"),
    ("2026-12-15", "08:30", "Retail Sales (Nov)", "medium"),
    ("2026-12-16", "14:00", "FOMC Decision", "high"),
    ("2026-12-22", "08:30", "GDP Q3 Final", "medium"),
]

INTRADAY_VOLATILITY = {
    (9, 30, 10, 0):  ("Market open surge - wide spreads, fakeouts common", 0.7),
    (10, 0, 10, 30): ("Opening range settling - direction establishing", 0.3),
    (11, 30, 13, 0): ("Lunch lull - low volume, choppy, avoid new entries", 0.1),
    (14, 0, 14, 30): ("Bond market close - can trigger SPY moves", 0.3),
    (15, 30, 16, 0): ("Power hour / MOC imbalance - high vol, fast moves", 0.6),
}


def _get_events_for_date(target_date):
    target_str = target_date.strftime("%Y-%m-%d")
    events = []
    for d, t, name, impact in ECONOMIC_CALENDAR:
        if d == target_str:
            events.append({"name": name, "time": t, "impact": impact, "date": d})
    if target_date.weekday() == 3:
        events.append({"name": "Initial Jobless Claims", "time": "08:30",
                        "impact": "medium", "date": target_str})
    return events


def get_upcoming_events(now, days_ahead=14):
    upcoming = []
    today = now.date()
    for day_offset in range(days_ahead + 1):
        d = today + timedelta(days=day_offset)
        for ev in _get_events_for_date(d):
            ev["days_until"] = day_offset
            if day_offset == 0:
                h, m = map(int, ev["time"].split(":"))
                ev_minutes = h * 60 + m
                current_minutes = now.hour * 60 + now.minute
                ev["minutes_until"] = ev_minutes - current_minutes
            else:
                ev["minutes_until"] = None
            upcoming.append(ev)
    return upcoming


def get_event_context(et_now):
    """et_now should be a datetime in Eastern Time."""
    upcoming = get_upcoming_events(et_now, days_ahead=14)
    today_events = [e for e in upcoming if e["days_until"] == 0]
    current_minutes = et_now.hour * 60 + et_now.minute

    today_upcoming = [e for e in today_events
                      if e["minutes_until"] is not None and e["minutes_until"] > -10]
    today_upcoming.sort(key=lambda e: abs(e["minutes_until"]))
    nearest = today_upcoming[0] if today_upcoming else None
    minutes_to = nearest["minutes_until"] if nearest else None

    event_penalty = 0.0
    if nearest and minutes_to is not None:
        if minutes_to <= 0:
            event_penalty = 0.9
        elif minutes_to <= 5:
            event_penalty = 0.85
        elif minutes_to <= 15:
            event_penalty = 0.6
        elif minutes_to <= 30:
            event_penalty = 0.4
        elif minutes_to <= 60:
            event_penalty = 0.2
        if nearest["impact"] == "medium":
            event_penalty *= 0.6

    zone_label = None
    zone_severity = 0.0
    for (sh, sm, eh, em), (label, sev) in INTRADAY_VOLATILITY.items():
        start = sh * 60 + sm
        end = eh * 60 + em
        if start <= current_minutes < end:
            zone_label = label
            zone_severity = sev
            break

    return {
        "events_today": today_events,
        "upcoming_events": upcoming,
        "nearest_event": nearest,
        "minutes_to_nearest": minutes_to,
        "event_penalty": event_penalty,
        "intraday_zone": zone_label,
        "intraday_severity": zone_severity,
    }


# ---------------------------------------------------------------------------
# Hold time estimator
# ---------------------------------------------------------------------------

def estimate_hold_time(signal_type, score, dte, vol_ratio, rsi, event_ctx):
    et_now = now_et()
    time_decimal = et_now.hour + et_now.minute / 60.0
    reasons = []

    if dte == 0:
        base = 15
        reasons.append(f"0DTE -> base {base} min scalp window")
    elif dte == 1:
        base = 30
        reasons.append(f"1DTE -> base {base} min hold")
    else:
        base = 45
        reasons.append(f"2DTE -> base {base} min hold")

    if score >= 85:
        base = int(base * 1.5)
        reasons.append(f"A+ signal -> extended to {base} min (strong conviction)")
    elif score >= 75:
        base = int(base * 1.3)
        reasons.append(f"A signal -> extended to {base} min")
    elif score >= 65:
        reasons.append("B signal -> standard hold")
    elif score >= 50:
        base = int(base * 0.7)
        reasons.append(f"C signal -> shortened to {base} min (marginal)")
    else:
        base = int(base * 0.5)
        reasons.append(f"Weak signal -> shortened to {base} min")

    if vol_ratio > 2.5:
        base = int(base * 0.8)
        reasons.append(f"Very high volume ({vol_ratio:.1f}x) -> may exhaust quickly")
    elif vol_ratio < 1.3:
        base = int(base * 0.7)
        reasons.append(f"Low volume ({vol_ratio:.1f}x) -> weak follow-through likely")

    if signal_type == "CALL" and rsi > 72:
        base = int(base * 0.6)
        reasons.append(f"RSI {rsi:.0f} overbought -> reversal risk")
    elif signal_type == "PUT" and rsi < 28:
        base = int(base * 0.6)
        reasons.append(f"RSI {rsi:.0f} oversold -> bounce risk")

    # Shorten hold times in afternoon regardless - moves are less reliable
    if time_decimal >= 14.5:
        base = int(base * 0.7)
        reasons.append(f"Late session -> shortened to {base} min (less follow-through)")
    elif time_decimal >= 13.0:
        base = int(base * 0.85)
        reasons.append(f"Afternoon -> slightly shortened to {base} min")

    # Lunch hour (11:30-1:00) - low liquidity, moves are choppy
    if 11.5 <= time_decimal < 13.0:
        base = int(base * 0.8)
        reasons.append(f"Lunch chop zone -> shortened to {base} min")

    nearest = event_ctx.get("nearest_event")
    mins_to = event_ctx.get("minutes_to_nearest")
    penalty = event_ctx.get("event_penalty", 0)

    if nearest and mins_to is not None:
        if mins_to > 0 and mins_to < base:
            base = max(3, mins_to - 5)
            reasons.append(f"[!] {nearest['name']} in {mins_to} min -> exit before event (hold {base} min)")
        elif mins_to <= 0 and mins_to > -15:
            base = min(base, 5)
            reasons.append(f"[!] {nearest['name']} just released -> quick scalp only ({base} min)")
        elif penalty > 0.3:
            base = int(base * (1 - penalty * 0.5))
            reasons.append(f"{nearest['name']} approaching -> shortened to {base} min")

    zone = event_ctx.get("intraday_zone")
    zone_sev = event_ctx.get("intraday_severity", 0)
    if zone:
        if zone_sev >= 0.5:
            base = int(base * 0.7)
            reasons.append(f"{zone} -> shortened to {base} min")
        else:
            reasons.append(f"{zone}")

    market_close = et_now.replace(hour=15, minute=55, second=0, microsecond=0)
    if dte == 0:
        market_close = et_now.replace(hour=15, minute=45, second=0, microsecond=0)
    mkt_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If before market open, base exit off open time
    if et_now < mkt_open:
        minutes_to_close = max(0, (market_close - mkt_open).total_seconds() / 60)
    else:
        minutes_to_close = max(0, (market_close - et_now).total_seconds() / 60)

    if base > minutes_to_close and minutes_to_close > 0:
        base = max(3, int(minutes_to_close))
        reasons.append(f"Near market close -> capped at {base} min")

    base = max(3, base)

    # Compute exit time in ET, clamped to market hours
    if et_now < mkt_open:
        exit_time_et = mkt_open + timedelta(minutes=base)
    else:
        exit_time_et = et_now + timedelta(minutes=base)
    # Never show exit past market close
    if exit_time_et > market_close:
        exit_time_et = market_close
    exit_str = exit_time_et.strftime("%I:%M %p ET")

    if penalty > 0.5 or base <= 5:
        confidence = "low"
    elif score >= 70 and penalty < 0.2:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "hold_minutes": base,
        "hold_label": f"~{base} min" if base < 60 else f"~{base // 60}h {base % 60}m",
        "exit_by": exit_str,
        "reasoning": reasons,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# DTE Recommender - suggests 0, 1, or 2 DTE
# ---------------------------------------------------------------------------

def recommend_dte(signal_score, signal_grade, vol_ratio, rsi, event_ctx, mtf_multiplier=1.0):
    """
    Recommend which DTE to trade based on market conditions.
    Returns dict with recommended_dte, reasoning, and alt options.
    """
    et_now = now_et()
    hour = et_now.hour
    minute = et_now.minute
    time_decimal = hour + minute / 60.0
    reasons = []
    dte_scores = {0: 50, 1: 50, 2: 30}  # base preference

    # --- Time of day ---
    if time_decimal < 10.5:  # Before 10:30 AM ET
        dte_scores[0] += 30
        reasons.append("Early session -> 0DTE has max theta burn ahead")
    elif time_decimal < 12.0:  # 10:30 AM - 12:00 PM ET
        dte_scores[0] += 15
        dte_scores[1] += 10
        reasons.append("Late morning -> 0DTE still viable, 1DTE safer after lunch")
    elif time_decimal < 13.0:  # 12:00 - 1:00 PM ET
        dte_scores[0] -= 20
        dte_scores[1] += 30
        reasons.append("Noon -> shifting to 1DTE, 0DTE decay accelerating")
    elif time_decimal < 14.0:  # 1:00 - 2:00 PM ET
        dte_scores[0] -= 40
        dte_scores[1] += 35
        reasons.append("Early afternoon -> 1DTE strongly preferred, 0DTE too risky")
    elif time_decimal < 15.0:  # 2:00 - 3:00 PM ET
        dte_scores[0] -= 60
        dte_scores[1] += 40
        reasons.append("Late afternoon -> 1DTE only, 0DTE is a coin flip at this point")
    else:  # After 3:00 PM ET
        dte_scores[0] -= 100  # hard kill 0DTE
        dte_scores[1] += 50
        dte_scores[2] += 15
        reasons.append("Final hour -> 1DTE required, 0DTE is gambling not trading")

    # --- Signal strength ---
    # After noon, even strong signals shouldn't override 1DTE preference
    if signal_grade in ("A+", "A"):
        if time_decimal < 12.0:
            dte_scores[0] += 20
            reasons.append(f"Strong signal ({signal_grade}) -> 0DTE to maximize leverage")
        else:
            dte_scores[1] += 15
            reasons.append(f"Strong signal ({signal_grade}) -> good for 1DTE, too late for 0DTE")
    elif signal_grade == "B":
        dte_scores[0] += 5
        dte_scores[1] += 10
        reasons.append("Solid signal (B) -> 0DTE fine, 1DTE gives cushion")
    elif signal_grade == "C":
        dte_scores[0] -= 15
        dte_scores[1] += 15
        dte_scores[2] += 10
        reasons.append("Marginal signal (C) -> longer DTE for room to be right")
    else:
        dte_scores[0] -= 30
        dte_scores[1] -= 10
        dte_scores[2] += 25
        reasons.append(f"Weak signal ({signal_grade}) -> 2DTE or skip the trade")

    # --- Volume ---
    if vol_ratio > 2.0:
        dte_scores[0] += 10
        reasons.append(f"High volume ({vol_ratio:.1f}x) -> momentum supports 0DTE")
    elif vol_ratio < 1.2:
        dte_scores[0] -= 15
        dte_scores[1] += 10
        reasons.append(f"Low volume ({vol_ratio:.1f}x) -> needs time to develop, longer DTE")

    # --- RSI extremes ---
    if rsi > 75 or rsi < 25:
        dte_scores[0] -= 15
        dte_scores[1] += 10
        reasons.append(f"RSI extreme ({rsi:.0f}) -> reversal risk, longer DTE safer")

    # --- Event proximity ---
    penalty = event_ctx.get("event_penalty", 0)
    nearest = event_ctx.get("nearest_event")
    mins_to = event_ctx.get("minutes_to_nearest")

    if penalty > 0.5 and nearest:
        dte_scores[0] -= 25
        dte_scores[1] += 15
        dte_scores[2] += 20
        reasons.append(f"Major event soon ({nearest['name']}) -> longer DTE to survive vol spike")
    elif penalty > 0.2 and nearest:
        dte_scores[0] -= 10
        dte_scores[1] += 10
        reasons.append(f"Event approaching ({nearest['name']}) -> slight preference for 1DTE")

    # --- Day of week ---
    dow = et_now.weekday()
    if dow == 4:  # Friday
        dte_scores[0] -= 20  # 0DTE on Friday is already riskier
        if time_decimal > 12:
            dte_scores[0] -= 30  # Friday afternoon 0DTE = no
            dte_scores[1] += 15
            reasons.append("Friday afternoon -> 0DTE expires today, 1DTE carries to Monday")
        elif time_decimal > 10.5:
            dte_scores[0] -= 10
            reasons.append("Friday midday -> 0DTE window shrinking fast")

    # --- Multi-timeframe confirmation ---
    if mtf_multiplier < 0.75:
        dte_scores[0] -= 30
        dte_scores[1] += 15
        dte_scores[2] += 10
        reasons.append("Higher TFs conflict -> 0DTE too risky without trend support, prefer 1-2DTE")
    elif mtf_multiplier < 0.90:
        dte_scores[0] -= 15
        dte_scores[1] += 10
        reasons.append("Higher TFs partially conflict -> lean toward 1DTE for safety")
    elif mtf_multiplier > 1.05:
        dte_scores[0] += 10
        reasons.append("Higher TFs confirm -> 0DTE viable with trend support")

    # Pick winner
    best_dte = max(dte_scores, key=dte_scores.get)
    sorted_dtes = sorted(dte_scores.items(), key=lambda x: x[1], reverse=True)

    # Build recommendation text
    if best_dte == 0:
        rec_text = "0DTE - Max profit potential, fastest theta capture"
    elif best_dte == 1:
        rec_text = "1DTE - Balanced risk/reward, overnight cushion"
    else:
        rec_text = "2DTE - Conservative, more time to be right"

    # Check if it's close between top two
    margin = sorted_dtes[0][1] - sorted_dtes[1][1]
    if margin < 10:
        alt_dte = sorted_dtes[1][0]
        rec_text += f" (close call with {alt_dte}DTE)"

    return {
        "recommended_dte": best_dte,
        "dte_scores": dte_scores,
        "recommendation": rec_text,
        "reasoning": reasons,
        "margin": margin,
    }


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_to_grade(score):
    if score >= 85: return "A+"
    elif score >= 75: return "A"
    elif score >= 65: return "B"
    elif score >= 50: return "C"
    elif score >= 35: return "D"
    else: return "F"


def grade_color(grade):
    return {
        "A+": THEME["grade_ap"], "A": THEME["grade_a"], "B": THEME["grade_b"],
        "C": THEME["grade_c"], "D": THEME["grade_d"], "F": THEME["grade_f"],
    }.get(grade, THEME["text_dim"])


# ---------------------------------------------------------------------------
# Multi-Timeframe (MTF) confirmation layer
# ---------------------------------------------------------------------------

def analyze_timeframe(df, signal_type):
    """
    Analyze a single timeframe's agreement with the proposed signal.
    Returns a dict with direction cues from that timeframe.
    """
    if df is None or df.empty or len(df) < 5:
        return None

    close = df["Close"]
    current_price = float(close.iloc[-1])

    # VWAP direction (if available)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = float((tp * df["Volume"]).cumsum().iloc[-1] / df["Volume"].cumsum().iloc[-1])
    price_vs_vwap = "above" if current_price > vwap else "below"

    # EMA trend
    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema9_val = float(ema9.iloc[-1])
    ema21_val = float(ema21.iloc[-1])
    ema_bullish = ema9_val > ema21_val

    # EMA slope (is ema9 rising or falling?)
    if len(ema9) >= 3:
        ema9_slope = float(ema9.iloc[-1]) - float(ema9.iloc[-3])
    else:
        ema9_slope = 0

    # MACD
    macd_l, macd_s, macd_h = compute_macd(close)
    macd_hist = float(macd_h.iloc[-1])
    macd_bullish = macd_hist > 0

    # Last candle direction
    last_open = float(df["Open"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    last_candle_green = last_close > last_open

    # RSI
    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50

    # How many cues agree with the signal?
    if signal_type == "CALL":
        agrees = sum([
            price_vs_vwap == "above",
            ema_bullish,
            ema9_slope > 0,
            macd_bullish,
            last_candle_green,
            50 < rsi < 70,  # healthy bullish range
        ])
        conflicts = sum([
            price_vs_vwap == "below",
            not ema_bullish,
            ema9_slope < 0,
            not macd_bullish,
            not last_candle_green,
            rsi < 40,  # bearish momentum
        ])
    else:  # PUT
        agrees = sum([
            price_vs_vwap == "below",
            not ema_bullish,
            ema9_slope < 0,
            not macd_bullish,
            not last_candle_green,
            30 < rsi < 50,  # healthy bearish range
        ])
        conflicts = sum([
            price_vs_vwap == "above",
            ema_bullish,
            ema9_slope > 0,
            macd_bullish,
            last_candle_green,
            rsi > 60,  # bullish momentum
        ])

    return {
        "agrees": agrees,
        "conflicts": conflicts,
        "total_cues": 6,
        "vwap": price_vs_vwap,
        "ema_bullish": ema_bullish,
        "ema9_slope": ema9_slope,
        "macd_bullish": macd_bullish,
        "macd_hist": macd_hist,
        "candle_green": last_candle_green,
        "rsi": rsi,
    }


def compute_mtf_confirmation(signal_type, df_5m, df_15m):
    """
    Compute a multi-timeframe confirmation multiplier.
    Returns (multiplier, reasoning_list) where multiplier is 0.6 - 1.15.
    
    Logic:
    - Both timeframes strongly agree: 1.15x boost
    - Both agree: 1.05x mild boost
    - Mixed (one agrees, one neutral): 1.0x no change
    - One conflicts: 0.85x penalty
    - Both conflict: 0.65x heavy penalty
    """
    analysis_5m = analyze_timeframe(df_5m, signal_type)
    analysis_15m = analyze_timeframe(df_15m, signal_type)

    reasons = []
    multiplier = 1.0

    if analysis_5m is None and analysis_15m is None:
        return 1.0, ["MTF data unavailable - no adjustment"], None

    results = {}

    # Score each timeframe: agrees/total -> ratio
    for label, analysis in [("5m", analysis_5m), ("15m", analysis_15m)]:
        if analysis is None:
            results[label] = "unavailable"
            continue
        agree_ratio = analysis["agrees"] / analysis["total_cues"]
        conflict_ratio = analysis["conflicts"] / analysis["total_cues"]

        if agree_ratio >= 0.65:
            results[label] = "confirming"
            direction = "bullish" if signal_type == "CALL" else "bearish"
            reasons.append(f"{label}: {analysis['agrees']}/{analysis['total_cues']} cues {direction} (confirming)")
        elif conflict_ratio >= 0.50:
            results[label] = "conflicting"
            direction = "bullish" if signal_type == "PUT" else "bearish"
            reasons.append(f"{label}: {analysis['conflicts']}/{analysis['total_cues']} cues {direction} (CONFLICTING)")
        else:
            results[label] = "neutral"
            reasons.append(f"{label}: mixed signals ({analysis['agrees']} agree, {analysis['conflicts']} conflict)")

    # Compute multiplier based on combined result
    confirms = sum(1 for v in results.values() if v == "confirming")
    conflicts = sum(1 for v in results.values() if v == "conflicting")
    neutrals = sum(1 for v in results.values() if v == "neutral")

    if confirms == 2:
        multiplier = 1.15
        reasons.append("=> Both 5m+15m confirm: +15% score boost")
    elif confirms == 1 and conflicts == 0:
        multiplier = 1.05
        reasons.append("=> One TF confirms, none conflict: +5% boost")
    elif conflicts == 0:
        multiplier = 1.0
        reasons.append("=> Mixed/neutral higher TFs: no adjustment")
    elif conflicts == 1 and confirms == 0:
        multiplier = 0.85
        reasons.append("=> One TF conflicts: -15% score penalty")
    elif conflicts == 1 and confirms == 1:
        multiplier = 0.92
        reasons.append("=> TFs disagree with each other: -8% penalty")
    elif conflicts == 2:
        multiplier = 0.65
        reasons.append("=> Both 5m+15m conflict: -35% heavy penalty")

    mtf_detail = {
        "5m": analysis_5m,
        "15m": analysis_15m,
        "results": results,
        "multiplier": multiplier,
    }

    return multiplier, reasons, mtf_detail


def evaluate_signal(df, signal_type, event_ctx=None):
    breakdown = {}
    weights = {
        "vwap": 15, "volume": 14, "rsi": 8, "ema_trend": 13,
        "range_position": 5, "momentum": 6, "event_timing": 12,
        "macd_confirm": 7, "trend_alignment": 4,
        "rsi_divergence": 6, "bb_squeeze": 5, "candle_pattern": 5,
    }
    raw_scores = {}

    close = df["Close"]
    current_price = float(close.iloc[-1])
    current_vwap = float(df["VWAP"].iloc[-1])

    # 1. VWAP distance - require meaningful break, not just 1 cent over
    vwap_diff_pct = (current_price - current_vwap) / current_vwap * 100
    if signal_type == "CALL":
        raw_scores["vwap"] = np.clip(vwap_diff_pct / 0.30, 0, 1)
    else:
        raw_scores["vwap"] = np.clip(-vwap_diff_pct / 0.30, 0, 1)
    # Penalty for being too close to VWAP (likely to reverse)
    if abs(vwap_diff_pct) < 0.03:
        raw_scores["vwap"] *= 0.3
    breakdown["vwap"] = f"Price {vwap_diff_pct:+.3f}% from VWAP"

    # 2. Volume (use last completed bar, not the still-forming one)
    if len(df) > 22:
        last_vol = float(df["Volume"].iloc[-2])
        avg_vol = float(df["Volume"].iloc[-22:-2].mean())
    elif len(df) > 3:
        last_vol = float(df["Volume"].iloc[-2])
        avg_vol = float(df["Volume"].iloc[:-2].mean())
    else:
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].mean())
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0
    raw_scores["volume"] = np.clip((vol_ratio - 1.0) / 2.0, 0, 1)
    breakdown["volume"] = f"Volume {vol_ratio:.2f}x avg"

    # 3. RSI - penalize signals fighting the momentum
    if "_rsi" in df.columns:
        rsi = float(df["_rsi"].iloc[-1]) if not pd.isna(df["_rsi"].iloc[-1]) else 50
    else:
        rsi_series = compute_rsi(close)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
    if signal_type == "CALL":
        if 50 <= rsi <= 68:
            raw_scores["rsi"] = 1.0 - abs(rsi - 58) / 20
        elif rsi > 68:
            raw_scores["rsi"] = max(0, 1.0 - (rsi - 68) / 15)  # stricter overbought penalty
        elif rsi < 40:
            raw_scores["rsi"] = 0  # CALL when RSI is bearish = bad
        else:
            raw_scores["rsi"] = max(0, (rsi - 30) / 30)
    else:
        if 32 <= rsi <= 50:
            raw_scores["rsi"] = 1.0 - abs(rsi - 42) / 20
        elif rsi < 32:
            raw_scores["rsi"] = max(0, 1.0 - (32 - rsi) / 15)  # stricter oversold penalty
        elif rsi > 60:
            raw_scores["rsi"] = 0  # PUT when RSI is bullish = bad
        else:
            raw_scores["rsi"] = max(0, (70 - rsi) / 30)
    breakdown["rsi"] = f"RSI {rsi:.1f}"

    # 4. EMA trend - also check if EMA9 crossed EMA21 recently
    if "_ema9" in df.columns:
        ema9 = df["_ema9"]
        ema21 = df["_ema21"]
    else:
        ema9 = compute_ema(close, 9)
        ema21 = compute_ema(close, 21)
    ema_diff = (float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / float(ema21.iloc[-1]) * 100
    if signal_type == "CALL":
        raw_scores["ema_trend"] = np.clip(ema_diff / 0.15, 0, 1)
    else:
        raw_scores["ema_trend"] = np.clip(-ema_diff / 0.15, 0, 1)
    # Penalize if EMAs are pointing the wrong way
    if len(ema9) >= 3:
        ema9_slope = float(ema9.iloc[-1]) - float(ema9.iloc[-3])
        if signal_type == "CALL" and ema9_slope < 0:
            raw_scores["ema_trend"] *= 0.5
        elif signal_type == "PUT" and ema9_slope > 0:
            raw_scores["ema_trend"] *= 0.5
    breakdown["ema_trend"] = f"EMA9-EMA21 gap {ema_diff:+.3f}%"

    # 5. Range position - penalize chasing at extremes
    day_high = float(df["High"].max())
    day_low = float(df["Low"].min())
    day_range = day_high - day_low if day_high != day_low else 1
    range_pct = (current_price - day_low) / day_range
    if signal_type == "CALL":
        # Don't reward buying at the very top of the range
        if range_pct > 0.9:
            raw_scores["range_position"] = 0.3  # chasing the high
        else:
            raw_scores["range_position"] = range_pct
    else:
        if range_pct < 0.1:
            raw_scores["range_position"] = 0.3  # chasing the low
        else:
            raw_scores["range_position"] = 1.0 - range_pct
    breakdown["range_position"] = f"Range position {range_pct:.0%} (low->high)"

    # 6. Candle momentum - require at least 2 bars, penalize single bar spikes
    last_n = close.iloc[-6:]
    changes = last_n.diff().dropna().values
    if signal_type == "CALL":
        streak = sum(1 for c in reversed(changes) if c > 0)
    else:
        streak = sum(1 for c in reversed(changes) if c < 0)
    if streak >= 2:
        raw_scores["momentum"] = np.clip(streak / 4, 0, 1)
    elif streak == 1:
        raw_scores["momentum"] = 0.2  # single bar moves are unreliable
    else:
        raw_scores["momentum"] = 0
    direction_word = "green" if signal_type == "CALL" else "red"
    breakdown["momentum"] = f"{streak} consecutive {direction_word} bars"

    # 7. Event timing
    if event_ctx:
        penalty = event_ctx.get("event_penalty", 0)
        nearest = event_ctx.get("nearest_event")
        mins_to = event_ctx.get("minutes_to_nearest")
        raw_scores["event_timing"] = 1.0 - penalty
        if nearest and mins_to is not None and penalty > 0:
            breakdown["event_timing"] = f"[!] {nearest['name']} in {mins_to} min (penalty {penalty:.0%})"
        else:
            breakdown["event_timing"] = "No imminent events [OK]"
    else:
        raw_scores["event_timing"] = 1.0
        breakdown["event_timing"] = "No event data"

    # 8. MACD confirmation (NEW) - does MACD agree with signal direction?
    if "_macd" in df.columns and "_macd_signal" in df.columns:
        macd_val = float(df["_macd"].iloc[-1])
        macd_sig = float(df["_macd_signal"].iloc[-1])
        macd_hist = float(df["_macd_hist"].iloc[-1])
    else:
        macd_l, macd_s, macd_h = compute_macd(close)
        macd_val = float(macd_l.iloc[-1])
        macd_sig = float(macd_s.iloc[-1])
        macd_hist = float(macd_h.iloc[-1])

    if signal_type == "CALL":
        if macd_hist > 0 and macd_val > macd_sig:
            raw_scores["macd_confirm"] = min(1.0, abs(macd_hist) / 0.3)
        elif macd_hist > 0:
            raw_scores["macd_confirm"] = 0.4
        else:
            raw_scores["macd_confirm"] = 0  # MACD bearish, calling bullish = bad
    else:
        if macd_hist < 0 and macd_val < macd_sig:
            raw_scores["macd_confirm"] = min(1.0, abs(macd_hist) / 0.3)
        elif macd_hist < 0:
            raw_scores["macd_confirm"] = 0.4
        else:
            raw_scores["macd_confirm"] = 0  # MACD bullish, calling bearish = bad
    breakdown["macd_confirm"] = f"MACD hist {macd_hist:+.3f}"

    # 9. Trend alignment (NEW) - is price above/below key EMAs consistently?
    price_above_ema9 = current_price > float(ema9.iloc[-1])
    price_above_ema21 = current_price > float(ema21.iloc[-1])
    price_above_vwap = current_price > current_vwap
    if signal_type == "CALL":
        alignment = sum([price_above_ema9, price_above_ema21, price_above_vwap])
        raw_scores["trend_alignment"] = alignment / 3.0
    else:
        alignment = sum([not price_above_ema9, not price_above_ema21, not price_above_vwap])
        raw_scores["trend_alignment"] = alignment / 3.0
    breakdown["trend_alignment"] = f"{alignment}/3 indicators aligned"

    # 10. RSI Divergence (NEW)
    rsi_series_full = df["_rsi"] if "_rsi" in df.columns else compute_rsi(close)
    div_type, div_strength = detect_rsi_divergence(close, rsi_series_full, lookback=20)
    if div_type == "bullish" and signal_type == "CALL":
        raw_scores["rsi_divergence"] = min(1.0, div_strength * 2)
        breakdown["rsi_divergence"] = f"Bullish RSI divergence ({div_strength:.0%})"
    elif div_type == "bearish" and signal_type == "PUT":
        raw_scores["rsi_divergence"] = min(1.0, div_strength * 2)
        breakdown["rsi_divergence"] = f"Bearish RSI divergence ({div_strength:.0%})"
    elif div_type and ((div_type == "bullish" and signal_type == "PUT") or
                       (div_type == "bearish" and signal_type == "CALL")):
        raw_scores["rsi_divergence"] = 0
        breakdown["rsi_divergence"] = f"{div_type.title()} divergence OPPOSES {signal_type.lower()}"
    else:
        raw_scores["rsi_divergence"] = 0.5
        breakdown["rsi_divergence"] = "No divergence detected"

    # 11. Bollinger Band Squeeze (NEW)
    if "_bb_bandwidth" in df.columns:
        bw = df["_bb_bandwidth"]
        pct_b_val = float(df["_bb_pct_b"].iloc[-1]) if not pd.isna(df["_bb_pct_b"].iloc[-1]) else 0.5
    else:
        _, _, _, bw, pct_b = compute_bollinger_bands(close)
        pct_b_val = float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else 0.5

    is_squeeze, squeeze_intensity = detect_bb_squeeze(bw)
    if is_squeeze:
        if (signal_type == "CALL" and pct_b_val > 0.5) or (signal_type == "PUT" and pct_b_val < 0.5):
            raw_scores["bb_squeeze"] = 0.5 + squeeze_intensity * 0.5
            breakdown["bb_squeeze"] = f"BB squeeze ({squeeze_intensity:.0%}) + price aligned"
        else:
            raw_scores["bb_squeeze"] = 0.3
            breakdown["bb_squeeze"] = f"BB squeeze ({squeeze_intensity:.0%}) direction unclear"
    else:
        if signal_type == "CALL":
            raw_scores["bb_squeeze"] = np.clip(pct_b_val, 0, 1) * 0.6
        else:
            raw_scores["bb_squeeze"] = np.clip(1 - pct_b_val, 0, 1) * 0.6
        breakdown["bb_squeeze"] = f"BB %B: {pct_b_val:.2f} (no squeeze)"

    # 12. Candle Pattern Recognition (NEW)
    patterns = detect_candle_patterns(df)
    pattern_score = 0
    pattern_names = []
    for pname, pbias, pstrength in patterns:
        if (pbias == "bullish" and signal_type == "CALL") or \
           (pbias == "bearish" and signal_type == "PUT"):
            pattern_score += pstrength
            pattern_names.append(f"{pname} ({pbias})")
        elif pbias != "neutral":
            pattern_score -= pstrength * 0.5
    raw_scores["candle_pattern"] = np.clip(pattern_score, 0, 1)
    breakdown["candle_pattern"] = " + ".join(pattern_names) if pattern_names else "No significant patterns"

    total = sum(raw_scores[k] * weights[k] for k in weights)
    total = round(min(total, 100), 1)
    grade = score_to_grade(total)

    return {
        "score": total, "grade": grade, "breakdown": breakdown,
        "raw": raw_scores, "weights": weights, "rsi": rsi,
        "ema9": float(ema9.iloc[-1]), "ema21": float(ema21.iloc[-1]), "vol_ratio": vol_ratio,
    }


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------

def find_support_resistance(df, n_touches=2, tolerance_pct=0.08):
    """
    Find support and resistance levels from price action.
    Uses pivot highs/lows and clusters them within tolerance.
    Returns list of (price, type, touches) tuples, limited to strongest.
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    if len(highs) < 10:
        return []

    # Find local pivot highs and lows (5-bar pivots for stronger levels)
    pivot_highs = []
    pivot_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and \
           highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            pivot_highs.append(highs[i])
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and \
           lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            pivot_lows.append(lows[i])

    # Cluster nearby pivots into levels
    all_pivots = [(p, "resistance") for p in pivot_highs] + \
                 [(p, "support") for p in pivot_lows]
    if not all_pivots:
        return []

    all_pivots.sort(key=lambda x: x[0])
    current_price = closes[-1]
    price_range = max(highs) - min(lows) if max(highs) != min(lows) else 1
    tol = price_range * tolerance_pct

    levels = []
    used = [False] * len(all_pivots)
    for i, (price, ptype) in enumerate(all_pivots):
        if used[i]:
            continue
        cluster = [price]
        cluster_types = [ptype]
        used[i] = True
        for j in range(i + 1, len(all_pivots)):
            if used[j]:
                continue
            if abs(all_pivots[j][0] - price) <= tol:
                cluster.append(all_pivots[j][0])
                cluster_types.append(all_pivots[j][1])
                used[j] = True
        if len(cluster) >= n_touches:
            avg_price = sum(cluster) / len(cluster)
            # Label based on where it is relative to current price
            if avg_price > current_price:
                level_type = "resistance"
            else:
                level_type = "support"
            levels.append((avg_price, level_type, len(cluster)))

    # Also add day high / low as key levels
    day_high = max(highs)
    day_low = min(lows)
    # Check they aren't too close to existing levels
    for lvl_price, _, _ in levels:
        if day_high is not None and abs(day_high - lvl_price) < tol:
            day_high = None
        if day_low is not None and abs(day_low - lvl_price) < tol:
            day_low = None
    if day_high is not None:
        levels.append((day_high, "resistance", 1))
    if day_low is not None:
        levels.append((day_low, "support", 1))

    levels.sort(key=lambda x: x[2], reverse=True)  # sort by touches
    levels = levels[:6]  # keep only the strongest
    levels.sort(key=lambda x: x[0])  # re-sort by price
    return levels


# ---------------------------------------------------------------------------
# Candlestick chart widget
# ---------------------------------------------------------------------------

class CandlestickWidget(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 0.5, 1, 1], hspace=0.08)
        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_vol = self.fig.add_subplot(gs[1], sharex=self.ax_price)
        self.ax_rsi = self.fig.add_subplot(gs[2], sharex=self.ax_price)
        self.ax_macd = self.fig.add_subplot(gs[3], sharex=self.ax_price)
        for ax in [self.ax_price, self.ax_vol, self.ax_rsi, self.ax_macd]:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaa", labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
        for ax in [self.ax_price, self.ax_vol, self.ax_rsi]:
            ax.tick_params(labelbottom=False)

        # S/R state
        self.show_auto_sr = True
        self.show_bb_bands = True
        self.show_vwap_bands = False
        self.manual_lines = []
        self._last_ylim = None
        self._chart_data = None
        self._chart_labels = None

        # Double-click to add manual line
        self.mpl_connect("button_press_event", self._on_click)
        # Crosshair tooltip on hover
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("axes_leave_event", self._on_mouse_leave)

    def _on_mouse_move(self, event):
        """Show OHLCV tooltip on hover over price chart."""
        if event.inaxes != self.ax_price or event.xdata is None:
            QToolTip.hideText()
            return
        if self._chart_data is not None and self._chart_labels is not None:
            idx = int(round(event.xdata))
            if 0 <= idx < len(self._chart_data):
                row = self._chart_data.iloc[idx]
                lbl = self._chart_labels[idx] if idx < len(self._chart_labels) else ""
                tip = (f"{lbl}\nO: ${row['Open']:.2f}  H: ${row['High']:.2f}\n"
                       f"L: ${row['Low']:.2f}  C: ${row['Close']:.2f}\n"
                       f"Vol: {row['Volume']:,.0f}")
                QToolTip.showText(QCursor.pos(), tip, self)

    def _on_mouse_leave(self, event):
        QToolTip.hideText()

    def _on_click(self, event):
        """Double-click on price panel to add a manual horizontal line."""
        if event.dblclick and event.inaxes == self.ax_price and event.ydata is not None:
            price = round(event.ydata, 2)
            # If clicking near an existing manual line, remove it instead
            for existing in self.manual_lines:
                if self._last_ylim:
                    rng = self._last_ylim[1] - self._last_ylim[0]
                    if abs(existing - price) < rng * 0.01:
                        self.manual_lines.remove(existing)
                        self._redraw_lines()
                        return
            self.manual_lines.append(price)
            self._redraw_lines()

    def _redraw_lines(self):
        """Redraw just the horizontal lines without full chart refresh."""
        # Remove old manual line artists
        to_remove = [l for l in self.ax_price.lines if getattr(l, '_is_manual_line', False)]
        for l in to_remove:
            l.remove()
        # Draw manual lines
        for price in self.manual_lines:
            line = self.ax_price.axhline(price, color="#e040fb", linewidth=1.0,
                                         linestyle="--", alpha=0.9)
            line._is_manual_line = True
            # Price label on right edge
            self.ax_price.text(self.ax_price.get_xlim()[1], price, f" ${price:.2f}",
                              fontsize=6, color="#e040fb", va="center",
                              clip_on=True)
        self.draw_idle()

    def clear_manual_lines(self):
        self.manual_lines.clear()

    def add_manual_line(self, price):
        if price not in self.manual_lines:
            self.manual_lines.append(price)

    def remove_manual_line(self, price):
        self.manual_lines = [p for p in self.manual_lines if abs(p - price) > 0.005]

    def update_chart(self, full_df, n_bars=15):
        for ax in [self.ax_price, self.ax_vol, self.ax_rsi, self.ax_macd]:
            ax.cla()
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaa", labelsize=6)

        data = full_df.iloc[-n_bars:]
        full_close = full_df["Close"]
        if len(data) < 2:
            self.draw()
            return

        # Store for crosshair tooltip
        self._chart_data = data

        x = np.arange(len(data))
        opens = data["Open"].values
        closes = data["Close"].values
        highs = data["High"].values
        lows = data["Low"].values
        volumes = data["Volume"].values

        colors_up = "#26a69a"
        colors_dn = "#ef5350"
        cwick = [colors_up if c >= o else colors_dn for o, c in zip(opens, closes)]
        cbody = cwick[:]

        body_bot = np.minimum(opens, closes)
        body_h = np.abs(closes - opens)
        body_h = np.where(body_h == 0, 0.01, body_h)

        self.ax_price.vlines(x, lows, highs, color=cwick, linewidth=0.6)
        self.ax_price.bar(x, body_h, bottom=body_bot, width=0.6,
                          color=cbody, edgecolor=cbody, linewidth=0.5)

        if "VWAP" in data.columns:
            self.ax_price.plot(x, data["VWAP"].values, color="#ffab00", linewidth=1.2,
                               label="VWAP", alpha=0.9)

        ema9_vals = (full_df["_ema9"] if "_ema9" in full_df.columns else compute_ema(full_close, 9)).iloc[-n_bars:].values
        ema21_vals = (full_df["_ema21"] if "_ema21" in full_df.columns else compute_ema(full_close, 21)).iloc[-n_bars:].values
        self.ax_price.plot(x, ema9_vals, color="#42a5f5", linewidth=0.8, label="EMA 9", alpha=0.7)
        self.ax_price.plot(x, ema21_vals, color="#ab47bc", linewidth=0.8, label="EMA 21", alpha=0.7)

        # Bollinger Bands
        if self.show_bb_bands and "_bb_upper" in full_df.columns:
            bb_u = full_df["_bb_upper"].iloc[-n_bars:].values
            bb_l = full_df["_bb_lower"].iloc[-n_bars:].values
            self.ax_price.plot(x, bb_u, color="#ffffff", linewidth=0.5, linestyle=":", alpha=0.3, label="BB")
            self.ax_price.plot(x, bb_l, color="#ffffff", linewidth=0.5, linestyle=":", alpha=0.3)
            self.ax_price.fill_between(x, bb_l, bb_u, alpha=0.04, color="#ffffff")

        # VWAP deviation bands
        if self.show_vwap_bands and "_vwap_u1" in full_df.columns:
            u1 = full_df["_vwap_u1"].iloc[-n_bars:].values
            l1 = full_df["_vwap_l1"].iloc[-n_bars:].values
            self.ax_price.fill_between(x, l1, u1, alpha=0.06, color="#ffab00")

        # --- Auto S/R levels (cached until data changes) ---
        if self.show_auto_sr:
            data_key = len(full_df)
            if not hasattr(self, '_sr_cache') or self._sr_cache[0] != data_key:
                self._sr_cache = (data_key, find_support_resistance(full_df))
            levels = self._sr_cache[1]
            for price_level, level_type, touches in levels:
                if level_type == "resistance":
                    color = "#ef5350"
                    label_prefix = "R"
                else:
                    color = "#26a69a"
                    label_prefix = "S"
                self.ax_price.axhline(price_level, color=color, linewidth=0.7,
                                      linestyle=":", alpha=0.6)
                self.ax_price.text(len(x) - 1, price_level,
                                  f" {label_prefix} ${price_level:.2f} ({touches}x)",
                                  fontsize=5.5, color=color, va="bottom" if level_type == "resistance" else "top",
                                  alpha=0.8, clip_on=True)

        # --- Manual lines ---
        for mprice in self.manual_lines:
            line = self.ax_price.axhline(mprice, color="#e040fb", linewidth=1.0,
                                         linestyle="--", alpha=0.9)
            line._is_manual_line = True
            self.ax_price.text(len(x) - 1, mprice, f" ${mprice:.2f}",
                              fontsize=6, color="#e040fb", va="center", clip_on=True)

        vcol = [colors_up + "55" if c >= o else colors_dn + "55" for o, c in zip(opens, closes)]
        self.ax_vol.bar(x, volumes, width=0.5, color=vcol)
        self.ax_vol.set_ylim(0, volumes.max() * 4 if volumes.max() > 0 else 1)
        self.ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}"
        ))

        self.ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        self.ax_price.legend(loc="upper left", fontsize=7, facecolor="#1e1e1e",
                             edgecolor="#444", labelcolor="#ccc")

        # Store ylim for click proximity detection
        self._last_ylim = self.ax_price.get_ylim()

        rsi_full = full_df["_rsi"] if "_rsi" in full_df.columns else compute_rsi(full_close, 14)
        rsi_vals = rsi_full.iloc[-n_bars:].values
        self.ax_rsi.plot(x, rsi_vals, color="#42a5f5", linewidth=1.0)
        self.ax_rsi.axhline(70, color="#ef5350", linewidth=0.6, linestyle="--", alpha=0.6)
        self.ax_rsi.axhline(30, color="#26a69a", linewidth=0.6, linestyle="--", alpha=0.6)
        self.ax_rsi.axhline(50, color="#666", linewidth=0.4, linestyle=":", alpha=0.4)
        self.ax_rsi.fill_between(x, 30, 70, alpha=0.05, color="white")
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.set_ylabel("RSI", fontsize=7, color="#aaa")
        self.ax_rsi.yaxis.set_major_locator(mticker.FixedLocator([30, 50, 70]))

        if "_macd" in full_df.columns:
            macd_line = full_df["_macd"]
            signal_line = full_df["_macd_signal"]
            histogram = full_df["_macd_hist"]
        else:
            macd_line, signal_line, histogram = compute_macd(full_close)
        macd_vals = macd_line.iloc[-n_bars:].values
        sig_vals = signal_line.iloc[-n_bars:].values
        hist_vals = histogram.iloc[-n_bars:].values
        hist_colors = [colors_up if h >= 0 else colors_dn for h in hist_vals]
        self.ax_macd.bar(x, hist_vals, width=0.5, color=hist_colors, alpha=0.5)
        self.ax_macd.plot(x, macd_vals, color="#42a5f5", linewidth=1.0, label="MACD")
        self.ax_macd.plot(x, sig_vals, color="#ff7043", linewidth=0.8, label="Signal")
        self.ax_macd.axhline(0, color="#666", linewidth=0.4, linestyle=":")
        self.ax_macd.set_ylabel("MACD", fontsize=7, color="#aaa")
        self.ax_macd.legend(loc="upper left", fontsize=6, facecolor="#1e1e1e",
                            edgecolor="#444", labelcolor="#ccc")

        if hasattr(data.index, "strftime"):
            # Choose format based on data resolution
            if len(data) >= 2:
                td = (data.index[-1] - data.index[0]).total_seconds() / max(len(data) - 1, 1)
                if td >= 86000:  # daily
                    labels = data.index.strftime("%m/%d")
                elif td >= 3000:  # hourly
                    labels = data.index.strftime("%m/%d %H:%M")
                else:  # intraday
                    labels = data.index.strftime("%H:%M")
            else:
                labels = data.index.strftime("%H:%M")
        else:
            labels = [str(i) for i in range(len(data))]
        self._chart_labels = list(labels)  # store for tooltip
        step = max(1, len(x) // 10)
        self.ax_macd.set_xticks(x[::step])
        self.ax_macd.set_xticklabels(labels[::step], rotation=45, fontsize=6, color="#aaa")

        try:
            self.fig.tight_layout(pad=0.5)
        except Exception:
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, hspace=0.08)
        self.draw()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class SPYderScalpApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPYderScalp")
        self.setGeometry(0, 0, 1366, 720)
        self.last_signal_time = None
        self.last_signal_type = None    # Track direction for whipsaw detection
        self.signal_cooldown = 300
        self.consecutive_same_signal = 0  # Track repeated signals for confirmation
        self.is_monitoring = False
        self.platform = platform.system()
        self.prediction_history = _load_prediction_history()  # persisted across sessions
        self.swing_prediction = None  # open/close swing forecast
        self.session_pnl = []  # session P&L entries
        self._settings = load_settings()
        self._build_ui()
        self._apply_saved_settings()
        # Refresh history display if we loaded data from a previous session
        if self.prediction_history:
            self._update_history_display()
            self._update_track_summary()
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_signals)
        # Timer to check prediction outcomes every 30 seconds
        self.prediction_check_timer = QTimer()
        self.prediction_check_timer.timeout.connect(self._check_prediction_outcomes)
        self.prediction_check_timer.start(30000)
        # Memory cleanup every 10 minutes
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_memory)
        self.cleanup_timer.start(600000)
        # Live clock - ticks every second
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(1000)
        # Populate calendar immediately so it doesn't just say "Scanning..."
        try:
            event_ctx = get_event_context(now_et())
            self._update_events(event_ctx)
        except Exception:
            pass
        # Inline scanner: refresh every 5 minutes, initial scan after 3s
        self.scanner_timer = QTimer()
        self.scanner_timer.timeout.connect(self._run_inline_scanner)
        self.scanner_timer.start(300000)  # 5 min
        QTimer.singleShot(3000, self._run_inline_scanner)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(2)

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        title = QLabel("SPYderScalp")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        top_bar.addWidget(title)

        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet("background:#333;color:#5eb8a2;border:1px solid #555;padding:4px 12px;border-radius:3px;")
        self.btn_start.clicked.connect(self.start_monitoring)
        top_bar.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background:#333;color:#cf7b78;border:1px solid #555;padding:4px 12px;border-radius:3px;")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_stop.setEnabled(False)
        top_bar.addWidget(self.btn_stop)

        self.btn_scan = QPushButton("Scan Now")
        self.btn_scan.setStyleSheet("background:#333;color:#7ba7c9;border:1px solid #555;padding:4px 12px;border-radius:3px;")
        self.btn_scan.clicked.connect(self.manual_scan)
        top_bar.addWidget(self.btn_scan)

        self.btn_scanner = QPushButton("Scan Options")
        self.btn_scanner.setStyleSheet("background:#333;color:#a48bbf;border:1px solid #555;padding:4px 12px;border-radius:3px;")
        self.btn_scanner.clicked.connect(self._open_scanner)
        top_bar.addWidget(self.btn_scanner)

        top_bar.addSpacing(12)
        self.cb_calls = QCheckBox("Calls")
        self.cb_calls.setChecked(True)
        top_bar.addWidget(self.cb_calls)
        self.cb_puts = QCheckBox("Puts")
        self.cb_puts.setChecked(True)
        top_bar.addWidget(self.cb_puts)

        top_bar.addWidget(QLabel("Min:"))
        self.combo_min_grade = QComboBox()
        self.combo_min_grade.addItems(["F", "D", "C", "B", "A", "A+"])
        self.combo_min_grade.setCurrentText("C")
        self.combo_min_grade.setFixedWidth(50)
        top_bar.addWidget(self.combo_min_grade)

        top_bar.addWidget(QLabel("Vol:"))
        self.spin_vol = QSpinBox()
        self.spin_vol.setRange(100, 500)
        self.spin_vol.setSingleStep(10)
        self.spin_vol.setValue(150)
        self.spin_vol.setSuffix("%")
        self.spin_vol.setFixedWidth(70)
        top_bar.addWidget(self.spin_vol)

        top_bar.addStretch()
        self.lbl_update = QLabel("")
        self.lbl_update.setFont(QFont("Arial", 8))
        self.lbl_update.setStyleSheet("color:gray;")
        top_bar.addWidget(self.lbl_update)
        outer.addLayout(top_bar)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: chart
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        price_row = QHBoxLayout()
        price_row.setSpacing(12)
        self.lbl_price = QLabel("SPY --")
        self.lbl_price.setFont(QFont("Arial", 11, QFont.Bold))
        price_row.addWidget(self.lbl_price)
        self.lbl_vwap = QLabel("VWAP --")
        self.lbl_vwap.setFont(QFont("Arial", 9))
        self.lbl_vwap.setStyleSheet("color:#ffab00;")
        price_row.addWidget(self.lbl_vwap)
        self.lbl_volume = QLabel("Vol --")
        self.lbl_volume.setFont(QFont("Arial", 9))
        price_row.addWidget(self.lbl_volume)
        self.lbl_rsi = QLabel("RSI --")
        self.lbl_rsi.setFont(QFont("Arial", 9))
        price_row.addWidget(self.lbl_rsi)
        self.lbl_ema = QLabel("EMA --")
        self.lbl_ema.setFont(QFont("Arial", 9))
        price_row.addWidget(self.lbl_ema)
        price_row.addStretch()
        left_layout.addLayout(price_row)

        # S/R controls row
        sr_row = QHBoxLayout()
        sr_row.setSpacing(6)
        self.cb_auto_sr = QCheckBox("S/R Levels")
        self.cb_auto_sr.setChecked(True)
        self.cb_auto_sr.setFont(QFont("Arial", 8))
        self.cb_auto_sr.stateChanged.connect(self._toggle_auto_sr)
        sr_row.addWidget(self.cb_auto_sr)

        sr_row.addWidget(QLabel("|"))

        lbl_add = QLabel("Add line $")
        lbl_add.setFont(QFont("Arial", 8))
        sr_row.addWidget(lbl_add)
        self.spin_manual_price = QDoubleSpinBox()
        self.spin_manual_price.setRange(0, 9999)
        self.spin_manual_price.setDecimals(2)
        self.spin_manual_price.setSingleStep(0.50)
        self.spin_manual_price.setValue(0)
        self.spin_manual_price.setFixedWidth(80)
        self.spin_manual_price.setFont(QFont("Arial", 8))
        sr_row.addWidget(self.spin_manual_price)

        btn_add_line = QPushButton("+")
        btn_add_line.setFixedWidth(24)
        btn_add_line.setStyleSheet("background:#e040fb;color:white;border-radius:2px;font-weight:bold;")
        btn_add_line.clicked.connect(self._add_manual_line)
        sr_row.addWidget(btn_add_line)

        btn_clear = QPushButton("Clear Lines")
        btn_clear.setFont(QFont("Arial", 8))
        btn_clear.setStyleSheet("background:#333;color:#999;border:1px solid #555;padding:2px 8px;border-radius:2px;")
        btn_clear.clicked.connect(self._clear_manual_lines)
        sr_row.addWidget(btn_clear)

        hint = QLabel("(dbl-click chart to add/remove)")
        hint.setFont(QFont("Arial", 7))
        hint.setStyleSheet("color:#666;")
        sr_row.addWidget(hint)

        # BB and VWAP band toggles
        self.cb_bb_bands = QCheckBox("BB")
        self.cb_bb_bands.setChecked(True)
        self.cb_bb_bands.setFont(QFont("Arial", 8))
        self.cb_bb_bands.stateChanged.connect(lambda s: setattr(self.candle_chart, 'show_bb_bands', bool(s)))
        sr_row.addWidget(self.cb_bb_bands)

        self.cb_vwap_bands = QCheckBox("VWAP±σ")
        self.cb_vwap_bands.setChecked(False)
        self.cb_vwap_bands.setFont(QFont("Arial", 8))
        self.cb_vwap_bands.stateChanged.connect(lambda s: setattr(self.candle_chart, 'show_vwap_bands', bool(s)))
        sr_row.addWidget(self.cb_vwap_bands)

        sr_row.addStretch()
        left_layout.addLayout(sr_row)

        # Chart timeframe selector row
        tf_row = QHBoxLayout()
        tf_row.setSpacing(2)
        lbl_tf = QLabel("Chart:")
        lbl_tf.setFont(QFont("Arial", 8))
        lbl_tf.setStyleSheet("color:#aaa;")
        tf_row.addWidget(lbl_tf)

        self._chart_tf = "1m"  # current chart timeframe
        self._chart_tf_buttons = {}
        for tf_label in ["1m", "5m", "10m", "15m", "1h", "1d"]:
            btn = QPushButton(tf_label)
            btn.setFont(QFont("Arial", 7, QFont.Bold))
            btn.setFixedHeight(20)
            btn.setFixedWidth(34)
            btn.setCursor(Qt.PointingHandCursor)
            is_active = (tf_label == "1m")
            if is_active:
                btn.setStyleSheet("background:#444;color:#bbb;border:1px solid #666;border-radius:2px;padding:1px 3px;")
            else:
                btn.setStyleSheet("background:#2a2a2a;color:#666;border:1px solid #444;border-radius:2px;padding:1px 3px;")
            btn.clicked.connect(lambda checked, t=tf_label: self._set_chart_timeframe(t))
            tf_row.addWidget(btn)
            self._chart_tf_buttons[tf_label] = btn

        tf_row.addStretch()
        left_layout.addLayout(tf_row)

        self.candle_chart = CandlestickWidget(self)
        self.candle_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.candle_chart, stretch=1)
        splitter.addWidget(left_widget)

        # RIGHT: signals panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(3)

        # --- Status indicator ---
        status_row = QHBoxLayout()
        self.lbl_status_dot = QLabel("*")
        self.lbl_status_dot.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_status_dot.setStyleSheet("color:#888;")
        status_row.addWidget(self.lbl_status_dot)
        self.lbl_status_text = QLabel("Idle")
        self.lbl_status_text.setFont(QFont("Arial", 9))
        self.lbl_status_text.setStyleSheet("color:#888;")
        status_row.addWidget(self.lbl_status_text)
        status_row.addStretch()
        # Win/loss summary
        self.lbl_track_summary = QLabel("")
        self.lbl_track_summary.setFont(QFont("Arial", 8))
        self.lbl_track_summary.setStyleSheet("color:#aaa;")
        status_row.addWidget(self.lbl_track_summary)
        right_layout.addLayout(status_row)

        # --- Big signal visual: arrow + grade ---
        sig_visual = QHBoxLayout()
        sig_visual.setSpacing(8)
        self.lbl_arrow = QLabel("")
        self.lbl_arrow.setFont(QFont("Arial", 48, QFont.Bold))
        self.lbl_arrow.setAlignment(Qt.AlignCenter)
        self.lbl_arrow.setFixedWidth(60)
        self.lbl_arrow.setFixedHeight(60)
        sig_visual.addWidget(self.lbl_arrow)

        sig_info = QVBoxLayout()
        sig_info.setSpacing(0)
        self.lbl_signal = QLabel("No signal")
        self.lbl_signal.setFont(QFont("Arial", 14, QFont.Bold))
        sig_info.addWidget(self.lbl_signal)

        grade_hold = QHBoxLayout()
        grade_hold.setSpacing(8)
        self.lbl_grade = QLabel("")
        self.lbl_grade.setFont(QFont("Arial", 20, QFont.Bold))
        grade_hold.addWidget(self.lbl_grade)
        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("%v / 100")
        self.quality_bar.setFixedHeight(16)
        grade_hold.addWidget(self.quality_bar, stretch=1)
        sig_info.addLayout(grade_hold)
        sig_visual.addLayout(sig_info, stretch=1)
        right_layout.addLayout(sig_visual)

        # --- Hold / DTE / Exit row ---
        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        self.lbl_hold_time = QLabel("Hold: --")
        self.lbl_hold_time.setFont(QFont("Arial", 9, QFont.Bold))
        info_row.addWidget(self.lbl_hold_time)
        self.lbl_exit_by = QLabel("")
        self.lbl_exit_by.setFont(QFont("Arial", 9))
        info_row.addWidget(self.lbl_exit_by)
        self.lbl_hold_confidence = QLabel("")
        self.lbl_hold_confidence.setFont(QFont("Arial", 9))
        info_row.addWidget(self.lbl_hold_confidence)
        info_row.addStretch()
        self.lbl_dte_rec = QLabel("DTE: --")
        self.lbl_dte_rec.setFont(QFont("Arial", 9, QFont.Bold))
        info_row.addWidget(self.lbl_dte_rec)
        self.lbl_dte_detail = QLabel("")
        self.lbl_dte_detail.setFont(QFont("Arial", 7))
        self.lbl_dte_detail.setStyleSheet("color:#aaa;")
        info_row.addWidget(self.lbl_dte_detail)
        right_layout.addLayout(info_row)

        # --- Swing prediction section (expanded) ---
        swing_box = QGroupBox("Open/Close Swing Forecast")
        swing_box.setFont(QFont("Arial", 9, QFont.Bold))
        swing_box.setStyleSheet("QGroupBox{color:#aaa;border:1px solid #333;border-radius:3px;margin-top:6px;padding-top:12px;}")
        swing_lay = QVBoxLayout(swing_box)
        swing_lay.setContentsMargins(4, 4, 4, 4)
        swing_lay.setSpacing(2)
        self.lbl_swing = QTextEdit()
        self.lbl_swing.setReadOnly(True)
        self.lbl_swing.setFont(QFont("Consolas", 9))
        self.lbl_swing.setStyleSheet("color:#111; background:#d8d8d8; border:none;")
        self.lbl_swing.setPlainText("Loading historical data...")
        swing_lay.addWidget(self.lbl_swing)

        # --- Splitter: forecast on top (bigger), tabs on bottom (smaller) ---
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(swing_box)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 8))
        tab_style = "background:#1a1a2e; color:#ccc; border:1px solid #333;"

        tab_bd = QWidget()
        QVBoxLayout(tab_bd).setContentsMargins(4, 4, 4, 4)
        self.lbl_breakdown = QTextEdit()
        self.lbl_breakdown.setReadOnly(True)
        self.lbl_breakdown.setFont(QFont("Consolas", 8))
        self.lbl_breakdown.setStyleSheet(tab_style)
        tab_bd.layout().addWidget(self.lbl_breakdown)
        self.tabs.addTab(tab_bd, "Signal")

        tab_hold = QWidget()
        QVBoxLayout(tab_hold).setContentsMargins(4, 4, 4, 4)
        self.lbl_hold_reasons = QTextEdit()
        self.lbl_hold_reasons.setReadOnly(True)
        self.lbl_hold_reasons.setFont(QFont("Consolas", 8))
        self.lbl_hold_reasons.setStyleSheet(tab_style)
        tab_hold.layout().addWidget(self.lbl_hold_reasons)
        self.tabs.addTab(tab_hold, "Hold")

        # History tab - prediction tracking
        tab_hist = QWidget()
        hist_layout = QVBoxLayout(tab_hist)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_top = QHBoxLayout()
        hist_top.addStretch()
        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.setFont(QFont("Arial", 7))
        btn_export_csv.setStyleSheet("background:#333;color:#7ba7c9;border:1px solid #555;padding:2px 8px;border-radius:2px;")
        btn_export_csv.clicked.connect(self._export_history_csv)
        hist_top.addWidget(btn_export_csv)
        btn_clear_hist = QPushButton("Clear History")
        btn_clear_hist.setFont(QFont("Arial", 7))
        btn_clear_hist.setStyleSheet("background:#333;color:#999;border:1px solid #555;padding:2px 8px;border-radius:2px;")
        btn_clear_hist.clicked.connect(self._clear_history)
        hist_top.addWidget(btn_clear_hist)
        hist_layout.addLayout(hist_top)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Consolas", 8))
        self.history_text.setStyleSheet(tab_style)
        self.history_text.setPlainText("No predictions yet. Signals will be tracked here.")
        hist_layout.addWidget(self.history_text)
        self.tabs.addTab(tab_hist, "History")

        tab_cal = QWidget()
        QVBoxLayout(tab_cal).setContentsMargins(4, 4, 4, 4)
        self.lbl_events = QTextEdit()
        self.lbl_events.setReadOnly(True)
        self.lbl_events.setFont(QFont("Consolas", 8))
        self.lbl_events.setStyleSheet(tab_style)
        self.lbl_events.setPlainText("Scanning...")
        tab_cal.layout().addWidget(self.lbl_events)
        self.tabs.addTab(tab_cal, "Calendar")

        tab_log = QWidget()
        log_layout = QVBoxLayout(tab_log)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_top = QHBoxLayout()
        log_top.addStretch()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.setFont(QFont("Arial", 7))
        btn_clear_log.setStyleSheet("background:#333;color:#999;border:1px solid #555;padding:2px 8px;border-radius:2px;")
        btn_clear_log.clicked.connect(self._clear_log)
        log_top.addWidget(btn_clear_log)
        log_layout.addLayout(log_top)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 8))
        self.log_text.setStyleSheet(tab_style)
        log_layout.addWidget(self.log_text)
        self.tabs.addTab(tab_log, "Log")

        right_splitter.addWidget(self.tabs)
        # Give forecast ~65% of vertical space, tabs ~35%
        right_splitter.setSizes([400, 170])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)
        right_layout.addWidget(right_splitter, stretch=1)
        splitter.addWidget(right_widget)
        splitter.setSizes([820, 540])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        left_widget.setMinimumWidth(400)
        right_widget.setMinimumWidth(280)
        outer.addWidget(splitter, stretch=1)

        # --- Bottom: Inline Value Scanner ---
        scanner_box = QGroupBox("Top Options (Value Scanner)")
        scanner_box.setFont(QFont("Arial", 9, QFont.Bold))
        scanner_box.setStyleSheet("QGroupBox{color:#aaa;border:1px solid #333;border-radius:3px;margin-top:4px;padding-top:10px;}")
        scanner_box.setMaximumHeight(240)
        scanner_lay = QVBoxLayout(scanner_box)
        scanner_lay.setContentsMargins(4, 2, 4, 2)
        scanner_lay.setSpacing(1)

        self.scanner_table = QTableWidget()
        self.scanner_table.setColumnCount(9)
        self.scanner_table.setHorizontalHeaderLabels(
            ["Type", "Strike", "DTE", "Bid", "Ask", "Mid", "Vol", "Score", "Signals"])
        self.scanner_table.setFont(QFont("Consolas", 9))
        self.scanner_table.setStyleSheet(
            "QTableWidget{background:#1a1a2e;color:#ccc;border:none;gridline-color:#333;}"
            "QHeaderView::section{background:#252540;color:#aaa;border:1px solid #333;padding:2px;font-size:9px;}"
            "QTableWidget::item:selected{background:#333;}")
        self.scanner_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scanner_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scanner_table.verticalHeader().setVisible(False)
        self.scanner_table.horizontalHeader().setStretchLastSection(True)
        self.scanner_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scanner_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Size columns
        hdr = self.scanner_table.horizontalHeader()
        for i in range(8):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.Stretch)
        self.scanner_table.setRowCount(0)
        scanner_lay.addWidget(self.scanner_table)

        self.lbl_scanner_status = QLabel("Scanner: idle")
        self.lbl_scanner_status.setFont(QFont("Arial", 8))
        self.lbl_scanner_status.setStyleSheet("color:#666;")
        scanner_lay.addWidget(self.lbl_scanner_status)

        outer.addWidget(scanner_box)

    # -----------------------------------------------------------------------
    # S/R line controls
    # -----------------------------------------------------------------------
    def _apply_saved_settings(self):
        """Restore persisted settings to UI controls."""
        s = self._settings
        self.combo_min_grade.setCurrentText(s.get("min_grade", "C"))
        self.spin_vol.setValue(s.get("vol_threshold", 150))
        self.cb_calls.setChecked(s.get("show_calls", True))
        self.cb_puts.setChecked(s.get("show_puts", True))
        self.cb_auto_sr.setChecked(s.get("show_auto_sr", True))
        self.cb_bb_bands.setChecked(s.get("show_bb", True))
        self.cb_vwap_bands.setChecked(s.get("show_vwap_bands", False))
        if s.get("chart_tf", "1m") != "1m":
            self._set_chart_timeframe(s["chart_tf"])

    def _save_current_settings(self):
        """Persist current UI settings to disk."""
        save_settings({
            "min_grade": self.combo_min_grade.currentText(),
            "vol_threshold": self.spin_vol.value(),
            "show_calls": self.cb_calls.isChecked(),
            "show_puts": self.cb_puts.isChecked(),
            "chart_tf": getattr(self, '_chart_tf', '1m'),
            "show_auto_sr": self.cb_auto_sr.isChecked(),
            "show_bb": self.cb_bb_bands.isChecked(),
            "show_vwap_bands": self.cb_vwap_bands.isChecked(),
            "alert_sound": True,
        })

    def closeEvent(self, event):
        """Save settings on exit."""
        self._save_current_settings()
        super().closeEvent(event)

    def _tick_clock(self):
        """Update live clock every second."""
        et = now_et()
        self.lbl_update.setText(et.strftime("%I:%M:%S %p ET"))
        if market_is_open(et):
            self.lbl_update.setStyleSheet(f"color:{THEME['green']};")
        else:
            self.lbl_update.setStyleSheet(f"color:{THEME['text_dim']};")

    def _export_history_csv(self):
        """Export prediction history to CSV file."""
        if not self.prediction_history:
            self._log("[!] No history to export")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export History", f"spyderscalp_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "CSV Files (*.csv)")
        if not filepath:
            return
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "time", "signal", "grade", "score", "entry_price",
                    "exit_price", "pnl_pct", "result", "hold_minutes"])
                writer.writeheader()
                for pred in self.prediction_history:
                    writer.writerow({
                        "time": pred["time"],
                        "signal": pred["signal"],
                        "grade": pred["grade"],
                        "score": pred["score"],
                        "entry_price": pred["entry_price"],
                        "exit_price": pred.get("exit_price", ""),
                        "pnl_pct": pred.get("pnl_pct", ""),
                        "result": pred.get("result", "PENDING"),
                        "hold_minutes": pred.get("hold_minutes", ""),
                    })
            self._log(f"[OK] Exported {len(self.prediction_history)} records to {filepath}")
        except Exception as e:
            self._log(f"[!] Export failed: {e}")
    def _toggle_auto_sr(self, state):
        self.candle_chart.show_auto_sr = bool(state)

    def _add_manual_line(self):
        price = self.spin_manual_price.value()
        if price > 0:
            self.candle_chart.add_manual_line(price)
            self.spin_manual_price.setValue(0)

    def _clear_manual_lines(self):
        self.candle_chart.clear_manual_lines()

    def _set_chart_timeframe(self, tf):
        """Switch chart timeframe and refresh."""
        self._chart_tf = tf
        # Update button styles
        for label, btn in self._chart_tf_buttons.items():
            if label == tf:
                btn.setStyleSheet("background:#444;color:#bbb;border:1px solid #666;border-radius:2px;padding:1px 3px;")
            else:
                btn.setStyleSheet("background:#2a2a2a;color:#666;border:1px solid #444;border-radius:2px;padding:1px 3px;")
        # Fetch and display chart data for selected timeframe
        self._refresh_chart_for_timeframe()

    def _refresh_chart_for_timeframe(self):
        """Fetch data for the selected chart timeframe and update chart."""
        tf = self._chart_tf
        # Map timeframe label to yfinance params
        tf_map = {
            "1m":  {"period": "1d",  "interval": "1m",  "bars": 15},
            "5m":  {"period": "5d",  "interval": "5m",  "bars": 30},
            "10m": {"period": "5d",  "interval": "15m", "bars": 30},  # yf has no 10m; use 15m as closest
            "15m": {"period": "5d",  "interval": "15m", "bars": 30},
            "1h":  {"period": "1mo", "interval": "1h",  "bars": 40},
            "1d":  {"period": "6mo", "interval": "1d",  "bars": 90},
        }
        params = tf_map.get(tf, tf_map["1m"])
        try:
            df = yf_download_safe("SPY", period=params["period"], interval=params["interval"])
            if df is None:
                self._log(f"[!] No data for {tf} timeframe")
                return
            if len(df) < 2:
                return

            df = self._calc_vwap(df)
            # Compute indicators for chart display
            close = df["Close"]
            df["_rsi"] = compute_rsi(close, 14)
            df["_ema9"] = compute_ema(close, 9)
            df["_ema21"] = compute_ema(close, 21)
            macd_l, macd_s, macd_h = compute_macd(close)
            df["_macd"] = macd_l
            df["_macd_signal"] = macd_s
            df["_macd_hist"] = macd_h

            # Store for reference by chart
            self._chart_df = df
            self.candle_chart.update_chart(df, n_bars=params["bars"])
            self.candle_chart.repaint()

            # Update x-axis date format based on timeframe
            actual_tf = tf if tf != "10m" else "15m"
            self._log(f"[>] Chart: {len(df)} bars ({actual_tf})")
        except Exception as e:
            self._log(f"[!] Chart fetch failed for {tf}: {e}")

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------
    def _update_status(self, state):
        """Update the monitoring status indicator."""
        states = {
            "monitoring": ("*", "#26a69a", "Monitoring (every 60s)"),
            "scanning": ("*", "#ffab00", "Scanning..."),
            "idle": ("*", "#888", "Idle - press Start"),
            "stopped": ("*", "#ef5350", "Stopped"),
        }
        dot, color, text = states.get(state, states["idle"])
        self.lbl_status_dot.setText(dot)
        self.lbl_status_dot.setStyleSheet(f"color:{color};")
        self.lbl_status_text.setText(text)
        self.lbl_status_text.setStyleSheet(f"color:{color};")

    def start_monitoring(self):
        et = now_et()
        if not market_is_open(et):
            self._log(f"[!] Market is closed ({et.strftime('%I:%M %p ET %a')}). Will scan once to show latest data.")
            self._update_status("stopped")
            self.lbl_signal.setText("Market Closed")
            self.lbl_signal.setStyleSheet("color:#888;")
            # Still do one scan to populate the chart, but don't start the timer
            try:
                df, data_mode = self._fetch_data()
                if df is not None and not df.empty:
                    df = self._calc_vwap(df)
                    self.candle_chart.update_chart(df, n_bars=15)
                    self.candle_chart.repaint()
                    current_price = float(df["Close"].iloc[-1])
                    self.lbl_price.setText(f"SPY ${current_price:.2f}")
                    self.lbl_update.setText(et.strftime("%I:%M:%S %p ET"))
                    self._log(f"[>] Loaded last session data for preview")
            except Exception:
                pass
            return

        self.is_monitoring = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.monitor_timer.start(60000)
        self._update_status("monitoring")
        self._log("[OK] Monitoring started - scanning every 60 seconds")
        self.check_signals()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.monitor_timer.stop()
        self._update_status("stopped")
        self._log("[STOP] Monitoring stopped")

    def manual_scan(self):
        self._log("[SCAN] Manual scan initiated...")
        self._update_status("scanning")
        self.check_signals()
        if self.is_monitoring:
            self._update_status("monitoring")
        else:
            self._update_status("idle")

    # -----------------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------------
    def _calc_vwap(self, df):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        # Bollinger Bands
        close = df["Close"]
        bb_u, bb_m, bb_l, bb_bw, bb_pb = compute_bollinger_bands(close)
        df["_bb_upper"] = bb_u
        df["_bb_middle"] = bb_m
        df["_bb_lower"] = bb_l
        df["_bb_bandwidth"] = bb_bw
        df["_bb_pct_b"] = bb_pb
        # VWAP deviation bands
        _, vu1, vl1, vu2, vl2 = compute_vwap_bands(df)
        df["_vwap_u1"] = vu1
        df["_vwap_l1"] = vl1
        df["_vwap_u2"] = vu2
        df["_vwap_l2"] = vl2
        return df

    def _min_grade_score(self):
        mapping = {"F": 0, "D": 35, "C": 50, "B": 65, "A": 75, "A+": 85}
        return mapping.get(self.combo_min_grade.currentText(), 0)

    def _fetch_data(self):
        strategies = [
            {"period": "1d", "interval": "1m", "label": "live"},
            {"period": "5d", "interval": "5m", "label": "delayed"},
            {"period": "5d", "interval": "1d", "label": "daily"},
        ]
        for strat in strategies:
            try:
                df = yf_download_safe("SPY", period=strat["period"], interval=strat["interval"])
                if df is None or len(df) < (15 if strat["label"] != "daily" else 2):
                    continue
                self._log(f"[>] Data: {len(df)} bars ({strat['label']}, {strat['interval']})")
                return df, strat["label"]
            except Exception as ex:
                self._log(f"[!] Fetch {strat['label']} failed: {type(ex).__name__}: {ex}")
                continue
        return None, None

    def _fetch_mtf_data(self):
        """Fetch 5m and 15m data for multi-timeframe confirmation. Cached 2 min."""
        now = datetime.now()
        if hasattr(self, '_mtf_cache') and self._mtf_cache:
            cache_time, cached_5m, cached_15m = self._mtf_cache
            if (now - cache_time).seconds < 120:
                return cached_5m, cached_15m

        df_5m = yf_download_safe("SPY", period="5d", interval="5m")
        if df_5m is not None and len(df_5m) < 10:
            df_5m = None
        df_15m = yf_download_safe("SPY", period="5d", interval="15m")
        if df_15m is not None and len(df_15m) < 10:
            df_15m = None

        self._mtf_cache = (now, df_5m, df_15m)
        return df_5m, df_15m

    def check_signals(self):
        # Only scan during market hours (9:30 AM - 4:00 PM ET, weekdays)
        et = now_et()
        if not market_is_open(et):
            if self.is_monitoring:
                self.stop_monitoring()
                self._log(f"[STOP] Market closed ({et.strftime('%I:%M %p ET')}) - monitoring paused until next open")
                self._update_status("stopped")
                self.lbl_signal.setText("Market Closed")
                self.lbl_signal.setStyleSheet("color:#888;")
                self.lbl_arrow.setText("")
                self.lbl_arrow.setStyleSheet("")
            return

        try:
            df, data_mode = self._fetch_data()
            if df is None or df.empty:
                self._log("[!] Could not fetch data - check your internet connection")
                return
            if data_mode != "live":
                self._log(f"[!] Market may be closed - using {data_mode} data for preview")
            if len(df) < 15:
                self._log("[!] Insufficient data bars")
                return

            df = self._calc_vwap(df)

            # Pre-compute indicators ONCE and cache on DataFrame
            close = df["Close"]
            if "_rsi" not in df.columns:
                df["_rsi"] = compute_rsi(close, 14)
                df["_ema9"] = compute_ema(close, 9)
                df["_ema21"] = compute_ema(close, 21)
                macd_l, macd_s, macd_h = compute_macd(close)
                df["_macd"] = macd_l
                df["_macd_signal"] = macd_s
                df["_macd_hist"] = macd_h

            self.candle_chart.update_chart(df, n_bars=15)
            # Only refresh for non-1m timeframe if selected (don't redraw twice)
            if hasattr(self, '_chart_tf') and self._chart_tf != "1m":
                self._refresh_chart_for_timeframe()
            else:
                self.candle_chart.repaint()

            current_price = float(df["Close"].iloc[-1])
            current_vwap = float(df["VWAP"].iloc[-1])

            # Volume: use the last COMPLETED bar (-2), not the still-forming bar (-1)
            if len(df) > 22:
                last_vol = float(df["Volume"].iloc[-2])
                avg_vol = float(df["Volume"].iloc[-22:-2].mean())
            elif len(df) > 3:
                last_vol = float(df["Volume"].iloc[-2])
                avg_vol = float(df["Volume"].iloc[:-2].mean())
            else:
                last_vol = float(df["Volume"].iloc[-1])
                avg_vol = float(df["Volume"].mean())
            vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0
            rsi_val = float(df["_rsi"].iloc[-1]) if not pd.isna(df["_rsi"].iloc[-1]) else 0.0
            ema9 = float(df["_ema9"].iloc[-1])
            ema21 = float(df["_ema21"].iloc[-1])

            self.lbl_price.setText(f"SPY ${current_price:.2f}")
            self.lbl_vwap.setText(f"VWAP ${current_vwap:.2f}")
            self.lbl_volume.setText(f"Vol {vol_ratio:.2f}x")
            self.lbl_rsi.setText(f"RSI {rsi_val:.1f}")
            self.lbl_ema.setText(f"EMA9 ${ema9:.2f} / 21 ${ema21:.2f}")
            self.lbl_update.setText(now_et().strftime("%I:%M:%S %p ET"))

            vol_threshold = self.spin_vol.value() / 100.0
            signal_type = None
            gated = False  # True if conditions are close but didn't pass all gates

            # Determine signal direction - require multiple confirmations
            price_above_vwap = current_price > current_vwap
            price_below_vwap = current_price < current_vwap
            vol_ok = vol_ratio > vol_threshold

            # Require price to be meaningfully away from VWAP (not just 1 cent)
            vwap_distance_pct = abs(current_price - current_vwap) / current_vwap * 100
            vwap_clear = vwap_distance_pct > 0.02  # at least 0.02% from VWAP

            # EMA confirmation: at least EMA9 should agree with direction
            ema9_confirms_call = ema9 > ema21
            ema9_confirms_put = ema9 < ema21

            # MACD histogram direction
            macd_hist = float(df["_macd_hist"].iloc[-1]) if "_macd_hist" in df.columns else 0
            macd_confirms_call = macd_hist > 0
            macd_confirms_put = macd_hist < 0

            if self.cb_calls.isChecked() and price_above_vwap and vol_ok and vwap_clear:
                # Require at least 1 of: EMA confirmation or MACD confirmation
                if ema9_confirms_call or macd_confirms_call:
                    signal_type = "CALL"
            if self.cb_puts.isChecked() and price_below_vwap and vol_ok and vwap_clear:
                if ema9_confirms_put or macd_confirms_put:
                    signal_type = "PUT"

            # If no full signal, still determine a direction for display
            if signal_type is None:
                gate_reasons = []
                # Figure out which direction price leans
                if price_above_vwap and self.cb_calls.isChecked():
                    signal_type = "CALL"
                    gated = True
                    if not vol_ok:
                        gate_reasons.append(f"vol {vol_ratio:.2f}x < {vol_threshold:.2f}x threshold")
                    if not vwap_clear:
                        gate_reasons.append(f"too close to VWAP ({vwap_distance_pct:.3f}%)")
                    if not ema9_confirms_call and not macd_confirms_call:
                        gate_reasons.append("no EMA/MACD confirmation")
                elif price_below_vwap and self.cb_puts.isChecked():
                    signal_type = "PUT"
                    gated = True
                    if not vol_ok:
                        gate_reasons.append(f"vol {vol_ratio:.2f}x < {vol_threshold:.2f}x threshold")
                    if not vwap_clear:
                        gate_reasons.append(f"too close to VWAP ({vwap_distance_pct:.3f}%)")
                    if not ema9_confirms_put and not macd_confirms_put:
                        gate_reasons.append("no EMA/MACD confirmation")
                else:
                    # Price at VWAP or direction unchecked — truly no signal
                    reasons = []
                    if not price_above_vwap and not price_below_vwap:
                        reasons.append(f"price = VWAP (${current_price:.2f})")
                    elif price_above_vwap and not self.cb_calls.isChecked():
                        reasons.append("price > VWAP but Calls unchecked")
                    elif price_below_vwap and not self.cb_puts.isChecked():
                        reasons.append("price < VWAP but Puts unchecked")
                    direction = "above" if price_above_vwap else "below"
                    diff = abs(current_price - current_vwap)
                    self._log(f"    No signal: ${current_price:.2f} ({direction} VWAP by ${diff:.2f}) | vol {vol_ratio:.2f}x | {', '.join(reasons)}")

                if gated:
                    direction = "above" if price_above_vwap else "below"
                    diff = abs(current_price - current_vwap)
                    self._log(f"    Weak {signal_type}: ${current_price:.2f} ({direction} VWAP by ${diff:.2f}) | vol {vol_ratio:.2f}x | gates: {', '.join(gate_reasons)}")

            if signal_type:
                event_ctx = get_event_context(now_et())
                result = evaluate_signal(df, signal_type, event_ctx)

                # Multi-timeframe confirmation layer
                df_5m, df_15m = self._fetch_mtf_data()
                mtf_mult, mtf_reasons, mtf_detail = compute_mtf_confirmation(
                    signal_type, df_5m, df_15m)

                # Apply MTF multiplier to score
                original_score = result["score"]
                adjusted_score = round(min(original_score * mtf_mult, 100), 1)
                result["score"] = adjusted_score
                result["grade"] = score_to_grade(adjusted_score)
                result["mtf_multiplier"] = mtf_mult
                result["mtf_reasons"] = mtf_reasons
                result["mtf_detail"] = mtf_detail
                result["original_score"] = original_score

                if mtf_mult != 1.0:
                    self._log(f"    MTF: {original_score:.1f} x {mtf_mult:.2f} = {adjusted_score:.1f} ({result['grade']})")
                    for r in mtf_reasons:
                        self._log(f"      {r}")

                # Always show the evaluation in the UI
                self._update_gauge(signal_type, result)
                self._update_events(event_ctx)

                # DTE recommendation
                dte_rec = recommend_dte(result["score"], result["grade"],
                                        result["vol_ratio"], result["rsi"], event_ctx,
                                        mtf_multiplier=mtf_mult)
                self._update_dte_display(dte_rec)
                rec_dte = dte_rec["recommended_dte"]

                hold = estimate_hold_time(signal_type, result["score"], rec_dte,
                                          result["vol_ratio"], result["rsi"], event_ctx)
                self._update_hold_display(hold)

                # If gated (didn't pass all confirmation gates), show info but don't alert
                if gated:
                    self.lbl_signal.setText(f"{signal_type} {result['grade']} (watching)")
                    self.lbl_signal.setStyleSheet(f"color:#888;")
                    # Add gate reasons to breakdown
                    gate_note = "\n\n  [!] NOT CONFIRMED:\n" + "\n".join(f"    - {r}" for r in gate_reasons)
                    current_bd = self.lbl_breakdown.toPlainText()
                    self.lbl_breakdown.setPlainText(current_bd + gate_note)
                    # Still track prediction but don't alert
                    self._record_prediction(signal_type, current_price, result["score"],
                                            result["grade"], hold.get("hold_minutes", 15))
                    return

                if result["score"] < self._min_grade_score():
                    self.lbl_signal.setText(f"{signal_type} signal (below min grade - skipped)")
                    return

                # Cooldown check
                if self.last_signal_time and (datetime.now() - self.last_signal_time).seconds < self.signal_cooldown:
                    # Direction flip within cooldown = whipsaw, require A grade to override
                    if self.last_signal_type and self.last_signal_type != signal_type:
                        if result["grade"] not in ("A+", "A"):
                            self.lbl_signal.setText(f"{signal_type} {result['grade']} (whipsaw - need A+ to flip)")
                            self._log(f"    [!] Whipsaw detected: {self.last_signal_type}->{signal_type} within cooldown, need A+ to override")
                            return
                        else:
                            self._log(f"    [!] Whipsaw override: strong {result['grade']} signal, allowing direction flip")
                    else:
                        self.lbl_signal.setText(f"{signal_type} {result['grade']} (cooldown)")
                        return

                self.last_signal_time = datetime.now()
                self.last_signal_type = signal_type
                self._trigger_alert(signal_type, current_price, current_vwap, vol_ratio, result, event_ctx, dte_rec)
            else:
                # Truly no direction — price at VWAP or direction unchecked
                self.lbl_signal.setText("Waiting for signal")
                self.lbl_signal.setStyleSheet("color:#888;")
                event_ctx = get_event_context(now_et())
                self._update_events(event_ctx)

        except Exception as e:
            import traceback
            self._log(f"[X] Error: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    # -----------------------------------------------------------------------
    # UI updates & alerting
    # -----------------------------------------------------------------------
    def _update_gauge(self, signal_type, result):
        g = result["grade"]
        s = result["score"]
        color = grade_color(g)
        self.lbl_signal.setText(f"{signal_type} Signal")
        self.lbl_signal.setStyleSheet(f"color:{color};")
        self.lbl_grade.setText(g)
        self.lbl_grade.setStyleSheet(f"color:{color};")
        self.quality_bar.setValue(int(s))

        # Big arrow visual
        if signal_type == "CALL":
            self.lbl_arrow.setText("^")
            self.lbl_arrow.setStyleSheet(f"color:{color}; background:#1a2e1a; border-radius:6px;")
        else:
            self.lbl_arrow.setText("v")
            self.lbl_arrow.setStyleSheet(f"color:{color}; background:#2e1a1a; border-radius:6px;")

        bd_lines = []
        for key, desc in result["breakdown"].items():
            raw = result["raw"][key]
            w = result["weights"][key]
            pts = raw * w
            bar = "#" * int(raw * 10) + "." * (10 - int(raw * 10))
            bd_lines.append(f"  {key:<16s} [{bar}] {pts:5.1f}/{w}  ({desc})")
        # MTF confirmation line
        mtf_mult = result.get("mtf_multiplier", 1.0)
        orig_score = result.get("original_score")
        if mtf_mult != 1.0 and orig_score is not None:
            if mtf_mult > 1.0:
                mtf_icon = "[OK]"
            elif mtf_mult >= 0.85:
                mtf_icon = "[!]"
            else:
                mtf_icon = "[!!]"
            bd_lines.append(f"")
            bd_lines.append(f"  {mtf_icon} MTF: {orig_score:.1f} x {mtf_mult:.2f} = {result['score']:.1f}")
            mtf_results = result.get("mtf_detail", {})
            if mtf_results:
                res = mtf_results.get("results", {})
                for tf, status in res.items():
                    if status != "unavailable":
                        bd_lines.append(f"       {tf}: {status}")
        self.lbl_breakdown.setPlainText("\n".join(bd_lines))

    def _update_events(self, event_ctx):
        today_events = event_ctx.get("events_today", [])
        upcoming = event_ctx.get("upcoming_events", [])
        zone = event_ctx.get("intraday_zone")
        lines = []
        lines.append("-- TODAY --")
        if today_events:
            for ev in today_events:
                mins = ev.get("minutes_until", 0)
                tag = "[HIGH]" if ev["impact"] == "high" else "[MED]"
                if mins is not None and mins > 0:
                    lines.append(f"  {tag} {ev['time']} ET  {ev['name']}  ({mins} min away)")
                elif mins is not None and mins > -15:
                    lines.append(f"  {tag} {ev['time']} ET  {ev['name']}  JUST RELEASED")
                elif mins is not None:
                    lines.append(f"  {tag} {ev['time']} ET  {ev['name']}  (passed)")
                else:
                    lines.append(f"  {tag} {ev['time']} ET  {ev['name']}")
        else:
            lines.append("  [OK] No major events today")
        if zone:
            lines.append(f"  >> {zone}")
        future = [e for e in upcoming if e["days_until"] > 0]
        if future:
            lines.append("")
            lines.append("-- UPCOMING --")
            count = 0
            for ev in future:
                if count >= 12:
                    remaining = len(future) - count
                    lines.append(f"  ... and {remaining} more events")
                    break
                tag = "[HIGH]" if ev["impact"] == "high" else "[MED]"
                try:
                    dt = datetime.strptime(ev["date"], "%Y-%m-%d")
                    day_label = dt.strftime("%a %b %d")
                except Exception:
                    day_label = ev["date"]
                when = "TOMORROW" if ev["days_until"] == 1 else f"in {ev['days_until']}d"
                lines.append(f"  {tag} {day_label}  {ev['time']} ET  {ev['name']}  ({when})")
                count += 1
        self.lbl_events.setPlainText("Economic Calendar:\n" + "\n".join(lines))

    def _update_hold_display(self, hold):
        conf_colors = {"high": "#26a69a", "medium": "#ffab00", "low": "#ef5350"}
        conf = hold["confidence"]
        self.lbl_hold_time.setText(f"Hold: {hold['hold_label']}")
        self.lbl_hold_time.setStyleSheet(f"color: {conf_colors.get(conf, '#aaa')};")
        self.lbl_exit_by.setText(f"Exit by: {hold['exit_by']}")
        self.lbl_hold_confidence.setText(f"Confidence: {conf.upper()}")
        self.lbl_hold_confidence.setStyleSheet(f"color: {conf_colors.get(conf, '#aaa')};")
        self.lbl_hold_reasons.setPlainText("\n".join(f"  * {r}" for r in hold["reasoning"]))

    def _update_dte_display(self, dte_rec):
        rec = dte_rec["recommended_dte"]
        scores = dte_rec["dte_scores"]
        # Color: green for 0DTE (aggressive), yellow for 1, orange for 2
        colors = {0: "#26a69a", 1: "#ffab00", 2: "#e65100"}
        self.lbl_dte_rec.setText(f"Rec: {rec}DTE")
        self.lbl_dte_rec.setStyleSheet(f"color: {colors.get(rec, '#aaa')};")
        # Show score comparison
        parts = [f"{d}DTE:{s}" for d, s in sorted(scores.items())]
        self.lbl_dte_detail.setText(f"{dte_rec['recommendation']}  ({' | '.join(parts)})")

    def _trigger_alert(self, signal_type, price, vwap, vol_ratio, result, event_ctx, dte_rec=None):
        rec_dte = dte_rec["recommended_dte"] if dte_rec else self._get_nearest_dte()
        hold = estimate_hold_time(signal_type, result["score"], rec_dte,
                                  vol_ratio, result["rsi"], event_ctx)
        self._update_hold_display(hold)
        options_rec = self._get_options_rec(signal_type, price)

        try:
            if desktop_notification:
                desktop_notification.notify(
                    title=f"SPYderScalp {signal_type} {result['grade']}",
                    message=f"Score {result['score']}/100\nSPY ${price:.2f}  VWAP ${vwap:.2f}\nHold {hold['hold_label']} -> exit {hold['exit_by']}",
                    timeout=10,
                )
        except Exception:
            pass

        self._play_sound()
        explanation = self._build_signal_explanation(signal_type, price, vwap, vol_ratio, result, event_ctx, hold)

        # Record prediction for tracking
        self._record_prediction(signal_type, price, result["score"], result["grade"], hold["hold_minutes"])

        log = (
            f"\n{'=' * 60}\n"
            f"[!] {signal_type} SIGNAL - Grade {result['grade']} ({result['score']}/100)\n"
            f"    {now_et().strftime('%I:%M:%S %p ET')}\n"
            f"{'=' * 60}\n"
            f"  SPY ${price:.2f}   VWAP ${vwap:.2f}   Vol {vol_ratio:.2f}x\n"
            f"  RSI {result['rsi']:.1f}   EMA9 ${result['ema9']:.2f}   EMA21 ${result['ema21']:.2f}\n\n"
        )
        for key, desc in result["breakdown"].items():
            raw = result["raw"][key]
            w = result["weights"][key]
            pts = raw * w
            log += f"    {key:<16s}  {pts:5.1f}/{w:2d}  {desc}\n"
        log += f"\nWHY THIS GRADE:\n{explanation}\n"
        log += f"\n{options_rec}\n"
        log += f"\nHOLD: {hold['hold_label']}  |  Exit by {hold['exit_by']}  |  Confidence: {hold['confidence'].upper()}\n"
        for r in hold["reasoning"]:
            log += f"    * {r}\n"
        if dte_rec:
            log += f"\nDTE RECOMMENDATION: {dte_rec['recommended_dte']}DTE\n"
            log += f"  {dte_rec['recommendation']}\n"
            for r in dte_rec["reasoning"]:
                log += f"    * {r}\n"
        # MTF confirmation details
        mtf_reasons = result.get("mtf_reasons", [])
        mtf_mult = result.get("mtf_multiplier", 1.0)
        orig_score = result.get("original_score")
        if mtf_reasons:
            log += f"\nMULTI-TIMEFRAME CONFIRMATION:\n"
            if orig_score is not None and mtf_mult != 1.0:
                log += f"  Base score: {orig_score:.1f} x {mtf_mult:.2f} = {result['score']:.1f}\n"
            for r in mtf_reasons:
                log += f"    * {r}\n"
        log += f"{'=' * 60}\n"
        self._log(log)

    def _build_signal_explanation(self, signal_type, price, vwap, vol_ratio, result, event_ctx, hold):
        lines = []
        score = result["score"]
        grade = result["grade"]
        raw = result["raw"]
        rsi = result["rsi"]
        direction = "bullish" if signal_type == "CALL" else "bearish"
        above_below = "above" if signal_type == "CALL" else "below"

        if grade in ("A+", "A"):
            lines.append(f"  Strong {direction} setup. Multiple indicators aligned,")
            lines.append(f"  giving high confidence in the trade.")
        elif grade == "B":
            lines.append(f"  Solid {direction} setup with most indicators confirming,")
            lines.append(f"  but one or two factors are holding back a top grade.")
        elif grade == "C":
            lines.append(f"  Marginal {direction} signal. Basic conditions met but several")
            lines.append(f"  indicators are not strongly confirming. Proceed with caution.")
        elif grade == "D":
            lines.append(f"  Weak {direction} signal. Most indicators not confirming.")
            lines.append(f"  Consider waiting for a better setup.")
        else:
            lines.append(f"  Very weak signal - looks like noise rather than a real move.")
        lines.append("")

        diff = abs(price - vwap)
        vwap_raw = raw.get("vwap", 0)
        if vwap_raw > 0.7:
            lines.append(f"  [OK] VWAP: Price ${diff:.2f} {above_below} VWAP - clear break,")
            lines.append(f"       {direction} institutional flow confirmed.")
        elif vwap_raw > 0.3:
            lines.append(f"  [~] VWAP: Price ${diff:.2f} {above_below} VWAP - break is there")
            lines.append(f"      but not decisive. Could be a false breakout.")
        else:
            lines.append(f"  [X] VWAP: Price barely {above_below} VWAP (${diff:.2f}) -")
            lines.append(f"      right at the line, could easily reverse.")

        vol_raw = raw.get("volume", 0)
        if vol_raw > 0.7:
            lines.append(f"  [OK] Volume: {vol_ratio:.1f}x avg - strong participation,")
            lines.append(f"       real buying/selling pressure behind this move.")
        elif vol_raw > 0.3:
            lines.append(f"  [~] Volume: {vol_ratio:.1f}x avg - decent but not exceptional.")
        else:
            lines.append(f"  [X] Volume: {vol_ratio:.1f}x avg - low conviction,")
            lines.append(f"      moves on thin volume are prone to quick reversals.")

        if signal_type == "CALL":
            if rsi > 70:
                lines.append(f"  [~] RSI: {rsi:.0f} - overbought. Move may be running out of steam.")
            elif 55 <= rsi <= 68:
                lines.append(f"  [OK] RSI: {rsi:.0f} - healthy bullish momentum, not overbought.")
            elif rsi < 40:
                lines.append(f"  [X] RSI: {rsi:.0f} - momentum is actually bearish.")
            else:
                lines.append(f"  [~] RSI: {rsi:.0f} - neutral, no strong confirmation.")
        else:
            if rsi < 30:
                lines.append(f"  [~] RSI: {rsi:.0f} - oversold. A bounce could happen any moment.")
            elif 32 <= rsi <= 45:
                lines.append(f"  [OK] RSI: {rsi:.0f} - healthy bearish momentum, not oversold.")
            elif rsi > 55:
                lines.append(f"  [X] RSI: {rsi:.0f} - momentum is actually bullish.")
            else:
                lines.append(f"  [~] RSI: {rsi:.0f} - neutral, no strong confirmation.")

        ema_raw = raw.get("ema_trend", 0)
        if ema_raw > 0.7:
            lines.append(f"  [OK] Trend: EMA 9/21 aligned {direction}ly.")
        elif ema_raw > 0.3:
            lines.append(f"  [~] Trend: EMAs weakly aligned. Could be transitional.")
        else:
            lines.append(f"  [X] Trend: EMAs not aligned with this {signal_type}.")
            lines.append(f"      Trading against the short-term trend.")

        # MACD confirmation
        macd_raw = raw.get("macd_confirm", 0)
        if macd_raw > 0.7:
            lines.append(f"  [OK] MACD: Histogram confirms {direction} momentum.")
        elif macd_raw > 0.3:
            lines.append(f"  [~] MACD: Weakly confirming - histogram fading.")
        elif macd_raw > 0:
            lines.append(f"  [X] MACD: Not confirming this {signal_type}.")
            lines.append(f"      Momentum divergence - this move may not hold.")
        else:
            lines.append(f"  [X] MACD: Pointing opposite direction - significant headwind.")

        # Trend alignment
        align_raw = raw.get("trend_alignment", 0)
        if align_raw >= 0.9:
            lines.append(f"  [OK] All indicators aligned - clean setup.")
        elif align_raw >= 0.5:
            lines.append(f"  [~] Mixed alignment - some indicators disagree.")
        else:
            lines.append(f"  [X] Poor alignment - fighting the trend on multiple fronts.")

        # Time of day warning
        et_now = now_et()
        time_decimal = et_now.hour + et_now.minute / 60.0
        if time_decimal >= 15.0:
            lines.append(f"  [!!] FINAL HOUR: Moves are less reliable, use tight stops.")
        elif time_decimal >= 14.0:
            lines.append(f"  [!] Late afternoon: Follow-through weaker, consider smaller size.")
        elif 11.5 <= time_decimal < 13.0:
            lines.append(f"  [~] Lunch hour: Choppier conditions, expect more noise.")

        evt_raw = raw.get("event_timing", 1.0)
        nearest = event_ctx.get("nearest_event") if event_ctx else None
        if evt_raw < 0.5 and nearest:
            mins = event_ctx.get("minutes_to_nearest", 0)
            lines.append(f"  [!!] EVENT RISK: {nearest['name']} is {mins} min away!")
            lines.append(f"       Vol spike will likely overwhelm technicals. Score penalized.")
        elif evt_raw < 0.8 and nearest:
            mins = event_ctx.get("minutes_to_nearest", 0)
            lines.append(f"  [!] Event: {nearest['name']} in ~{mins} min -")
            lines.append(f"      use smaller size and plan to exit before release.")
        else:
            lines.append(f"  [OK] No imminent economic events - clear window to trade.")

        # Multi-timeframe confirmation
        mtf_mult = result.get("mtf_multiplier", 1.0)
        mtf_detail = result.get("mtf_detail")
        if mtf_detail and mtf_detail is not None:
            results = mtf_detail.get("results", {})
            lines.append("")
            if mtf_mult >= 1.1:
                lines.append(f"  [OK] MULTI-TIMEFRAME: 5m+15m both confirm this {signal_type} ({mtf_mult:.0%} boost)")
                lines.append(f"       Higher timeframes aligned - high-conviction setup.")
            elif mtf_mult >= 1.0:
                conf_tfs = [k for k, v in results.items() if v == "confirming"]
                if conf_tfs:
                    lines.append(f"  [OK] MTF: {', '.join(conf_tfs)} confirms (mild boost)")
                else:
                    lines.append(f"  [~] MTF: Higher timeframes are neutral - proceed normally.")
            elif mtf_mult >= 0.85:
                conf_tfs = [k for k, v in results.items() if v == "conflicting"]
                lines.append(f"  [!] MTF WARNING: {', '.join(conf_tfs)} conflicts with this {signal_type}")
                lines.append(f"      Score reduced by {(1-mtf_mult)*100:.0f}%. Consider smaller size.")
            else:
                lines.append(f"  [!!] MTF CONFLICT: Both 5m and 15m oppose this {signal_type}!")
                lines.append(f"       Score reduced by {(1-mtf_mult)*100:.0f}%. High risk of failure.")

        return "\n".join(lines)

    def _get_cached_expirations(self):
        """Cache options expirations for 5 minutes to avoid repeated API calls."""
        now = datetime.now()
        if hasattr(self, '_exp_cache') and self._exp_cache:
            cache_time, exps = self._exp_cache
            if (now - cache_time).seconds < 300:
                return exps
        try:
            spy = yf.Ticker("SPY")
            exps = spy.options
            self._exp_cache = (now, exps)
            return exps
        except Exception:
            return []

    def _get_nearest_dte(self):
        exps = self._get_cached_expirations()
        today = datetime.now().date()
        for exp in exps[:5]:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 0 <= dte <= 2:
                    return dte
            except Exception:
                pass
        return 1

    def _get_options_rec(self, signal_type, price):
        try:
            expirations = self._get_cached_expirations()
            if not expirations:
                return "[!] No options data"
            today = datetime.now().date()
            recs = []
            for exp in expirations[:5]:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 0 <= dte <= 2:
                    if signal_type == "CALL":
                        strikes = [round(price), round(price) + 1, round(price) + 2]
                    else:
                        strikes = [round(price), round(price) - 1, round(price) - 2]
                    recs.append(f"  {exp} ({dte} DTE): strikes ${strikes[0]}, ${strikes[1]}, ${strikes[2]}")
            return "\n".join(recs) if recs else "[!] No 0-2 DTE expirations found"
        except Exception as e:
            return f"[!] Options error: {e}"

    def _play_sound(self):
        try:
            if self.platform == "Darwin":
                os.system("afplay /System/Library/Sounds/Glass.aiff &")
            elif self.platform == "Windows" and winsound:
                winsound.Beep(1000, 500)
            else:
                print("\a")
        except Exception:
            pass

    def _log(self, msg):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        # Auto-trim log to last 500 lines
        doc = self.log_text.document()
        if doc.blockCount() > 500:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            for _ in range(doc.blockCount() - 500):
                cursor.movePosition(cursor.Down, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # remove trailing newline

    def _clear_log(self):
        self.log_text.clear()
        self._log("[OK] Log cleared")

    def _clear_history(self):
        self.prediction_history.clear()
        _save_prediction_history(self.prediction_history)
        self.history_text.setPlainText("History cleared.")
        self._update_track_summary()
        self._log("[OK] Prediction history cleared")

    def _cleanup_memory(self):
        """Free memory from old data. Called periodically or manually."""
        import gc
        # Trim prediction history to last 100
        if len(self.prediction_history) > 100:
            # Keep only resolved + last 100
            self.prediction_history = self.prediction_history[-100:]
            self._update_history_display()
            self._log("[OK] Trimmed prediction history to 100 entries")
        # Close scanner window if hidden
        if hasattr(self, '_scanner_win') and self._scanner_win and not self._scanner_win.isVisible():
            self._scanner_win.close()
            self._scanner_win = None
        gc.collect()

    def _open_scanner(self):
        """Refresh the inline value scanner panel."""
        self._run_inline_scanner()

    def _run_inline_scanner(self):
        """Scan options and populate the bottom panel."""
        self.lbl_scanner_status.setText("Scanner: scanning...")
        self.lbl_scanner_status.setStyleSheet("color:#ffab00;")
        QApplication.processEvents()
        try:
            # Get current SPY price
            df = yf_download_safe("SPY", period="1d", interval="1m")
            if df is None or df.empty:
                self.lbl_scanner_status.setText("Scanner: no price data")
                self.lbl_scanner_status.setStyleSheet("color:#ef5350;")
                return
            spy_price = float(df["Close"].iloc[-1])

            results = scan_options_value(spy_price, expirations_to_scan=3)
            if not results:
                self.scanner_table.setRowCount(0)
                self.lbl_scanner_status.setText("Scanner: no opportunities found")
                self.lbl_scanner_status.setStyleSheet("color:#888;")
                return

            # Populate table with all results (top 5 visible without scrolling)
            self.scanner_table.setRowCount(len(results))
            for i, opp in enumerate(results):
                type_item = QTableWidgetItem(opp["type"])
                type_color = QColor("#5eb8a2") if opp["type"] == "CALL" else QColor("#cf7b78")
                type_item.setForeground(type_color)

                strike_item = QTableWidgetItem(f"${opp['strike']:.0f}")
                dte_item = QTableWidgetItem(f"{opp['dte']}d")
                bid_item = QTableWidgetItem(f"${opp['bid']:.2f}")
                ask_item = QTableWidgetItem(f"${opp['ask']:.2f}")
                mid_item = QTableWidgetItem(f"${opp['mid']:.2f}")
                vol_item = QTableWidgetItem(f"{opp['volume']}")
                score_item = QTableWidgetItem(f"{opp['score']:.0f}")

                # Color score
                sc = opp["score"]
                if sc >= 30:
                    score_item.setForeground(QColor("#5eb8a2"))
                elif sc >= 20:
                    score_item.setForeground(QColor("#ffab00"))
                else:
                    score_item.setForeground(QColor("#888"))

                signals_text = ", ".join(opp.get("signals", []))
                signals_item = QTableWidgetItem(signals_text)
                signals_item.setForeground(QColor("#aaa"))

                self.scanner_table.setItem(i, 0, type_item)
                self.scanner_table.setItem(i, 1, strike_item)
                self.scanner_table.setItem(i, 2, dte_item)
                self.scanner_table.setItem(i, 3, bid_item)
                self.scanner_table.setItem(i, 4, ask_item)
                self.scanner_table.setItem(i, 5, mid_item)
                self.scanner_table.setItem(i, 6, vol_item)
                self.scanner_table.setItem(i, 7, score_item)
                self.scanner_table.setItem(i, 8, signals_item)

            et = now_et()
            self.lbl_scanner_status.setText(
                f"Scanner: {len(results)} opportunities | top score: {results[0]['score']:.0f} | {et.strftime('%I:%M %p ET')}")
            self.lbl_scanner_status.setStyleSheet("color:#5eb8a2;")

        except Exception as e:
            self.lbl_scanner_status.setText(f"Scanner: error - {e}")
            self.lbl_scanner_status.setStyleSheet("color:#ef5350;")

    # -------------------------------------------------------------------
    # Prediction tracking
    # -------------------------------------------------------------------
    def _record_prediction(self, signal_type, price, score, grade, hold_minutes):
        """Record a signal as a prediction to track."""
        et = now_et()

        # Dedup: don't record if same direction prediction was recorded < 5 min ago and still pending
        if self.prediction_history:
            last = self.prediction_history[-1]
            if (last["signal"] == signal_type
                    and last["result"] in (None, "PENDING")
                    and isinstance(last.get("timestamp"), datetime)):
                age_seconds = (et - last["timestamp"]).total_seconds()
                if age_seconds < 300:  # 5 min dedup window
                    return last  # still tracking the previous one

        pred = {
            "time": et.strftime("%I:%M %p"),
            "timestamp": et,
            "signal": signal_type,
            "entry_price": price,
            "score": score,
            "grade": grade,
            "hold_minutes": hold_minutes,
            "check_time": et + timedelta(minutes=hold_minutes),
            "exit_price": None,
            "result": None,
            "pnl_pct": None,
        }
        self.prediction_history.append(pred)
        _save_prediction_history(self.prediction_history)
        self._update_history_display()
        return pred

    def _check_prediction_outcomes(self):
        """Called by timer - checks if any pending predictions have reached their exit time."""
        # Always refresh display so countdowns update
        has_pending = any(p["result"] in (None, "PENDING") for p in self.prediction_history)
        if has_pending:
            self._update_history_display()

        if not market_is_open():
            return  # Don't fetch when market is closed

        et = now_et()
        # Find all predictions that need checking
        pending = [p for p in self.prediction_history
                   if (p["result"] is None or p["result"] == "PENDING")
                   and isinstance(p.get("check_time"), datetime)
                   and et >= p["check_time"]]
        if not pending:
            return

        # Single fetch for all pending predictions
        try:
            df = yf_download_safe("SPY", period="1d", interval="1m")
            if df is None:
                return
            exit_price = float(df["Close"].iloc[-1])
        except Exception:
            return

        for pred in pending:
            pred["exit_price"] = exit_price
            entry = pred["entry_price"]
            if pred["signal"] == "CALL":
                pnl = (exit_price - entry) / entry * 100
            else:
                pnl = (entry - exit_price) / entry * 100
            pred["pnl_pct"] = round(pnl, 3)

            # Thresholds scale with hold time - longer holds need bigger moves to win
            hold_min = pred.get("hold_minutes", 15)
            if hold_min <= 10:
                win_threshold = 0.02   # quick scalp
                loss_threshold = -0.02
            elif hold_min <= 20:
                win_threshold = 0.03
                loss_threshold = -0.03
            else:
                win_threshold = 0.05   # longer holds should produce more
                loss_threshold = -0.04

            if pnl > win_threshold:
                pred["result"] = "WIN"
            elif pnl < loss_threshold:
                pred["result"] = "LOSS"
            else:
                pred["result"] = "FLAT"

        # Mark remaining not-yet-due as PENDING
        for pred in self.prediction_history:
            if pred["result"] is None:
                pred["result"] = "PENDING"

        self._update_history_display()
        self._update_track_summary()
        _save_prediction_history(self.prediction_history)

    def _update_track_summary(self):
        """Update the win/loss summary in the status bar."""
        resolved = [p for p in self.prediction_history if p["result"] in ("WIN", "LOSS", "FLAT")]
        if not resolved:
            self.lbl_track_summary.setText("")
            return
        wins = sum(1 for p in resolved if p["result"] == "WIN")
        losses = sum(1 for p in resolved if p["result"] == "LOSS")
        flats = sum(1 for p in resolved if p["result"] == "FLAT")
        total = len(resolved)
        rate = wins / total * 100 if total > 0 else 0
        color = "#26a69a" if rate >= 50 else "#ef5350"
        self.lbl_track_summary.setText(f"W:{wins} L:{losses} F:{flats} ({rate:.0f}%)")
        self.lbl_track_summary.setStyleSheet(f"color:{color};")

    def _update_history_display(self):
        """Refresh the History tab with all tracked predictions."""
        et = now_et()
        lines = []
        lines.append(f"{'TIME':<10} {'SIG':<5} {'GRADE':<6} {'ENTRY':>8} {'EXIT':>8} {'P&L':>7} {'RESULT'}")
        lines.append("-" * 65)
        for pred in reversed(self.prediction_history):
            t = pred["time"]
            sig = pred["signal"]
            grade = pred["grade"]
            entry = f"${pred['entry_price']:.2f}"
            ex = f"${pred['exit_price']:.2f}" if pred["exit_price"] else "--"
            pnl = f"{pred['pnl_pct']:+.3f}%" if pred["pnl_pct"] is not None else "--"
            result = pred["result"] or "PENDING"
            if result == "WIN":
                marker = "[OK]"
            elif result == "LOSS":
                marker = "[X]"
            elif result == "FLAT":
                marker = "[~]"
            else:
                marker = "[...]"
                # Add countdown for pending predictions
                check_time = pred.get("check_time")
                if isinstance(check_time, datetime):
                    remaining = (check_time - et).total_seconds()
                    if remaining > 0:
                        mins = int(remaining // 60)
                        secs = int(remaining % 60)
                        marker = f"[{mins}:{secs:02d}]"
                    else:
                        marker = "[checking]"
            lines.append(f"{t:<10} {sig:<5} {grade:<6} {entry:>8} {ex:>8} {pnl:>7} {marker} {result}")

        # Summary at bottom
        resolved = [p for p in self.prediction_history if p["result"] in ("WIN", "LOSS", "FLAT")]
        if resolved:
            wins = sum(1 for p in resolved if p["result"] == "WIN")
            losses = sum(1 for p in resolved if p["result"] == "LOSS")
            total = len(resolved)
            avg_pnl = sum(p["pnl_pct"] for p in resolved) / total
            best = max(resolved, key=lambda p: p["pnl_pct"])
            worst = min(resolved, key=lambda p: p["pnl_pct"])
            lines.append("")
            lines.append(f"Total: {total}  |  Wins: {wins}  |  Rate: {wins/total*100:.0f}%  |  Avg P&L: {avg_pnl:+.3f}%")
            lines.append(f"Best: {best['pnl_pct']:+.3f}% ({best['grade']} {best['signal']})  |  Worst: {worst['pnl_pct']:+.3f}% ({worst['grade']} {worst['signal']})")

            # Grade breakdown
            grade_stats = {}
            for p in resolved:
                g = p["grade"]
                if g not in grade_stats:
                    grade_stats[g] = {"wins": 0, "total": 0, "pnl_sum": 0}
                grade_stats[g]["total"] += 1
                grade_stats[g]["pnl_sum"] += p["pnl_pct"]
                if p["result"] == "WIN":
                    grade_stats[g]["wins"] += 1
            lines.append("")
            lines.append("By Grade:")
            for g in ["A+", "A", "B", "C", "D", "F"]:
                if g in grade_stats:
                    gs = grade_stats[g]
                    r = gs["wins"] / gs["total"] * 100
                    avg = gs["pnl_sum"] / gs["total"]
                    lines.append(f"  {g}: {gs['wins']}/{gs['total']} ({r:.0f}%) avg P&L {avg:+.3f}%")

            # Signal direction breakdown
            call_resolved = [p for p in resolved if p["signal"] == "CALL"]
            put_resolved = [p for p in resolved if p["signal"] == "PUT"]
            if call_resolved and put_resolved:
                lines.append("")
                call_wins = sum(1 for p in call_resolved if p["result"] == "WIN")
                put_wins = sum(1 for p in put_resolved if p["result"] == "WIN")
                lines.append(f"CALL: {call_wins}/{len(call_resolved)} ({call_wins/len(call_resolved)*100:.0f}%)  |  "
                             f"PUT: {put_wins}/{len(put_resolved)} ({put_wins/len(put_resolved)*100:.0f}%)")

        self.history_text.setPlainText("\n".join(lines))

    # -------------------------------------------------------------------
    # Open/Close swing predictor
    # -------------------------------------------------------------------
    def _fetch_swing_prediction(self):
        """Fetch historical data in a background thread."""
        self._swing_worker = SwingFetchWorker()
        self._swing_worker.finished.connect(self._process_swing_data)
        self._swing_worker.start()

    def _process_swing_data(self, df):
        """Process swing prediction data (called from worker thread signal)."""
        try:
            if df is None or df.empty or len(df) < 20:
                self.lbl_swing.setText("Insufficient historical data")
                return

            # Calculate daily moves
            df["OC_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100
            df["range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100
            df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

            et = now_et()
            today_dow = et.weekday()

            # Day-of-week stats
            df["dow"] = df.index.dayofweek
            dow_data = df[df["dow"] == today_dow]["OC_pct"]
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            dow_name = dow_names[today_dow] if today_dow < 5 else "?"

            # Recent trend (last 5 days)
            recent = df["OC_pct"].iloc[-5:]
            recent_avg = recent.mean()
            recent_up = sum(1 for x in recent if x > 0)

            # Overall stats
            last_20 = df.iloc[-20:]
            avg_oc = last_20["OC_pct"].mean()
            avg_range = last_20["range_pct"].mean()
            up_days = sum(1 for x in last_20["OC_pct"] if x > 0)
            down_days = 20 - up_days

            # Predict direction
            signals_up = 0
            signals_dn = 0
            reasons = []

            # 1. Day-of-week bias
            if len(dow_data) >= 4:
                dow_avg = dow_data.mean()
                dow_up = sum(1 for x in dow_data if x > 0)
                dow_rate = dow_up / len(dow_data) * 100
                if dow_avg > 0.05:
                    signals_up += 1
                    reasons.append(f"{dow_name}s tend up: avg {dow_avg:+.2f}% ({dow_rate:.0f}% green)")
                elif dow_avg < -0.05:
                    signals_dn += 1
                    reasons.append(f"{dow_name}s tend down: avg {dow_avg:+.2f}% ({100-dow_rate:.0f}% red)")
                else:
                    reasons.append(f"{dow_name}s neutral: avg {dow_avg:+.2f}%")

            # 2. Recent momentum
            if recent_up >= 4:
                signals_up += 1
                reasons.append(f"Recent momentum: {recent_up}/5 up days (avg {recent_avg:+.2f}%)")
            elif recent_up <= 1:
                signals_dn += 1
                reasons.append(f"Recent weakness: {recent_up}/5 up days (avg {recent_avg:+.2f}%)")
            else:
                reasons.append(f"Mixed recent: {recent_up}/5 up (avg {recent_avg:+.2f}%)")

            # 3. 20-day trend
            if avg_oc > 0.05:
                signals_up += 1
                reasons.append(f"20-day bullish bias: avg {avg_oc:+.2f}%, {up_days}/{20} up")
            elif avg_oc < -0.05:
                signals_dn += 1
                reasons.append(f"20-day bearish bias: avg {avg_oc:+.2f}%, {down_days}/{20} down")

            # 4. Mean reversion after big moves
            yesterday_oc = df["OC_pct"].iloc[-1]
            if abs(yesterday_oc) > avg_range * 0.6:
                if yesterday_oc > 0:
                    signals_dn += 1
                    reasons.append(f"Yesterday big up ({yesterday_oc:+.2f}%) -> reversion bias")
                else:
                    signals_up += 1
                    reasons.append(f"Yesterday big down ({yesterday_oc:+.2f}%) -> bounce bias")

            # 5. Consecutive day streaks (3+ days same direction often reverses)
            last_3 = df["OC_pct"].iloc[-3:]
            if all(x > 0 for x in last_3):
                signals_dn += 1
                reasons.append(f"3+ green days in a row -> mean reversion risk")
            elif all(x < 0 for x in last_3):
                signals_up += 1
                reasons.append(f"3+ red days in a row -> bounce is overdue")

            # 6. Range contraction/expansion (tight range = breakout coming)
            last_5_range = df["range_pct"].iloc[-5:].mean()
            if last_5_range < avg_range * 0.6:
                reasons.append(f"[!] Range contracting ({last_5_range:.2f}% vs avg {avg_range:.2f}%) -> breakout likely")
            elif last_5_range > avg_range * 1.4:
                reasons.append(f"Range expanding ({last_5_range:.2f}%) -> volatile, expect continuation")

            # 7. Gap analysis (overnight gap tends to fill)
            if not pd.isna(df["gap_pct"].iloc[-1]):
                gap = df["gap_pct"].iloc[-1]
                if gap > 0.15:
                    signals_dn += 1
                    reasons.append(f"Gap up {gap:+.2f}% -> gap fill bias (bearish intraday)")
                elif gap < -0.15:
                    signals_up += 1
                    reasons.append(f"Gap down {gap:+.2f}% -> gap fill bias (bullish intraday)")

            # Build prediction
            if signals_up > signals_dn:
                direction = "UP"
                confidence = min(90, 50 + (signals_up - signals_dn) * 15)
            elif signals_dn > signals_up:
                direction = "DOWN"
                confidence = min(90, 50 + (signals_dn - signals_up) * 15)
            else:
                direction = "NEUTRAL"
                confidence = 50

            # Direction arrow for header
            dir_arrow = {"UP": "^", "DOWN": "v", "NEUTRAL": "-"}.get(direction, "?")

            # Day-of-week breakdown table
            dow_table_lines = []
            dow_names_all = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            for i, dn in enumerate(dow_names_all):
                dd = df[df["dow"] == i]["OC_pct"]
                if len(dd) >= 2:
                    d_avg = dd.mean()
                    d_up = sum(1 for x in dd if x > 0)
                    d_rate = d_up / len(dd) * 100
                    marker = " <<" if i == today_dow else ""
                    dow_table_lines.append(f"    {dn}:  avg {d_avg:+.2f}%  |  {d_rate:.0f}% green  ({len(dd)} samples){marker}")

            # Last 5 days detail
            last5_lines = []
            for idx in range(-min(5, len(df)), 0):
                row = df.iloc[idx]
                d = df.index[idx]
                oc = row["OC_pct"]
                rng = row["range_pct"]
                tag = "+" if oc > 0 else "-" if oc < 0 else "="
                try:
                    day_str = d.strftime("%a %m/%d")
                except Exception:
                    day_str = str(d)[:10]
                last5_lines.append(f"    {day_str}  [{tag}]  O->C: {oc:+.2f}%  range: {rng:.2f}%")

            # Volatility stats
            vol_20 = last_20["OC_pct"].std()
            max_up = last_20["OC_pct"].max()
            max_dn = last_20["OC_pct"].min()

            lines = [
                f"  [{dir_arrow}] Prediction: {direction}  ({confidence}% confidence)",
                f"",
                f"  -- 20-Day Overview --",
                f"    Avg O->C:     {avg_oc:+.2f}%",
                f"    Avg range:    {avg_range:.2f}%",
                f"    Volatility:   {vol_20:.2f}% std dev",
                f"    Up/Down:      {up_days} up / {down_days} down",
                f"    Best day:     {max_up:+.2f}%",
                f"    Worst day:    {max_dn:+.2f}%",
                f"",
                f"  -- Day-of-Week Breakdown --",
            ] + dow_table_lines + [
                f"",
                f"  -- Last 5 Sessions --",
            ] + last5_lines + [
                f"",
                f"  -- Signals --",
            ] + [f"    {r}" for r in reasons]

            self.lbl_swing.setPlainText("\n".join(lines))
            self.lbl_swing.setStyleSheet("color:#111; background:#d8d8d8; border:none;")
            self.swing_prediction = {"direction": direction, "confidence": confidence}

        except Exception as e:
            self.lbl_swing.setText(f"Could not load historical data: {e}")
            self.lbl_swing.setStyleSheet("color:#888;")


# ---------------------------------------------------------------------------
# Options Value Scanner - finds mispriced / good-value SPY contracts
# ---------------------------------------------------------------------------

def _black_scholes_call(S, K, T, r, sigma):
    """Basic BS call price for comparison. T in years, r annual rate."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    from math import log, sqrt, exp
    from statistics import NormalDist
    nd = NormalDist()
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * nd.cdf(d1) - K * exp(-r * T) * nd.cdf(d2)


def _black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    from math import log, sqrt, exp
    from statistics import NormalDist
    nd = NormalDist()
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * nd.cdf(-d2) - S * nd.cdf(-d1)


def scan_options_value(spy_price, expirations_to_scan=3, r=0.045):
    """
    Scan SPY options chains for value opportunities.
    Returns a list of dicts, each representing a potential value trade.

    Checks:
    1. Spread value    - wide bid-ask where midpoint is favorable
    2. IV discount     - contracts with lower IV than neighbors
    3. Intrinsic edge  - midpoint below intrinsic value (free money)
    4. Volume spike    - unusual volume vs open interest
    5. Model discount  - market price below Black-Scholes fair value
    """
    spy = yf.Ticker("SPY")
    try:
        all_exps = spy.options
    except Exception:
        return []

    if not all_exps:
        return []

    today = datetime.now().date()
    opportunities = []

    exps_to_check = []
    for exp_str in all_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 0 <= dte <= 5:
            exps_to_check.append((exp_str, exp_date, dte))
        if len(exps_to_check) >= expirations_to_scan:
            break

    for exp_str, exp_date, dte in exps_to_check:
        T = max(dte / 365.0, 1 / 365.0)  # time in years, min 1 day

        try:
            chain = spy.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
            if df.empty:
                continue

            # Filter to strikes near the money (+/- $8)
            near_money = df[(df["strike"] >= spy_price - 8) &
                            (df["strike"] <= spy_price + 8)].copy()
            if near_money.empty:
                continue

            for _, row in near_money.iterrows():
                strike = row["strike"]
                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0
                last = row.get("lastPrice", 0) or 0
                volume = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                iv = row.get("impliedVolatility", 0) or 0

                if ask <= 0 or bid < 0:
                    continue

                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pct = spread / mid * 100 if mid > 0 else 999

                # Intrinsic value
                if opt_type == "CALL":
                    intrinsic = max(0, spy_price - strike)
                else:
                    intrinsic = max(0, strike - spy_price)

                # Black-Scholes theoretical price
                if iv > 0:
                    if opt_type == "CALL":
                        bs_price = _black_scholes_call(spy_price, strike, T, r, iv)
                    else:
                        bs_price = _black_scholes_put(spy_price, strike, T, r, iv)
                else:
                    bs_price = intrinsic

                signals = []
                score = 0

                # 1. Spread value: wide spread where midpoint is favorable
                if spread_pct > 15 and mid > 0.05:
                    if last > 0 and last < mid - spread * 0.1:
                        signals.append(f"Last ${last:.2f} below mid ${mid:.2f} -- fillable below ask")
                        score += 15

                # 2. Below intrinsic value (rare but happens with stale quotes)
                if mid < intrinsic * 0.95 and intrinsic > 0.10:
                    discount_pct = (1 - mid / intrinsic) * 100
                    signals.append(f"Mid ${mid:.2f} is {discount_pct:.0f}% below intrinsic ${intrinsic:.2f}")
                    score += 30

                # 3. Model discount: market ask below BS fair value
                if bs_price > 0 and ask < bs_price * 0.90 and bs_price > 0.10:
                    discount = (1 - ask / bs_price) * 100
                    signals.append(f"Ask ${ask:.2f} is {discount:.0f}% below BS value ${bs_price:.2f}")
                    score += 25

                # 4. IV discount vs neighbors: compare to adjacent strikes
                if iv > 0:
                    neighbor_ivs = near_money[
                        (near_money["strike"] != strike) &
                        (abs(near_money["strike"] - strike) <= 2)
                    ]["impliedVolatility"]
                    if len(neighbor_ivs) >= 2:
                        avg_neighbor_iv = neighbor_ivs.mean()
                        if avg_neighbor_iv > 0 and iv < avg_neighbor_iv * 0.85:
                            iv_discount = (1 - iv / avg_neighbor_iv) * 100
                            signals.append(f"IV {iv:.0%} is {iv_discount:.0f}% below neighbors ({avg_neighbor_iv:.0%})")
                            score += 20

                # 5. Unusual activity: volume spike relative to OI
                if oi > 10 and volume > oi * 2:
                    ratio = volume / oi
                    signals.append(f"Volume/OI ratio {ratio:.1f}x -- unusual activity")
                    score += 10
                elif volume > 5000 and oi > 0 and volume > oi * 1.5:
                    signals.append(f"Heavy volume {volume:,} vs OI {oi:,}")
                    score += 5

                # 6. Tight spread + decent volume = good liquidity value
                if spread_pct < 5 and volume > 500 and mid > 0.10:
                    signals.append(f"Tight spread ({spread_pct:.1f}%) + liquid ({volume:,} vol)")
                    score += 8

                # 7. Penny-priced with upside: cheap OTM with low ask
                if ask <= 0.05 and ask > 0 and dte <= 1:
                    otm_dist = abs(spy_price - strike)
                    if otm_dist < 3:
                        signals.append(f"Penny contract ${ask:.2f} -- {otm_dist:.1f} pts OTM, lottery ticket")
                        score += 5

                if signals and score >= 10:
                    opportunities.append({
                        "type": opt_type,
                        "exp": exp_str,
                        "dte": dte,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": last,
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "volume": volume,
                        "oi": oi,
                        "iv": iv,
                        "intrinsic": intrinsic,
                        "bs_price": bs_price,
                        "score": score,
                        "signals": signals,
                    })

    opportunities.sort(key=lambda x: x["score"], reverse=True)
    return opportunities


class OptionsValueScanner(QMainWindow):
    """Separate window that scans SPY option chains for value opportunities."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPYderScalp - Options Value Scanner")
        self.setGeometry(50, 50, 1050, 620)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("SPY Options Value Scanner")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        header.addWidget(title)

        header.addStretch()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setFont(QFont("Arial", 9))
        self.lbl_status.setStyleSheet("color:gray;")
        header.addWidget(self.lbl_status)

        self.btn_refresh = QPushButton("Rescan")
        self.btn_refresh.setStyleSheet("background:#333;color:#7ba7c9;border:1px solid #555;padding:4px 14px;border-radius:3px;")
        self.btn_refresh.clicked.connect(self.run_scan)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        # Filter row
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Show:"))
        self.cb_calls = QCheckBox("Calls")
        self.cb_calls.setChecked(True)
        self.cb_calls.stateChanged.connect(self._apply_filters)
        filt.addWidget(self.cb_calls)
        self.cb_puts = QCheckBox("Puts")
        self.cb_puts.setChecked(True)
        self.cb_puts.stateChanged.connect(self._apply_filters)
        filt.addWidget(self.cb_puts)

        filt.addSpacing(12)
        filt.addWidget(QLabel("Min score:"))
        self.spin_min_score = QSpinBox()
        self.spin_min_score.setRange(5, 80)
        self.spin_min_score.setValue(10)
        self.spin_min_score.valueChanged.connect(self._apply_filters)
        filt.addWidget(self.spin_min_score)

        filt.addStretch()
        self.lbl_count = QLabel("")
        self.lbl_count.setFont(QFont("Arial", 9))
        filt.addWidget(self.lbl_count)
        layout.addLayout(filt)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Score", "Type", "Exp", "DTE", "Strike", "Bid", "Ask",
            "Mid", "Spread%", "IV", "Vol/OI", "Signals"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(11, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setFont(QFont("Consolas", 9))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a2e; color: #ccc; gridline-color: #333;
                           alternate-background-color: #1e1e36; }
            QHeaderView::section { background: #2a2a4a; color: #ddd; padding: 4px;
                                   border: 1px solid #333; font-weight: bold; }
            QTableWidget::item:selected { background: #3a3a6a; }
        """)
        self.table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self.table, stretch=1)

        # Detail panel
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 9))
        self.detail_text.setMaximumHeight(140)
        self.detail_text.setStyleSheet("background:#1a1a2e; color:#ccc; border:1px solid #333;")
        self.detail_text.setPlainText("Click a row to see full analysis.")
        layout.addWidget(self.detail_text)

        # Legend
        legend = QLabel(
            "Score = composite value score  |  "
            "Signals: intrinsic edge, model discount, IV skew, volume spike, spread value"
        )
        legend.setFont(QFont("Arial", 8))
        legend.setStyleSheet("color:#666;")
        layout.addWidget(legend)

        self._all_results = []

    def run_scan(self):
        self.lbl_status.setText("Scanning...")
        self.lbl_status.setStyleSheet("color:#ffab00;")
        self.btn_refresh.setEnabled(False)
        # Use a timer to allow UI to update before blocking scan
        QTimer.singleShot(50, self._do_scan)

    def _do_scan(self):

        try:
            # Get current SPY price
            df = yf_download_safe("SPY", period="1d", interval="1m")
            if df is None:
                df = yf_download_safe("SPY", period="5d", interval="1d")
            if df is None:
                self.lbl_status.setText("[!] Could not get SPY price")
                self.lbl_status.setStyleSheet("color:#ef5350;")
                self.btn_refresh.setEnabled(True)
                return
            spy_price = float(df["Close"].iloc[-1])

            results = scan_options_value(spy_price)
            self._all_results = results
            self._apply_filters()

            count = len(results)
            self.lbl_status.setText(
                f"Found {count} opportunities  |  SPY ${spy_price:.2f}  |  "
                f"Scanned {datetime.now().strftime('%I:%M:%S %p')}"
            )
            self.lbl_status.setStyleSheet("color:#26a69a;")

        except Exception as e:
            self.lbl_status.setText(f"[X] Error: {e}")
            self.lbl_status.setStyleSheet("color:#ef5350;")
            traceback.print_exc()

        self.btn_refresh.setEnabled(True)

    def _apply_filters(self):
        show_calls = self.cb_calls.isChecked()
        show_puts = self.cb_puts.isChecked()
        min_score = self.spin_min_score.value()

        filtered = [
            r for r in self._all_results
            if r["score"] >= min_score
            and ((r["type"] == "CALL" and show_calls) or (r["type"] == "PUT" and show_puts))
        ]

        self.table.setRowCount(0)
        self.table.setRowCount(len(filtered))

        for i, opp in enumerate(filtered):
            score_item = QTableWidgetItem(str(opp["score"]))
            score_item.setTextAlignment(Qt.AlignCenter)
            if opp["score"] >= 30:
                score_item.setForeground(QColor("#26a69a"))
            elif opp["score"] >= 20:
                score_item.setForeground(QColor("#ffab00"))
            else:
                score_item.setForeground(QColor("#888"))

            type_item = QTableWidgetItem(opp["type"])
            type_item.setTextAlignment(Qt.AlignCenter)
            if opp["type"] == "CALL":
                type_item.setForeground(QColor("#26a69a"))
            else:
                type_item.setForeground(QColor("#ef5350"))

            self.table.setItem(i, 0, score_item)
            self.table.setItem(i, 1, type_item)
            self.table.setItem(i, 2, QTableWidgetItem(opp["exp"]))

            dte_item = QTableWidgetItem(str(opp["dte"]))
            dte_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 3, dte_item)

            self.table.setItem(i, 4, QTableWidgetItem(f"${opp['strike']:.0f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"${opp['bid']:.2f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"${opp['ask']:.2f}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"${opp['mid']:.2f}"))

            spread_item = QTableWidgetItem(f"{opp['spread_pct']:.1f}%")
            if opp['spread_pct'] > 20:
                spread_item.setForeground(QColor("#ef5350"))
            elif opp['spread_pct'] < 8:
                spread_item.setForeground(QColor("#26a69a"))
            self.table.setItem(i, 8, spread_item)

            self.table.setItem(i, 9, QTableWidgetItem(f"{opp['iv']:.0%}" if opp['iv'] > 0 else "--"))

            vol_oi = f"{opp['volume']:,}/{opp['oi']:,}" if opp['oi'] > 0 else f"{opp['volume']:,}"
            self.table.setItem(i, 10, QTableWidgetItem(vol_oi))

            sig_summary = " | ".join(s.split("--")[0].strip() for s in opp["signals"])
            self.table.setItem(i, 11, QTableWidgetItem(sig_summary))

        self.lbl_count.setText(f"Showing {len(filtered)} of {len(self._all_results)}")
        self._filtered = filtered

    def _on_row_clicked(self, row, col):
        if row < 0 or row >= len(self._filtered):
            return
        opp = self._filtered[row]

        lines = [
            f"{'=' * 56}",
            f"{opp['type']} ${opp['strike']:.0f}  |  Exp {opp['exp']} ({opp['dte']} DTE)",
            f"{'=' * 56}",
            f"",
            f"  Bid: ${opp['bid']:.2f}   Ask: ${opp['ask']:.2f}   Mid: ${opp['mid']:.2f}",
            f"  Spread: ${opp['spread']:.2f} ({opp['spread_pct']:.1f}%)",
            f"  Last: ${opp['last']:.2f}   Volume: {opp['volume']:,}   OI: {opp['oi']:,}",
            f"  IV: {opp['iv']:.1%}" if opp['iv'] > 0 else "  IV: N/A",
            f"  Intrinsic: ${opp['intrinsic']:.2f}   BS Model: ${opp['bs_price']:.2f}",
            f"",
            f"  VALUE SCORE: {opp['score']}",
            f"",
            f"  WHY:",
        ]
        for sig in opp["signals"]:
            lines.append(f"    * {sig}")

        lines.append("")

        # Trade suggestion
        if opp["mid"] > 0:
            if opp["spread_pct"] > 10:
                limit = opp["bid"] + opp["spread"] * 0.3
                lines.append(f"  SUGGESTION: Try limit order at ${limit:.2f} (30% above bid)")
                lines.append(f"  Wide spread -- avoid market orders. Be patient for a fill.")
            else:
                limit = opp["mid"]
                lines.append(f"  SUGGESTION: Limit order near mid ${limit:.2f}")

            if opp["score"] >= 30:
                lines.append(f"  Rating: STRONG value -- multiple signals aligned")
            elif opp["score"] >= 20:
                lines.append(f"  Rating: MODERATE value -- worth investigating")
            else:
                lines.append(f"  Rating: WEAK -- one signal, proceed with caution")

        self.detail_text.setPlainText("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = SPYderScalpApp()
    win.show()
    # Force the canvas to realize its size and paint
    win.candle_chart.setVisible(True)
    win.candle_chart.updateGeometry()
    win.candle_chart.draw_idle()
    # Auto-start monitoring on launch
    QTimer.singleShot(500, win.start_monitoring)
    # Fetch swing prediction after a short delay (non-blocking)
    QTimer.singleShot(2000, win._fetch_swing_prediction)
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Write crash info to log file so silent failures are debuggable
        try:
            with open(CRASH_LOG, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"CRASH at {datetime.now().isoformat()}\n")
                f.write(traceback.format_exc())
                f.write(f"{'='*60}\n")
        except Exception:
            pass
        raise
