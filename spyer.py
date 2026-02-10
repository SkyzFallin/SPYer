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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import yfinance as yf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout, QCheckBox,
    QProgressBar, QFrame, QComboBox, QSpinBox, QSplitter, QScrollArea,
    QTabWidget, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDoubleSpinBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor

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
    import os
elif platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None


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

def recommend_dte(signal_score, signal_grade, vol_ratio, rsi, event_ctx):
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
    elif time_decimal < 13.0:  # 10:30 AM - 1:00 PM ET
        dte_scores[0] += 15
        dte_scores[1] += 10
        reasons.append("Mid-morning -> 0DTE still good, 1DTE safer for lunch chop")
    elif time_decimal < 14.5:  # 1:00 - 2:30 PM ET
        dte_scores[0] -= 10
        dte_scores[1] += 25
        reasons.append("Afternoon -> 1DTE preferred, 0DTE theta accelerating against you")
    elif time_decimal < 15.5:  # 2:30 - 3:30 PM ET
        dte_scores[0] -= 20
        dte_scores[1] += 20
        reasons.append("Late afternoon -> 0DTE risky (fast decay), prefer 1DTE")
    else:  # After 3:30 PM ET
        dte_scores[0] -= 40
        dte_scores[1] += 30
        dte_scores[2] += 10
        reasons.append("Power hour -> 0DTE very risky, 1DTE to carry overnight if conviction")

    # --- Signal strength ---
    if signal_grade in ("A+", "A"):
        dte_scores[0] += 20
        reasons.append(f"Strong signal ({signal_grade}) -> 0DTE to maximize leverage")
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
        if time_decimal > 12:
            dte_scores[0] -= 10
            reasons.append("Friday afternoon -> 0DTE expires today, less room for error")

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
        "A+": "#00c853", "A": "#2e7d32", "B": "#558b2f",
        "C": "#f9a825", "D": "#e65100", "F": "#c62828",
    }.get(grade, "#888888")


def evaluate_signal(df, signal_type, event_ctx=None):
    breakdown = {}
    weights = {
        "vwap": 20, "volume": 18, "rsi": 12, "ema_trend": 18,
        "range_position": 8, "momentum": 8, "event_timing": 16,
    }
    raw_scores = {}

    close = df["Close"]
    current_price = close.iloc[-1]
    current_vwap = df["VWAP"].iloc[-1]

    # 1. VWAP distance
    vwap_diff_pct = (current_price - current_vwap) / current_vwap * 100
    if signal_type == "CALL":
        raw_scores["vwap"] = np.clip(vwap_diff_pct / 0.30, 0, 1)
    else:
        raw_scores["vwap"] = np.clip(-vwap_diff_pct / 0.30, 0, 1)
    breakdown["vwap"] = f"Price {vwap_diff_pct:+.3f}% from VWAP"

    # 2. Volume
    avg_vol = df["Volume"].iloc[-21:-1].mean() if len(df) > 21 else df["Volume"].mean()
    vol_ratio = df["Volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0
    raw_scores["volume"] = np.clip((vol_ratio - 1.0) / 2.0, 0, 1)
    breakdown["volume"] = f"Volume {vol_ratio:.2f}x avg"

    # 3. RSI
    rsi_series = compute_rsi(close)
    rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
    if signal_type == "CALL":
        if 50 <= rsi <= 70:
            raw_scores["rsi"] = 1.0 - abs(rsi - 60) / 20
        elif rsi > 70:
            raw_scores["rsi"] = max(0, 1.0 - (rsi - 70) / 20)
        else:
            raw_scores["rsi"] = max(0, (rsi - 30) / 20)
    else:
        if 30 <= rsi <= 50:
            raw_scores["rsi"] = 1.0 - abs(rsi - 40) / 20
        elif rsi < 30:
            raw_scores["rsi"] = max(0, 1.0 - (30 - rsi) / 20)
        else:
            raw_scores["rsi"] = max(0, (70 - rsi) / 20)
    breakdown["rsi"] = f"RSI {rsi:.1f}"

    # 4. EMA trend
    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema_diff = (ema9.iloc[-1] - ema21.iloc[-1]) / ema21.iloc[-1] * 100
    if signal_type == "CALL":
        raw_scores["ema_trend"] = np.clip(ema_diff / 0.15, 0, 1)
    else:
        raw_scores["ema_trend"] = np.clip(-ema_diff / 0.15, 0, 1)
    breakdown["ema_trend"] = f"EMA9-EMA21 gap {ema_diff:+.3f}%"

    # 5. Range position
    day_high = df["High"].max()
    day_low = df["Low"].min()
    day_range = day_high - day_low if day_high != day_low else 1
    range_pct = (current_price - day_low) / day_range
    if signal_type == "CALL":
        raw_scores["range_position"] = range_pct
    else:
        raw_scores["range_position"] = 1.0 - range_pct
    breakdown["range_position"] = f"Range position {range_pct:.0%} (low->high)"

    # 6. Candle momentum
    last_n = close.iloc[-6:]
    changes = last_n.diff().dropna().values
    if signal_type == "CALL":
        streak = sum(1 for c in reversed(changes) if c > 0)
    else:
        streak = sum(1 for c in reversed(changes) if c < 0)
    raw_scores["momentum"] = np.clip(streak / 4, 0, 1)
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

    total = sum(raw_scores[k] * weights[k] for k in weights)
    total = round(min(total, 100), 1)
    grade = score_to_grade(total)

    return {
        "score": total, "grade": grade, "breakdown": breakdown,
        "raw": raw_scores, "weights": weights, "rsi": rsi,
        "ema9": ema9.iloc[-1], "ema21": ema21.iloc[-1], "vol_ratio": vol_ratio,
    }


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------

def find_support_resistance(df, n_touches=2, tolerance_pct=0.05):
    """
    Find support and resistance levels from price action.
    Uses pivot highs/lows and clusters them within tolerance.
    Returns list of (price, type, touches) tuples.
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    if len(highs) < 5:
        return []

    # Find local pivot highs and lows (3-bar pivots)
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
        if abs(day_high - lvl_price) < tol:
            day_high = None
        if abs(day_low - lvl_price) < tol:
            day_low = None
    if day_high is not None:
        levels.append((day_high, "resistance", 1))
    if day_low is not None:
        levels.append((day_low, "support", 1))

    levels.sort(key=lambda x: x[0])
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
        self.manual_lines = []  # list of price floats
        self._last_ylim = None

        # Double-click to add manual line
        self.mpl_connect("button_press_event", self._on_click)

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

    def update_chart(self, full_df, n_bars=60):
        for ax in [self.ax_price, self.ax_vol, self.ax_rsi, self.ax_macd]:
            ax.cla()
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaa", labelsize=6)

        data = full_df.iloc[-n_bars:]
        full_close = full_df["Close"]
        if len(data) < 2:
            self.draw()
            return

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

        ema9_vals = compute_ema(full_close, 9).iloc[-n_bars:].values
        ema21_vals = compute_ema(full_close, 21).iloc[-n_bars:].values
        self.ax_price.plot(x, ema9_vals, color="#42a5f5", linewidth=0.8, label="EMA 9", alpha=0.7)
        self.ax_price.plot(x, ema21_vals, color="#ab47bc", linewidth=0.8, label="EMA 21", alpha=0.7)

        # --- Auto S/R levels ---
        if self.show_auto_sr:
            levels = find_support_resistance(full_df)
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

        rsi_full = compute_rsi(full_close, 14)
        rsi_vals = rsi_full.iloc[-n_bars:].values
        self.ax_rsi.plot(x, rsi_vals, color="#42a5f5", linewidth=1.0)
        self.ax_rsi.axhline(70, color="#ef5350", linewidth=0.6, linestyle="--", alpha=0.6)
        self.ax_rsi.axhline(30, color="#26a69a", linewidth=0.6, linestyle="--", alpha=0.6)
        self.ax_rsi.axhline(50, color="#666", linewidth=0.4, linestyle=":", alpha=0.4)
        self.ax_rsi.fill_between(x, 30, 70, alpha=0.05, color="white")
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.set_ylabel("RSI", fontsize=7, color="#aaa")
        self.ax_rsi.yaxis.set_major_locator(mticker.FixedLocator([30, 50, 70]))

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
            labels = data.index.strftime("%H:%M")
        else:
            labels = [str(i) for i in range(len(data))]
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
        self.signal_cooldown = 300
        self.is_monitoring = False
        self.platform = platform.system()
        self._build_ui()
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_signals)

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
        self.btn_start.setStyleSheet("background:#4CAF50;color:white;padding:4px 12px;border-radius:3px;")
        self.btn_start.clicked.connect(self.start_monitoring)
        top_bar.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background:#f44336;color:white;padding:4px 12px;border-radius:3px;")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_stop.setEnabled(False)
        top_bar.addWidget(self.btn_stop)

        self.btn_scan = QPushButton("Scan Now")
        self.btn_scan.setStyleSheet("background:#2196F3;color:white;padding:4px 12px;border-radius:3px;")
        self.btn_scan.clicked.connect(self.manual_scan)
        top_bar.addWidget(self.btn_scan)

        self.btn_scanner = QPushButton("Value Scanner")
        self.btn_scanner.setStyleSheet("background:#9C27B0;color:white;padding:4px 12px;border-radius:3px;")
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
        btn_clear.setStyleSheet("background:#555;color:white;padding:2px 8px;border-radius:2px;")
        btn_clear.clicked.connect(self._clear_manual_lines)
        sr_row.addWidget(btn_clear)

        hint = QLabel("(dbl-click chart to add/remove)")
        hint.setFont(QFont("Arial", 7))
        hint.setStyleSheet("color:#666;")
        sr_row.addWidget(hint)

        sr_row.addStretch()
        left_layout.addLayout(sr_row)

        self.candle_chart = CandlestickWidget(self)
        self.candle_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.candle_chart, stretch=1)
        splitter.addWidget(left_widget)

        # RIGHT: signals
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        sig_header = QHBoxLayout()
        self.lbl_signal = QLabel("No signal")
        self.lbl_signal.setFont(QFont("Arial", 13, QFont.Bold))
        sig_header.addWidget(self.lbl_signal)
        self.lbl_grade = QLabel("")
        self.lbl_grade.setFont(QFont("Arial", 22, QFont.Bold))
        self.lbl_grade.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sig_header.addWidget(self.lbl_grade)
        right_layout.addLayout(sig_header)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("%v / 100")
        self.quality_bar.setFixedHeight(18)
        right_layout.addWidget(self.quality_bar)

        hold_row = QHBoxLayout()
        self.lbl_hold_time = QLabel("Hold: --")
        self.lbl_hold_time.setFont(QFont("Arial", 10, QFont.Bold))
        hold_row.addWidget(self.lbl_hold_time)
        self.lbl_exit_by = QLabel("")
        self.lbl_exit_by.setFont(QFont("Arial", 9))
        self.lbl_exit_by.setAlignment(Qt.AlignRight)
        hold_row.addWidget(self.lbl_exit_by)
        self.lbl_hold_confidence = QLabel("")
        self.lbl_hold_confidence.setFont(QFont("Arial", 9))
        self.lbl_hold_confidence.setAlignment(Qt.AlignRight)
        hold_row.addWidget(self.lbl_hold_confidence)
        right_layout.addLayout(hold_row)

        # DTE recommendation row
        dte_row = QHBoxLayout()
        self.lbl_dte_rec = QLabel("DTE: --")
        self.lbl_dte_rec.setFont(QFont("Arial", 10, QFont.Bold))
        dte_row.addWidget(self.lbl_dte_rec)
        self.lbl_dte_detail = QLabel("")
        self.lbl_dte_detail.setFont(QFont("Arial", 8))
        self.lbl_dte_detail.setStyleSheet("color:#aaa;")
        dte_row.addWidget(self.lbl_dte_detail, stretch=1)
        right_layout.addLayout(dte_row)

        # Tabs
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
        QVBoxLayout(tab_log).setContentsMargins(4, 4, 4, 4)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 8))
        self.log_text.setStyleSheet(tab_style)
        tab_log.layout().addWidget(self.log_text)
        self.tabs.addTab(tab_log, "Log")

        right_layout.addWidget(self.tabs, stretch=1)
        splitter.addWidget(right_widget)
        splitter.setSizes([820, 540])
        outer.addWidget(splitter, stretch=1)

    # -----------------------------------------------------------------------
    # S/R line controls
    # -----------------------------------------------------------------------
    def _toggle_auto_sr(self, state):
        self.candle_chart.show_auto_sr = bool(state)

    def _add_manual_line(self):
        price = self.spin_manual_price.value()
        if price > 0:
            self.candle_chart.add_manual_line(price)
            self.spin_manual_price.setValue(0)

    def _clear_manual_lines(self):
        self.candle_chart.clear_manual_lines()

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------
    def start_monitoring(self):
        self.is_monitoring = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.monitor_timer.start(60000)
        self._log("[OK] Monitoring started - scanning every 60 seconds")
        self.check_signals()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.monitor_timer.stop()
        self._log("[STOP] Monitoring stopped")

    def manual_scan(self):
        self._log("[SCAN] Manual scan initiated...")
        self.check_signals()

    # -----------------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------------
    def _calc_vwap(self, df):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
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
                df = yf.download("SPY", period=strat["period"], interval=strat["interval"],
                                 progress=False, auto_adjust=True, multi_level_index=False)
                if df is not None and not df.empty and len(df) >= 2:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    min_bars = 15 if strat["label"] != "daily" else 2
                    if len(df) >= min_bars:
                        self._log(f"[>] Data: {len(df)} bars ({strat['label']}, {strat['interval']})")
                        return df, strat["label"]
            except Exception as ex:
                self._log(f"[!] Fetch {strat['label']} failed: {type(ex).__name__}: {ex}")
                continue
        return None, None

    def check_signals(self):
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
            self.candle_chart.update_chart(df, n_bars=60)

            current_price = df["Close"].iloc[-1]
            current_vwap = df["VWAP"].iloc[-1]
            avg_vol = df["Volume"].iloc[-21:-1].mean() if len(df) > 21 else df["Volume"].mean()
            vol_ratio = df["Volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0
            rsi_s = compute_rsi(df["Close"])
            rsi_val = rsi_s.iloc[-1] if not pd.isna(rsi_s.iloc[-1]) else 0
            ema9 = compute_ema(df["Close"], 9).iloc[-1]
            ema21 = compute_ema(df["Close"], 21).iloc[-1]

            self.lbl_price.setText(f"SPY ${current_price:.2f}")
            self.lbl_vwap.setText(f"VWAP ${current_vwap:.2f}")
            self.lbl_volume.setText(f"Vol {vol_ratio:.2f}x")
            self.lbl_rsi.setText(f"RSI {rsi_val:.1f}")
            self.lbl_ema.setText(f"EMA9 ${ema9:.2f} / 21 ${ema21:.2f}")
            self.lbl_update.setText(now_et().strftime("%I:%M:%S %p ET"))

            vol_threshold = self.spin_vol.value() / 100.0
            signal_type = None
            if self.cb_calls.isChecked() and current_price > current_vwap and vol_ratio > vol_threshold:
                signal_type = "CALL"
            if self.cb_puts.isChecked() and current_price < current_vwap and vol_ratio > vol_threshold:
                signal_type = "PUT"

            if signal_type:
                event_ctx = get_event_context(now_et())
                result = evaluate_signal(df, signal_type, event_ctx)
                self._update_gauge(signal_type, result)
                self._update_events(event_ctx)

                # DTE recommendation
                dte_rec = recommend_dte(result["score"], result["grade"],
                                        result["vol_ratio"], result["rsi"], event_ctx)
                self._update_dte_display(dte_rec)
                rec_dte = dte_rec["recommended_dte"]

                if result["score"] < self._min_grade_score():
                    self.lbl_signal.setText(f"{signal_type} signal (below min grade - skipped)")
                    hold = estimate_hold_time(signal_type, result["score"], rec_dte,
                                              result["vol_ratio"], result["rsi"], event_ctx)
                    self._update_hold_display(hold)
                    return

                if self.last_signal_time and (datetime.now() - self.last_signal_time).seconds < self.signal_cooldown:
                    self.lbl_signal.setText(f"{signal_type} {result['grade']} (cooldown)")
                    return

                self.last_signal_time = datetime.now()
                self._trigger_alert(signal_type, current_price, current_vwap, vol_ratio, result, event_ctx, dte_rec)
            else:
                self.lbl_signal.setText("No signal")
                self.lbl_signal.setStyleSheet("color:#888;")
                self.lbl_grade.setText("")
                self.quality_bar.setValue(0)
                self.lbl_breakdown.setPlainText("")
                self.lbl_hold_time.setText("Hold: --")
                self.lbl_exit_by.setText("")
                self.lbl_hold_confidence.setText("")
                self.lbl_hold_reasons.setPlainText("")
                self.lbl_dte_rec.setText("DTE: --")
                self.lbl_dte_rec.setStyleSheet("color:#888;")
                self.lbl_dte_detail.setText("")
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

        bd_lines = []
        for key, desc in result["breakdown"].items():
            raw = result["raw"][key]
            w = result["weights"][key]
            pts = raw * w
            bar = "#" * int(raw * 10) + "." * (10 - int(raw * 10))
            bd_lines.append(f"  {key:<16s} [{bar}] {pts:5.1f}/{w}  ({desc})")
        self.lbl_breakdown.setPlainText("\n".join(bd_lines))
        self.tabs.setCurrentIndex(0)

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
            elif rsi < 45:
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

        return "\n".join(lines)

    def _get_nearest_dte(self):
        try:
            spy = yf.Ticker("SPY")
            expirations = spy.options
            today = datetime.now().date()
            for exp in expirations[:5]:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 0 <= dte <= 2:
                    return dte
        except Exception:
            pass
        return 1

    def _get_options_rec(self, signal_type, price):
        try:
            spy = yf.Ticker("SPY")
            expirations = spy.options
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

    def _open_scanner(self):
        self._scanner_win = OptionsValueScanner()
        self._scanner_win.show()
        self._scanner_win.run_scan()


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
        self.btn_refresh.setStyleSheet("background:#2196F3;color:white;padding:4px 14px;border-radius:3px;")
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
        QApplication.processEvents()

        try:
            # Get current SPY price
            df = yf.download("SPY", period="1d", interval="1m",
                             progress=False, auto_adjust=True, multi_level_index=False)
            if df is None or df.empty:
                df = yf.download("SPY", period="5d", interval="1d",
                                 progress=False, auto_adjust=True, multi_level_index=False)
            if df is None or df.empty:
                self.lbl_status.setText("[!] Could not get SPY price")
                self.lbl_status.setStyleSheet("color:#ef5350;")
                self.btn_refresh.setEnabled(True)
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
