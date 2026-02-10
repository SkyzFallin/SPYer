"""
SPYer – SPY Intraday Options Signal App
Cross-Platform (Windows & macOS)

Enhanced with multi-indicator signal quality scoring:
  • VWAP cross with volume confirmation
  • RSI momentum check
  • EMA-9 / EMA-21 trend alignment
  • Price-range position (high/low of day)
  • Candle momentum (consecutive green/red bars)

Each indicator adds to a 0-100 quality score with a letter grade (A-F).
"""

import sys
import platform
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout, QCheckBox,
    QProgressBar, QFrame, QComboBox, QSpinBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor

try:
    from plyer import notification as desktop_notification
except ImportError:
    desktop_notification = None

# Platform-specific sound
if platform.system() == "Darwin":
    import os
elif platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None


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


def score_to_grade(score: float) -> str:
    if score >= 85:
        return "A+"
    elif score >= 75:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 35:
        return "D"
    else:
        return "F"


def grade_color(grade: str) -> str:
    """Return a hex colour for the grade."""
    return {
        "A+": "#00c853",
        "A": "#2e7d32",
        "B": "#558b2f",
        "C": "#f9a825",
        "D": "#e65100",
        "F": "#c62828",
    }.get(grade, "#888888")


def evaluate_signal(df: pd.DataFrame, signal_type: str) -> dict:
    """
    Evaluate a CALL or PUT signal across multiple indicators.
    Returns dict with total score (0-100), grade, and per-indicator breakdown.
    """
    breakdown = {}
    weights = {
        "vwap": 25,
        "volume": 20,
        "rsi": 15,
        "ema_trend": 20,
        "range_position": 10,
        "momentum": 10,
    }
    raw_scores = {}  # 0.0 – 1.0 per indicator

    close = df["Close"]
    current_price = close.iloc[-1]
    current_vwap = df["VWAP"].iloc[-1]

    # 1. VWAP distance (larger gap = stronger)
    vwap_diff_pct = (current_price - current_vwap) / current_vwap * 100
    if signal_type == "CALL":
        raw_scores["vwap"] = np.clip(vwap_diff_pct / 0.30, 0, 1)  # 0.30% = full score
        breakdown["vwap"] = f"Price {vwap_diff_pct:+.3f}% from VWAP"
    else:
        raw_scores["vwap"] = np.clip(-vwap_diff_pct / 0.30, 0, 1)
        breakdown["vwap"] = f"Price {vwap_diff_pct:+.3f}% from VWAP"

    # 2. Volume ratio strength
    avg_vol = df["Volume"].iloc[-21:-1].mean() if len(df) > 21 else df["Volume"].mean()
    vol_ratio = df["Volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0
    raw_scores["volume"] = np.clip((vol_ratio - 1.0) / 2.0, 0, 1)  # 3x = full score
    breakdown["volume"] = f"Volume {vol_ratio:.2f}x avg"

    # 3. RSI alignment
    rsi_series = compute_rsi(close)
    rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
    if signal_type == "CALL":
        # Ideal: RSI 50-70 (momentum without overbought)
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

    # 4. EMA-9 / EMA-21 trend alignment
    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema_diff = (ema9.iloc[-1] - ema21.iloc[-1]) / ema21.iloc[-1] * 100
    if signal_type == "CALL":
        raw_scores["ema_trend"] = np.clip(ema_diff / 0.15, 0, 1)
    else:
        raw_scores["ema_trend"] = np.clip(-ema_diff / 0.15, 0, 1)
    breakdown["ema_trend"] = f"EMA9-EMA21 gap {ema_diff:+.3f}%"

    # 5. Range position (where is price within today's H/L)
    day_high = df["High"].max()
    day_low = df["Low"].min()
    day_range = day_high - day_low if day_high != day_low else 1
    range_pct = (current_price - day_low) / day_range
    if signal_type == "CALL":
        raw_scores["range_position"] = range_pct
    else:
        raw_scores["range_position"] = 1.0 - range_pct
    breakdown["range_position"] = f"Range position {range_pct:.0%} (low→high)"

    # 6. Candle momentum – consecutive bars in signal direction
    last_n = close.iloc[-6:]  # last 5 changes
    changes = last_n.diff().dropna()
    if signal_type == "CALL":
        streak = sum(1 for c in reversed(changes) if c > 0)
    else:
        streak = sum(1 for c in reversed(changes) if c < 0)
    raw_scores["momentum"] = np.clip(streak / 4, 0, 1)
    breakdown["momentum"] = f"{streak} consecutive {'green' if signal_type == 'CALL' else 'red'} bars"

    # Weighted total
    total = sum(raw_scores[k] * weights[k] for k in weights)
    total = round(min(total, 100), 1)
    grade = score_to_grade(total)

    return {
        "score": total,
        "grade": grade,
        "breakdown": breakdown,
        "raw": raw_scores,
        "weights": weights,
        "rsi": rsi,
        "ema9": ema9.iloc[-1],
        "ema21": ema21.iloc[-1],
        "vol_ratio": vol_ratio,
    }


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class SPYerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPYer – SPY 0-2 DTE Options Signal Alert")
        self.setGeometry(100, 100, 960, 780)
        self.last_signal_time = None
        self.signal_cooldown = 300  # seconds
        self.is_monitoring = False
        self.platform = platform.system()
        self._build_ui()

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_signals)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Title
        title = QLabel("🎯 SPYer – Signal Monitor")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        subtitle = QLabel(f"Running on {self.platform}  •  Multi-indicator quality scoring")
        subtitle.setFont(QFont("Arial", 9))
        subtitle.setStyleSheet("color: gray;")
        subtitle.setAlignment(Qt.AlignCenter)
        main.addWidget(subtitle)

        # ---- Status row ----
        status_box = QGroupBox("Current Market Snapshot")
        sg = QGridLayout()

        self.lbl_price = QLabel("SPY Price: --")
        self.lbl_price.setFont(QFont("Arial", 12))
        sg.addWidget(self.lbl_price, 0, 0)

        self.lbl_vwap = QLabel("VWAP: --")
        self.lbl_vwap.setFont(QFont("Arial", 12))
        sg.addWidget(self.lbl_vwap, 0, 1)

        self.lbl_volume = QLabel("Volume Ratio: --")
        self.lbl_volume.setFont(QFont("Arial", 12))
        sg.addWidget(self.lbl_volume, 1, 0)

        self.lbl_rsi = QLabel("RSI(14): --")
        self.lbl_rsi.setFont(QFont("Arial", 12))
        sg.addWidget(self.lbl_rsi, 1, 1)

        self.lbl_ema = QLabel("EMA 9/21: --")
        self.lbl_ema.setFont(QFont("Arial", 12))
        sg.addWidget(self.lbl_ema, 2, 0)

        self.lbl_update = QLabel("Last Update: --")
        self.lbl_update.setFont(QFont("Arial", 10))
        self.lbl_update.setStyleSheet("color: gray;")
        sg.addWidget(self.lbl_update, 2, 1)

        status_box.setLayout(sg)
        main.addWidget(status_box)

        # ---- Signal quality gauge ----
        gauge_box = QGroupBox("Signal Quality")
        gauge_layout = QVBoxLayout()

        sig_row = QHBoxLayout()
        self.lbl_signal = QLabel("No signal")
        self.lbl_signal.setFont(QFont("Arial", 16, QFont.Bold))
        sig_row.addWidget(self.lbl_signal)

        self.lbl_grade = QLabel("")
        self.lbl_grade.setFont(QFont("Arial", 28, QFont.Bold))
        self.lbl_grade.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sig_row.addWidget(self.lbl_grade)
        gauge_layout.addLayout(sig_row)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("%v / 100")
        self.quality_bar.setFixedHeight(28)
        gauge_layout.addWidget(self.quality_bar)

        self.lbl_breakdown = QLabel("")
        self.lbl_breakdown.setFont(QFont("Consolas", 9))
        self.lbl_breakdown.setWordWrap(True)
        gauge_layout.addWidget(self.lbl_breakdown)

        gauge_box.setLayout(gauge_layout)
        main.addWidget(gauge_box)

        # ---- Settings row ----
        settings_box = QGroupBox("Settings")
        sl = QHBoxLayout()

        self.cb_calls = QCheckBox("Call Signals")
        self.cb_calls.setChecked(True)
        sl.addWidget(self.cb_calls)

        self.cb_puts = QCheckBox("Put Signals")
        self.cb_puts.setChecked(True)
        sl.addWidget(self.cb_puts)

        sl.addWidget(QLabel("Min Grade:"))
        self.combo_min_grade = QComboBox()
        self.combo_min_grade.addItems(["F", "D", "C", "B", "A", "A+"])
        self.combo_min_grade.setCurrentText("C")
        sl.addWidget(self.combo_min_grade)

        sl.addWidget(QLabel("Vol threshold:"))
        self.spin_vol = QSpinBox()
        self.spin_vol.setRange(100, 500)
        self.spin_vol.setSingleStep(10)
        self.spin_vol.setValue(150)
        self.spin_vol.setSuffix("%")
        sl.addWidget(self.spin_vol)

        settings_box.setLayout(sl)
        main.addWidget(settings_box)

        # ---- Buttons ----
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("▶ Start Monitoring")
        self.btn_start.setStyleSheet("background-color:#4CAF50;color:white;font-size:14px;padding:10px;border-radius:4px;")
        self.btn_start.clicked.connect(self.start_monitoring)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏸ Stop Monitoring")
        self.btn_stop.setStyleSheet("background-color:#f44336;color:white;font-size:14px;padding:10px;border-radius:4px;")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_scan = QPushButton("🔍 Scan Now")
        self.btn_scan.setStyleSheet("background-color:#2196F3;color:white;font-size:14px;padding:10px;border-radius:4px;")
        self.btn_scan.clicked.connect(self.manual_scan)
        btn_row.addWidget(self.btn_scan)

        main.addLayout(btn_row)

        # ---- Log ----
        log_box = QGroupBox("Signal History & Recommendations")
        ll = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setMaximumHeight(260)
        ll.addWidget(self.log_text)
        log_box.setLayout(ll)
        main.addWidget(log_box)

        footer = QLabel("ℹ️  Scans every 60 s during market hours (9:30 AM – 4:00 PM ET)  •  Data via Yahoo Finance (~15 min delay)")
        footer.setFont(QFont("Arial", 9))
        footer.setStyleSheet("color:gray;")
        main.addWidget(footer)

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------
    def start_monitoring(self):
        self.is_monitoring = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.monitor_timer.start(60000)
        self._log("✅ Monitoring started – scanning every 60 seconds")
        self.check_signals()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.monitor_timer.stop()
        self._log("⏸ Monitoring stopped")

    def manual_scan(self):
        self._log("🔍 Manual scan initiated…")
        self.check_signals()

    # -----------------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------------
    def _calc_vwap(self, df):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        return df

    def _min_grade_score(self) -> float:
        mapping = {"F": 0, "D": 35, "C": 50, "B": 65, "A": 75, "A+": 85}
        return mapping.get(self.combo_min_grade.currentText(), 0)

    def check_signals(self):
        try:
            spy = yf.Ticker("SPY")
            df = spy.history(period="1d", interval="1m")

            if df.empty or len(df) < 15:
                self._log("⚠️ Insufficient data")
                return

            df = self._calc_vwap(df)

            current_price = df["Close"].iloc[-1]
            current_vwap = df["VWAP"].iloc[-1]
            avg_vol = df["Volume"].iloc[-21:-1].mean() if len(df) > 21 else df["Volume"].mean()
            vol_ratio = df["Volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0

            rsi_s = compute_rsi(df["Close"])
            rsi_val = rsi_s.iloc[-1] if not pd.isna(rsi_s.iloc[-1]) else 0
            ema9 = compute_ema(df["Close"], 9).iloc[-1]
            ema21 = compute_ema(df["Close"], 21).iloc[-1]

            # Update snapshot labels
            self.lbl_price.setText(f"SPY Price: ${current_price:.2f}")
            self.lbl_vwap.setText(f"VWAP: ${current_vwap:.2f}")
            self.lbl_volume.setText(f"Volume Ratio: {vol_ratio:.2f}x")
            self.lbl_rsi.setText(f"RSI(14): {rsi_val:.1f}")
            self.lbl_ema.setText(f"EMA 9: ${ema9:.2f}  |  21: ${ema21:.2f}")
            self.lbl_update.setText(f"Last Update: {datetime.now().strftime('%I:%M:%S %p')}")

            vol_threshold = self.spin_vol.value() / 100.0
            signal_type = None

            if self.cb_calls.isChecked() and current_price > current_vwap and vol_ratio > vol_threshold:
                signal_type = "CALL"
            if self.cb_puts.isChecked() and current_price < current_vwap and vol_ratio > vol_threshold:
                signal_type = "PUT"

            if signal_type:
                result = evaluate_signal(df, signal_type)
                self._update_gauge(signal_type, result)

                # Filter by minimum grade
                if result["score"] < self._min_grade_score():
                    self.lbl_signal.setText(f"{signal_type} signal (below min grade – skipped)")
                    return

                # Cooldown check
                if self.last_signal_time and (datetime.now() - self.last_signal_time).seconds < self.signal_cooldown:
                    self.lbl_signal.setText(f"{signal_type} {result['grade']} (cooldown)")
                    return

                self.last_signal_time = datetime.now()
                self._trigger_alert(signal_type, current_price, current_vwap, vol_ratio, result)
            else:
                self.lbl_signal.setText("No signal")
                self.lbl_signal.setStyleSheet("color:#888;")
                self.lbl_grade.setText("")
                self.quality_bar.setValue(0)
                self.lbl_breakdown.setText("")

        except Exception as e:
            self._log(f"❌ Error: {e}")

    # -----------------------------------------------------------------------
    # UI updates & alerting
    # -----------------------------------------------------------------------
    def _update_gauge(self, signal_type, result):
        g = result["grade"]
        s = result["score"]
        color = grade_color(g)

        self.lbl_signal.setText(f"{'🟢' if signal_type == 'CALL' else '🔴'} {signal_type} Signal")
        self.lbl_signal.setStyleSheet(f"color:{color};")
        self.lbl_grade.setText(g)
        self.lbl_grade.setStyleSheet(f"color:{color};")
        self.quality_bar.setValue(int(s))

        bd_lines = []
        for key, desc in result["breakdown"].items():
            raw = result["raw"][key]
            w = result["weights"][key]
            pts = raw * w
            bar = "█" * int(raw * 10) + "░" * (10 - int(raw * 10))
            bd_lines.append(f"  {key:<16s} {bar}  {pts:5.1f}/{w}  ({desc})")
        self.lbl_breakdown.setText("\n".join(bd_lines))

    def _trigger_alert(self, signal_type, price, vwap, vol_ratio, result):
        options_rec = self._get_options_rec(signal_type, price)

        # Desktop notification
        try:
            if desktop_notification:
                desktop_notification.notify(
                    title=f"SPYer {signal_type} {result['grade']} 🚨",
                    message=f"Score {result['score']}/100\nSPY ${price:.2f}  VWAP ${vwap:.2f}\nVol {vol_ratio:.1f}x",
                    timeout=10,
                )
        except Exception:
            pass

        self._play_sound()

        log = (
            f"\n{'=' * 64}\n"
            f"🚨  {signal_type} SIGNAL  –  Grade {result['grade']}  ({result['score']}/100)\n"
            f"    {datetime.now().strftime('%I:%M:%S %p')}\n"
            f"{'=' * 64}\n"
            f"  SPY ${price:.2f}   VWAP ${vwap:.2f}   Vol {vol_ratio:.2f}x\n"
            f"  RSI {result['rsi']:.1f}   EMA9 ${result['ema9']:.2f}   EMA21 ${result['ema21']:.2f}\n"
        )
        for key, desc in result["breakdown"].items():
            raw = result["raw"][key]
            w = result["weights"][key]
            pts = raw * w
            log += f"    {key:<16s}  {pts:5.1f}/{w:2d}  {desc}\n"
        log += f"\n{options_rec}\n{'=' * 64}\n"
        self._log(log)

    def _get_options_rec(self, signal_type, price):
        try:
            spy = yf.Ticker("SPY")
            expirations = spy.options
            if not expirations:
                return "⚠️ No options data"

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
                    recs.append(f"  📅 {exp} ({dte} DTE): strikes ${strikes[0]}, ${strikes[1]}, ${strikes[2]}")

            return "\n".join(recs) if recs else "⚠️ No 0-2 DTE expirations found"
        except Exception as e:
            return f"⚠️ Options error: {e}"

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = SPYerApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
