"""AI-assisted, risk-controlled NSE equity trading dashboard.

Install: pip install streamlit yfinance pandas numpy ta scikit-learn xgboost dhanhq
Secrets required: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
Optional secret: LIVE_TRADING_ENABLED=true (otherwise signals are paper-only).
Run: streamlit run app.py
"""
from __future__ import annotations

import math
import os
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from xgboost import XGBClassifier

try:
    from dhanhq import DhanContext, dhanhq
except ImportError:
    DhanContext = None
    dhanhq = None


st.set_page_config(page_title="Quant Barrier Trader", page_icon="📈", layout="wide")

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
# DhanHQ NSE_EQ security IDs. Override or extend through DHAN_SECURITY_IDS in secrets.
DEFAULT_SECURITY_IDS = {"RELIANCE.NS": "2885", "TCS.NS": "11536", "INFY.NS": "1594", "HDFCBANK.NS": "1333", "ICICIBANK.NS": "4963"}
FEATURES = ["log_return", "rsi_14", "macd_hist_norm", "atr_ratio", "stoch_k", "stoch_d", "bb_width", "bb_percent_b", "vroc", "sma_spread_ratio"]


def secret(name: str, default: Any = None) -> Any:
    """Read a Streamlit secret, then environment variable, without printing it."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def add_log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs = [f"[{stamp}] {message}", *st.session_state.get("logs", [])][:250]


def security_ids() -> dict[str, str]:
    configured = secret("DHAN_SECURITY_IDS", {})
    if isinstance(configured, str):
        # Format: RELIANCE.NS=2885,TCS.NS=11536
        configured = dict(item.split("=", 1) for item in configured.split(",") if "=" in item)
    return {**DEFAULT_SECURITY_IDS, **dict(configured or {})}


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"Yahoo Finance returned no daily data for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Stationary inputs only; absolute price levels never enter the model."""
    df = raw.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    bb = BollingerBands(close, window=20, window_dev=2)
    sma10 = SMAIndicator(close, window=10).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    result = pd.DataFrame(index=df.index)
    result["log_return"] = np.log(close / close.shift(1))
    result["rsi_14"] = RSIIndicator(close, window=14).rsi() / 100.0
    result["macd_hist_norm"] = macd / close.replace(0, np.nan)
    result["atr_ratio"] = atr / close.replace(0, np.nan)
    result["stoch_k"] = stoch.stoch() / 100.0
    result["stoch_d"] = stoch.stoch_signal() / 100.0
    result["bb_width"] = bb.bollinger_wband() / 100.0
    result["bb_percent_b"] = bb.bollinger_pband()
    result["vroc"] = volume.pct_change(10).replace([np.inf, -np.inf], np.nan)
    result["sma_spread_ratio"] = (sma10 / sma50.replace(0, np.nan)) - 1.0
    result["atr"] = atr  # retained only for barrier construction / trade risk; not a model feature
    result["close"] = close  # retained only for barriers / display; not a model feature
    return result.replace([np.inf, -np.inf], np.nan)


def triple_barrier_labels(frame: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """1 when the +1.5 ATR barrier occurs before the -1.0 ATR barrier within 5 sessions."""
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    for i in range(len(frame) - horizon):
        entry, atr = frame["close"].iloc[i], frame["atr"].iloc[i]
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue
        upper, lower = entry + 1.5 * atr, entry - 1.0 * atr
        future = frame.iloc[i + 1 : i + horizon + 1]
        for _, candle in future.iterrows():
            if candle["High"] >= upper and candle["Low"] <= lower:
                labels.iloc[i] = 0
                break
            if candle["High"] >= upper:
                labels.iloc[i] = 1
                break
            if candle["Low"] <= lower:
                labels.iloc[i] = 0
                break
        else:
            labels.iloc[i] = 0
    return labels


def make_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    engineered = feature_frame(raw)
    engineered[["High", "Low"]] = raw[["High", "Low"]]
    engineered["target"] = triple_barrier_labels(engineered)
    clean = engineered.dropna(subset=FEATURES + ["target"])
    return clean[FEATURES], clean["target"].astype(int), engineered


def build_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", learning_rate=0.02, max_depth=4,
        n_estimators=300, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=1, tree_method="hist",
    )


def train_model(ticker: str) -> dict[str, Any]:
    raw = load_history(ticker)
    X, y, engineered = make_dataset(raw)
    if len(X) < 160 or y.nunique() < 2:
        raise ValueError(f"Insufficient balanced labeled history for {ticker}.")
    split_at = int(len(X) * 0.80)
    X_train, X_test, y_train, y_test = X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Time-ordered holdout for {ticker} has only one class; cannot calculate reliable metrics.")
    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    folds = TimeSeriesSplit(n_splits=4)
    cv_f1 = []
    for train_idx, val_idx in folds.split(X_train):
        fold = build_model((y_train.iloc[train_idx] == 0).sum() / max((y_train.iloc[train_idx] == 1).sum(), 1))
        fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        cv_f1.append(f1_score(y_train.iloc[val_idx], (fold.predict_proba(X_train.iloc[val_idx])[:, 1] >= 0.5).astype(int), zero_division=0))
    model = build_model(class_weight)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "precision": precision_score(y_test, prediction, zero_division=0), "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0), "roc_auc": roc_auc_score(y_test, probability),
        "accuracy": accuracy_score(y_test, prediction), "cv_f1": float(np.mean(cv_f1)), "observations": len(X),
    }
    latest = engineered.dropna(subset=FEATURES).iloc[-1]
    return {"model": model, "metrics": metrics, "latest_features": latest[FEATURES].to_frame().T, "last_price": float(latest["close"]), "atr": float(latest["atr"]), "trained_at": datetime.now().isoformat()}


def dhan_client() -> Any:
    client_id, token = secret("DHAN_CLIENT_ID"), secret("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        raise RuntimeError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be configured in Streamlit secrets.")
    if DhanContext is None:
        raise RuntimeError("dhanhq is not installed. Run: pip install dhanhq")
    return dhanhq(DhanContext(str(client_id), str(token)))


def funds_and_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = dhan_client()
    funds = client.get_fund_limits() or {}
    positions = client.get_positions() or []
    return funds, positions if isinstance(positions, list) else []


def get_daily_pnl(positions: list[dict[str, Any]]) -> float:
    keys = ("realizedProfit", "realizedPnl", "unrealizedProfit", "unrealizedPnl", "dayPnL", "dayPnl")
    return sum(float(p.get(k, 0) or 0) for p in positions for k in keys)


def position_size(capital: float, price: float, risk_pct: float, stop_pct: float, cap_pct: float) -> int:
    risk_amount = capital * risk_pct
    stop_distance = price * stop_pct
    risk_quantity = math.floor(risk_amount / stop_distance) if stop_distance > 0 else 0
    capital_quantity = math.floor((capital * cap_pct) / price) if price > 0 else 0
    return max(0, min(risk_quantity, capital_quantity))


def execute_super_order(ticker: str, quantity: int, price: float, stop_pct: float, target_pct: float) -> Any:
    sid = security_ids().get(ticker)
    if not sid:
        raise RuntimeError(f"No Dhan security ID configured for {ticker}.")
    client = dhan_client()
    return client.place_super_order(
        security_id=str(sid), exchange_segment="NSE_EQ", transaction_type="BUY", quantity=quantity,
        order_type="MARKET", product_type="INTRADAY", price=0,
        target_price=round(price * (1 + target_pct), 2), stop_loss_price=round(price * (1 - stop_pct), 2),
        correlation_id=f"tb-{ticker.replace('.', '')}-{uuid.uuid4().hex[:12]}",
    )


def run_cycle(tickers: list[str], threshold: float, risk_pct: float, stop_pct: float, target_pct: float, cap_pct: float) -> dict[str, float]:
    try:
        funds, positions = funds_and_positions()
    except Exception as exc:
        add_log(f"Dhan connection error: {exc}")
        return {}
    capital = float(funds.get("availabelBalance", 0) or 0)
    ledger = float(funds.get("sodLimit", capital) or capital)
    daily_pnl = get_daily_pnl(positions)
    if ledger > 0 and daily_pnl <= -(ledger * 0.03):
        add_log(f"CIRCUIT BREAKER: daily P&L ₹{daily_pnl:,.2f} breached 3% of ₹{ledger:,.2f}; orders halted.")
        return {}
    open_ids = {str(p.get("securityId", "")) for p in positions if abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0}
    probabilities: dict[str, float] = {}
    for ticker in tickers:
        try:
            artifact = st.session_state.models.get(ticker) or train_model(ticker)
            st.session_state.models[ticker] = artifact
            prob = float(artifact["model"].predict_proba(artifact["latest_features"])[0, 1])
            probabilities[ticker] = prob
            quantity = position_size(capital, artifact["last_price"], risk_pct, stop_pct, cap_pct)
            add_log(f"{ticker}: P(upper barrier)={prob:.1%}; price ₹{artifact['last_price']:,.2f}; calculated qty={quantity}.")
            sid = security_ids().get(ticker)
            if prob < threshold or quantity < 1 or str(sid) in open_ids:
                continue
            if str(secret("LIVE_TRADING_ENABLED", "false")).lower() != "true":
                add_log(f"{ticker}: signal qualifies, but LIVE_TRADING_ENABLED is false (paper-only).")
                continue
            response = execute_super_order(ticker, quantity, artifact["last_price"], stop_pct, target_pct)
            add_log(f"{ticker}: Dhan Super Order response: {response}")
        except Exception as exc:
            add_log(f"{ticker}: cycle error: {exc}")
    return probabilities


def main() -> None:
    st.session_state.setdefault("models", {})
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_probabilities", {})
    st.title("📈 Quant Barrier Trader")
    with st.sidebar:
        st.header("Execution Controls")
        live_bot = st.toggle("Live Bot", value=False, help="Runs one controlled scan per 60-second dashboard refresh.")
        threshold = st.slider("Model confidence threshold", 0.50, 0.90, 0.70, 0.01, format="%.0f%%")
        risk_pct = st.slider("Risk per trade", 0.005, 0.03, 0.01, 0.005, format="%.1f%%")
        stop_pct = st.slider("Stop-loss", 0.003, 0.05, 0.01, 0.001, format="%.1f%%")
        target_pct = st.slider("Take-profit", 0.005, 0.10, 0.015, 0.001, format="%.1f%%")
        cap_pct = st.slider("Max capital per stock", 0.05, 0.40, 0.20, 0.01, format="%.0f%%")
        tickers = st.multiselect("Target tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
        if st.button("Retrain AI Models"):
            st.session_state.models = {}
            load_history.clear()
            add_log("Model cache cleared; models will retrain from fresh Yahoo Finance history.")
        if st.button("Exit All Positions", type="secondary"):
            try:
                response = dhan_client().exit_all_positions()
                add_log(f"EMERGENCY EXIT response: {response}")
                st.warning("Exit request sent to Dhan. Verify fill status in the broker order book.")
            except Exception as exc:
                st.error(f"Could not exit positions: {exc}")
    try:
        funds, positions = funds_and_positions()
        balance = float(funds.get("availabelBalance", 0) or 0)
        risk_exposed = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) * float(p.get("costPrice", 0) or 0) for p in positions)
        active_count = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0 for p in positions)
    except Exception as exc:
        balance, risk_exposed, active_count = 0.0, 0.0, 0
        add_log(f"Dashboard broker refresh failed: {exc}")
    precision_values = [v["metrics"]["precision"] for v in st.session_state.models.values()]
    cards = st.columns(4)
    cards[0].metric("Dhan ledger balance", f"₹{balance:,.2f}")
    cards[1].metric("Current risk exposed", f"₹{risk_exposed:,.2f}")
    cards[2].metric("Active positions", int(active_count))
    cards[3].metric("Model precision", f"{np.mean(precision_values):.1%}" if precision_values else "—")

    @st.fragment(run_every=60 if live_bot else None)
    def signal_panel() -> None:
        if st.button("Scan Now", type="primary") or live_bot:
            probs = run_cycle(tickers, threshold, risk_pct, stop_pct, target_pct, cap_pct)
            if probs:
                st.session_state.last_probabilities = probs
        st.subheader("Live Signal Meters")
        probabilities = st.session_state.last_probabilities
        if not probabilities:
            st.info("Run a scan to calculate barrier-hit probabilities.")
        else:
            for ticker in tickers:
                p = probabilities.get(ticker)
                if p is not None:
                    st.write(f"{ticker} — **{p:.1%}** P(upper barrier hit first)")
                    st.progress(p)
    signal_panel()
    st.subheader("Model Validation")
    if st.session_state.models:
        report = pd.DataFrame({ticker: data["metrics"] for ticker, data in st.session_state.models.items()}).T
        st.dataframe(report.style.format("{:.2%}", subset=["precision", "recall", "f1", "roc_auc", "accuracy", "cv_f1"]), use_container_width=True)
    st.subheader("Live Execution Console")
    st.code("\n".join(st.session_state.logs) or "No events yet.", language="text")
    st.caption("Target: upper barrier +1.5 × ATR; lower barrier −1.0 × ATR; vertical barrier 5 trading days. Orders use Dhan Super Orders for broker-managed target and stop-loss protection.")


if __name__ == "__main__":
    main()
