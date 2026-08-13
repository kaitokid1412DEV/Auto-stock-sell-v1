"""
AI-assisted, risk-controlled NSE equity trading dashboard.

Install:
    pip install streamlit yfinance pandas numpy ta scikit-learn xgboost dhanhq

Secrets required:
    APP_PASSWORD, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
Optional secret:
    LIVE_TRADING_ENABLED=true (otherwise signals are paper-only).

Run:
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
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
DEFAULT_SECURITY_IDS = {
    "RELIANCE.NS": "2885",
    "TCS.NS": "11536",
    "INFY.NS": "1594",
    "HDFCBANK.NS": "1333",
    "ICICIBANK.NS": "4963",
}
FEATURES = [
    "log_return",
    "rsi_14",
    "macd_hist_norm",
    "atr_ratio",
    "stoch_k",
    "stoch_d",
    "bb_width",
    "bb_percent_b",
    "vroc",
    "sma_spread_ratio",
]


def secret(name: str, default: Any = None) -> Any:
    """Read a Streamlit secret or environment variable securely."""
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
        configured = dict(item.split("=", 1) for item in configured.split(",") if "=" in item)
    return {**DEFAULT_SECURITY_IDS, **dict(configured or {})}


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"No market data retrieved for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Stationary feature generation preventing price-level data leakage."""
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

    result["atr"] = atr
    result["close"] = close
    return result.replace([np.inf, -np.inf], np.nan)


def triple_barrier_labels(frame: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Triple-barrier labelling (+1.5 ATR Target / -1.0 ATR Stop over horizon window)."""
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    for i in range(len(frame) - horizon):
        entry, atr = frame["close"].iloc[i], frame["atr"].iloc[i]
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue

        upper, lower = entry + (1.5 * atr), entry - (1.0 * atr)
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
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.02,
        max_depth=4,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def train_model(ticker: str) -> dict[str, Any]:
    raw = load_history(ticker)
    X, y, engineered = make_dataset(raw)

    if len(X) < 160 or y.nunique() < 2:
        raise ValueError(f"Insufficient history or unbalanced classes for {ticker}.")

    split_at = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Unbalanced dataset splits for {ticker}.")

    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    folds = TimeSeriesSplit(n_splits=4)
    cv_f1 = []

    for train_idx, val_idx in folds.split(X_train):
        fold_weight = (y_train.iloc[train_idx] == 0).sum() / max((y_train.iloc[train_idx] == 1).sum(), 1)
        fold = build_model(fold_weight)
        fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = (fold.predict_proba(X_train.iloc[val_idx])[:, 1] >= 0.5).astype(int)
        cv_f1.append(f1_score(y_train.iloc[val_idx], preds, zero_division=0))

    model = build_model(class_weight)
    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "accuracy": accuracy_score(y_test, prediction),
        "cv_f1": float(np.mean(cv_f1)),
        "observations": len(X),
    }

    latest = engineered.dropna(subset=FEATURES).iloc[-1]
    return {
        "model": model,
        "metrics": metrics,
        "latest_features": latest[FEATURES].to_frame().T,
        "last_price": float(latest["close"]),
        "atr": float(latest["atr"]),
        "trained_at": datetime.now().isoformat(),
    }


def dhan_client() -> Any:
    client_id, token = secret("DHAN_CLIENT_ID"), secret("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        raise RuntimeError("Missing Dhan API credentials in Streamlit secrets.")
    if DhanContext is None:
        raise RuntimeError("dhanhq package not found.")
    return dhanhq(DhanContext(str(client_id), str(token)))


def funds_and_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = dhan_client()
    funds = client.get_fund_limits() or {}
    positions = client.get_positions() or []
    return funds, positions if isinstance(positions, list) else []


def get_daily_pnl(positions: list[dict[str, Any]]) -> float:
    keys = ("realizedProfit", "realizedPnl", "unrealizedProfit", "unrealizedPnl", "dayPnL", "dayPnl")
    return sum(float(p.get(k, 0) or 0) for p in positions for k in keys)


def ATR_position_size(capital: float, price: float, atr: float, risk_pct: float, cap_pct: float) -> int:
    risk_amount = capital * risk_pct
    stop_distance = atr * 1.0

    risk_quantity = math.floor(risk_amount / stop_distance) if stop_distance > 0 else 0
    capital_quantity = math.floor((capital * cap_pct) / price) if price > 0 else 0

    return max(0, min(risk_quantity, capital_quantity))


def execute_super_order(ticker: str, quantity: int, price: float, atr: float) -> Any:
    sid = security_ids().get(ticker)
    if not sid:
        raise RuntimeError(f"No Dhan Security ID mapping found for {ticker}.")
    client = dhan_client()

    target_price = round(price + (1.5 * atr), 2)
    stop_loss_price = round(price - (1.0 * atr), 2)

    return client.place_super_order(
        security_id=str(sid),
        exchange_segment="NSE_EQ",
        transaction_type="BUY",
        quantity=quantity,
        order_type="MARKET",
        product_type="INTRADAY",
        price=0,
        target_price=target_price,
        stop_loss_price=stop_loss_price,
        correlation_id=f"tb-{ticker.replace('.', '')}-{uuid.uuid4().hex[:12]}",
    )


def run_cycle(
    tickers: list[str],
    threshold: float,
    risk_pct: float,
    cap_pct: float,
) -> dict[str, float]:
    try:
        funds, positions = funds_and_positions()
    except Exception as exc:
        add_log(f"Connection Error: {exc}")
        return {}

    capital = float(funds.get("availabelBalance", 0) or 0)
    ledger = float(funds.get("sodLimit", capital) or capital)
    daily_pnl = get_daily_pnl(positions)

    if ledger > 0 and daily_pnl <= -(ledger * 0.03):
        add_log(f"CIRCUIT BREAKER: Daily PnL (₹{daily_pnl:,.2f}) breached 3.0% limit. Execution suspended.")
        return {}

    open_ids = {str(p.get("securityId", "")) for p in positions if abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0}
    probabilities: dict[str, float] = {}

    for ticker in tickers:
        try:
            artifact = st.session_state.models.get(ticker) or train_model(ticker)
            st.session_state.models[ticker] = artifact

            prob = float(artifact["model"].predict_proba(artifact["latest_features"])[0, 1])
            probabilities[ticker] = prob

            price, atr = artifact["last_price"], artifact["atr"]
            quantity = ATR_position_size(capital, price, atr, risk_pct, cap_pct)

            add_log(f"{ticker}: Signal={prob * 100:.1f}% | Price=₹{price:,.2f} | Quantity={quantity}")

            sid = security_ids().get(ticker)
            if prob < threshold or quantity < 1 or str(sid) in open_ids:
                continue

            if str(secret("LIVE_TRADING_ENABLED", "false")).lower() != "true":
                add_log(f"{ticker}: Execution skipped (Paper-Trading Mode).")
                continue

            response = execute_super_order(ticker, quantity, price, atr)
            add_log(f"{ticker}: Dhan Super Order Placed -> {response}")
        except Exception as exc:
            add_log(f"{ticker}: Processing error -> {exc}")

    return probabilities


def login() -> bool:
    required = secret("APP_PASSWORD")
    if not required:
        st.error("Set APP_PASSWORD in Streamlit secrets.")
        return False
    if st.session_state.get("authenticated"):
        return True

    st.title("🔐 Quant Barrier Trader")
    supplied = st.text_input("Password", type="password")
    if st.button("Unlock", type="primary"):
        if hashlib.sha256(supplied.encode()).digest() == hashlib.sha256(str(required).encode()).digest():
            st.session_state.authenticated = True
            st.rerun()
        st.error("Invalid password.")
    return False


def main() -> None:
    if not login():
        return

    st.session_state.setdefault("models", {})
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_probabilities", {})
    st.session_state.setdefault("bot_state", "STOPPED")  # Options: STOPPED, RUNNING, PAUSED

    st.title("📈 Quant Barrier Trader")

    # --- SIDEBAR & RISK PARAMETERS ---
    with st.sidebar:
        st.header("Risk & Parameter Setup")
        
        # FIXED: Proper float formatting strings (%.1f%%) to avoid rounding display bugs like '1%-1%'
        threshold = st.slider("Confidence Threshold", min_value=0.50, max_value=0.90, value=0.70, step=0.01, format="%.1f%%")
        risk_pct = st.slider("Risk Per Trade", min_value=0.005, max_value=0.030, value=0.010, step=0.005, format="%.1f%%")
        cap_pct = st.slider("Max Capital Allocation", min_value=0.05, max_value=0.40, value=0.20, step=0.01, format="%.1f%%")
        
        tickers = st.multiselect("Target Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)

        st.divider()
        if st.button("Retrain AI Models", use_container_width=True):
            st.session_state.models = {}
            load_history.clear()
            add_log("Models reset. Retraining from historical market feed...")

        if st.button("Exit All Positions", type="secondary", use_container_width=True):
            try:
                response = dhan_client().exit_all_positions()
                add_log(f"EMERGENCY EXIT EXECUTED: {response}")
                st.warning("Position exit requests transmitted.")
            except Exception as exc:
                st.error(f"Failed to terminate positions: {exc}")

    # --- PORTFOLIO OVERVIEW CARDS ---
    try:
        funds, positions = funds_and_positions()
        balance = float(funds.get("availabelBalance", 0) or 0)
        risk_exposed = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) * float(p.get("costPrice", 0) or 0) for p in positions)
        active_count = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0 for p in positions)
    except Exception as exc:
        balance, risk_exposed, active_count = 0.0, 0.0, 0
        add_log(f"Portfolio metrics error: {exc}")

    precision_values = [v["metrics"]["precision"] for v in st.session_state.models.values()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available Balance", f"₹{balance:,.2f}")
    c2.metric("Capital Exposed", f"₹{risk_exposed:,.2f}")
    c3.metric("Open Positions", int(active_count))
    c4.metric("Avg Model Precision", f"{np.mean(precision_values) * 100:.1f}%" if precision_values else "—")

    st.divider()

    # --- USER FEED EXECUTION CONTROL CONSOLE ---
    st.subheader("🤖 Live User Feed & Bot Console")
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])

    with ctrl_col1:
        if st.button("▶️ Start Feed", type="primary", use_container_width=True):
            st.session_state.bot_state = "RUNNING"
            add_log("USER FEED STARTED: Automated scan cycle activated.")
            st.rerun()

    with ctrl_col2:
        if st.button("⏸️ Pause Feed", use_container_width=True):
            st.session_state.bot_state = "PAUSED"
            add_log("USER FEED PAUSED: Automated scans suspended.")
            st.rerun()

    with ctrl_col3:
        if st.button("⏹️ Stop Feed", use_container_width=True):
            st.session_state.bot_state = "STOPPED"
            add_log("USER FEED STOPPED: Bot turned off.")
            st.rerun()

    with ctrl_col4:
        state = st.session_state.bot_state
        if state == "RUNNING":
            st.success("STATUS: **ACTIVE (Scanning Every 60s)**")
        elif state == "PAUSED":
            st.warning("STATUS: **PAUSED (Awaiting Resume)**")
        else:
            st.error("STATUS: **STOPPED (Feed Offline)**")

    # --- SIGNAL METER (AUTO REFRESH VIA FRAGMENT) ---
    is_active = st.session_state.bot_state == "RUNNING"

    @st.fragment(run_every=60 if is_active else None)
    def signal_panel() -> None:
        manual_scan = st.button("🔍 Manual Scan Now")
        
        if manual_scan or is_active:
            if st.session_state.bot_state != "PAUSED" or manual_scan:
                probs = run_cycle(tickers, threshold, risk_pct, cap_pct)
                if probs:
                    st.session_state.last_probabilities = probs

        st.markdown("#### Real-time Barrier Hit Probabilities")
        probabilities = st.session_state.last_probabilities
        if not probabilities:
            st.info("No scan data available. Start the feed or run a manual scan.")
        else:
            for ticker in tickers:
                p = probabilities.get(ticker)
                if p is not None:
                    col_t, col_p = st.columns([1, 4])
                    col_t.write(f"**{ticker}**")
                    col_p.progress(p, text=f"P(Target Barrier Hit): {p * 100:.1f}%")

    signal_panel()

    st.divider()

    # --- MODEL VALIDATION REPORT ---
    st.subheader("📊 AI Model Cross-Validation Metrics")
    if st.session_state.models:
        report = pd.DataFrame({ticker: data["metrics"] for ticker, data in st.session_state.models.items()}).T
        # FIXED: Proper percentage formatting across dataframe columns
        st.dataframe(
            report.style.format("{:.1%}", subset=["precision", "recall", "f1", "roc_auc", "accuracy", "cv_f1"]),
            use_container_width=True,
        )
    else:
        st.info("Models will automatically train when the feed starts or a manual scan is triggered.")

    # --- EXECUTION LOGS CONSOLE ---
    st.subheader("🖥️ Execution Console Log")
    st.code("\n".join(st.session_state.logs) or "No active events logged.", language="text")
    st.caption("Target Barrier: +1.5×ATR | Stop Barrier: -1.0×ATR | Horizon: 5 Sessions. Orders routed via Dhan Super Orders.")


if __name__ == "__main__":
    main()
