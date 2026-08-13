"""AI-assisted, risk-controlled NSE equity trading dashboard with Dhan Real-Time Sync.

Install: pip install streamlit yfinance pandas numpy ta scikit-learn xgboost dhanhq plotly
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


st.set_page_config(page_title="Quant AI Terminal Pro", page_icon="⚡", layout="wide")

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
DEFAULT_SECURITY_IDS = {"RELIANCE.NS": "2885", "TCS.NS": "11536", "INFY.NS": "1594", "HDFCBANK.NS": "1333", "ICICIBANK.NS": "4963"}
FEATURES = ["log_return", "rsi_14", "macd_hist_norm", "atr_ratio", "stoch_k", "stoch_d", "bb_width", "bb_percent_b", "vroc", "sma_spread_ratio"]


def secret(name: str, default: Any = None) -> Any:
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


def dhan_client() -> Any:
    client_id, token = secret("DHAN_CLIENT_ID"), secret("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        raise RuntimeError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be configured in Streamlit secrets.")
    if DhanContext is None:
        raise RuntimeError("dhanhq is not installed. Run: pip install dhanhq")
    return dhanhq(DhanContext(str(client_id), str(token)))


@st.cache_data(ttl=300, show_spinner=False)
def load_history_with_dhan_sync(ticker: str) -> pd.DataFrame:
    """Downloads historical data & syncs latest live quotes directly from Dhan API."""
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"No daily data found for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Synchronize latest tick from Dhan if credentials are available
    sid = security_ids().get(ticker)
    if sid:
        try:
            client = dhan_client()
            # Try fetching live market quote for Dhan real-time sync
            quote = client.get_market_quote(securities={"NSE_EQ": [int(sid)]})
            if quote and "data" in quote and str(sid) in quote["data"]:
                live_data = quote["data"][str(sid)]
                last_price = float(live_data.get("last_price", 0))
                if last_price > 0:
                    today = pd.Timestamp.now().strftime("%Y-%m-%d")
                    if today in df.index:
                        df.loc[today, "Close"] = last_price
                        df.loc[today, "High"] = max(df.loc[today, "High"], last_price)
                        df.loc[today, "Low"] = min(df.loc[today, "Low"], last_price)
                    else:
                        new_row = pd.DataFrame([{
                            "Open": float(live_data.get("open", last_price)),
                            "High": float(live_data.get("high", last_price)),
                            "Low": float(live_data.get("low", last_price)),
                            "Close": last_price,
                            "Volume": float(live_data.get("volume", 0))
                        }], index=[pd.Timestamp(today)])
                        df = pd.concat([df, new_row])
        except Exception:
            pass  # Fall back to pure YFinance data cleanly if Dhan Quote API is unavailable
    return df


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
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
    raw = load_history_with_dhan_sync(ticker)
    X, y, engineered = make_dataset(raw)
    if len(X) < 160 or y.nunique() < 2:
        raise ValueError(f"Insufficient labeled history for {ticker}.")
    split_at = int(len(X) * 0.80)
    X_train, X_test, y_train, y_test = X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]
    
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
        "raw_latest": latest,
        "trained_at": datetime.now().isoformat()
    }


def explain_movement(latest: pd.Series) -> list[str]:
    """Generates human-readable explanations of why the AI evaluates the stock up or down."""
    reasons = []
    rsi = latest.get("rsi_14", 0.5) * 100
    macd_norm = latest.get("macd_hist_norm", 0)
    sma_spread = latest.get("sma_spread_ratio", 0)
    vroc = latest.get("vroc", 0)
    bb_b = latest.get("bb_percent_b", 0.5)

    if rsi > 70:
        reasons.append("⚠️ **Overbought Momentum**: RSI is above 70, signaling short-term exhaustion risk.")
    elif rsi < 30:
        reasons.append("🟢 **Oversold Bounce Potential**: RSI is below 30, signaling deep oversold conditions.")
    else:
        reasons.append(f"🔵 **Balanced Momentum**: RSI sits healthy at {rsi:.1f}.")

    if macd_norm > 0:
        reasons.append("🟢 **Positive MACD Spread**: Short-term trend momentum is accelerating upward.")
    else:
        reasons.append("🔴 **Negative MACD Spread**: Bearish histogram divergence detected.")

    if sma_spread > 0:
        reasons.append("🟢 **Golden Trend Structure**: 10-day Moving Average is above 50-day MA.")
    else:
        reasons.append("🔴 **Death Cross Alignment**: 10-day MA is trading below 50-day MA.")

    if vroc > 0.15:
        reasons.append("⚡ **Institutional Volume Surge**: Volume Rate-of-Change expanded by >15%.")
    
    if bb_b > 1.0:
        reasons.append("📈 **Upper Bollinger Band Breakout**: Price extended beyond volatility channels.")
    elif bb_b < 0.0:
        reasons.append("📉 **Lower Bollinger Band Breakdown**: Price pierced lower volatility boundary.")

    return reasons


def funds_and_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        client = dhan_client()
        funds = client.get_fund_limits() or {}
        positions = client.get_positions() or []
        return funds, positions if isinstance(positions, list) else []
    except Exception:
        return {}, []


def position_size(capital: float, price: float, risk_pct: float, stop_pct: float, cap_pct: float) -> tuple[int, float, float]:
    risk_amount = capital * risk_pct
    stop_distance = price * stop_pct
    risk_quantity = math.floor(risk_amount / stop_distance) if stop_distance > 0 else 0
    capital_quantity = math.floor((capital * cap_pct) / price) if price > 0 else 0
    qty = max(0, min(risk_quantity, capital_quantity))
    max_loss = qty * stop_distance
    total_cost = qty * price
    return qty, max_loss, total_cost


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
    funds, positions = funds_and_positions()
    capital = float(funds.get("availabelBalance", 100000) or 100000)
    probabilities: dict[str, float] = {}
    
    for ticker in tickers:
        try:
            artifact = st.session_state.models.get(ticker) or train_model(ticker)
            st.session_state.models[ticker] = artifact
            prob = float(artifact["model"].predict_proba(artifact["latest_features"])[0, 1])
            probabilities[ticker] = prob
            qty, max_loss, cost = position_size(capital, artifact["last_price"], risk_pct, stop_pct, cap_pct)
            
            add_log(f"{ticker}: P(Barrier)={prob:.1%} | Price: ₹{artifact['last_price']:,.2f} | Size: {qty} shares (Max Risk: ₹{max_loss:,.2f})")
            
            sid = security_ids().get(ticker)
            if prob < threshold or qty < 1:
                continue
            if str(secret("LIVE_TRADING_ENABLED", "false")).lower() != "true":
                add_log(f"{ticker}: ⚡ High confidence signal ({prob:.1%}), but LIVE_TRADING_ENABLED is false (Paper Mode).")
                continue
            
            response = execute_super_order(ticker, qty, artifact["last_price"], stop_pct, target_pct)
            add_log(f"🚀 {ticker}: Dhan Super Order Sent! Response: {response}")
        except Exception as exc:
            add_log(f"❌ {ticker} cycle error: {exc}")
    return probabilities


def process_console_command(cmd: str, tickers: list[str], threshold: float, risk_pct: float, stop_pct: float, target_pct: float, cap_pct: float):
    c = cmd.strip().lower()
    add_log(f"💻 Executed Console Command: {cmd}")
    if c == "/start":
        st.session_state.live_bot_active = True
        add_log("🟢 Live Bot Engine Started.")
    elif c == "/stop":
        st.session_state.live_bot_active = False
        add_log("🔴 Live Bot Engine Halted.")
    elif c == "/scan":
        probs = run_cycle(tickers, threshold, risk_pct, stop_pct, target_pct, cap_pct)
        st.session_state.last_probabilities = probs
        add_log("🔎 Market Scan Complete.")
    elif c == "/retrain":
        st.session_state.models = {}
        load_history_with_dhan_sync.clear()
        add_log("🔄 Cleared AI model cache. Retraining models...")
    elif c == "/exit":
        try:
            res = dhan_client().exit_all_positions()
            add_log(f"🚨 EMERGENCY EXIT ALL: {res}")
        except Exception as e:
            add_log(f"❌ Exit Error: {e}")
    elif c == "/status":
        add_log(f"Status: Bot Active={st.session_state.get('live_bot_active', False)} | Models Cached={len(st.session_state.get('models', {}))}")
    else:
        add_log("⚠️ Unknown command. Available: /start, /stop, /scan, /retrain, /exit, /status")


def main() -> None:
    st.session_state.setdefault("models", {})
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_probabilities", {})
    st.session_state.setdefault("live_bot_active", False)

    # Styling header
    st.markdown("<h1 style='text-align: center;'>⚡ QUANT AI TERMINAL PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Real-Time Dhan Sync | XGBoost Triple-Barrier Classifier | Automated Risk Sizing</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Control Engine")
        live_bot = st.toggle("Live Bot Active", value=st.session_state.live_bot_active)
        st.session_state.live_bot_active = live_bot
        
        st.subheader("🎯 Strategy Thresholds")
        threshold = st.slider("Model Confidence Threshold", 0.50, 0.90, 0.70, 0.01, format="%.0f%%")
        
        st.subheader("🛡️ Risk Management Parameters")
        risk_pct = st.slider("Risk Per Trade (% Ledger)", 0.005, 0.03, 0.01, 0.005, format="%.1f%%")
        stop_pct = st.slider("Stop-Loss Distance", 0.003, 0.05, 0.01, 0.001, format="%.1f%%")
        target_pct = st.slider("Take-Profit Target", 0.005, 0.10, 0.015, 0.001, format="%.1f%%")
        cap_pct = st.slider("Max Allocation Per Stock", 0.05, 0.40, 0.20, 0.01, format="%.0f%%")
        
        tickers = st.multiselect("Active Watchlist", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
        
        if st.button("🔄 Retrain AI Models"):
            st.session_state.models = {}
            load_history_with_dhan_sync.clear()
            add_log("Models purged. Retraining initialized.")
            
        if st.button("🚨 Emergency Exit All Positions", type="secondary"):
            try:
                response = dhan_client().exit_all_positions()
                add_log(f"EMERGENCY EXIT EXECUTED: {response}")
            except Exception as exc:
                add_log(f"Emergency exit failed: {exc}")

    # Top Metrics Row
    funds, positions = funds_and_positions()
    balance = float(funds.get("availabelBalance", 0) or 0)
    risk_exposed = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) * float(p.get("costPrice", 0) or 0) for p in positions)
    active_count = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0 for p in positions)
    precision_values = [v["metrics"]["precision"] for v in st.session_state.models.values()]

    cards = st.columns(4)
    cards[0].metric("Dhan Ledger Balance", f"₹{balance:,.2f}")
    cards[1].metric("Capital at Risk", f"₹{risk_exposed:,.2f}")
    cards[2].metric("Open Positions", int(active_count))
    cards[3].metric("Avg Model Precision", f"{np.mean(precision_values):.1%}" if precision_values else "—")

    st.markdown("---")

    # Signal & Analysis Panel
    @st.fragment(run_every=60 if st.session_state.live_bot_active else None)
    def live_signal_panel():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Live Probability Gauges & Market Reasoning")
            if st.button("🔍 Scan Watchlist Now", type="primary") or st.session_state.live_bot_active:
                probs = run_cycle(tickers, threshold, risk_pct, stop_pct, target_pct, cap_pct)
                if probs:
                    st.session_state.last_probabilities = probs

            probs = st.session_state.last_probabilities
            if not probs:
                st.info("Click 'Scan Watchlist Now' or enable Live Bot to evaluate trade probability.")
            else:
                for ticker in tickers:
                    p = probs.get(ticker)
                    if p is not None:
                        artifact = st.session_state.models.get(ticker)
                        price = artifact["last_price"] if artifact else 0
                        
                        st.write(f"### {ticker} — ₹{price:,.2f}")
                        st.progress(p)
                        st.caption(f"Target Hit Probability: **{p:.1%}** | Threshold Required: {threshold:.0%}")
                        
                        # Show Stock Movement Analysis ("Why is it going Up/Down?")
                        if artifact and "raw_latest" in artifact:
                            with st.expander(f"🔎 AI Technical Breakdown: Why is {ticker} moving?"):
                                reasons = explain_movement(artifact["raw_latest"])
                                for r in reasons:
                                    st.markdown(f"- {r}")

        with col2:
            st.subheader("🛡️ Risk & Position Calculator")
            selected_ticker = st.selectbox("Simulate Trade Risk", tickers)
            if selected_ticker in st.session_state.models:
                art = st.session_state.models[selected_ticker]
                px = art["last_price"]
                qty, max_loss, total_cost = position_size(balance if balance > 0 else 100000, px, risk_pct, stop_pct, cap_pct)
                
                st.write(f"**Current Price:** ₹{px:,.2f}")
                st.write(f"**Max Order Size:** {qty} shares")
                st.write(f"**Total Capital Required:** ₹{total_cost:,.2f}")
                st.write(f"**Max Drawdown Risk:** ₹{max_loss:,.2f}")
                st.write(f"**Target Exit (+{target_pct:.1%}):** ₹{px * (1 + target_pct):,.2f}")
                st.write(f"**Stop-Loss Exit (-{stop_pct:.1%}):** ₹{px * (1 - stop_pct):,.2f}")

    live_signal_panel()

    st.markdown("---")

    # Command Line Console & Logs
    st.subheader("💻 Interactive Terminal Console")
    st.caption("Type slash commands to control the bot: `/start`, `/stop`, `/scan`, `/retrain`, `/exit`, `/status`")
    
    cmd_input = st.text_input("Console Input:", key="console_cmd_input", placeholder="Type command here e.g. /scan and press Enter")
    if cmd_input:
        process_console_command(cmd_input, tickers, threshold, risk_pct, stop_pct, target_pct, cap_pct)

    st.code("\n".join(st.session_state.logs) or "Terminal ready. Execute command or scan to view output.", language="text")

    # Model Analytics Table
    if st.session_state.models:
        st.subheader("📈 Model Training Scoreboard")
        report = pd.DataFrame({ticker: data["metrics"] for ticker, data in st.session_state.models.items()}).T
        st.dataframe(report.style.format("{:.2%}", subset=["precision", "recall", "f1", "roc_auc", "accuracy", "cv_f1"]), use_container_width=True)


if __name__ == "__main__":
    main()
