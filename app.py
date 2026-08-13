"""
AI-assisted, risk-controlled NSE equity trading dashboard.

Install:
    pip install streamlit yfinance pandas numpy ta scikit-learn xgboost dhanhq plotly feedparser

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

import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
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

# Expanded Top 50+ NSE Equities
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LTIM.NS", "ITC.NS", "HINDUNILVR.NS",
    "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HCLTECH.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "ASIANPAINT.NS", "COALINDIA.NS", "NESTLEIND.NS", "GRASIM.NS",
    "JSWSTEEL.NS", "TECHM.NS", "INDUSINDBK.NS", "ONGC.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "TATACONSUM.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "DIVISLAB.NS", "BPCL.NS",
    "HINDALCO.NS", "BEL.NS", "HAL.NS", "TRENT.NS", "VBL.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS"
]

DEFAULT_SECURITY_IDS = {
    "RELIANCE.NS": "2885", "TCS.NS": "11536", "INFY.NS": "1594", "HDFCBANK.NS": "1333", "ICICIBANK.NS": "4963",
    "BHARTIARTL.NS": "10604", "SBIN.NS": "3045", "ITC.NS": "1660", "HINDUNILVR.NS": "1394", "LT.NS": "11483"
}

FEATURES = [
    "log_return", "rsi_14", "macd_hist_norm", "atr_ratio", 
    "stoch_k", "stoch_d", "bb_width", "bb_percent_b", "vroc", "sma_spread_ratio"
]


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


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"No market data retrieved for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close", "Volume"])


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
    try:
        client = dhan_client()
        funds = client.get_fund_limits() or {}
        positions = client.get_positions() or []
        return funds, positions if isinstance(positions, list) else []
    except Exception:
        return {}, []


def fetch_stock_news(ticker: str) -> list[dict[str, str]]:
    """Fetches real-time stock market news using RSS feed endpoints."""
    clean_ticker = ticker.replace(".NS", "")
    feed_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+NSE+India&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:5]:
        articles.append({"title": entry.title, "link": entry.link, "published": entry.get("published", "")})
    return articles


def execute_cmd(cmd_text: str) -> None:
    parts = cmd_text.strip().split()
    if not parts:
        return
    command = parts[0].lower()

    if command == "/start":
        st.session_state.bot_state = "RUNNING"
        add_log("CMD EXEC: /start -> Trading feed active.")
    elif command == "/stop":
        st.session_state.bot_state = "STOPPED"
        add_log("CMD EXEC: /stop -> Trading feed suspended.")
    elif command == "/risk":
        if len(parts) > 1:
            try:
                val = float(parts[1]) / 100.0
                st.session_state.cmd_risk_pct = max(0.005, min(0.05, val))
                add_log(f"CMD EXEC: /risk -> Risk ratio set to {st.session_state.cmd_risk_pct * 100:.1f}%")
            except ValueError:
                add_log("CMD ERROR: Usage -> /risk [value] (e.g. /risk 1.5)")
        else:
            add_log(f"CMD READ: Current Risk setting = {st.session_state.get('cmd_risk_pct', 0.01) * 100:.1f}%")
    elif command == "/stocks":
        _, positions = funds_and_positions()
        active = [p for p in positions if abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0]
        if not active:
            add_log("CMD READ: /stocks -> No open active positions.")
        else:
            add_log("--- CURRENT BOUGHT POSITIONS ---")
            for p in active:
                sym = p.get("tradingSymbol", p.get("securityId", "Stock"))
                qty = p.get("netQty", p.get("netQuantity", 0))
                price = p.get("costPrice", 0)
                add_log(f"Holdings: {sym} | Qty: {qty} | Entry Price: ₹{float(price):,.2f}")
    else:
        add_log(f"CMD UNKNOWN: Command '{command}' not recognized. Options: /start, /stop, /risk [val], /stocks")


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
    st.session_state.setdefault("bot_state", "STOPPED")
    st.session_state.setdefault("cmd_risk_pct", 0.01)

    st.title("📈 AI-Powered NSE Quant Trading Platform")

    # --- TOP METRICS OVERVIEW ---
    funds, positions = funds_and_positions()
    balance = float(funds.get("availabelBalance", 0) or 0)
    risk_exposed = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) * float(p.get("costPrice", 0) or 0) for p in positions)
    active_count = sum(abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0 for p in positions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available Balance", f"₹{balance:,.2f}")
    c2.metric("Capital Exposed", f"₹{risk_exposed:,.2f}")
    c3.metric("Open Positions", int(active_count))
    c4.metric("Engine Status", st.session_state.bot_state)

    st.divider()

    # --- MAIN NAVIGATION TABS ---
    tab_console, tab_portfolio, tab_models = st.tabs(["💻 Execution & CLI Console", "📊 Portfolio & Insights", "🤖 Model Validation"])

    # -------------------------------------------------------------
    # TAB 1: COMMAND LINE CONSOLE & SIGNALS
    # -------------------------------------------------------------
    with tab_console:
        st.subheader("Interactive CLI Terminal Controls")
        
        # CMD Line Input
        cmd_input = st.text_input("Command Line Terminal", placeholder="Enter command (/start, /stop, /risk 1.5, /stocks)...", key="cli_input")
        if cmd_input:
            execute_cmd(cmd_input)
            st.session_state.cli_input = ""

        # Console Log Stream
        st.code("\n".join(st.session_state.logs) or "Terminal ready. Enter command above.", language="bash")

        st.divider()
        st.subheader("Market Scan Probabilities")
        
        is_active = st.session_state.bot_state == "RUNNING"
        
        @st.fragment(run_every=60 if is_active else None)
        def signal_panel() -> None:
            probabilities = st.session_state.last_probabilities
            if not probabilities:
                st.info("Run scan to view real-time predictions.")
            else:
                for ticker in DEFAULT_TICKERS[:10]:
                    p = probabilities.get(ticker)
                    if p is not None:
                        c_sym, c_bar = st.columns([1, 4])
                        c_sym.write(f"**{ticker}**")
                        c_bar.progress(p, text=f"Hit Probability: {p * 100:.1f}%")

        signal_panel()

    # -------------------------------------------------------------
    # TAB 2: PORTFOLIO, BAR GRAPHS, PREDICTIONS & NEWS
    # -------------------------------------------------------------
    with tab_portfolio:
        st.subheader("Portfolio Performance & Stock Analytics")

        active_positions = [p for p in positions if abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0)) > 0]
        
        if active_positions:
            portfolio_data = []
            for p in active_positions:
                ticker_name = p.get("tradingSymbol", p.get("securityId", "Stock"))
                qty = abs(float(p.get("netQty", p.get("netQuantity", 0)) or 0))
                buy_price = float(p.get("costPrice", 0) or 0)
                tot_val = qty * buy_price
                
                # Fetch ML Prediction if trained
                pred_prob = "N/A"
                if ticker_name in st.session_state.models:
                    pred_prob = f"{st.session_state.models[ticker_name]['metrics']['f1'] * 100:.1f}%"

                portfolio_data.append({
                    "Stock": ticker_name,
                    "Quantity": qty,
                    "Buy Price (₹)": buy_price,
                    "Total Value (₹)": tot_val,
                    "Model Confidence": pred_prob
                })
            
            df_port = pd.DataFrame(portfolio_data)

            # BAR GRAPH: Stock vs Total Value / Quantity Bought
            fig = px.bar(
                df_port, 
                x="Stock", 
                y="Total Value (₹)", 
                color="Quantity", 
                title="Bought Stocks: Volume & Total Valuation Overview",
                text_auto=".2s"
            )
            st.plotly_chart(fig, use_container_width=True)

            # HOLDINGS TABLE
            st.markdown("### Purchased Positions Detail")
            st.dataframe(df_port, use_container_width=True)

        else:
            st.info("No active bought stock positions detected in connected Dhan trading account.")

        st.divider()
        st.subheader("📰 Live Stock News Feed")
        
        selected_stock = st.selectbox("Select Stock for News & Analytics", DEFAULT_TICKERS)
        if selected_stock:
            news_items = fetch_stock_news(selected_stock)
            if news_items:
                for item in news_items:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"Published: {item['published']}")
            else:
                st.write("No active news articles retrieved.")

    # -------------------------------------------------------------
    # TAB 3: MODEL VALIDATION METRICS
    # -------------------------------------------------------------
    with tab_models:
        st.subheader("XGBoost Predictive Metrics")
        if st.session_state.models:
            report = pd.DataFrame({ticker: data["metrics"] for ticker, data in st.session_state.models.items()}).T
            st.dataframe(
                report.style.format("{:.1%}", subset=["precision", "recall", "f1", "roc_auc", "accuracy", "cv_f1"]),
                use_container_width=True,
            )
        else:
            st.info("No trained models found. Models will train when the CLI engine runs scan operations.")


if __name__ == "__main__":
    main()
