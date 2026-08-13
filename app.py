from __future__ import annotations

import hashlib
import io
import math
import os
import sys
import uuid
from datetime import datetime
from typing import Any
import xml.etree.ElementTree as ET

import feedparser
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


st.set_page_config(page_title="Quant Barrier & Sentiment Trader", page_icon="📈", layout="wide")

# Expanded Market Tickers (50 US + 50 Indian)
TOP_50_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "LLY",
    "JPM", "WMT", "V", "XOM", "UNH", "MA", "ORCL", "PG", "COST", "HD",
    "JNJ", "BAC", "ABBV", "KO", "MRK", "NFLX", "CVX", "CRM", "AMD", "PEP",
    "TMSO", "LIN", "TMO", "ACN", "CSCO", "DIS", "MCD", "ABT", "DHR", "INTC",
    "TXN", "PM", "VZ", "AMGN", "IBM", "PFE", "UNP", "LOW", "SPGI", "HON"
]

TOP_50_INDIA = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "HINDUNILVR.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "ULTRACEMCO.NS", "TITAN.NS",
    "POWERGRID.NS", "TATASTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS", "WIPRO.NS", "ASIANPAINT.NS", "ONGC.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS",
    "TATAMOTORS.NS", "ADANIPORTS.NS", "NESTLEIND.NS", "GRASIM.NS", "TECHM.NS", "SBILIFE.NS", "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS", "HDFCLIFE.NS",
    "BRITANNIA.NS", "BPCL.NS", "HINDALCO.NS", "TATACONSUM.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "BEL.NS"
]

DEFAULT_TICKERS = TOP_50_US[:5] + TOP_50_INDIA[:5]

DEFAULT_SECURITY_IDS = {
    "RELIANCE.NS": "2885", "TCS.NS": "11536", "INFY.NS": "1594", 
    "HDFCBANK.NS": "1333", "ICICIBANK.NS": "4963"
}

FEATURES = [
    "log_return", "rsi_14", "macd_hist_norm", "atr_ratio", 
    "stoch_k", "stoch_d", "bb_width", "bb_percent_b", "vroc", 
    "sma_spread_ratio", "news_sentiment"
]


def secret(name: str, default: Any = None) -> Any:
    """Read a Streamlit secret or environment variable."""
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_news(ticker: str) -> list[dict[str, str]]:
    """Fetch live Google News RSS feed for given ticker."""
    query_ticker = ticker.replace(".NS", "")
    rss_url = f"https://news.google.com/rss/search?q={query_ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    news_items = []
    
    POSITIVE_WORDS = {"up", "gain", "bull", "surge", "growth", "profit", "buy", "high", "beat", "positive", "record"}
    NEGATIVE_WORDS = {"down", "loss", "bear", "drop", "fall", "slump", "sell", "low", "miss", "negative", "warn"}

    for entry in feed.entries[:8]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")
        
        # Primitive sentiment analysis (lexicon based)
        words = set(title.lower().split() + summary.lower().split())
        pos_score = len(words.intersection(POSITIVE_WORDS))
        neg_score = len(words.intersection(NEGATIVE_WORDS))
        
        score = 0.0
        if pos_score + neg_score > 0:
            score = (pos_score - neg_score) / (pos_score + neg_score)
            
        news_items.append({
            "title": title,
            "link": entry.get("link", "#"),
            "published": published,
            "score": score
        })
    return news_items


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError(f"Yahoo Finance returned no daily data for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def feature_frame(raw: pd.DataFrame, news_sentiment_score: float = 0.0) -> pd.DataFrame:
    """Stationary inputs only; absolute price levels never enter model."""
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
    
    # News sentiment features
    result["news_sentiment"] = news_sentiment_score
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


def make_dataset(raw: pd.DataFrame, news_score: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    engineered = feature_frame(raw, news_sentiment_score=news_score)
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
    news = fetch_ticker_news(ticker)
    news_score = np.mean([item["score"] for item in news]) if news else 0.0
    
    X, y, engineered = make_dataset(raw, news_score)
    if len(X) < 160 or y.nunique() < 2:
        raise ValueError(f"Insufficient balanced labeled history for {ticker}.")
        
    split_at = int(len(X) * 0.80)
    X_train, X_test, y_train, y_test = X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]
    
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Time-ordered holdout for {ticker} has only one class.")
        
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
        "model": model, "metrics": metrics, "latest_features": latest[FEATURES].to_frame().T,
        "last_price": float(latest["close"]), "atr": float(latest["atr"]), 
        "news": news, "news_score": news_score, "trained_at": datetime.now().isoformat()
    }


def dhan_client() -> Any:
    client_id, token = secret("DHAN_CLIENT_ID"), secret("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        raise RuntimeError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be configured.")
    if DhanContext is None:
        raise RuntimeError("dhanhq package not installed.")
    return dhanhq(DhanContext(str(client_id), str(token)))


def funds_and_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        client = dhan_client()
        funds = client.get_fund_limits() or {}
        positions = client.get_positions() or []
        return funds, positions if isinstance(positions, list) else []
    except Exception:
        return {}, []


def position_size(capital: float, price: float, risk_pct: float, stop_pct: float, cap_pct: float) -> int:
    risk_amount = capital * risk_pct
    stop_distance = price * stop_pct
    risk_quantity = math.floor(risk_amount / stop_distance) if stop_distance > 0 else 0
    capital_quantity = math.floor((capital * cap_pct) / price) if price > 0 else 0
    return max(0, min(risk_quantity, capital_quantity))


def run_cycle(tickers: list[str], threshold: float, risk_pct: float, stop_pct: float, target_pct: float, cap_pct: float) -> dict[str, float]:
    funds, positions = funds_and_positions()
    capital = float(funds.get("availabelBalance", 100000) or 100000) # Default fallback paper capital
    probabilities: dict[str, float] = {}
    
    for ticker in tickers:
        try:
            artifact = st.session_state.models.get(ticker) or train_model(ticker)
            st.session_state.models[ticker] = artifact
            prob = float(artifact["model"].predict_proba(artifact["latest_features"])[0, 1])
            probabilities[ticker] = prob
            quantity = position_size(capital, artifact["last_price"], risk_pct, stop_pct, cap_pct)
            add_log(f"{ticker}: P(Barrier)={prob:.1%} | News Score={artifact['news_score']:.2f} | Price=${artifact['last_price']:,.2f}")
        except Exception as exc:
            add_log(f"{ticker}: Execution Error: {exc}")
    return probabilities


def login() -> bool:
    required = secret("APP_PASSWORD")
    if not required:
        st.error("APP_PASSWORD must be set in Streamlit secrets before application can be unlocked.")
        return False
    if st.session_state.get("authenticated"):
        return True
    st.title("🔐 Quant Barrier & Sentiment Dashboard")
    supplied = st.text_input("Application password", type="password")
    if st.button("Unlock", type="primary"):
        if hashlib.sha256(supplied.encode()).digest() == hashlib.sha256(str(required).encode()).digest():
            st.session_state.authenticated = True
            st.rerun()
        st.error("Invalid password.")
    return False


def execute_cmd(command: str) -> str:
    """Evaluates Python/Terminal commands inside session scope."""
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        if command.startswith("!"):
            # Shell pass-through command
            import subprocess
            res = subprocess.run(command[1:], shell=True, capture_output=True, text=True)
            return res.stdout + res.stderr
        else:
            # Python expression or block evaluation
            exec_globals = {
                "st": st, "pd": pd, "np": np, "yf": yf,
                "models": st.session_state.get("models", {}),
                "logs": st.session_state.get("logs", []),
                "run_cycle": run_cycle
            }
            exec(command, exec_globals)
            return buffer.getvalue() or "Command executed successfully."
    except Exception as e:
        return f"Error executing command: {e}"
    finally:
        sys.stdout = sys.__stdout__


def main() -> None:
    if not login():
        return
        
    st.session_state.setdefault("models", {})
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_probabilities", {})
    st.session_state.setdefault("cmd_history", [])

    st.title("📈 Quant Barrier & News-Assisted Trader")
    
    with st.sidebar:
        st.header("Execution Controls")
        live_bot = st.toggle("Live Bot Scan", value=False)
        threshold = st.slider("Model confidence threshold", 0.50, 0.90, 0.65, 0.01, format="%.0f%%")
        risk_pct = st.slider("Risk per trade", 0.005, 0.03, 0.01, 0.005, format="%.1f%%")
        stop_pct = st.slider("Stop-loss", 0.003, 0.05, 0.01, 0.001, format="%.1f%%")
        target_pct = st.slider("Take-profit", 0.005, 0.10, 0.015, 0.001, format="%.1f%%")
        cap_pct = st.slider("Max capital per stock", 0.05, 0.40, 0.20, 0.01, format="%.0f%%")
        
        market_choice = st.radio("Select Universe Quick-List:", ["Default Mixed", "Top 50 US", "Top 50 India"])
        if market_choice == "Top 50 US":
            active_defaults = TOP_50_US
        elif market_choice == "Top 50 India":
            active_defaults = TOP_50_INDIA
        else:
            active_defaults = DEFAULT_TICKERS

        tickers = st.multiselect("Target Tickers", active_defaults + TOP_50_US + TOP_50_INDIA, default=active_defaults[:5])
        
        if st.button("Retrain AI Models"):
            st.session_state.models = {}
            load_history.clear()
            fetch_ticker_news.clear()
            add_log("Models & news cache reset; retraining initiated.")

    # High-level Metrics Cards
    precision_values = [v["metrics"]["precision"] for v in st.session_state.models.values()]
    cards = st.columns(4)
    cards[0].metric("Monitored Tickers", len(tickers))
    cards[1].metric("Active Trained Models", len(st.session_state.models))
    cards[2].metric("Average Precision", f"{np.mean(precision_values):.1%}" if precision_values else "—")
    cards[3].metric("Engine Status", "Active" if live_bot else "Standby")

    # Main Tabs Area
    tab_dashboard, tab_news, tab_console = st.tabs(["📊 Prediction Signals", "📰 Live News & Sentiment", "💻 Interactive Console"])

    with tab_dashboard:
        if st.button("Scan Tickers Now", type="primary") or live_bot:
            probs = run_cycle(tickers, threshold, risk_pct, stop_pct, target_pct, cap_pct)
            if probs:
                st.session_state.last_probabilities = probs
        
        probabilities = st.session_state.last_probabilities
        if not probabilities:
            st.info("Run a scan to calculate barrier-hit predictions and sentiment integration.")
        else:
            for ticker in tickers:
                p = probabilities.get(ticker)
                if p is not None:
                    st.write(f"**{ticker}** — Prediction Confidence: **{p:.1%}**")
                    st.progress(p)

        st.subheader("Model Validation Metrics")
        if st.session_state.models:
            report = pd.DataFrame({ticker: data["metrics"] for ticker, data in st.session_state.models.items()}).T
            st.dataframe(report.style.format("{:.2%}", subset=["precision", "recall", "f1", "roc_auc", "accuracy", "cv_f1"]), use_container_width=True)

    with tab_news:
        st.subheader("Live Financial Feed & Predictions Sentiment Analysis")
        selected_news_ticker = st.selectbox("View news for ticker:", tickers)
        if selected_news_ticker:
            news_items = fetch_ticker_news(selected_news_ticker)
            if news_items:
                avg_score = np.mean([item["score"] for item in news_items])
                st.write(f"**Aggregated Sentiment Score:** `{avg_score:.2f}` (-1.0 Negative to +1.0 Positive)")
                for item in news_items:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"[{item['title']}]({item['link']})")
                        st.caption(f"Published: {item['published']}")
                    with col2:
                        st.write(f"Sentiment: `{item['score']:.2f}`")
                    st.divider()
            else:
                st.info("No current news items found for this ticker.")

    with tab_console:
        st.subheader("Interactive Command Console")
        st.caption("Enter Python expressions or `!shell_command` (e.g., `print(models.keys())` or `!pip list`).")
        
        cmd_input = st.text_input("Terminal Command Input", key="cmd_input")
        if st.button("Execute", type="primary") and cmd_input:
            out = execute_cmd(cmd_input)
            st.session_state.cmd_history.append((cmd_input, out))
        
        if st.session_state.cmd_history:
            st.write("### Output Logs")
            for cmd, out in reversed(st.session_state.cmd_history):
                st.code(f"> {cmd}\n{out}", language="bash")
                
    st.subheader("Live Execution Console")
    st.code("\n".join(st.session_state.logs) or "No execution events yet.", language="text")


if __name__ == "__main__":
    main()
