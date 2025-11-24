import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ta
from fix import fix_ohlc

st.set_page_config(page_title="NSE Live Trading Dashboard", layout="wide")

st.title("📈 NSE Live Trading Dashboard — Merged Version (Signals + 1-Min Live)")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("NSE Ticker", "HDFCBANK.NS")

timeframe = st.sidebar.selectbox(
    "Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=0
)

period_map = {
    "1m": "1d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "1d": "1y"
}

period = period_map[timeframe]

refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 10, 120, 30)

# Auto-refresh
st.experimental_autorefresh(interval=refresh_sec * 1000, key="refresh")

# --------------------------- FETCH DATA ---------------------------
df = yf.download(ticker, period=period, interval=timeframe)

if df.empty:
    st.error("No data found. Try another ticker.")
    st.stop()

df = fix_ohlc(df)

# --------------------------- INDICATORS ---------------------------
df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

df["EMA12"] = df["Close"].ewm(span=12).mean()
df["EMA26"] = df["Close"].ewm(span=26).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9).mean()

# --------------------------- SIGNALS ---------------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

df["MA_diff"] = df["MA20"] - df["MA50"]
df["MA_diff_prev"] = df["MA_diff"].shift(1)

df["MACD_diff"] = df["MACD"] - df["Signal"]
df["MACD_diff_prev"] = df["MACD_diff"].shift(1)

df["SignalType"] = 0
df["SignalReason"] = ""

for i in range(1, len(df)):
    try:
        ma_up = df["MA_diff_prev"].iat[i] <= 0 and df["MA_diff"].iat[i] > 0
        ma_down = df["MA_diff_prev"].iat[i] >= 0 and df["MA_diff"].iat[i] < 0
        macd_up = df["MACD_diff_prev"].iat[i] <= 0 and df["MACD_diff"].iat[i] > 0
        macd_down = df["MACD_diff_prev"].iat[i] >= 0 and df["MACD_diff"].iat[i] < 0
        rsi_val = df["RSI"].iat[i]

        if ma_up and macd_up and rsi_val < 70:
            df.at[df.index[i], "SignalType"] = 1
            df.at[df.index[i], "SignalReason"] = f"BUY — MA20>MA50 & MACD>Signal & RSI={rsi_val:.1f}"

        elif ma_down and macd_down and rsi_val > 30:
            df.at[df.index[i], "SignalType"] = -1
            df.at[df.index[i], "SignalReason"] = f"SELL — MA20<MA50 & MACD<Signal & RSI={rsi_val:.1f}"

    except:
        continue

signals = df[df["SignalType"] != 0].copy()
signals["Datetime"] = signals.index
signals["Price"] = signals["Close"]

# --------------------------- DASHBOARD ---------------------------
st.subheader("📌 Live Price Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
col2.metric("Open", f"₹{df['Open'].iloc[-1]:.2f}")
col3.metric("High", f"₹{df['High'].iloc[-1]:.2f}")
col4.metric("Low", f"₹{df['Low'].iloc[-1]:.2f}")

# --------------------------- CANDLESTICK WITH SIGNALS ---------------------------
st.subheader("🕯 Candlestick Chart + Signals")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
))

fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA 20"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA 50"))

# Buy markers
buy = df[df["SignalType"] == 1]
fig.add_trace(go.Scatter(
    x=buy.index, y=buy["Low"] * 0.995,
    mode="markers", marker=dict(symbol="triangle-up", size=12),
    name="BUY", text=buy["SignalReason"]
))

# Sell markers
sell = df[df["SignalType"] == -1]
fig.add_trace(go.Scatter(
    x=sell.index, y=sell["High"] * 1.005,
    mode="markers", marker=dict(symbol="triangle-down", size=12),
    name="SELL", text=sell["SignalReason"]
))

fig.update_layout(height=550, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --------------------------- RSI ---------------------------
st.subheader("📉 RSI (14)")
st.line_chart(df["RSI"])

# --------------------------- MACD ---------------------------
st.subheader("📉 MACD")
fig_macd = px.line(df, x=df.index, y=["MACD", "Signal"])
st.plotly_chart(fig_macd, use_container_width=True)

# --------------------------- SIGNAL TABLE ---------------------------
st.subheader("🔔 Latest Signals")
st.dataframe(signals[["Datetime", "Price", "SignalType", "SignalReason"]].tail(15))

# --------------------------- RAW DATA ---------------------------
st.subheader("📄 Raw Data (Last 50 Rows)")
st.dataframe(df.tail(50))