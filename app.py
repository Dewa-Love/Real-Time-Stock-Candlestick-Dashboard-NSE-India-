import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ta
from fix import fix_ohlc

# -------------------------------- PAGE CONFIG --------------------------------
st.set_page_config(
    page_title="NSE Live Dashboard",
    page_icon="📈",
    layout="wide",
)

# -------------------------------- AUTO REFRESH --------------------------------
refresh_sec = st.sidebar.slider("🔄 Auto-refresh (seconds)", 10, 120, 30)
st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

# -------------------------------- CSS THEMING --------------------------------
st.markdown("""
<style>

/* PAGE BACKGROUND */
body {
    background-color: #0d0f16;
}

/* GLOW EFFECT */
.glow {
    box-shadow: 0px 0px 30px rgba(0, 255, 255, 0.25);
    transition: 0.3s ease;
}
.glow:hover {
    box-shadow: 0px 0px 40px rgba(0, 255, 255, 0.55);
}

/* GRADIENT HEADER */
.title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #00f7ff, #007bff, #6f00ff);
    -webkit-background-clip: text;
    color: transparent;
}

/* SECTION HEADERS */
.section-header {
    font-size: 30px;
    font-weight: 600;
    background: linear-gradient(90deg, #00f7ff, #8b5cf6);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 10px;
}

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(135deg, #151823, #1d2233);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #2e3450;
    color: white;
    transition: 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: #00f7ff;
}

/* TABLE STYLING */
table {
    background-color: #11131a !important;
    color: white !important;
}

/* DIVIDER */
.section {
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #2e3450;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------- TITLE --------------------------------
st.markdown("<div class='title'>🚀 NSE Live Trading Dashboard — Premium Edition</div>", unsafe_allow_html=True)

# -------------------------------- SIDEBAR --------------------------------
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("🔎 NSE Ticker", "HDFCBANK.NS")

timeframe = st.sidebar.selectbox(
    "⏱️ Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"]
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

# -------------------------------- FETCH DATA --------------------------------
df = yf.download(ticker, period=period, interval=timeframe)

if df.empty:
    st.error("❌ Could not load data. Yahoo Finance blocked this request. Try again later.")
    st.stop()

df = fix_ohlc(df)

# -------------------------------- INDICATORS --------------------------------
df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

df["EMA12"] = df["Close"].ewm(span=12).mean()
df["EMA26"] = df["Close"].ewm(span=26).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9).mean()

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# -------------------------------- SIGNALS LOGIC --------------------------------
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

        if ma_up and macd_up:
            df.at[df.index[i], "SignalType"] = 1
            df.at[df.index[i], "SignalReason"] = "BUY Signal — Bullish crossover"

        elif ma_down and macd_down:
            df.at[df.index[i], "SignalType"] = -1
            df.at[df.index[i], "SignalReason"] = "SELL Signal — Bearish crossover"

    except:
        continue

signals = df[df["SignalType"] != 0]

# -------------------------------- METRICS --------------------------------
st.markdown("<div class='section-header'>📌 Key Price Metrics</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='metric-card glow'><h3>₹{df['Close'].iloc[-1]:.2f}</h3><p>Current Price</p></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card glow'><h3>₹{df['Open'].iloc[-1]:.2f}</h3><p>Open</p></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card glow'><h3>₹{df['High'].iloc[-1]:.2f}</h3><p>High</p></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card glow'><h3>₹{df['Low'].iloc[-1]:.2f}</h3><p>Low</p></div>", unsafe_allow_html=True)

# -------------------------------- CANDLE CHART --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🕯 Candlestick Chart + Signals</div>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color="#00ffbf",
    decreasing_line_color="#ff4d6d",
))

fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#00d4ff")))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color="#ffcc00")))

# BUY & SELL markers
buy = df[df["SignalType"] == 1]
sell = df[df["SignalType"] == -1]

fig.add_trace(go.Scatter(
    x=buy.index, y=buy["Low"] * 0.998,
    mode="markers",
    marker=dict(symbol="triangle-up", size=15, color="#00ffbf"),
    name="BUY"
))

fig.add_trace(go.Scatter(
    x=sell.index, y=sell["High"] * 1.002,
    mode="markers",
    marker=dict(symbol="triangle-down", size=15, color="#ff4d6d"),
    name="SELL"
))

fig.update_layout(
    height=600,
    plot_bgcolor="#0d0f16",
    paper_bgcolor="#0d0f16",
    font=dict(color="white"),
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------- RSI --------------------------------
st.markdown("<div class='section-header'>📉 RSI</div>", unsafe_allow_html=True)
st.line_chart(df["RSI"])

# -------------------------------- MACD --------------------------------
st.markdown("<div class='section-header'>📉 MACD</div>", unsafe_allow_html=True)
st.line_chart(df[["MACD", "Signal"]])

# -------------------------------- SIGNAL TABLE --------------------------------
st.markdown("<div class='section-header'>🔔 Signals</div>", unsafe_allow_html=True)
st.dataframe(signals[["Close", "SignalType", "SignalReason"]])

# -------------------------------- RAW DATA --------------------------------
st.markdown("<div class='section-header'>📄 Raw Data (Last 50 rows)</div>", unsafe_allow_html=True)
st.dataframe(df.tail(50))
