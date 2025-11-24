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

# -------------------------------- CUSTOM CSS --------------------------------
st.markdown("""
<style>

/* GLOBAL DARK THEME */
body {
    background-color: #0e1117;
    color: white;
}

/* HEADERS */
h1, h2, h3, h4 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;
    padding: 20px;
}

/* METRIC CARDS */
.metric-card {
    background-color: #1f2937;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #374151;
}

/* SECTION DIVIDER */
.section {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #2d3748;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------- APP TITLE --------------------------------
st.title("🚀 NSE Live Trading Dashboard — Premium Edition")
st.markdown("### 📌 Live Market Overview with Technical Indicators + Buy/Sell Signals\n")

# -------------------------------- SIDEBAR --------------------------------
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input("🔎 NSE Ticker", "HDFCBANK.NS")

timeframe = st.sidebar.selectbox(
    "⏱️ Interval",
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

refresh_sec = st.sidebar.slider("🔄 Auto-refresh (seconds)", 10, 120, 30)

# NEW Streamlit auto-refresh API
st_autorefresh = st.autorefresh(interval=refresh_sec * 1000, key="refresh")

st.sidebar.markdown("---")
st.sidebar.markdown("📊 *Dashboard updates live based on selected interval.*")

# -------------------------------- FETCH DATA --------------------------------
df = yf.download(ticker, period=period, interval=timeframe)

if df.empty:
    st.error("❌ No data found. Try another stock symbol.")
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

# -------------------------------- SIGNAL LOGIC --------------------------------
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

# -------------------------------- METRICS --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📌 Key Price Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='metric-card'><h3>₹{df['Close'].iloc[-1]:.2f}</h3><p>Current Price</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>₹{df['Open'].iloc[-1]:.2f}</h3><p>Open</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>₹{df['High'].iloc[-1]:.2f}</h3><p>High</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>₹{df['Low'].iloc[-1]:.2f}</h3><p>Low</p></div>", unsafe_allow_html=True)

# -------------------------------- CANDLESTICK --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("🕯 Candlestick Chart + Buy/Sell Signals")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color='#10B981',
    decreasing_line_color='#EF4444'
))

fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA 20", line=dict(color="#3B82F6")))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA 50", line=dict(color="#FACC15")))

buy = df[df["SignalType"] == 1]
sell = df[df["SignalType"] == -1]

fig.add_trace(go.Scatter(
    x=buy.index, y=buy["Low"] * 0.995,
    mode="markers", marker=dict(symbol="triangle-up", size=14, color="#22C55E"),
    name="BUY", text=buy["SignalReason"]
))

fig.add_trace(go.Scatter(
    x=sell.index, y=sell["High"] * 1.005,
    mode="markers", marker=dict(symbol="triangle-down", size=14, color="#EF4444"),
    name="SELL", text=sell["SignalReason"]
))

fig.update_layout(
    height=600,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------- RSI --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📉 RSI (14-period)")
st.line_chart(df["RSI"])

# -------------------------------- MACD --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📉 MACD Indicator")

fig_macd = px.line(df, x=df.index, y=["MACD", "Signal"])
fig_macd.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white")
)
st.plotly_chart(fig_macd, use_container_width=True)

# -------------------------------- SIGNAL TABLE --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("🔔 Recent Buy/Sell Signals")

st.dataframe(signals[["Datetime", "Price", "SignalType", "SignalReason"]].tail(20))

# -------------------------------- RAW DATA --------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📄 Raw OHLC Data")
st.dataframe(df.tail(50))
