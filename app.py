import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ta
from datetime import datetime, timedelta
import numpy as np
from fix import fix_ohlc

# Try to import ML predictor (optional)
ML_AVAILABLE = False
try:
    from ml_predictor import EnsemblePredictor, TrendPredictor, SupportResistancePredictor
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML Predictor not available. Download ml_predictor.py to enable AI predictions.")
    EnsemblePredictor = None
    TrendPredictor = None
    SupportResistancePredictor = None

# -------------------------------- PAGE CONFIG --------------------------------
st.set_page_config(
    page_title="StockPulse - Advanced Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------- CUSTOM CSS --------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0a0e1a;
        color: #e4e7eb;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #151b2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #2dd4bf;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #14b8a6;
    }
    
    /* Header Section */
    .stock-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        margin-bottom: 2rem;
    }
    
    .stock-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .stock-badge {
        display: inline-block;
        padding: 4px 12px;
        background: #065f46;
        color: #10b981;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .stock-meta {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    /* Price Display */
    .price-large {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2dd4bf;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .price-change-positive {
        color: #10b981;
        font-size: 1.125rem;
        font-weight: 600;
    }
    
    .price-change-negative {
        color: #ef4444;
        font-size: 1.125rem;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1e293b;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: #2dd4bf;
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(45, 212, 191, 0.1);
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-subtext {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1e293b;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0f172a;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid #1e293b;
        border-radius: 8px;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        border-color: #14b8a6;
        color: #ffffff;
    }
    
    /* Scorecard */
    .scorecard-item {
        background: #0f172a;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #2dd4bf;
        margin-bottom: 0.75rem;
    }
    
    .scorecard-label {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    
    .scorecard-desc {
        color: #64748b;
        font-size: 0.75rem;
    }
    
    .score-badge {
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        float: right;
    }
    
    .score-good {
        background: #065f46;
        color: #10b981;
    }
    
    .score-average {
        background: #92400e;
        color: #fbbf24;
    }
    
    .score-poor {
        background: #7f1d1d;
        color: #ef4444;
    }
    
    /* Analyst Ratings */
    .rating-bar {
        background: linear-gradient(90deg, #10b981 0%, #fbbf24 50%, #ef4444 100%);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .rating-label {
        display: inline-block;
        text-align: center;
        padding: 0.5rem;
    }
    
    .rating-number {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Tables */
    .dataframe {
        background: #0f172a !important;
        color: #e4e7eb !important;
        border: 1px solid #1e293b !important;
    }
    
    .dataframe thead th {
        background: #1e293b !important;
        color: #2dd4bf !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .dataframe tbody tr:hover {
        background: #1e293b !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(45, 212, 191, 0.2);
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #064e3b;
        border-color: #10b981;
        color: #d1fae5;
    }
    
    .alert-warning {
        background: #78350f;
        border-color: #f59e0b;
        color: #fef3c7;
    }
    
    .alert-info {
        background: #164e63;
        border-color: #06b6d4;
        color: #cffafe;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #2dd4bf !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------- SIDEBAR --------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    ticker = st.text_input("🔍 Stock Symbol", "RELIANCE.NS", help="Enter NSE ticker (e.g., RELIANCE.NS)")
    
    timeframe = st.selectbox(
        "⏱️ Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "1d"],
        index=2
    )
    
    period_map = {
        "1m": "1d", "5m": "5d", "15m": "1mo",
        "30m": "1mo", "1h": "3mo", "1d": "1y"
    }
    period = period_map[timeframe]
    
    st.markdown("---")
    
    refresh_sec = st.slider("🔄 Auto-refresh (seconds)", 10, 300, 60)
    
    st.markdown("---")
    
    show_advanced = st.checkbox("📊 Advanced Indicators", value=True)
    show_volume = st.checkbox("📈 Volume Analysis", value=True)
    
    st.markdown("---")
    st.markdown("### 📌 Watchlist")
    watchlist = st.multiselect(
        "Quick Access",
        ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "SBIN.NS"],
        default=["RELIANCE.NS"]
    )

# Auto-refresh
st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

# -------------------------------- FETCH DATA --------------------------------
@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        return fix_ohlc(df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return {}

df = fetch_stock_data(ticker, period, timeframe)

if df is None or df.empty:
    st.error("❌ Could not load data. Please check the ticker symbol and try again.")
    st.stop()

stock_info = fetch_stock_info(ticker)

# -------------------------------- CALCULATE INDICATORS --------------------------------
# Moving Averages
df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
df["SMA200"] = ta.trend.sma_indicator(df["Close"], window=200)

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["BB_High"] = bollinger.bollinger_hband()
df["BB_Low"] = bollinger.bollinger_lband()
df["BB_Mid"] = bollinger.bollinger_mavg()

# RSI
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

# MACD
df["EMA12"] = df["Close"].ewm(span=12).mean()
df["EMA26"] = df["Close"].ewm(span=26).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# Stochastic
stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
df["Stoch_K"] = stoch.stoch()
df["Stoch_D"] = stoch.stoch_signal()

# ATR (Average True Range)
df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)

# Volume indicators
df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()

# -------------------------------- SIGNAL GENERATION --------------------------------
df["Signal"] = 0
df["Signal_Type"] = ""

for i in range(50, len(df)):
    signals = []
    
    # EMA Crossover
    if df["EMA20"].iloc[i] > df["EMA50"].iloc[i] and df["EMA20"].iloc[i-1] <= df["EMA50"].iloc[i-1]:
        signals.append("BUY")
    elif df["EMA20"].iloc[i] < df["EMA50"].iloc[i] and df["EMA20"].iloc[i-1] >= df["EMA50"].iloc[i-1]:
        signals.append("SELL")
    
    # RSI
    if df["RSI"].iloc[i] < 30:
        signals.append("BUY (Oversold)")
    elif df["RSI"].iloc[i] > 70:
        signals.append("SELL (Overbought)")
    
    # MACD
    if df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i] and df["MACD"].iloc[i-1] <= df["MACD_Signal"].iloc[i-1]:
        signals.append("BUY (MACD)")
    elif df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i] and df["MACD"].iloc[i-1] >= df["MACD_Signal"].iloc[i-1]:
        signals.append("SELL (MACD)")
    
    if signals:
        df.at[df.index[i], "Signal"] = 1 if "BUY" in str(signals) else -1
        df.at[df.index[i], "Signal_Type"] = " | ".join(signals)

# -------------------------------- HEADER SECTION --------------------------------
current_price = df["Close"].iloc[-1]
prev_close = df["Close"].iloc[-2] if len(df) > 1 else current_price
price_change = current_price - prev_close
price_change_pct = (price_change / prev_close) * 100

company_name = stock_info.get("longName", ticker.replace(".NS", ""))

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"""
    <div class='stock-header'>
        <div class='stock-title'>
            📊 {company_name} 
            <span class='stock-badge'>NSE: {ticker.replace('.NS', '')}</span>
        </div>
        <div class='stock-meta'>
            {stock_info.get('sector', 'N/A')} • {stock_info.get('industry', 'N/A')} • Market Cap: ₹{stock_info.get('marketCap', 0)/10000000:.2f} Cr
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    change_class = "price-change-positive" if price_change >= 0 else "price-change-negative"
    arrow = "▲" if price_change >= 0 else "▼"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='price-large'>₹{current_price:.2f}</div>
        <div class='{change_class}'>{arrow} {price_change:.2f} ({price_change_pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------- KEY METRICS --------------------------------
st.markdown("<div class='section-header'>📊 Key Metrics</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics_data = [
    ("P/E Ratio", stock_info.get("trailingPE", "N/A"), f"Sector: {stock_info.get('sector', 'N/A')[:10]}"),
    ("EPS", stock_info.get("trailingEps", "N/A"), "TTM"),
    ("P/B Ratio", stock_info.get("priceToBook", "N/A"), "Book Value"),
    ("Div Yield", f"{stock_info.get('dividendYield', 0)*100:.2f}%" if stock_info.get('dividendYield') else "N/A", "Annual"),
    ("ROE", f"{stock_info.get('returnOnEquity', 0)*100:.2f}%" if stock_info.get('returnOnEquity') else "N/A", "Return"),
    ("52W High", f"₹{stock_info.get('fiftyTwoWeekHigh', 0):.2f}" if stock_info.get('fiftyTwoWeekHigh') else "N/A", "Peak"),
]

for col, (label, value, subtext) in zip([col1, col2, col3, col4, col5, col6], metrics_data):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-subtext'>{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------- TABS LAYOUT --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🎯 Signals", "📰 Company Info", "⚖️ Comparison", "📊 Analytics", "🤖 AI Predictions"])

with tab1:
    st.markdown("### 🕯️ Price Action & Indicators")
    
    # Main candlestick chart
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
    ))
    
    # Add EMAs
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA20"],
        name="EMA 20",
        line=dict(color="#2dd4bf", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"],
        name="EMA 50",
        line=dict(color="#fbbf24", width=2)
    ))
    
    if show_advanced:
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_High"],
            name="BB Upper",
            line=dict(color="#8b5cf6", width=1, dash="dash"),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Low"],
            name="BB Lower",
            line=dict(color="#8b5cf6", width=1, dash="dash"),
            fill='tonexty',
            opacity=0.2
        ))
    
    # Buy/Sell Signals
    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals["Low"] * 0.998,
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="#10b981", line=dict(width=1, color="#ffffff")),
        name="BUY",
        text=buy_signals["Signal_Type"],
        hovertemplate="<b>BUY Signal</b><br>%{text}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals["High"] * 1.002,
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(width=1, color="#ffffff")),
        name="SELL",
        text=sell_signals["Signal_Type"],
        hovertemplate="<b>SELL Signal</b><br>%{text}<extra></extra>"
    ))
    
    fig.update_layout(
        height=600,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e4e7eb", family="Inter"),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    if show_volume:
        st.markdown("### 📊 Volume Analysis")
        
        colors = ['#10b981' if df["Close"].iloc[i] >= df["Open"].iloc[i] else '#ef4444' 
                  for i in range(len(df))]
        
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7
        ))
        
        fig_volume.add_trace(go.Scatter(
            x=df.index,
            y=df["Volume_SMA"],
            name="Volume SMA",
            line=dict(color="#2dd4bf", width=2)
        ))
        
        fig_volume.update_layout(
            height=200,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Volume"),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)

with tab2:
    st.markdown("### 📈 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        st.markdown("#### RSI (14)")
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            fill='tozeroy',
            line=dict(color="#2dd4bf", width=2),
            name="RSI"
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold")
        
        fig_rsi.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 100]),
            showlegend=False
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Current RSI value
        current_rsi = df["RSI"].iloc[-1]
        rsi_status = "Overbought 🔴" if current_rsi > 70 else "Oversold 🟢" if current_rsi < 30 else "Neutral ⚪"
        st.markdown(f"**Current RSI:** {current_rsi:.2f} - {rsi_status}")
    
    with col2:
        # MACD
        st.markdown("#### MACD")
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            line=dict(color="#2dd4bf", width=2),
            name="MACD"
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            line=dict(color="#fbbf24", width=2),
            name="Signal"
        ))
        
        colors = ['#10b981' if val >= 0 else '#ef4444' for val in df["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            marker_color=colors,
            name="Histogram",
            opacity=0.5
        ))
        
        fig_macd.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # MACD status
        macd_status = "Bullish 🟢" if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1] else "Bearish 🔴"
        st.markdown(f"**MACD Status:** {macd_status}")
    
    if show_advanced:
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            # Stochastic Oscillator
            st.markdown("#### Stochastic Oscillator")
            fig_stoch = go.Figure()
            
            fig_stoch.add_trace(go.Scatter(
                x=df.index, y=df["Stoch_K"],
                line=dict(color="#2dd4bf", width=2),
                name="%K"
            ))
            
            fig_stoch.add_trace(go.Scatter(
                x=df.index, y=df["Stoch_D"],
                line=dict(color="#fbbf24", width=2),
                name="%D"
            ))
            
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="#ef4444")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="#10b981")
            
            fig_stoch.update_layout(
                height=250,
                plot_bgcolor="#0a0e1a",
                paper_bgcolor="#0a0e1a",
                font=dict(color="#e4e7eb"),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", range=[0, 100]),
                showlegend=True
            )
            
            st.plotly_chart(fig_stoch, use_container_width=True)
        
        with col4:
            # ATR (Volatility)
            st.markdown("#### Average True Range (Volatility)")
            fig_atr = go.Figure()
            
            fig_atr.add_trace(go.Scatter(
                x=df.index, y=df["ATR"],
                fill='tozeroy',
                line=dict(color="#8b5cf6", width=2),
                name="ATR"
            ))
            
            fig_atr.update_layout(
                height=250,
                plot_bgcolor="#0a0e1a",
                paper_bgcolor="#0a0e1a",
                font=dict(color="#e4e7eb"),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                showlegend=False
            )
            
            st.plotly_chart(fig_atr, use_container_width=True)
            
            st.markdown(f"**Current ATR:** ₹{df['ATR'].iloc[-1]:.2f}")

with tab3:
    st.markdown("### 🎯 Trading Signals & Analysis")
    
    # Stock Scorecard
    st.markdown("#### 📋 Stock Scorecard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance metrics
        performance_score = "good" if price_change_pct > 2 else "average" if price_change_pct > -2 else "poor"
        
        growth_score = "good" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "poor"
        
        profitability_score = "good" if stock_info.get("returnOnEquity", 0) > 0.15 else "average"
        
        st.markdown(f"""
        <div class='scorecard-item'>
            <div class='scorecard-label'>📊 Performance</div>
            <div class='scorecard-desc'>Price trend: {'upward' if price_change_pct > 0 else 'downward'}</div>
            <span class='score-badge score-{performance_score}'>{performance_score.title()}</span>
        </div>
        
        <div class='scorecard-item'>
            <div class='scorecard-label'>📈 Growth</div>
            <div class='scorecard-desc'>Moving average trend</div>
            <span class='score-badge score-{growth_score}'>{growth_score.title()}</span>
        </div>
        
        <div class='scorecard-item'>
            <div class='scorecard-label'>💰 Profitability</div>
            <div class='scorecard-desc'>Return on Equity metrics</div>
            <span class='score-badge score-{profitability_score}'>{profitability_score.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Overall rating
        st.markdown("#### 🎯 Analyst Ratings")
        
        # Calculate simple score
        score = 0
        if price_change_pct > 0: score += 1
        if df["RSI"].iloc[-1] > 30 and df["RSI"].iloc[-1] < 70: score += 1
        if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]: score += 1
        if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]: score += 1
        
        # Simulate ratings (in real app, use actual analyst data)
        buy_count = max(20, int(score * 6.5))
        hold_count = max(5, int((4 - score) * 2))
        sell_count = max(3, int((4 - score) * 1.5))
        
        total_analysts = buy_count + hold_count + sell_count
        
        st.markdown(f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <div class='rating-label' style='color: #10b981;'>
                <div style='font-size: 0.75rem; color: #64748b;'>BUY</div>
                <div class='rating-number' style='color: #10b981;'>{buy_count}</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Analysts</div>
            </div>
            <div class='rating-label' style='color: #fbbf24;'>
                <div style='font-size: 0.75rem; color: #64748b;'>HOLD</div>
                <div class='rating-number' style='color: #fbbf24;'>{hold_count}</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Analysts</div>
            </div>
            <div class='rating-label' style='color: #ef4444;'>
                <div style='font-size: 0.75rem; color: #64748b;'>SELL</div>
                <div class='rating-number' style='color: #ef4444;'>{sell_count}</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Analysts</div>
            </div>
            <div class='rating-bar' style='margin-top: 1rem;'></div>
            <div style='font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;'>
                Consensus: <strong style='color: #2dd4bf;'>Strong Buy</strong> from {total_analysts} analysts
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Signals Table
    st.markdown("#### 📊 Recent Trading Signals")
    
    recent_signals = df[df["Signal"] != 0].tail(10)[["Close", "Signal", "Signal_Type"]].copy()
    recent_signals["Signal"] = recent_signals["Signal"].map({1: "🟢 BUY", -1: "🔴 SELL"})
    recent_signals.columns = ["Price (₹)", "Action", "Reason"]
    
    if not recent_signals.empty:
        st.dataframe(recent_signals, use_container_width=True, height=400)
    else:
        st.info("No recent signals generated. Continue monitoring...")

with tab4:
    st.markdown("### 🏢 Company Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Company Essentials")
        essentials = {
            "Market Cap": f"₹{stock_info.get('marketCap', 0)/10000000:.2f} Cr",
            "P/E Ratio": f"{stock_info.get('trailingPE', 'N/A')}",
            "EPS": f"₹{stock_info.get('trailingEps', 'N/A')}",
            "Book Value": f"₹{stock_info.get('bookValue', 'N/A')}",
            "P/B Ratio": f"{stock_info.get('priceToBook', 'N/A')}",
        }
        
        for key, value in essentials.items():
            st.markdown(f"**{key}:** {value}")
    
    with col2:
        st.markdown("#### 📈 Performance")
        performance = {
            "52W High": f"₹{stock_info.get('fiftyTwoWeekHigh', 0):.2f}",
            "52W Low": f"₹{stock_info.get('fiftyTwoWeekLow', 0):.2f}",
            "50D Avg": f"₹{stock_info.get('fiftyDayAverage', 0):.2f}",
            "200D Avg": f"₹{stock_info.get('twoHundredDayAverage', 0):.2f}",
            "Beta": f"{stock_info.get('beta', 'N/A')}",
        }
        
        for key, value in performance.items():
            st.markdown(f"**{key}:** {value}")
    
    with col3:
        st.markdown("#### 💰 Financials")
        financials = {
            "Revenue": f"₹{stock_info.get('totalRevenue', 0)/10000000:.2f} Cr",
            "Profit Margin": f"{stock_info.get('profitMargins', 0)*100:.2f}%",
            "ROE": f"{stock_info.get('returnOnEquity', 0)*100:.2f}%",
            "Debt/Equity": f"{stock_info.get('debtToEquity', 'N/A')}",
            "Dividend Yield": f"{stock_info.get('dividendYield', 0)*100:.2f}%",
        }
        
        for key, value in financials.items():
            st.markdown(f"**{key}:** {value}")
    
    st.markdown("---")
    
    # Financial Breakdown Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Profit Margin Breakdown")
        
        # Simulated margin data
        profit_margin = stock_info.get('profitMargins', 0.15) * 100
        operating_margin = stock_info.get('operatingMargins', 0.20) * 100
        gross_margin = stock_info.get('grossMargins', 0.35) * 100
        
        margin_data = pd.DataFrame({
            'Margin Type': ['Gross Margin', 'Operating Margin', 'Profit Margin'],
            'Percentage': [gross_margin, operating_margin, profit_margin]
        })
        
        fig_margins = px.bar(
            margin_data, 
            x='Margin Type', 
            y='Percentage',
            color='Percentage',
            color_continuous_scale=['#ef4444', '#fbbf24', '#10b981'],
            text='Percentage'
        )
        
        fig_margins.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_margins.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Percentage (%)"),
            showlegend=False
        )
        
        st.plotly_chart(fig_margins, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Revenue Distribution (Simulated)")
        
        # Simulated revenue distribution
        revenue_data = pd.DataFrame({
            'Category': ['Product Sales', 'Services', 'Licensing', 'Other'],
            'Amount': [45, 30, 15, 10]
        })
        
        fig_revenue = px.pie(
            revenue_data, 
            values='Amount', 
            names='Category',
            color_discrete_sequence=['#2dd4bf', '#10b981', '#fbbf24', '#8b5cf6'],
            hole=0.4
        )
        
        fig_revenue.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#0a0e1a', width=2))
        )
        
        fig_revenue.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    st.markdown("---")
    
    # Financial Metrics Over Time
    st.markdown("#### 📈 Key Financial Metrics Trend")
    
    # Simulated quarterly data
    quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2']
    revenue_trend = [85, 92, 88, 95, 98, 102]
    profit_trend = [12, 15, 13, 16, 17, 18]
    eps_trend = [2.5, 3.0, 2.8, 3.2, 3.4, 3.6]
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Bar(
        x=quarters,
        y=revenue_trend,
        name='Revenue (₹Cr)',
        marker_color='#2dd4bf',
        yaxis='y',
        opacity=0.7
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=quarters,
        y=profit_trend,
        name='Profit (₹Cr)',
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=quarters,
        y=eps_trend,
        name='EPS (₹)',
        mode='lines+markers',
        line=dict(color='#fbbf24', width=3),
        marker=dict(size=8),
        yaxis='y3'
    ))
    
    fig_trends.update_layout(
        height=400,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e4e7eb"),
        xaxis=dict(gridcolor="#1e293b", title="Quarter"),
        yaxis=dict(
            title=dict(text="Revenue (₹Cr)", font=dict(color="#2dd4bf")),
            tickfont=dict(color="#2dd4bf"),
            gridcolor="#1e293b"
        ),
        yaxis2=dict(
            title=dict(text="Profit (₹Cr)", font=dict(color="#10b981")),
            tickfont=dict(color="#10b981"),
            anchor="free",
            overlaying="y",
            side="left",
            position=0.05
        ),
        yaxis3=dict(
            title=dict(text="EPS (₹)", font=dict(color="#fbbf24")),
            tickfont=dict(color="#fbbf24"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 📝 Business Summary")
    summary = stock_info.get('longBusinessSummary', 'No description available.')
    st.markdown(f"<div style='background: #0f172a; padding: 1.5rem; border-radius: 12px; border: 1px solid #1e293b;'>{summary}</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("### ⚖️ Peer Comparison")
    
    # Simulated peer data (in production, fetch real data)
    peers = {
        "Company": [company_name, "Peer 1", "Peer 2", "Peer 3", "Peer 4"],
        "Price (₹)": [current_price, current_price*0.9, current_price*1.1, current_price*0.85, current_price*1.05],
        "P/E": [stock_info.get('trailingPE', 20), 20.5, 18.3, 22.1, 19.8],
        "Market Cap (Cr)": [
            stock_info.get('marketCap', 0)/10000000,
            stock_info.get('marketCap', 0)/10000000 * 0.8,
            stock_info.get('marketCap', 0)/10000000 * 1.2,
            stock_info.get('marketCap', 0)/10000000 * 0.7,
            stock_info.get('marketCap', 0)/10000000 * 0.95,
        ],
        "ROE %": [stock_info.get('returnOnEquity', 0.15)*100, 12.5, 15.2, 10.8, 14.3],
        "Div Yield %": [stock_info.get('dividendYield', 0.02)*100, 2.1, 1.8, 2.5, 1.9],
        "52W Change %": [price_change_pct, 15.2, -5.3, 22.1, 8.7],
    }
    
    peer_df = pd.DataFrame(peers)
    
    # Highlight current company
    st.dataframe(
        peer_df.style.apply(lambda x: ['background-color: #1e293b' if i == 0 else '' for i in range(len(x))], axis=0),
        use_container_width=True,
        height=250
    )
    
    st.markdown("---")
    
    # Multi-metric comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 P/E Ratio Comparison")
        colors = ['#2dd4bf' if i == 0 else '#64748b' for i in range(len(peer_df))]
        fig_pe = px.bar(
            peer_df, 
            x="Company", 
            y="P/E",
            color="Company",
            color_discrete_sequence=colors,
            text="P/E"
        )
        fig_pe.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_pe.update_layout(
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=False,
            height=300,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="P/E Ratio")
        )
        st.plotly_chart(fig_pe, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 ROE % Comparison")
        fig_roe = px.bar(
            peer_df, 
            x="Company", 
            y="ROE %",
            color="Company",
            color_discrete_sequence=colors,
            text="ROE %"
        )
        fig_roe.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_roe.update_layout(
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=False,
            height=300,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Return on Equity (%)")
        )
        st.plotly_chart(fig_roe, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📈 Market Cap Distribution")
        fig_mcap = px.pie(
            peer_df,
            values='Market Cap (Cr)',
            names='Company',
            color_discrete_sequence=['#2dd4bf', '#10b981', '#fbbf24', '#8b5cf6', '#ef4444'],
            hole=0.4
        )
        fig_mcap.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#0a0e1a', width=2))
        )
        fig_mcap.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=False
        )
        st.plotly_chart(fig_mcap, use_container_width=True)
    
    with col4:
        st.markdown("#### 📊 52-Week Performance")
        # Create performance comparison
        perf_colors = ['#10b981' if x > 0 else '#ef4444' for x in peer_df['52W Change %']]
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Bar(
            x=peer_df['Company'],
            y=peer_df['52W Change %'],
            marker_color=perf_colors,
            text=peer_df['52W Change %'],
            texttemplate='%{text:.1f}%',
            textposition='outside'
        ))
        
        fig_perf.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Change (%)"),
            showlegend=False
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    st.markdown("---")
    
    # Radar chart for multi-dimensional comparison
    st.markdown("#### 🎯 Multi-Dimensional Performance Radar")
    
    # Normalize metrics for radar chart (0-100 scale)
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 50
        return ((value - min_val) / (max_val - min_val)) * 100
    
    categories = ['P/E Efficiency', 'ROE', 'Div Yield', 'Market Share', 'Growth']
    
    # Current company normalized scores
    pe_norm = 100 - normalize(peer_df['P/E'].iloc[0], peer_df['P/E'].min(), peer_df['P/E'].max())
    roe_norm = normalize(peer_df['ROE %'].iloc[0], peer_df['ROE %'].min(), peer_df['ROE %'].max())
    div_norm = normalize(peer_df['Div Yield %'].iloc[0], peer_df['Div Yield %'].min(), peer_df['Div Yield %'].max())
    mcap_norm = normalize(peer_df['Market Cap (Cr)'].iloc[0], peer_df['Market Cap (Cr)'].min(), peer_df['Market Cap (Cr)'].max())
    growth_norm = normalize(peer_df['52W Change %'].iloc[0], peer_df['52W Change %'].min(), peer_df['52W Change %'].max())
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[pe_norm, roe_norm, div_norm, mcap_norm, growth_norm],
        theta=categories,
        fill='toself',
        name=company_name,
        line_color='#2dd4bf',
        fillcolor='rgba(45, 212, 191, 0.3)'
    ))
    
    # Add industry average
    fig_radar.add_trace(go.Scatterpolar(
        r=[50, 50, 50, 50, 50],
        theta=categories,
        fill='toself',
        name='Peer Average',
        line_color='#64748b',
        fillcolor='rgba(100, 116, 139, 0.1)',
        line_dash='dash'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#1e293b'
            ),
            bgcolor='#0a0e1a'
        ),
        showlegend=True,
        height=400,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e4e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

with tab6:
    st.markdown("### 📊 Advanced Analytics & Insights")
    
    # Price Distribution Analysis
    st.markdown("#### 📈 Price Distribution & Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📊 Price Range Distribution")
        
        # Create price bins
        price_range = df['Close'].max() - df['Close'].min()
        bins = 5
        df['Price_Bin'] = pd.cut(df['Close'], bins=bins)
        price_dist = df['Price_Bin'].value_counts().sort_index()
        
        fig_price_dist = px.bar(
            x=[f"₹{interval.left:.0f}-{interval.right:.0f}" for interval in price_dist.index],
            y=price_dist.values,
            labels={'x': 'Price Range', 'y': 'Frequency'},
            color=price_dist.values,
            color_continuous_scale='teal'
        )
        
        fig_price_dist.update_layout(
            height=250,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title="Price Range (₹)"),
            yaxis=dict(gridcolor="#1e293b", title="Count"),
            showlegend=False
        )
        
        st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        st.markdown("##### 📊 Returns Distribution")
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change() * 100
        
        fig_returns = px.histogram(
            df.dropna(),
            x='Returns',
            nbins=30,
            color_discrete_sequence=['#2dd4bf']
        )
        
        fig_returns.update_layout(
            height=250,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title="Returns (%)"),
            yaxis=dict(gridcolor="#1e293b", title="Frequency"),
            showlegend=False
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col3:
        st.markdown("##### 📊 Volume Distribution")
        
        volume_bins = pd.cut(df['Volume'], bins=5)
        volume_dist = volume_bins.value_counts().sort_index()
        
        labels = [f"{int(interval.left/1000000)}M-{int(interval.right/1000000)}M" for interval in volume_dist.index]
        
        fig_vol_pie = px.pie(
            values=volume_dist.values,
            names=labels,
            color_discrete_sequence=['#2dd4bf', '#10b981', '#fbbf24', '#8b5cf6', '#ef4444'],
            hole=0.4
        )
        
        fig_vol_pie.update_traces(
            textposition='inside',
            textinfo='percent',
            marker=dict(line=dict(color='#0a0e1a', width=2))
        )
        
        fig_vol_pie.update_layout(
            height=250,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_vol_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Trading Activity Analysis
    st.markdown("#### 📊 Trading Activity Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🟢🔴 Bullish vs Bearish Days")
        
        bullish_days = (df['Close'] > df['Open']).sum()
        bearish_days = (df['Close'] < df['Open']).sum()
        neutral_days = (df['Close'] == df['Open']).sum()
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Bullish', 'Bearish', 'Neutral'],
            'Days': [bullish_days, bearish_days, neutral_days],
            'Color': ['#10b981', '#ef4444', '#64748b']
        })
        
        fig_sentiment = px.pie(
            sentiment_data,
            values='Days',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Bullish': '#10b981', 'Bearish': '#ef4444', 'Neutral': '#64748b'},
            hole=0.5
        )
        
        fig_sentiment.update_traces(
            textposition='outside',
            textinfo='percent+label+value',
            marker=dict(line=dict(color='#0a0e1a', width=3))
        )
        
        fig_sentiment.update_layout(
            height=350,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb", size=12),
            showlegend=False,
            annotations=[dict(text=f'Total<br>{len(df)} Days', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Stats
        bull_pct = (bullish_days / len(df)) * 100
        st.markdown(f"""
        <div style='background: #0f172a; padding: 1rem; border-radius: 10px; border: 1px solid #1e293b; margin-top: 1rem;'>
            <div style='color: #10b981; font-size: 1.2rem; font-weight: 600;'>Bullish Trend: {bull_pct:.1f}%</div>
            <div style='color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;'>
                Average Bullish Gain: +{df[df['Close'] > df['Open']]['Returns'].mean():.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### 📊 Volume by Price Movement")
        
        # Categorize by price movement
        df['Movement'] = pd.cut(df['Returns'].fillna(0), 
                                bins=[-float('inf'), -2, -0.5, 0.5, 2, float('inf')],
                                labels=['Strong Down', 'Down', 'Flat', 'Up', 'Strong Up'])
        
        movement_volume = df.groupby('Movement', observed=True)['Volume'].sum()
        
        colors_map = {
            'Strong Down': '#7f1d1d',
            'Down': '#ef4444',
            'Flat': '#64748b',
            'Up': '#10b981',
            'Strong Up': '#065f46'
        }
        
        colors = [colors_map[m] for m in movement_volume.index]
        
        fig_vol_movement = go.Figure()
        
        fig_vol_movement.add_trace(go.Bar(
            x=movement_volume.index,
            y=movement_volume.values,
            marker_color=colors,
            text=[f"{v/1000000:.1f}M" for v in movement_volume.values],
            textposition='outside'
        ))
        
        fig_vol_movement.update_layout(
            height=350,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title="Price Movement"),
            yaxis=dict(gridcolor="#1e293b", title="Total Volume"),
            showlegend=False
        )
        
        st.plotly_chart(fig_vol_movement, use_container_width=True)
        
        # High volume insight
        high_vol_days = df[df['Volume'] > df['Volume'].quantile(0.75)]
        avg_return_high_vol = high_vol_days['Returns'].mean()
        vol_color = '#10b981' if avg_return_high_vol > 0 else '#ef4444'
        
        st.markdown(f"""
        <div style='background: #0f172a; padding: 1rem; border-radius: 10px; border: 1px solid #1e293b; margin-top: 1rem;'>
            <div style='color: {vol_color}; font-size: 1.2rem; font-weight: 600;'>High Volume Days: {len(high_vol_days)}</div>
            <div style='color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;'>
                Average Return on High Volume: {avg_return_high_vol:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical Indicator Distribution
    st.markdown("#### 📊 Technical Indicator Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### RSI Zones Distribution")
        
        # RSI zones
        rsi_zones = pd.cut(df['RSI'].dropna(), 
                          bins=[0, 30, 50, 70, 100],
                          labels=['Oversold (<30)', 'Weak (30-50)', 'Strong (50-70)', 'Overbought (>70)'])
        
        rsi_dist = rsi_zones.value_counts()
        
        colors_rsi = ['#10b981', '#fbbf24', '#2dd4bf', '#ef4444']
        
        fig_rsi_zones = px.bar(
            x=rsi_dist.index,
            y=rsi_dist.values,
            color=rsi_dist.index,
            color_discrete_sequence=colors_rsi,
            text=rsi_dist.values
        )
        
        fig_rsi_zones.update_traces(textposition='outside')
        fig_rsi_zones.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Days"),
            showlegend=False
        )
        
        st.plotly_chart(fig_rsi_zones, use_container_width=True)
    
    with col2:
        st.markdown("##### MACD Signal Strength")
        
        df['MACD_Strength'] = abs(df['MACD_Hist'])
        strength_bins = pd.cut(df['MACD_Strength'].dropna(),
                              bins=5,
                              labels=['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong'])
        
        strength_dist = strength_bins.value_counts().sort_index()
        
        fig_macd_strength = px.pie(
            values=strength_dist.values,
            names=strength_dist.index,
            color_discrete_sequence=['#94a3b8', '#64748b', '#fbbf24', '#2dd4bf', '#10b981'],
            hole=0.4
        )
        
        fig_macd_strength.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#0a0e1a', width=2))
        )
        
        fig_macd_strength.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            showlegend=False
        )
        
        st.plotly_chart(fig_macd_strength, use_container_width=True)
    
    with col3:
        st.markdown("##### Volatility (ATR) Trend")
        
        df['ATR_Trend'] = df['ATR'].rolling(window=5).mean()
        
        fig_atr_trend = go.Figure()
        
        fig_atr_trend.add_trace(go.Scatter(
            x=df.index,
            y=df['ATR'],
            name='ATR',
            line=dict(color='#8b5cf6', width=1),
            opacity=0.5
        ))
        
        fig_atr_trend.add_trace(go.Scatter(
            x=df.index,
            y=df['ATR_Trend'],
            name='ATR Trend',
            line=dict(color='#2dd4bf', width=3)
        ))
        
        fig_atr_trend.update_layout(
            height=300,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="ATR Value"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_atr_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Performance Metrics
    st.markdown("#### 📈 Performance Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    volatility = df['Returns'].std()
    sharpe_ratio = (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(252) if df['Returns'].std() != 0 else 0
    max_drawdown = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
    
    metrics_display = [
        ("Total Return", f"{total_return:+.2f}%", "Period Performance"),
        ("Volatility", f"{volatility:.2f}%", "Daily Std Dev"),
        ("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Risk-Adjusted Return"),
        ("Max Drawdown", f"{max_drawdown:.2f}%", "Largest Drop")
    ]
    
    for col, (label, value, desc) in zip([col1, col2, col3, col4], metrics_display):
        with col:
            color = "#10b981" if "+" in value or (label == "Sharpe Ratio" and sharpe_ratio > 1) else "#ef4444" if label == "Max Drawdown" else "#2dd4bf"
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color: {color};'>{value}</div>
                <div class='metric-subtext'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Correlation Matrix (if multiple indicators)
    st.markdown("#### 🔗 Indicator Correlation Heatmap")
    
    correlation_data = df[['Close', 'Volume', 'RSI', 'MACD', 'ATR']].corr()
    
    fig_corr = px.imshow(
        correlation_data,
        labels=dict(color="Correlation"),
        x=correlation_data.columns,
        y=correlation_data.columns,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        text_auto='.2f'
    )
    
    fig_corr.update_layout(
        height=400,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e4e7eb"),
        xaxis=dict(side='bottom')
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

with tab7:
    st.markdown("### 🤖 AI-Powered Market Predictions")
    
    st.markdown("""
    <div style='background: #0f172a; padding: 1rem; border-radius: 10px; border: 1px solid #2dd4bf; margin-bottom: 2rem;'>
        <p style='color: #2dd4bf; font-weight: 600; margin: 0;'>⚡ Hybrid ML System Active</p>
        <p style='color: #94a3b8; font-size: 0.875rem; margin: 0.5rem 0 0 0;'>
            Combining LSTM Neural Networks + Facebook Prophet + Technical Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        prediction_days = st.slider("📅 Prediction Horizon (Days)", 5, 30, 20)
    
    with col2:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        train_new_model = st.checkbox("🔄 Train Fresh Model (takes 30-60s)", value=False)
    
    with col3:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Generate Predictions", type="primary"):
            st.session_state.run_predictions = True
    
    # Initialize predictors
    if 'run_predictions' not in st.session_state:
        st.session_state.run_predictions = False
    
    if st.session_state.run_predictions or 'ml_results' not in st.session_state:
        with st.spinner("🧠 AI Models analyzing market data..."):
            try:
                ensemble = EnsemblePredictor()
                trend_predictor = TrendPredictor()
                sr_predictor = SupportResistancePredictor()
                
                # Get predictions
                ml_results = ensemble.predict_all(df, days_ahead=prediction_days, train_models=train_new_model)
                
                # Store in session
                st.session_state.ml_results = ml_results
                st.session_state.prediction_timestamp = datetime.now()
                
                st.success("✅ Predictions generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Tip: Install required packages: pip install tensorflow prophet scikit-learn")
                ml_results = {}
    
    else:
        ml_results = st.session_state.get('ml_results', {})
    
    if not ml_results:
        st.info("👆 Click 'Generate Predictions' to see AI forecasts")
        st.stop()
    
    # Display timestamp
    if 'prediction_timestamp' in st.session_state:
        st.markdown(f"<div style='text-align: right; color: #64748b; font-size: 0.875rem;'>Generated: {st.session_state.prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 1: TREND PREDICTION
    st.markdown("### 🎯 Trend Direction Forecast")
    
    if 'trend_prediction' in ml_results:
        trend_data = ml_results['trend_prediction']
        
        col1, col2, col3 = st.columns(3)
        
        # Trend badge color
        if trend_data['trend'] == 'BULLISH':
            trend_color = '#10b981'
            trend_icon = '📈'
        elif trend_data['trend'] == 'BEARISH':
            trend_color = '#ef4444'
            trend_icon = '📉'
        else:
            trend_color = '#fbbf24'
            trend_icon = '➡️'
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; border: 2px solid {trend_color};'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{trend_icon}</div>
                <div class='metric-label'>PREDICTED TREND</div>
                <div class='metric-value' style='color: {trend_color}; font-size: 2rem;'>{trend_data['trend']}</div>
                <div class='metric-subtext'>Medium-term (1-4 weeks)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>CONFIDENCE SCORE</div>
                <div class='metric-value' style='color: #2dd4bf; font-size: 2.5rem;'>{trend_data['confidence']:.0f}%</div>
                <div class='metric-subtext'>Model Certainty</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>SIGNAL STRENGTH</div>
                <div class='metric-value' style='color: #fbbf24; font-size: 2.5rem;'>{abs(trend_data['score'])}/10</div>
                <div class='metric-subtext'>{'Strong' if abs(trend_data['score']) >= 5 else 'Moderate'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Supporting signals
        st.markdown("#### 📊 Supporting Technical Signals")
        
        signals = trend_data['signals']
        
        col1, col2, col3, col4 = st.columns(4)
        
        signals_display = [
            ("MA Crossover", "✅ Bullish" if signals['ma_cross'] else "❌ Bearish", signals['ma_cross']),
            ("RSI Level", f"{signals['rsi']:.1f}", signals['rsi'] >= 40 and signals['rsi'] <= 60),
            ("MACD", "✅ Bullish" if signals['macd_bullish'] else "❌ Bearish", signals['macd_bullish']),
            ("Volume", "✅ Surge" if signals['volume_surge'] else "⚪ Normal", signals['volume_surge'])
        ]
        
        for col, (label, value, is_positive) in zip([col1, col2, col3, col4], signals_display):
            color = '#10b981' if is_positive else '#64748b'
            with col:
                st.markdown(f"""
                <div style='background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #1e293b; text-align: center;'>
                    <div style='color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.5rem;'>{label}</div>
                    <div style='color: {color}; font-weight: 600;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 2: PRICE PREDICTIONS
    st.markdown("### 📈 Price Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'ensemble_predictions' in ml_results or 'lstm_predictions' in ml_results or 'prophet_predictions' in ml_results:
            st.markdown("#### 🎯 Hybrid Model Prediction")
            
            # Use ensemble if available, otherwise fall back
            if 'ensemble_predictions' in ml_results:
                predictions = ml_results['ensemble_predictions']
                model_name = "Ensemble (LSTM + Prophet)"
            elif 'lstm_predictions' in ml_results:
                predictions = ml_results['lstm_predictions']
                model_name = "LSTM Neural Network"
            else:
                predictions = ml_results['prophet_predictions']
                model_name = "Facebook Prophet"
            
            # Create prediction chart
            fig_forecast = go.Figure()
            
            # Historical data
            historical_days = min(60, len(df))
            fig_forecast.add_trace(go.Scatter(
                x=df.index[-historical_days:],
                y=df['Close'].iloc[-historical_days:],
                mode='lines',
                name='Historical Price',
                line=dict(color='#64748b', width=2)
            ))
            
            # Predicted price
            fig_forecast.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#2dd4bf', width=3),
                marker=dict(size=6)
            ))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Upper_Bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Lower_Bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(45, 212, 191, 0.2)',
                showlegend=True
            ))
            
            fig_forecast.update_layout(
                height=400,
                plot_bgcolor="#0a0e1a",
                paper_bgcolor="#0a0e1a",
                font=dict(color="#e4e7eb"),
                xaxis=dict(gridcolor="#1e293b", title="Date"),
                yaxis=dict(gridcolor="#1e293b", title="Price (₹)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Price targets
            current_price = df['Close'].iloc[-1]
            predicted_end = predictions['Predicted_Price'].iloc[-1]
            change = ((predicted_end - current_price) / current_price) * 100
            
            st.markdown(f"""
            <div style='background: #0f172a; padding: 1.5rem; border-radius: 10px; border: 1px solid #1e293b; margin-top: 1rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='color: #94a3b8; font-size: 0.875rem;'>Current Price</div>
                        <div style='color: #f1f5f9; font-size: 1.5rem; font-weight: 600;'>₹{current_price:.2f}</div>
                    </div>
                    <div style='font-size: 2rem;'>→</div>
                    <div>
                        <div style='color: #94a3b8; font-size: 0.875rem;'>{prediction_days}-Day Target</div>
                        <div style='color: {"#10b981" if change > 0 else "#ef4444"}; font-size: 1.5rem; font-weight: 600;'>₹{predicted_end:.2f}</div>
                    </div>
                    <div>
                        <div style='color: #94a3b8; font-size: 0.875rem;'>Expected Change</div>
                        <div style='color: {"#10b981" if change > 0 else "#ef4444"}; font-size: 1.5rem; font-weight: 600;'>{change:+.2f}%</div>
                    </div>
                </div>
                <div style='color: #64748b; font-size: 0.75rem; margin-top: 1rem; text-align: center;'>Model: {model_name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("💡 Install TensorFlow and Prophet for price predictions")
    
    with col2:
        st.markdown("#### 📊 Prediction Summary")
        
        if 'ensemble_predictions' in ml_results or 'lstm_predictions' in ml_results:
            predictions = ml_results.get('ensemble_predictions', ml_results.get('lstm_predictions'))
            
            # Statistics
            pred_mean = predictions['Predicted_Price'].mean()
            pred_std = predictions['Predicted_Price'].std()
            pred_min = predictions['Lower_Bound'].min()
            pred_max = predictions['Upper_Bound'].max()
            
            current_price = df['Close'].iloc[-1]
            
            stats_data = [
                ("Average Predicted", f"₹{pred_mean:.2f}"),
                ("Price Range", f"₹{pred_min:.2f} - ₹{pred_max:.2f}"),
                ("Volatility", f"±₹{pred_std:.2f}"),
                ("Potential Gain", f"{((pred_max - current_price)/current_price*100):.2f}%"),
                ("Potential Loss", f"{((pred_min - current_price)/current_price*100):.2f}%")
            ]
            
            for label, value in stats_data:
                st.markdown(f"""
                <div style='background: #0f172a; padding: 0.75rem; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 0.5rem;'>
                    <div style='color: #94a3b8; font-size: 0.75rem;'>{label}</div>
                    <div style='color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Model accuracy
        if 'lstm_training' in ml_results:
            lstm_acc = ml_results['lstm_training']['accuracy']
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                <div style='color: #d1fae5; font-size: 0.875rem;'>LSTM Model Accuracy</div>
                <div style='color: #ffffff; font-size: 2rem; font-weight: 700;'>{lstm_acc:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 3: SUPPORT & RESISTANCE
    st.markdown("### 🎯 Support & Resistance Levels")
    
    if 'support_resistance' in ml_results:
        sr_data = ml_results['support_resistance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Resistance Levels")
            
            if sr_data['resistance_levels']:
                for i, (level, dist) in enumerate(zip(sr_data['resistance_levels'], sr_data['resistance_distance'])):
                    st.markdown(f"""
                    <div style='background: #0f172a; padding: 1rem; border-radius: 8px; border-left: 3px solid #ef4444; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div style='color: #94a3b8; font-size: 0.75rem;'>R{i+1}</div>
                                <div style='color: #f1f5f9; font-size: 1.25rem; font-weight: 600;'>₹{level:.2f}</div>
                            </div>
                            <div style='text-align: right;'>
                                <div style='color: #ef4444; font-weight: 600;'>+{dist:.2f}%</div>
                                <div style='color: #64748b; font-size: 0.75rem;'>away</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No resistance levels detected")
        
        with col2:
            st.markdown("#### 🟢 Support Levels")
            
            if sr_data['support_levels']:
                for i, (level, dist) in enumerate(zip(sr_data['support_levels'], sr_data['support_distance'])):
                    st.markdown(f"""
                    <div style='background: #0f172a; padding: 1rem; border-radius: 8px; border-left: 3px solid #10b981; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div style='color: #94a3b8; font-size: 0.75rem;'>S{i+1}</div>
                                <div style='color: #f1f5f9; font-size: 1.25rem; font-weight: 600;'>₹{level:.2f}</div>
                            </div>
                            <div style='text-align: right;'>
                                <div style='color: #10b981; font-weight: 600;'>-{dist:.2f}%</div>
                                <div style='color: #64748b; font-size: 0.75rem;'>below</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No support levels detected")
        
        # S/R Chart
        st.markdown("#### 📊 Support & Resistance Visualization")
        
        fig_sr = go.Figure()
        
        # Price line
        fig_sr.add_trace(go.Scatter(
            x=df.index[-60:],
            y=df['Close'].iloc[-60:],
            mode='lines',
            name='Price',
            line=dict(color='#64748b', width=2)
        ))
        
        # Resistance lines
        for i, level in enumerate(sr_data['resistance_levels']):
            fig_sr.add_hline(
                y=level,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"R{i+1}: ₹{level:.2f}",
                annotation_position="right"
            )
        
        # Support lines
        for i, level in enumerate(sr_data['support_levels']):
            fig_sr.add_hline(
                y=level,
                line_dash="dash",
                line_color="#10b981",
                annotation_text=f"S{i+1}: ₹{level:.2f}",
                annotation_position="right"
            )
        
        fig_sr.update_layout(
            height=400,
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e4e7eb"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Price (₹)"),
            showlegend=True
        )
        
        st.plotly_chart(fig_sr, use_container_width=True)
    
    # SECTION 4: BREAKOUT PREDICTION
    if 'breakout_prediction' in ml_results:
        breakout_data = ml_results['breakout_prediction']
        
        st.markdown("---")
        st.markdown("### 🚀 Breakout Analysis")
        
        if breakout_data['breakout_likely']:
            color = '#10b981' if breakout_data['direction'] == 'UP' else '#ef4444'
            icon = '🚀' if breakout_data['direction'] == 'UP' else '📉'
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 2rem; border-radius: 12px; border: 2px solid {color};'>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>{icon}</div>
                    <div style='color: {color}; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;'>
                        BREAKOUT ALERT: {breakout_data['direction']}
                    </div>
                    <div style='color: #94a3b8; font-size: 1rem; margin-bottom: 1rem;'>
                        Target: ₹{breakout_data['target']:.2f}
                    </div>
                    <div style='color: #2dd4bf; font-size: 1.5rem; font-weight: 600;'>
                        Confidence: {breakout_data['confidence']}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #0f172a; padding: 2rem; border-radius: 12px; border: 1px solid #1e293b; text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 1rem;'>📊</div>
                <div style='color: #94a3b8; font-size: 1.25rem;'>
                    Stock is currently <span style='color: #fbbf24; font-weight: 600;'>CONSOLIDATING</span>
                </div>
                <div style='color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;'>
                    No immediate breakout expected. Monitor key levels.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DISCLAIMER
    st.markdown("""
    <div style='background: #78350f; padding: 1rem; border-radius: 10px; border: 1px solid #f59e0b;'>
        <div style='color: #fef3c7; font-weight: 600; margin-bottom: 0.5rem;'>⚠️ Important Disclaimer</div>
        <div style='color: #fde68a; font-size: 0.875rem;'>
            These predictions are generated by machine learning models and should not be considered as financial advice. 
            Past performance does not guarantee future results. Always do your own research and consult with a financial advisor 
            before making investment decisions. ML models can be wrong, especially during unprecedented market events.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------- FOOTER --------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.markdown(f"**Data Source:** Yahoo Finance")

with col3:
    st.markdown(f"**Refresh Rate:** {refresh_sec}s")

st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 2rem;'>⚠️ This is for educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
