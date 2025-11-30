import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import requests
import xml.etree.ElementTree as ET

# --- Configuration ---
st.set_page_config(
    page_title="ProStock | AI-Powered Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa; /* Softer white/gray background */
        color: #212529;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 28px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* News Card Styling */
    .news-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #0d6efd;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* TV Container */
    .tv-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        background-color: #000;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 10px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e7f1ff;
        color: #0d6efd;
        font-weight: bold;
    }

    /* Fix Layout Overlap */
    .block-container {
        padding-top: 4rem;
        max-width: 95%;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=60)
def get_stock_data(ticker, interval, period, start=None, end=None):
    try:
        if interval == "1d" and start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Fallback for weekends/holidays
        if data.empty and period == "1d":
            data = yf.download(ticker, period="5d", interval=interval, progress=False)
        
        return data
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news
    except Exception:
        return {}, []

@st.cache_data(ttl=300)
def get_exchange_rate(pair="KRW=X"):
    try:
        data = yf.Ticker(pair).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        return None
    return None

def calculate_currency_conversion(amount, from_curr, to_curr):
    if from_curr == to_curr: return amount, 1.0
    try:
        # Try direct pair
        ticker = f"{from_curr}{to_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            return amount * rate, rate
        # Try inverse
        ticker = f"{to_curr}{from_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = 1.0 / data['Close'].iloc[-1]
            return amount * rate, rate
    except: pass
    return None, None

def calculate_technicals(data):
    if len(data) < 2: return data
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * std
    data['BB_Lower'] = data['BB_Middle'] - 2 * std
    return data

def get_fear_and_greed_proxy():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        sp500 = yf.Ticker("^GSPC").history(period="6mo")
        if sp500.empty: return 50, "Neutral"
        current_sp = sp500['Close'].iloc[-1]
        avg_sp = sp500['Close'].mean()
        fear_score = max(0, min(100, 100 - (vix - 10) * 2.5))
        momentum_score = max(0, min(100, 50 + ((current_sp - avg_sp) / avg_sp) * 500))
        final_score = (fear_score * 0.4) + (momentum_score * 0.6)
        if final_score < 25: label = "Extreme Fear"
        elif final_score < 45: label = "Fear"
        elif final_score < 55: label = "Neutral"
        elif final_score < 75: label = "Greed"
        else: label = "Extreme Greed"
        return int(final_score), label
    except: return 50, "Neutral"

def safe_extract_news_title(item):
    if not isinstance(item, dict): return None
    if 'title' in item and item['title']: return item['title']
    if 'content' in item and isinstance(item['content'], dict):
        if 'title' in item['content'] and item['content']['title']: return item['content']['title']
    for key, value in item.items():
        if isinstance(value, dict):
            res = safe_extract_news_title(value)
            if res: return res
    return None

def analyze_news_sentiment(news_items):
    if not news_items: return 0, 0, 0, "Neutral"
    polarities = []
    for item in news_items:
        title = safe_extract_news_title(item)
        if title:
            blob = TextBlob(title)
            polarities.append(blob.sentiment.polarity)
    if not polarities: return 0, 0, 0, "Neutral"
    pos = sum(1 for p in polarities if p > 0.05)
    neg = sum(1 for p in polarities if p < -0.05)
    neu = len(polarities) - pos - neg
    avg_pol = np.mean(polarities)
    if avg_pol > 0.05: label = "Positive"
    elif avg_pol < -0.05: label = "Negative"
    else: label = "Neutral"
    return pos, neg, neu, label

def generate_ai_report(ticker, price, sma, rsi, fg_score, fg_label, news_label):
    report = f"### 🧠 AI Executive Summary for {ticker}\n\n"
    report += f"**1. Market Sentiment:** {fg_label} ({fg_score}/100).\n"
    report += f"**2. News Analysis:** {news_label} sentiment detected.\n"
    trend = "Bullish 🟢" if price > sma else "Bearish 🔴"
    rsi_state = "Overbought ⚠️" if rsi > 70 else "Oversold 🛒" if rsi < 30 else "Neutral ⚖️"
    report += f"**3. Technicals:** {trend} trend, RSI is {rsi_state}."
    return report

@st.cache_data(ttl=600)
def fetch_rss_feed(url):
    """Fetches and parses an RSS feed."""
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall('.//item')[:10]:
            items.append({
                'title': item.find('title').text,
                'link': item.find('link').text,
                'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else "Recent"
            })
        return items
    except Exception as e:
        return []

# --- Sidebar Navigation ---
st.sidebar.markdown("## 📈 ProStock Terminal")
mode = st.sidebar.radio("Navigation", ["Asset Terminal", "Media & News"])
st.sidebar.markdown("---")

if mode == "Asset Terminal":
    # --- Asset Class Selection ---
    market_type = st.sidebar.selectbox("Market Type", ["Stocks", "Commodities", "Currencies/Forex"])
    ticker = ""
    if market_type == "Stocks":
        ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    elif market_type == "Commodities":
        commodities = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Copper": "HG=F", "Natural Gas": "NG=F", "Corn": "ZC=F", "Soybeans": "ZS=F"}
        selected_comm = st.sidebar.selectbox("Select Commodity", list(commodities.keys()))
        ticker = commodities[selected_comm]
    elif market_type == "Currencies/Forex":
        currencies = {"USD/KRW (Won)": "KRW=X", "EUR/USD": "EURUSD=X", "JPY/USD": "JPY=X", "GBP/USD": "GBPUSD=X", "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
        selected_curr = st.sidebar.selectbox("Select Pair", list(currencies.keys()))
        ticker = currencies[selected_curr]

    # Timeframe
    st.sidebar.markdown("### ⏱️ Timeframe")
    timeframe = st.sidebar.selectbox("Select Interval", ["1 Minute", "5 Minute", "1 Hour", "1 Day"], label_visibility="collapsed")
    if timeframe == "1 Minute": interval, period = "1m", "1d"
    elif timeframe == "5 Minute": interval, period = "5m", "5d"
    elif timeframe == "1 Hour": interval, period = "1h", "1mo"
    else: interval, period = "1d", "1y"

    if interval == "1d":
        start_date = st.sidebar.date_input("Start", value=datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End", value=datetime.now())
    else:
        st.sidebar.caption(f"Live Feed: Last {period}")

    # Indicators
    st.sidebar.markdown("### 📊 Indicators")
    show_sma = st.sidebar.toggle("SMA", value=True)
    sma_period = st.sidebar.number_input("SMA Period", value=20) if show_sma else 20
    show_ema = st.sidebar.toggle("EMA")
    ema_period = st.sidebar.number_input("EMA Period", value=50) if show_ema else 50
    show_bb = st.sidebar.toggle("Bollinger Bands")
    show_rsi = st.sidebar.toggle("RSI")

    # Currency Converter
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧮 Currency Converter", expanded=False):
        cc_amount = st.number_input("Amount", value=100.0, min_value=0.0)
        c1, c2 = st.columns(2)
        with c1: cc_from = st.selectbox("From", ["USD", "KRW", "EUR", "JPY", "GBP", "CNY", "BTC"], index=0)
        with c2: cc_to = st.selectbox("To", ["KRW", "USD", "EUR", "JPY", "GBP", "CNY", "BTC"], index=0)
        if st.button("Convert"):
            res, rate = calculate_currency_conversion(cc_amount, cc_from, cc_to)
            if res: st.success(f"{cc_amount} {cc_from} = {res:,.2f} {cc_to}")
            else: st.error("Failed")

    if st.sidebar.button("🔄 Refresh Data", type="primary"): st.rerun()

    # --- Main Asset Logic ---
    if ticker:
        s_date = start_date if interval == "1d" else None
        e_date = end_date if interval == "1d" else None
        data = get_stock_data(ticker, interval, period, s_date, e_date)
        info, news = get_stock_info(ticker)

        if data is not None and len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data = calculate_technicals(data)
            
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            delta = current_price - prev_close
            pct = (delta / prev_close) * 100 if prev_close else 0
            
            market_cap = info.get('marketCap', 0)
            volume = info.get('volume', 0)
            currency = info.get('currency', 'USD')

            usd_krw_rate = get_exchange_rate("KRW=X")
            price_display = f"{currency} {current_price:,.2f}"
            price_sub_display = f"(₩{current_price * usd_krw_rate:,.0f})" if currency == 'USD' and usd_krw_rate else ""

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(f"Price ({ticker})", price_display, f"{delta:,.2f} ({pct:+.2f}%)")
                if price_sub_display: st.caption(f"≈ {price_sub_display}")
            m2.metric("Market Cap", f"${market_cap/1e9:,.1f}B" if market_cap else "N/A")
            m3.metric("Volume", f"{volume/1e6:,.1f}M" if volume else "N/A")
            m4.metric("Sector", info.get('sector', 'N/A'))

            tabs = st.tabs(["📈 Chart", "🧠 AI Analysis", "📰 News", "🔢 Raw Data"] + (["📋 Fundamentals"] if market_type == "Stocks" else []))

            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price', increasing_line_color='#00C853', decreasing_line_color='#FF3D00'))
                if show_sma: fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='#FFA000', width=1.5), name='SMA'))
                if show_bb:
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper BB'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Lower BB'))
                fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, yaxis=dict(title=f'Price ({currency})'))
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                fg_score, fg_label = get_fear_and_greed_proxy()
                pos, neg, neu, news_label = analyze_news_sentiment(news)
                c1, c2 = st.columns(2)
                with c1:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=fg_score, title={'text': f"Fear/Greed: {fg_label}"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 45], 'color': "#FF5252"}, {'range': [55, 100], 'color': "#00E676"}]}))
                    fig_g.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_g, use_container_width=True)
                with c2:
                    st.subheader("Sentiment Analysis")
                    st.write(f"**News Tone:** {news_label}")
                    st.progress((pos / (pos+neg+neu+0.1)))
                
                if len(data) > sma_period:
                    report = generate_ai_report(ticker, current_price, data['SMA'].iloc[-1], data['RSI'].iloc[-1], fg_score, fg_label, news_label)
                    st.markdown(f"""<div class="ai-analysis-box">{report.replace(chr(10), '<br>')}</div>""", unsafe_allow_html=True)

            with tabs[2]:
                if news:
                    for item in news[:10]:
                        t = safe_extract_news_title(item) or "No Title"
                        l = item.get('link') or item.get('url') or "#"
                        pub = item.get('publisher', 'Unknown')
                        st.markdown(f"""<div class="news-card"><a href="{l}" target="_blank" style="text-decoration: none; color: #0d6efd; font-weight: bold;">{t}</a><div style="font-size: 12px; color: gray;">{pub}</div></div>""", unsafe_allow_html=True)
                else: st.info("No news found.")

            with tabs[3]:
                st.dataframe(data.tail(50), use_container_width=True)

            if market_type == "Stocks":
                with tabs[4]:
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Summary:** {info.get('longBusinessSummary', 'N/A')}")
        else:
            st.error("No data available.")
    else:
        st.info("Select an asset.")

# --- Media & News Mode ---
elif mode == "Media & News":
    st.title("📺 Global Finance Media Center")
    
    # 1. Live TV Section
    st.subheader("🔴 Live Financial News TV")
    st.caption("Official free live streams from major financial networks.")
    
    tv_col1, tv_col2 = st.columns(2)
    
    with tv_col1:
        st.markdown("**Bloomberg TV (Global)**")
        # Bloomberg's official YouTube Live URL (Standard)
        st.video("https://www.youtube.com/watch?v=dp8PhLsUcFE")
        
        st.markdown("**Sky News (Business/Global)**")
        st.video("https://www.youtube.com/watch?v=9Auq9mYxFEE")

    with tv_col2:
        st.markdown("**CNA (Asia Markets)**")
        st.video("https://www.youtube.com/watch?v=XWq5kBlakcQ")
        
        st.markdown("**ABC News (Australia/Global Business)**")
        st.video("https://www.youtube.com/watch?v=W1ilCy6XrmI")

    st.markdown("---")

    # 2. Agency News Feed
    st.subheader("📰 Major Agency Feeds (Live)")
    st.caption("Real-time headlines from CNBC, BBC, and CNN.")
    
    agency_tabs = st.tabs(["CNBC Finance", "BBC Business", "CNN Business"])
    
    with agency_tabs[0]:
        items = fetch_rss_feed("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664")
        if items:
            for i in items:
                st.markdown(f"""
                <div class="news-card" style="border-left: 4px solid #fab005;">
                    <a href="{i['link']}" target="_blank" style="text-decoration: none; color: #212529; font-weight: 600; font-size: 16px;">{i['title']}</a>
                    <div style="font-size: 12px; color: #868e96; margin-top: 4px;">CNBC • {i['pubDate']}</div>
                </div>
                """, unsafe_allow_html=True)
        else: st.warning("Feed unavailable.")

    with agency_tabs[1]:
        items = fetch_rss_feed("http://feeds.bbci.co.uk/news/business/rss.xml")
        if items:
            for i in items:
                st.markdown(f"""
                <div class="news-card" style="border-left: 4px solid #bb0000;">
                    <a href="{i['link']}" target="_blank" style="text-decoration: none; color: #212529; font-weight: 600; font-size: 16px;">{i['title']}</a>
                    <div style="font-size: 12px; color: #868e96; margin-top: 4px;">BBC • {i['pubDate']}</div>
                </div>
                """, unsafe_allow_html=True)
        else: st.warning("Feed unavailable.")

    with agency_tabs[2]:
        items = fetch_rss_feed("http://rss.cnn.com/rss/money_latest.rss")
        if items:
            for i in items:
                st.markdown(f"""
                <div class="news-card" style="border-left: 4px solid #cc0000;">
                    <a href="{i['link']}" target="_blank" style="text-decoration: none; color: #212529; font-weight: 600; font-size: 16px;">{i['title']}</a>
                    <div style="font-size: 12px; color: #868e96; margin-top: 4px;">CNN • {i['pubDate']}</div>
                </div>
                """, unsafe_allow_html=True)
        else: st.warning("Feed unavailable.")
