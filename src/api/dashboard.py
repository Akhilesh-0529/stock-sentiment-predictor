import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.live_api import fetch_price_history, fetch_news, news_to_sentiment_series
from src.numerical.features import generate_features
from src.models.predictor import Predictor

st.set_page_config(layout="wide", page_title="Stock Sentiment Predictor")

st.sidebar.title("📊 Controls")
symbol = st.sidebar.text_input("Symbol", value="AAPL")
period = st.sidebar.selectbox("Period", ["7d", "30d", "60d"], index=1)
interval = st.sidebar.selectbox("Interval", ["1h", "1d"], index=0)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()

st.title(f"📈 {symbol} — Stock Sentiment Analysis")

try:
    with st.spinner("Fetching price data..."):
        price_df = fetch_price_history(symbol, period=period, interval=interval)
except Exception as e:
    st.error(f"❌ Price fetch failed: {e}")
    st.stop()

# Price chart
st.header("💹 Price History")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=price_df.index,
    open=price_df['open'],
    high=price_df['high'],
    low=price_df['low'],
    close=price_df['close'],
    name='Price'
))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=400)
st.plotly_chart(fig, use_container_width=True)

# Sentiment analysis
st.header("📰 Sentiment Analysis")
with st.spinner("Fetching news..."):
    news_df = fetch_news(symbol, from_days=7)
    sentiment_series = news_to_sentiment_series(news_df, price_df.index, window="1H")

col1, col2 = st.columns(2)
with col1:
    st.metric("Average Sentiment", f"{sentiment_series.mean():.3f}")
with col2:
    st.metric("News Articles", len(news_df))

fig_sent = px.line(sentiment_series, title="Sentiment Over Time")
st.plotly_chart(fig_sent, use_container_width=True)

# Features
st.header("🔧 Technical Features")
features = generate_features(price_df)
st.dataframe(features.tail(10), use_container_width=True)

# Model predictions
st.header("🤖 Model Predictions")
predictor = Predictor()
try:
    with st.spinner("Generating predictions..."):
        preds = predictor.predict(features.tail(30))
        pred_df = pd.DataFrame({
            'Predicted Return': preds
        }, index=features.tail(30).index)
        
        fig_pred = px.line(pred_df, title="Predicted Returns (Last 30 periods)")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        last_pred = preds[-1]
        if last_pred > 0:
            st.success(f"📈 Bullish signal: +{last_pred:.4f}")
        else:
            st.warning(f"📉 Bearish signal: {last_pred:.4f}")
except Exception as e:
    st.info("⚠️ Model not trained yet. Run training notebook first.")

# News preview
if not news_df.empty:
    st.header("📑 Recent News")
    for idx, row in news_df.head(5).iterrows():
        with st.expander(f"{row['title'][:80]}..."):
            st.write(row['description'])
            st.caption(f"Published: {idx}")
else:
    st.info("ℹ️ No news available. Set NEWSAPI_KEY environment variable.")