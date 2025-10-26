import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.live_api import fetch_price_history, fetch_news, news_to_sentiment_series
from src.numerical.features import generate_features
from src.models.predictor import Predictor
from src.backtest.strategy import TradingStrategy
import json
import os
from datetime import datetime

st.set_page_config(layout="wide", page_title="Stock Sentiment Predictor Pro")

# Sidebar
st.sidebar.title("📊 Configuration")
symbol = st.sidebar.text_input("Symbol", value="AAPL")
period = st.sidebar.selectbox("Period", ["7d", "30d", "60d", "1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1h", "1d"], index=1)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Live Analysis", "🔄 Backtest", "📊 Report", "🤖 Model Info", "⚙️ Training"])

# Fetch data once for all tabs
try:
    with st.spinner("Fetching data..."):
        price_df = fetch_price_history(symbol, period=period, interval=interval)
        news_df = fetch_news(symbol, from_days=7)
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# Tab 1: Live Analysis
with tab1:
    st.title(f"📈 {symbol} — Live Analysis")
    
    # Price chart - FIX: convert to float
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = float(price_df['close'].iloc[-1])
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        daily_change = float(price_df['close'].pct_change().iloc[-1]) * 100
        st.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{daily_change:.2f}%")
    with col3:
        volume = float(price_df['volume'].iloc[-1])
        st.metric("Volume", f"{volume:,.0f}")
    with col4:
        volatility = float(price_df['close'].pct_change().std()) * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['open'],
        high=price_df['high'],
        low=price_df['low'],
        close=price_df['close'],
        name='Price'
    ))
    fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price (USD)", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment & Predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📰 Sentiment Analysis")
        if not news_df.empty:
            sentiment_series = news_to_sentiment_series(news_df, price_df.index, window="1D")
            avg_sentiment = float(sentiment_series.mean())
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            fig_sent = px.line(sentiment_series, title="Sentiment Trend")
            st.plotly_chart(fig_sent, use_container_width=True)
            
            st.write("Recent Headlines:")
            for idx, row in news_df.head(3).iterrows():
                st.caption(f"• {row['title'][:100]}...")
        else:
            st.info("No news data available")
            sentiment_series = pd.Series(0.0, index=price_df.index)
    
    with col2:
        st.subheader("🤖 Model Prediction")
        model_path = f"models/{symbol}_model.pkl"
        if os.path.exists(model_path):
            predictor = Predictor(model_path=model_path)
            features = generate_features(price_df)
            
            # Add sentiment feature if not present
            if 'sentiment' not in features.columns:
                if not news_df.empty:
                    sentiment_aligned = news_to_sentiment_series(news_df, features.index, window="1H")
                    features['sentiment'] = sentiment_aligned
                else:
                    features['sentiment'] = 0.0
            
            try:
                preds = predictor.predict(features.tail(1))
                pred_return = float(preds[-1]) * 100
                
                if pred_return > 0.5:
                    st.success(f"🟢 BUY Signal: +{pred_return:.2f}%")
                elif pred_return < -0.5:
                    st.error(f"🔴 SELL Signal: {pred_return:.2f}%")
                else:
                    st.warning(f"🟡 HOLD: {pred_return:.2f}%")
                
                # Show prediction chart
                recent_preds = predictor.predict(features.tail(30))
                fig_pred = px.line(x=features.tail(30).index, y=recent_preds, 
                                   title="Recent Predictions", labels={"y": "Predicted Return"})
                st.plotly_chart(fig_pred, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(f"Model not trained for {symbol}. Go to Training tab.")

# Tab 2: Backtest
with tab2:
    st.title("🔄 Strategy Backtesting")
    
    model_path = f"models/{symbol}_model.pkl"
    if os.path.exists(model_path):
        predictor = Predictor(model_path=model_path)
        features = generate_features(price_df)
        
        # Add sentiment feature
        if 'sentiment' not in features.columns:
            if not news_df.empty:
                sentiment_aligned = news_to_sentiment_series(news_df, features.index, window="1H")
                features['sentiment'] = sentiment_aligned
            else:
                features['sentiment'] = 0.0
        
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        with col2:
            commission = st.number_input("Commission (%)", value=0.1, step=0.1) / 100
        
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                strategy = TradingStrategy(initial_capital=initial_capital, commission=commission)
                
                predictions = pd.Series(predictor.predict(features), index=features.index)
                signals, metrics = strategy.backtest(predictions, price_df['close'].loc[features.index])
                
                # Store metrics in session state for report
                st.session_state['backtest_metrics'] = metrics
                st.session_state['backtest_signals'] = signals
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics['total_return_pct']}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']}%")
                with col4:
                    st.metric("Win Rate", f"{metrics['win_rate_pct']}%")
                
                # Portfolio value chart
                fig = px.line(signals, x=signals.index, y='portfolio_value', 
                             title="Portfolio Value Over Time")
                fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", 
                             annotation_text="Initial Capital")
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade signals
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=price_df.index, y=price_df['close'], 
                                         mode='lines', name='Price'))
                
                buy_signals = signals[signals['signal'] == 1]
                sell_signals = signals[signals['signal'] == -1]
                
                fig2.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], 
                                         mode='markers', name='Buy', 
                                         marker=dict(color='green', size=10, symbol='triangle-up')))
                fig2.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], 
                                         mode='markers', name='Sell',
                                         marker=dict(color='red', size=10, symbol='triangle-down')))
                
                fig2.update_layout(title="Trade Signals", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Train a model first to run backtests")

# Tab 3: Report Generation
with tab3:
    st.title("📊 Analysis Report")
    
    st.write("Generate a comprehensive analysis report for this stock.")
    
    report_name = st.text_input("Report Name", value=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}")
    
    if st.button("📄 Generate Report"):
        with st.spinner("Analyzing data and generating report..."):
            report = generate_analysis_report(symbol, price_df, news_df)
            
            # Display report
            st.markdown(report)
            
            # Download button
            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=report,
                file_name=f"{report_name}.md",
                mime="text/markdown"
            )

# Tab 4: Model Info
with tab4:
    st.title("🤖 Model Information")
    
    metadata_path = f"models/{symbol}_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Info")
            st.json({
                "Symbol": metadata['symbol'],
                "Trained At": metadata['trained_at'],
                "Samples": metadata['n_samples'],
                "Features": metadata['features']
            })
        
        with col2:
            st.subheader("Performance Metrics")
            perf = metadata['performance']
            st.metric("R² Score", f"{perf['r2']:.4f}")
            st.metric("MAE", f"{perf['mae']:.6f}")
            st.metric("Directional Accuracy", f"{perf['directional_accuracy']*100:.2f}%")
    else:
        st.info("No model metadata available")

# Tab 5: Training
with tab5:
    st.title("⚙️ Model Training")
    
    st.write("Train a new model for this symbol:")
    
    if st.button("🚀 Train Model"):
        with st.spinner(f"Training model for {symbol}..."):
            from src.models.training_pipeline import TrainingPipeline
            
            pipeline = TrainingPipeline([symbol], lookback_days=365)
            try:
                model, metadata = pipeline.train_model(symbol)
                st.success(f"✅ Model trained successfully!")
                st.json(metadata)
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def generate_analysis_report(symbol: str, price_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
    """Generate comprehensive analysis report."""
    current_price = float(price_df['close'].iloc[-1])
    price_change = float(price_df['close'].pct_change().iloc[-1]) * 100
    volume = float(price_df['volume'].iloc[-1])
    avg_volume = float(price_df['volume'].mean())
    volatility = float(price_df['close'].pct_change().std()) * 100
    
    # Technical indicators
    features = generate_features(price_df)
    rsi = float(features['rsi_14'].iloc[-1])
    sma_5 = float(features['sma_5'].iloc[-1]) if 'sma_5' in features.columns else current_price
    sma_20 = float(features['sma_20'].iloc[-1]) if 'sma_20' in features.columns else current_price
    
    # Sentiment analysis
    sentiment_summary = "No sentiment data available"
    if not news_df.empty:
        sentiment_series = news_to_sentiment_series(news_df, price_df.index, window="1D")
        avg_sentiment = float(sentiment_series.mean())
        sentiment_summary = f"Average sentiment score: {avg_sentiment:.3f} ({len(news_df)} articles analyzed)"
    
    # Price trend
    if current_price > sma_20:
        trend = "Bullish (price above 20-day SMA)"
    else:
        trend = "Bearish (price below 20-day SMA)"
    
    # RSI interpretation
    if rsi > 70:
        rsi_signal = "Overbought (RSI > 70)"
    elif rsi < 30:
        rsi_signal = "Oversold (RSI < 30)"
    else:
        rsi_signal = "Neutral"
    
    # Get model prediction
    model_prediction = "N/A"
    model_path = f"models/{symbol}_model.pkl"
    if os.path.exists(model_path):
        try:
            predictor = Predictor(model_path=model_path)
            # Add sentiment feature
            if 'sentiment' not in features.columns:
                features['sentiment'] = 0.0
            pred = float(predictor.predict(features.tail(1))[0]) * 100
            if pred > 0.5:
                model_prediction = f"BUY (+{pred:.2f}%)"
            elif pred < -0.5:
                model_prediction = f"SELL ({pred:.2f}%)"
            else:
                model_prediction = f"HOLD ({pred:.2f}%)"
        except Exception:
            pass
    
    # Get backtest results if available
    backtest_summary = "Run backtest to see performance metrics"
    if 'backtest_metrics' in st.session_state:
        metrics = st.session_state['backtest_metrics']
        backtest_summary = f"""
- Total Return: {metrics['total_return_pct']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Max Drawdown: {metrics['max_drawdown_pct']}%
- Win Rate: {metrics['win_rate_pct']}%
- Total Trades: {metrics['total_trades']}
"""
    
    report = f"""
# Stock Analysis Report: {symbol}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

{symbol} is currently trading at **${current_price:.2f}**, showing a daily change of **{price_change:+.2f}%**.
The technical analysis suggests a **{trend}** trend with **{rsi_signal}** momentum conditions.

---

## Price Analysis

### Current Metrics
- **Current Price:** ${current_price:.2f}
- **Daily Change:** {price_change:+.2f}%
- **Volume:** {volume:,.0f} (Avg: {avg_volume:,.0f})
- **Volatility (Daily Std):** {volatility:.2f}%

### Technical Indicators
- **RSI (14):** {rsi:.2f} - {rsi_signal}
- **5-Day SMA:** ${sma_5:.2f}
- **20-Day SMA:** ${sma_20:.2f}
- **Trend:** {trend}

### Price Levels
- **52-Week High:** ${float(price_df['high'].max()):.2f}
- **52-Week Low:** ${float(price_df['low'].min()):.2f}
- **Current vs High:** {(current_price / float(price_df['high'].max()) - 1) * 100:.2f}%
- **Current vs Low:** {(current_price / float(price_df['low'].min()) - 1) * 100:.2f}%

---

## Sentiment Analysis

{sentiment_summary}

---

## Model Prediction

**Signal:** {model_prediction}

The ML model analyzes historical price patterns, technical indicators, and sentiment data to generate predictions.

---

## Backtest Results

{backtest_summary}

---

## Risk Assessment

### Volatility Analysis
- Daily volatility: {volatility:.2f}%
- Annualized volatility: {volatility * (252 ** 0.5):.2f}%
- Risk Level: {"High" if volatility > 3 else "Moderate" if volatility > 1.5 else "Low"}

### Market Position
- RSI Position: {rsi_signal}
- Trend: {trend}
- Volume: {"Above average" if volume > avg_volume else "Below average"}

---

## Recommendations

### Short-term (1-5 days)
Based on current technical indicators:
{f"- **BUY**: RSI is oversold, potential bounce expected" if rsi < 30 else 
 f"- **SELL**: RSI is overbought, potential pullback expected" if rsi > 70 else
 f"- **HOLD**: Wait for clearer signals"}

### Medium-term (1-4 weeks)
{f"- Price is above 20-day SMA, uptrend likely to continue" if current_price > sma_20 else
 f"- Price is below 20-day SMA, downtrend may persist"}

### Risk Management
- Set stop-loss at: ${current_price * 0.95:.2f} (-5%)
- Take profit target: ${current_price * 1.10:.2f} (+10%)
- Position size: Risk no more than 2% of portfolio

---

## Disclaimer

This report is generated for informational purposes only and should not be considered as financial advice.
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
Past performance does not guarantee future results.

---

*Report generated by Stock Sentiment Predictor Pro*
"""
    
    return report