import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.live_api import fetch_price_history
from src.numerical.features import generate_features
from src.models.predictor import Predictor
import os

st.set_page_config(layout="wide", page_title="Stock Predictor")

st.title("📈 Stock Sentiment Predictor")

# Sidebar
symbol = st.sidebar.text_input("Symbol", value="AAPL")
period = st.sidebar.selectbox("Period", ["7d", "30d", "60d"], index=1)

# Fetch data
try:
    price_df = fetch_price_history(symbol, period=period, interval="1d")
    
    # Display current price
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price", f"${float(price_df['close'].iloc[-1]):.2f}")
    with col2:
        change = float(price_df['close'].pct_change().iloc[-1]) * 100
        st.metric("Change", f"{change:.2f}%")
    with col3:
        vol = float(price_df['close'].pct_change().std()) * 100
        st.metric("Volatility", f"{vol:.2f}%")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['open'],
        high=price_df['high'],
        low=price_df['low'],
        close=price_df['close']
    ))
    fig.update_layout(title=f"{symbol} Price", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model prediction
    st.subheader("🤖 Model Prediction")
    
    model_path = f"models/{symbol}_model.pkl"
    if os.path.exists(model_path):
        try:
            predictor = Predictor(model_path=model_path)
            
            # Generate features (6 features)
            features = generate_features(price_df)
            st.write(f"Generated {len(features.columns)} features: {list(features.columns)}")
            
            # Add sentiment ONLY ONCE
            if 'sentiment' not in features.columns:
                features['sentiment'] = 0.0
            
            st.write(f"Total features with sentiment: {len(features.columns)} - {list(features.columns)}")
            
            # Predict on last row only
            X_pred = features.tail(1)
            st.write(f"Predicting with shape: {X_pred.shape}, columns: {list(X_pred.columns)}")
            
            pred = predictor.predict(X_pred)
            pred_pct = float(pred[0]) * 100
            
            if pred_pct > 0.5:
                st.success(f"🟢 BUY: +{pred_pct:.2f}%")
            elif pred_pct < -0.5:
                st.error(f"🔴 SELL: {pred_pct:.2f}%")
            else:
                st.warning(f"🟡 HOLD: {pred_pct:.2f}%")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Train a model first")
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training..."):
                from src.models.training_pipeline import TrainingPipeline
                pipeline = TrainingPipeline([symbol])
                model, metadata = pipeline.train_model(symbol)
                st.success("Training complete!")
                st.json(metadata)
    
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())