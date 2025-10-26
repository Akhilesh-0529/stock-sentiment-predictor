# 📈 Stock Sentiment Predictor

An advanced machine learning system for stock price prediction that combines **technical analysis**, **sentiment analysis**, and **state-of-the-art deep learning** models.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-success.svg)

## 🚀 Features

### Core Capabilities
- 📊 **Real-time stock data** via yfinance API
- 📰 **News sentiment analysis** using NewsAPI + VADER
- 🤖 **Multiple ML models**: Random Forest, XGBoost, LSTM, Transformer
- 📈 **Interactive dashboard** built with Streamlit
- 🔄 **Backtesting framework** with trading strategy simulation
- 📄 **Automated PDF/Markdown reports**

### Advanced ML Components
- 🎯 **Heterogeneous Ensemble**: Combines Random Forest + XGBoost + Ridge with a meta-learner.
- 📉 **Uncertainty Quantification**: Provides confidence intervals using quantile regression.
- 🔀 **Market Regime Detection**: Uses HMM-based adaptive models for different market conditions.
- 💬 **Sentiment-Adaptive Weighting**: Dynamically adjusts feature weights based on news volume.
- 🧠 **Deep Learning Models**: Includes LSTM, Transformer, and Temporal Fusion Transformer (TFT).

## 📦 Installation

```bash
git clone https://github.com/Akhilesh-0529/stock-sentiment-predictor.git
cd stock-sentiment-predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Mac users, install OpenMP for XGBoost
brew install libomp
```

## 🎯 Quick Start

### 1. Run Dashboard

```bash
streamlit run src/api/simple_dashboard.py
```

### 2. Train Models

```bash
# Train a model for a single stock
python -c "from src.models.training_pipeline import TrainingPipeline; TrainingPipeline(['AAPL']).train_model('AAPL')"
```

## 🏗️ Architecture

The project is structured to separate concerns, from data ingestion to model deployment.

```
stock-sentiment-predictor/
├── src/
│   ├── data/
│   ├── sentiment/
│   ├── numerical/
│   ├── models/
│   ├── utils/
│   ├── backtest/
│   └── api/
├── models/
└── reports/
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. Do not use this system for actual trading without extensive validation and professional advice.

## 📄 License

This project is licensed under the MIT License.
