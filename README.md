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

### 1. Train Models

```bash
# Train a model for a single stock
python -c "from src.models.training_pipeline import TrainingPipeline; TrainingPipeline(['AAPL']).train_model('AAPL')"

# Train models for multiple stocks
python -c "from src.models.training_pipeline import TrainingPipeline; TrainingPipeline(['AAPL', 'TSLA', 'GOOGL']).train_all()"
```

### 2. Run Dashboard

```bash
streamlit run src/api/simple_dashboard.py
```

### 3. Run API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

## 🏗️ Architecture

```
stock-sentiment-predictor/
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Historical data fetching
│   │   └── live_api.py            # Live data + NewsAPI
│   ├── sentiment/
│   │   └── processors.py          # VADER sentiment analysis
│   ├── numerical/
│   │   └── features.py            # Technical indicators
│   ├── models/
│   │   ├── trainer.py             # Model training logic
│   │   ├── predictor.py           # Prediction logic
│   │   └── training_pipeline.py   # Full training pipeline
│   ├── utils/
│   │   └── report_generator.py    # Advanced ML components
│   ├── backtest/
│   │   └── strategy.py            # Trading strategy backtesting
│   └── api/
│       ├── simple_dashboard.py    # Streamlit dashboard
│       └── main.py                # FastAPI server
├── models/                         # Saved trained models
└── reports/                        # Generated PDF/Markdown reports
```

## 🤖 Models Implemented

### Traditional ML
- **Random Forest**: Ensemble of decision trees.
- **XGBoost**: Efficient gradient boosting.
- **Ridge Regression**: Linear model with L2 regularization.

### Ensemble & Adaptive Systems
- **Heterogeneous Stacking**: RF + XGBoost + Ridge fed into a meta-learner.
- **Uncertainty Quantification**: Quantile regression for confidence intervals.
- **Market Regime Detection**: HMM-based adaptive models.
- **Sentiment Weighting**: Dynamic feature importance based on news volume.

### Deep Learning (PyTorch)
- **LSTM**: Bidirectional LSTM for capturing sequential patterns.
- **Transformer**: Multi-head attention mechanism for time-series.
- **Temporal Fusion Transformer (TFT)**: Advanced architecture combining LSTM and attention.

## 🧪 Testing

A comprehensive test suite is included to ensure all components work correctly.

```bash
# Run all tests
python src/utils/report_generator.py
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. Financial markets are volatile and unpredictable. Do not use this system for actual trading without extensive validation and professional advice.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and open a pull request.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.
