# 📈 Stock Sentiment Predictor

A machine learning-powered stock prediction system that combines technical analysis with sentiment analysis from news data.

## Features

- 📊 **Real-time stock data** via yfinance
- 📰 **News sentiment analysis** using NewsAPI
- 🤖 **ML predictions** with Random Forest
- 📈 **Interactive dashboard** built with Streamlit
- 🔄 **Backtesting** trading strategies
- 📊 **Automated reports** generation

## Installation

```bash
# Clone the repository
cd stock-sentiment-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

```bash
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

## Project Structure

```
stock-sentiment-predictor/
├── src/
│   ├── data/
│   │   ├── ingestion.py          # Data fetching
│   │   └── live_api.py           # Live data & news API
│   ├── sentiment/
│   │   └── processors.py         # Sentiment analysis
│   ├── numerical/
│   │   └── features.py           # Technical indicators
│   ├── models/
│   │   ├── trainer.py            # Model training
│   │   ├── predictor.py          # Predictions
│   │   └── training_pipeline.py  # Full training pipeline
│   ├── backtest/
│   │   └── strategy.py           # Trading strategy backtesting
│   └── api/
│       ├── simple_dashboard.py   # Streamlit dashboard
│       ├── enhanced_dashboard.py # Full-featured dashboard
│       └── main.py               # FastAPI server
├── models/                        # Trained models
├── requirements.txt
└── README.md
```

## Technical Features

The system generates 6 technical features:
- **return**: Price return
- **log_return**: Log returns
- **sma_5**: 5-day simple moving average
- **sma_20**: 20-day simple moving average
- **rsi_14**: 14-day Relative Strength Index
- **vol_change**: Volume change percentage

Plus 1 sentiment feature from news analysis = **7 total features**

## Model

- **Algorithm**: Random Forest Regressor
- **Pipeline**: StandardScaler → RandomForestRegressor
- **Target**: Next-day return prediction
- **Features**: 7 (6 technical + 1 sentiment)

## Configuration

### NewsAPI Key

Set your NewsAPI key (free at https://newsapi.org):

```bash
export NEWSAPI_KEY="your_key_here"
```

Or edit `src/config/settings.py`

## Usage Examples

### Python API

```python
from src.models.predictor import Predictor
from src.data.live_api import fetch_price_history
from src.numerical.features import generate_features

# Fetch data
price_df = fetch_price_history("AAPL", period="30d", interval="1d")

# Generate features
features = generate_features(price_df)
features['sentiment'] = 0.0  # Add sentiment

# Predict
predictor = Predictor(model_path="models/AAPL_model.pkl")
prediction = predictor.predict(features.tail(1))
print(f"Predicted return: {prediction[0] * 100:.2f}%")
```

### REST API

```bash
# Start server
uvicorn src.api.main:app --reload

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "30d"}'
```

## Dashboard Features

### Simple Dashboard
- Live price display
- Candlestick charts
- Model predictions (BUY/SELL/HOLD signals)
- One-click model training

### Enhanced Dashboard (Work in Progress)
- Multi-tab interface
- Backtesting with performance metrics
- Automated report generation
- Model performance visualization
- Trading signals overlay

## Performance Metrics

The system tracks:
- **R² Score**: Model fit quality
- **MAE**: Mean Absolute Error
- **Directional Accuracy**: % of correct predictions
- **Sharpe Ratio**: Risk-adjusted returns (backtesting)
- **Max Drawdown**: Largest peak-to-trough decline

## Roadmap

- [ ] Support for multiple stocks in one view
- [ ] Real-time alerts via email/Telegram
- [ ] Deep learning models (LSTM/Transformer)
- [ ] Options pricing predictions
- [ ] Portfolio optimization
- [ ] Paper trading integration

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Disclaimer

⚠️ **This is for educational purposes only.** 

Do not use this system for actual trading without proper testing and risk management. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## License

MIT License - see LICENSE file for details

## Contact

For questions or suggestions, open an issue on GitHub.

---

**Happy Trading! 📈🚀**# stock-sentiment-predictor
