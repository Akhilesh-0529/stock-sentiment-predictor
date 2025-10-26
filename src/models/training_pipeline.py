import pandas as pd
import numpy as np
from datetime import datetime
from src.data.ingestion import fetch_data
from src.numerical.features import generate_features
from src.models.trainer import Trainer
import json
import os


class TrainingPipeline:
    def __init__(self, symbols: list, lookback_days: int = 365):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.trainer = Trainer()
        
    def prepare_training_data(self, symbol: str):
        """Fetch and prepare complete training dataset."""
        print(f"Preparing data for {symbol}...")
        
        # Fetch historical data
        price_df = fetch_data(symbol, limit=self.lookback_days)
        
        # Generate technical features
        features = generate_features(price_df)
        
        # Add sentiment feature (default to 0 for now)
        features['sentiment'] = 0.0
        
        # Create target: next-day return
        price_df['target'] = price_df['close'].pct_change().shift(-1)
        target = price_df['target'].loc[features.index].dropna()
        
        # Align features and target
        features = features.loc[target.index]
        
        return features, target.values
    
    def train_model(self, symbol: str):
        """Train model for a specific symbol."""
        X, y = self.prepare_training_data(symbol)
        
        print(f"Training on {len(X)} samples with {len(X.columns)} features...")
        print(f"Features: {list(X.columns)}")
        
        model = self.trainer.train(X, y)
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "features": list(X.columns),
            "performance": self.evaluate_model(model, X, y)
        }
        
        os.makedirs("models", exist_ok=True)
        with open(f"models/{symbol}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save model with symbol-specific name
        self.trainer.model_path = f"models/{symbol}_model.pkl"
        import joblib
        joblib.dump(model, f"models/{symbol}_model.pkl")
        print(f"Model saved for {symbol}")
        return model, metadata
    
    def evaluate_model(self, model, X, y):
        """Calculate performance metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        preds = model.predict(X)
        
        return {
            "mse": float(mean_squared_error(y, preds)),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
            "directional_accuracy": float((np.sign(preds) == np.sign(y)).mean())
        }
    
    def train_all(self):
        """Train models for all symbols."""
        results = {}
        for symbol in self.symbols:
            try:
                model, metadata = self.train_model(symbol)
                results[symbol] = metadata
            except Exception as e:
                print(f"Failed to train {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        return results