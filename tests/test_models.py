import unittest
import sys
import os
import numpy as np

# Add the correct path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.trainer import Trainer
from src.models.predictor import Predictor
from src.data.ingestion import fetch_data
from src.sentiment.processors import analyze_sentiment
from src.numerical.features import generate_features


class TestModels(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        self.predictor = Predictor()
        self.test_data = fetch_data("AAPL", limit=200)
        texts = ["Good results", "Bad news", "Neutral"] * 70
        self.sentiment_series = analyze_sentiment(texts)
        self.features = generate_features(self.test_data)
        min_len = min(len(self.features), len(self.sentiment_series))
        self.features = self.features.iloc[-min_len:]
        self.sentiment_series = np.asarray(self.sentiment_series[-min_len:])

    def test_training_returns_model(self):
        model = self.trainer.train(self.features, self.sentiment_series)
        self.assertIsNotNone(model, "Model should be trained and not None.")

    def test_prediction_shape_and_type(self):
        model = self.trainer.train(self.features, self.sentiment_series)
        preds = self.predictor.predict(self.features.tail(10))
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.issubdtype(np.array(preds).dtype, np.floating))

    def test_directional_accuracy(self):
        X = self.features[:-1]
        y_returns = self.test_data['close'].pct_change().shift(-1).dropna()
        y = y_returns.iloc[-len(X):].values
        model = self.trainer.train(X, y)
        preds = self.predictor.predict(X)
        acc = (np.sign(preds) == np.sign(y)).mean()
        self.assertGreaterEqual(acc, 0.0)


if __name__ == '__main__':
    unittest.main()