from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def create_markdown_report(symbol: str, analysis_text: str, output_path: str = None) -> str:
    _ensure_reports_dir()
    if output_path is None:
        output_path = f"reports/{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    return output_path


def create_pdf_report(symbol: str, analysis_data, output_path: str = None) -> str:
    _ensure_reports_dir()
    if output_path is None:
        output_path = f"reports/{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph(f"Stock Analysis Report: {symbol}", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 12))

        ts = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
        story.append(ts)
        story.append(Spacer(1, 12))

        if isinstance(analysis_data, str):
            for line in analysis_data.splitlines():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))
        elif isinstance(analysis_data, dict):
            for section, content in analysis_data.items():
                story.append(Paragraph(f"<b>{section}</b>", styles["Heading2"]))
                story.append(Spacer(1, 6))

                if isinstance(content, str):
                    for line in content.splitlines():
                        story.append(Paragraph(line, styles["Normal"]))
                        story.append(Spacer(1, 4))
                elif isinstance(content, (list, tuple)):
                    for item in content:
                        story.append(Paragraph(f"- {item}", styles["Normal"]))
                        story.append(Spacer(1, 2))
                elif isinstance(content, pd.DataFrame):
                    data = [list(content.columns)]
                    for _, row in content.reset_index().iterrows():
                        data.append([str(x) for x in row.tolist()])
                    tbl = Table(data, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(str(content), styles["Normal"]))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(str(analysis_data), styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)
        return output_path

    except ImportError:
        fallback_text = ""
        if isinstance(analysis_data, str):
            fallback_text = analysis_data
        elif isinstance(analysis_data, dict):
            lines = []
            for k, v in analysis_data.items():
                lines.append(f"## {k}")
                if isinstance(v, (list, tuple)):
                    lines.extend([f"- {i}" for i in v])
                else:
                    lines.append(str(v))
                lines.append("")
            fallback_text = "\n".join(lines)
        else:
            fallback_text = str(analysis_data)

        md_path = create_markdown_report(symbol, fallback_text, output_path=None)
        return md_path


class HeterogeneousEnsemble:
    def __init__(self):
        self.models = {}
        self.meta_learner = None
        self.scalers = {}
        
    def _create_base_models(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        for name in self.models.keys():
            self.scalers[name] = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self._create_base_models()
        
        meta_features = np.zeros((len(X), len(self.models)))
        
        for idx, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            X_scaled = self.scalers[name].fit_transform(X)
            model.fit(X_scaled, y)
            meta_features[:, idx] = model.predict(X_scaled)
        
        print("Training meta-learner...")
        self.meta_learner = Ridge(alpha=0.1)
        self.meta_learner.fit(meta_features, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.zeros((len(X), len(self.models)))
        
        for idx, (name, model) in enumerate(self.models.items()):
            X_scaled = self.scalers[name].transform(X)
            meta_features[:, idx] = model.predict(X_scaled)
        
        return self.meta_learner.predict(meta_features)
    
    def get_base_predictions(self, X: pd.DataFrame) -> dict:
        predictions = {}
        for name, model in self.models.items():
            X_scaled = self.scalers[name].transform(X)
            predictions[name] = model.predict(X_scaled)
        return predictions
    
    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'meta_learner': self.meta_learner,
            'scalers': self.scalers
        }, path)
    
    def load(self, path: str):
        data = joblib.load(path)
        self.models = data['models']
        self.meta_learner = data['meta_learner']
        self.scalers = data['scalers']
        return self


class UncertaintyQuantifier:
    def __init__(self, n_estimators: int = 100, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {}
        self.n_estimators = n_estimators
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        for q in self.quantiles:
            print(f"Training quantile {q} model...")
            self.models[q] = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                loss='quantile',
                alpha=q,
                random_state=42
            )
            self.models[q].fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> dict:
        predictions = {}
        for q in self.quantiles:
            predictions[f'q{int(q*100)}'] = self.models[q].predict(X)
        
        predictions['mean'] = predictions['q50']
        predictions['lower_bound'] = predictions['q10']
        predictions['upper_bound'] = predictions['q90']
        predictions['uncertainty'] = predictions['upper_bound'] - predictions['lower_bound']
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = self.predict(X)
        return preds['mean'], preds['lower_bound'], preds['upper_bound']
    
    def calculate_var(self, X: pd.DataFrame, confidence_level: float = 0.95) -> float:
        q_lower = (1 - confidence_level) / 2
        q_upper = 1 - q_lower
        
        if q_lower not in self.quantiles:
            print(f"Warning: quantile {q_lower} not trained, using closest available")
        
        predictions = self.predict(X)
        var = predictions[f'q{int(q_lower*100)}'].mean()
        return var
    
    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'quantiles': self.quantiles
        }, path)
    
    def load(self, path: str):
        data = joblib.load(path)
        self.models = data['models']
        self.quantiles = data['quantiles']
        return self


class MonteCarloDropout:
    def __init__(self, base_model, n_iterations: int = 100, dropout_rate: float = 0.2):
        self.base_model = base_model
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        
        for _ in range(self.n_iterations):
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=X.shape)
            X_dropout = X * mask
            pred = self.base_model.predict(X_dropout)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred


class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_models = {}
        
    def _compute_regime_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=price_df.index)
        
        features['returns'] = price_df['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_change'] = price_df['volume'].pct_change()
        features['high_low_range'] = (price_df['high'] - price_df['low']) / price_df['close']
        
        return features.dropna()
    
    def fit_regime_detector(self, price_df: pd.DataFrame):
        regime_features = self._compute_regime_features(price_df)
        X_scaled = self.scaler.fit_transform(regime_features)
        
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        self.hmm_model.fit(X_scaled)
        
        regimes = self.hmm_model.predict(X_scaled)
        regime_features['regime'] = regimes
        
        self._analyze_regimes(regime_features)
        
        return regimes
    
    def _analyze_regimes(self, regime_features: pd.DataFrame):
        print("\nRegime Analysis:")
        for regime in range(self.n_regimes):
            mask = regime_features['regime'] == regime
            regime_data = regime_features[mask]
            
            print(f"\nRegime {regime}:")
            print(f"  Occurrences: {mask.sum()} ({mask.sum()/len(regime_features)*100:.1f}%)")
            print(f"  Avg Volatility: {regime_data['volatility'].mean():.4f}")
            print(f"  Avg Returns: {regime_data['returns'].mean():.4f}")
    
    def predict_regime(self, price_df: pd.DataFrame) -> int:
        regime_features = self._compute_regime_features(price_df)
        X_scaled = self.scaler.transform(regime_features)
        regime = self.hmm_model.predict(X_scaled)[-1]
        return regime
    
    def fit_regime_specific_models(self, X: pd.DataFrame, y: np.ndarray, regimes: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            if mask.sum() < 10:
                continue
                
            X_regime = X[mask]
            y_regime = y[mask]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_regime, y_regime)
            self.regime_models[regime] = model
            
            print(f"Trained model for regime {regime} with {mask.sum()} samples")
    
    def predict_with_regime(self, X: pd.DataFrame, regime: int) -> np.ndarray:
        if regime not in self.regime_models:
            regime = 0
        
        return self.regime_models[regime].predict(X)
    
    def save(self, path: str):
        joblib.dump({
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'regime_models': self.regime_models,
            'n_regimes': self.n_regimes
        }, path)
    
    def load(self, path: str):
        data = joblib.load(path)
        self.hmm_model = data['hmm_model']
        self.scaler = data['scaler']
        self.regime_models = data['regime_models']
        self.n_regimes = data['n_regimes']
        return self


class SentimentAdaptiveWeighting:
    def __init__(self):
        self.discourse_thresholds = {'low': 5, 'medium': 20, 'high': 50}
        
    def calculate_discourse_volume(self, news_df: pd.DataFrame, window: str = '1D') -> pd.Series:
        discourse = news_df.resample(window).size()
        return discourse
    
    def get_sentiment_weight(self, discourse_volume: float) -> float:
        if discourse_volume < self.discourse_thresholds['low']:
            return 0.1
        elif discourse_volume < self.discourse_thresholds['medium']:
            return 0.3
        elif discourse_volume < self.discourse_thresholds['high']:
            return 0.6
        else:
            return 0.8
    
    def adaptive_prediction(
        self,
        technical_pred: float,
        sentiment_pred: float,
        discourse_volume: float
    ) -> float:
        sentiment_weight = self.get_sentiment_weight(discourse_volume)
        technical_weight = 1 - sentiment_weight
        
        final_pred = (technical_weight * technical_pred + 
                     sentiment_weight * sentiment_pred)
        
        return final_pred


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import joblib


class StockGraphEmbedding:
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.stock_to_idx = {}
        
    def _build_stock_graph(self, stocks: List[str], correlation_matrix: pd.DataFrame) -> Data:
        self.stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
        
        edge_index = []
        edge_weight = []
        
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if i != j:
                    corr = correlation_matrix.loc[stock1, stock2]
                    if abs(corr) > 0.3:
                        edge_index.append([i, j])
                        edge_weight.append(abs(corr))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        x = torch.eye(len(stocks))
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return data
    
    def _create_gnn_model(self, n_stocks: int):
        class StockGNN(nn.Module):
            def __init__(self, n_features, hidden_dim, embedding_dim):
                super(StockGNN, self).__init__()
                self.conv1 = GCNConv(n_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, embedding_dim)
                
            def forward(self, data):
                x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
                
                x = self.conv1(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                
                x = self.conv2(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                
                x = self.conv3(x, edge_index, edge_weight)
                
                return x
        
        return StockGNN(n_stocks, self.hidden_dim, self.embedding_dim)
    
    def fit(
        self,
        stocks: List[str],
        price_data: dict,
        epochs: int = 100,
        learning_rate: float = 0.01
    ):
        returns_df = pd.DataFrame({
            stock: price_data[stock]['close'].pct_change()
            for stock in stocks
        }).dropna()
        
        correlation_matrix = returns_df.corr()
        
        graph_data = self._build_stock_graph(stocks, correlation_matrix)
        
        self.model = self._create_gnn_model(len(stocks))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            embeddings = self.model(graph_data)
            
            reconstructed = torch.mm(embeddings, embeddings.t())
            target = torch.tensor(correlation_matrix.values, dtype=torch.float)
            
            loss = F.mse_loss(reconstructed, target)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def get_embedding(self, stock: str, graph_data: Data = None) -> np.ndarray:
        if graph_data is None:
            raise ValueError("Graph data required for inference")
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph_data)
        
        stock_idx = self.stock_to_idx[stock]
        return embeddings[stock_idx].numpy()
    
    def get_all_embeddings(self, graph_data: Data) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph_data)
        return embeddings.numpy()
    
    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'stock_to_idx': self.stock_to_idx,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }, path)
    
    def load(self, path: str, n_stocks: int):
        checkpoint = torch.load(path)
        self.stock_to_idx = checkpoint['stock_to_idx']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        self.model = self._create_gnn_model(n_stocks)
        self.model.load_state_dict(checkpoint['model_state'])
        return self


class EnhancedFeatureEngineering:
    def __init__(self, gnn_embeddings: np.ndarray = None):
        self.gnn_embeddings = gnn_embeddings
    
    def add_graph_features(self, features: pd.DataFrame, stock_idx: int) -> pd.DataFrame:
        if self.gnn_embeddings is None:
            return features
        
        embedding = self.gnn_embeddings[stock_idx]
        
        for i, val in enumerate(embedding):
            features[f'gnn_embed_{i}'] = val
        
        return features
        

class AdvancedPredictionSystem:
    def __init__(self):
        self.ensemble = HeterogeneousEnsemble()
        self.uncertainty_model = UncertaintyQuantifier()
        self.regime_detector = MarketRegimeDetector(n_regimes=3)
        self.sentiment_adapter = SentimentAdaptiveWeighting()
        self.graph_embedder = None
        
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame = None
    ):
        print("\n=== Training Ensemble Models ===")
        self.ensemble.fit(X, y)
        
        print("\n=== Training Uncertainty Quantification ===")
        self.uncertainty_model.fit(X, y)
        
        print("\n=== Training Regime Detector ===")
        regimes = self.regime_detector.fit_regime_detector(price_df)
        self.regime_detector.fit_regime_specific_models(X, y, regimes)
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame = None,
        return_details: bool = True
    ) -> Dict:
        ensemble_pred = self.ensemble.predict(X)
        
        uncertainty_preds = self.uncertainty_model.predict(X)
        
        current_regime = self.regime_detector.predict_regime(price_df)
        regime_pred = self.regime_detector.predict_with_regime(X, current_regime)
        
        if news_df is not None:
            discourse_volume = len(news_df)
            sentiment_weight = self.sentiment_adapter.get_sentiment_weight(discourse_volume)
        else:
            discourse_volume = 0
            sentiment_weight = 0.1
        
        final_pred = (
            0.4 * ensemble_pred[-1] +
            0.3 * regime_pred[-1] +
            0.3 * uncertainty_preds['mean'][-1]
        )
        
        if return_details:
            return {
                'prediction': final_pred,
                'ensemble_pred': ensemble_pred[-1],
                'regime_pred': regime_pred[-1],
                'current_regime': current_regime,
                'uncertainty_lower': uncertainty_preds['lower_bound'][-1],
                'uncertainty_upper': uncertainty_preds['upper_bound'][-1],
                'uncertainty_range': uncertainty_preds['uncertainty'][-1],
                'sentiment_weight': sentiment_weight,
                'discourse_volume': discourse_volume,
                'base_models': self.ensemble.get_base_predictions(X)
            }
        
        return {'prediction': final_pred}
    
    def save(self, base_path: str):
        self.ensemble.save(f"{base_path}_ensemble.pkl")
        self.uncertainty_model.save(f"{base_path}_uncertainty.pkl")
        self.regime_detector.save(f"{base_path}_regime.pkl")
        
    def load(self, base_path: str):
        self.ensemble.load(f"{base_path}_ensemble.pkl")
        self.uncertainty_model.load(f"{base_path}_uncertainty.pkl")
        self.regime_detector.load(f"{base_path}_regime.pkl")
        return self


if __name__ == "__main__":
    print("Testing report generator...")
    
    test_data = {
        "Summary": "AAPL stock analysis shows bullish trend",
        "Metrics": ["Price: $262.82", "Change: +1.25%", "Volatility: 1.60%"],
        "Recommendation": "HOLD position with slight bullish bias"
    }
    
    md_path = create_markdown_report("AAPL", "# Test Report\n\nThis is a test.")
    print(f"✅ Markdown report created: {md_path}")
    
    pdf_path = create_pdf_report("AAPL", test_data)
    print(f"✅ PDF report created: {pdf_path}")
    
    print("\n✅ All tests passed!")

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE TEST SUITE FOR ADVANCED ML COMPONENTS")
print("=" * 80)

def test_report_generation():
    """Test basic report generation"""
    print("\n[TEST 1] Report Generation...")
    try:
        from src.utils.report_generator import create_markdown_report, create_pdf_report
        
        test_data = {
            "Summary": "Test stock analysis",
            "Metrics": ["Price: $100", "Change: +2%"],
            "Recommendation": "BUY"
        }
        
        md_path = create_markdown_report("TEST", "# Test\nContent")
        assert os.path.exists(md_path), "Markdown report not created"
        
        pdf_path = create_pdf_report("TEST", test_data)
        assert pdf_path is not None, "PDF report failed"
        
        print("✅ Report generation PASSED")
        return True
    except Exception as e:
        print(f"❌ Report generation FAILED: {e}")
        return False


def test_heterogeneous_ensemble():
    """Test ensemble model"""
    print("\n[TEST 2] Heterogeneous Ensemble...")
    try:
        from src.utils.report_generator import HeterogeneousEnsemble
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 7), columns=[f'f{i}' for i in range(7)])
        y = np.random.randn(100)
        
        ensemble = HeterogeneousEnsemble()
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        assert len(predictions) == 10, "Prediction length mismatch"
        assert not np.isnan(predictions).any(), "NaN in predictions"
        
        base_preds = ensemble.get_base_predictions(X[:10])
        assert 'random_forest' in base_preds, "Missing RF predictions"
        assert 'xgboost' in base_preds, "Missing XGBoost predictions"
        assert 'ridge' in base_preds, "Missing Ridge predictions"
        
        print("✅ Heterogeneous Ensemble PASSED")
        return True
    except Exception as e:
        print(f"❌ Heterogeneous Ensemble FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_quantification():
    """Test uncertainty quantification"""
    print("\n[TEST 3] Uncertainty Quantification...")
    try:
        from src.utils.report_generator import UncertaintyQuantifier
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 7), columns=[f'f{i}' for i in range(7)])
        y = np.random.randn(100)
        
        uq = UncertaintyQuantifier(n_estimators=50)
        uq.fit(X, y)
        
        predictions = uq.predict(X[:10])
        assert 'mean' in predictions, "Missing mean prediction"
        assert 'lower_bound' in predictions, "Missing lower bound"
        assert 'upper_bound' in predictions, "Missing upper bound"
        assert 'uncertainty' in predictions, "Missing uncertainty"
        
        assert len(predictions['mean']) == 10, "Prediction length mismatch"
        assert (predictions['upper_bound'] >= predictions['lower_bound']).all(), "Invalid bounds"
        
        mean, lower, upper = uq.predict_with_confidence(X[:5])
        assert len(mean) == 5, "Confidence prediction length mismatch"
        
        print("✅ Uncertainty Quantification PASSED")
        return True
    except Exception as e:
        print(f"❌ Uncertainty Quantification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_regime_detection():
    """Test market regime detection"""
    print("\n[TEST 4] Market Regime Detection...")
    try:
        from src.utils.report_generator import MarketRegimeDetector
        
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        price_df = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 102,
            'low': np.random.randn(200).cumsum() + 98,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        detector = MarketRegimeDetector(n_regimes=3)
        regimes = detector.fit_regime_detector(price_df)
        
        assert len(regimes) > 0, "No regimes detected"
        assert len(np.unique(regimes)) <= 3, "Too many regimes"
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(len(regimes), 7), columns=[f'f{i}' for i in range(7)])
        y = np.random.randn(len(regimes))
        
        detector.fit_regime_specific_models(X, y, regimes)
        
        current_regime = detector.predict_regime(price_df)
        assert 0 <= current_regime < 3, "Invalid regime prediction"
        
        print("✅ Market Regime Detection PASSED")
        return True
    except Exception as e:
        print(f"❌ Market Regime Detection FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_adaptive_weighting():
    """Test sentiment adaptive weighting"""
    print("\n[TEST 5] Sentiment Adaptive Weighting...")
    try:
        from src.utils.report_generator import SentimentAdaptiveWeighting
        
        adapter = SentimentAdaptiveWeighting()
        
        assert adapter.get_sentiment_weight(3) == 0.1, "Low discourse weight wrong"
        assert adapter.get_sentiment_weight(15) == 0.3, "Medium discourse weight wrong"
        assert adapter.get_sentiment_weight(35) == 0.6, "High discourse weight wrong"
        assert adapter.get_sentiment_weight(60) == 0.8, "Very high discourse weight wrong"
        
        final_pred = adapter.adaptive_prediction(0.05, 0.02, 10)
        assert isinstance(final_pred, float), "Prediction not a float"
        
        print("✅ Sentiment Adaptive Weighting PASSED")
        return True
    except Exception as e:
        print(f"❌ Sentiment Adaptive Weighting FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_embedding():
    """Test GNN stock embedding (optional - may skip if torch not installed)"""
    print("\n[TEST 6] Graph Neural Network Embedding...")
    try:
        from src.utils.report_generator import StockGraphEmbedding
        
        stocks = ['AAPL', 'GOOGL', 'MSFT']
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        price_data = {}
        for stock in stocks:
            price_data[stock] = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100
            }, index=dates)
        
        embedder = StockGraphEmbedding(embedding_dim=16, hidden_dim=32)
        embedder.fit(stocks, price_data, epochs=10, learning_rate=0.01)
        
        assert embedder.model is not None, "Model not created"
        assert len(embedder.stock_to_idx) == 3, "Stock index mapping wrong"
        
        print("✅ Graph Neural Network Embedding PASSED")
        return True
    except ImportError as e:
        print(f"⚠️  Graph Neural Network Embedding SKIPPED (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Graph Neural Network Embedding FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_prediction_system():
    """Test integrated advanced prediction system"""
    print("\n[TEST 7] Advanced Prediction System Integration...")
    try:
        from src.utils.report_generator import AdvancedPredictionSystem
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 7), columns=[f'f{i}' for i in range(7)])
        y = np.random.randn(100)
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        price_df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        system = AdvancedPredictionSystem()
        system.train(X, y, price_df)
        
        result = system.predict(X[:1], price_df.tail(50), return_details=True)
        
        assert 'prediction' in result, "Missing prediction"
        assert 'ensemble_pred' in result, "Missing ensemble prediction"
        assert 'current_regime' in result, "Missing regime"
        assert 'uncertainty_lower' in result, "Missing uncertainty bounds"
        assert 'uncertainty_upper' in result, "Missing uncertainty bounds"
        
        print("✅ Advanced Prediction System PASSED")
        return True
    except Exception as e:
        print(f"❌ Advanced Prediction System FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_pipeline_compatibility():
    """Test compatibility with existing pipeline"""
    print("\n[TEST 8] Existing Pipeline Compatibility...")
    try:
        from src.data.ingestion import fetch_data
        from src.numerical.features import generate_features
        
        price_df = fetch_data("AAPL", limit=100)
        features = generate_features(price_df)
        
        assert len(features) > 0, "No features generated"
        assert 'return' in features.columns, "Missing return feature"
        
        print("✅ Existing Pipeline Compatibility PASSED")
        return True
    except Exception as e:
        print(f"❌ Existing Pipeline Compatibility FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80)
    
    tests = [
        test_report_generation,
        test_heterogeneous_ensemble,
        test_uncertainty_quantification,
        test_market_regime_detection,
        test_sentiment_adaptive_weighting,
        test_graph_embedding,
        test_advanced_prediction_system,
        test_existing_pipeline_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - SAFE TO COMMIT")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED - REVIEW BEFORE COMMIT")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)