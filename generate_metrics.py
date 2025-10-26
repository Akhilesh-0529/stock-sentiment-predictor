import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from src.models.predictor import Predictor
from src.data.live_api import fetch_price_history
from src.numerical.features import generate_features
import time
import warnings
warnings.filterwarnings('ignore')

def calculate_all_metrics(symbol='AAPL', period='365d'):
    """
    Calculate comprehensive metrics for a given stock symbol.
    """
    print(f"\n{'='*60}")
    print(f"CALCULATING METRICS FOR {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Fetch Data
    print("📊 Fetching historical data...")
    start_time = time.time()
    price_df = fetch_price_history(symbol, period=period, interval="1d")
    fetch_time = (time.time() - start_time) * 1000
    print(f"   ✓ Fetched {len(price_df)} days of data in {fetch_time:.0f}ms")
    
    # 2. Generate Features
    print("\n🔧 Generating features...")
    start_time = time.time()
    features = generate_features(price_df)
    features['sentiment'] = 0.0  # Placeholder
    feature_time = (time.time() - start_time) * 1000
    print(f"   ✓ Generated {features.shape[1]} features in {feature_time:.0f}ms")
    
    # 3. Split Data
    features = features.dropna()
    train_size = int(len(features) * 0.7)
    test_df = features[train_size:]
    
    feature_cols = [col for col in features.columns if col != 'return']
    X_test = test_df[feature_cols]
    y_test = test_df['return']
    
    print(f"\n📈 Data Split:")
    print(f"   Training samples: {train_size}")
    print(f"   Testing samples: {len(X_test)}")
    
    # 4. Load Model & Predict
    print(f"\n🤖 Loading model and making predictions...")
    try:
        predictor = Predictor(model_path=f"models/{symbol}_model.pkl")
        
        latencies = []
        for _ in range(100):
            start_time = time.time()
            _ = predictor.predict(X_test.iloc[0:1])
            latencies.append((time.time() - start_time) * 1000)
        avg_latency = np.mean(latencies)
        
        y_pred = predictor.predict(X_test)
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Average prediction latency: {avg_latency:.1f}ms")
        
    except FileNotFoundError:
        print(f"   ✗ Model not found for {symbol}. Please train it first:")
        print(f"     python -c \"from src.models.training_pipeline import TrainingPipeline; TrainingPipeline(['{symbol}']).train_model('{symbol}')\"")
        return None
    
    # 5. Calculate Metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    y_test_direction = np.sign(y_test)
    y_pred_direction = np.sign(y_pred)
    directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
    
    print(f"   R² Score:              {r2:.3f}")
    print(f"   MAE (Mean Abs Error):  {mae*100:.2f}%")
    print(f"   Directional Accuracy:  {directional_accuracy:.1f}%")
    
    # 6. Backtesting Metrics
    print(f"\n💰 BACKTESTING METRICS:")
    strategy_returns = np.where(y_pred_direction == 1, y_test.values, -y_test.values)
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = (cumulative_returns[-1] - 1) * 100
    
    if strategy_returns.std() > 0:
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
    
    print(f"   Total Return:          {total_return:+.1f}%")
    print(f"   Sharpe Ratio:          {sharpe_ratio:.2f}")
    print(f"   Max Drawdown:          {max_drawdown:.1f}%")
    print(f"   Win Rate:              {win_rate:.1f}%")
    
    # 7. Resume-Ready Summary
    print(f"\n📝 RESUME-READY BULLETS FOR {symbol}:")
    print(f"• Achieved {directional_accuracy:.1f}% directional accuracy on next-day predictions, trained on {train_size:,}+ samples.")
    print(f"• Engineered ML pipeline with <{avg_latency:.0f}ms prediction latency.")
    print(f"• Backtested strategy yielded a {sharpe_ratio:.2f} Sharpe ratio with a {max_drawdown:.1f}% max drawdown.")
    
    return {'symbol': symbol, 'directional_accuracy': directional_accuracy, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'train_samples': train_size}

def compare_multiple_stocks(symbols=['AAPL', 'TSLA', 'GOOGL'], period='365d'):
    all_metrics = [calculate_all_metrics(s, period) for s in symbols]
    all_metrics = [m for m in all_metrics if m]

    if not all_metrics:
        print("\nNo metrics calculated. Train your models first!")
        return

    print(f"\n\n{'='*60}")
    print(f"AGGREGATE METRICS ACROSS {len(all_metrics)} STOCKS")
    print(f"{'='*60}\n")
    
    avg_directional = np.mean([m['directional_accuracy'] for m in all_metrics])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
    total_samples = sum([m['train_samples'] for m in all_metrics])
    
    print(f"📊 Average Performance:")
    print(f"   Directional Accuracy:  {avg_directional:.1f}%")
    print(f"   Sharpe Ratio:          {avg_sharpe:.2f}")
    print(f"\n📈 Total Data Scale:")
    print(f"   Total training samples: {total_samples:,}")
    
    print(f"\n📝 MULTI-STOCK RESUME BULLET:")
    print(f"• Achieved {avg_directional:.1f}% average directional accuracy across {len(all_metrics)} stocks ({', '.join([m['symbol'] for m in all_metrics])}), trained on {total_samples:,}+ samples with an average {avg_sharpe:.2f} Sharpe ratio.")

if __name__ == "__main__":
    print("Metrics Calculator")
    choice = input("\nEnter choice (1=AAPL, 2=All, 3=Custom) or press Enter for All: ").strip()
    
    if choice == "1":
        calculate_all_metrics('AAPL', period='365d')
    elif choice == "3":
        symbol = input("Enter stock symbol: ").strip().upper()
        calculate_all_metrics(symbol, period='365d')
    else:
        compare_multiple_stocks(['AAPL', 'TSLA', 'GOOGL'], period='365d')
    
    print(f"\n{'='*60}\n✅ Metrics calculation complete!\n{'='*60}\n")
