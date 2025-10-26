import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical features from OHLCV data - EXACTLY 6 features."""
    df = df.copy()
    
    # Generate features
    df['return'] = df['close'].pct_change().fillna(0)
    df['log_return'] = np.log1p(df['return'])
    df['sma_5'] = df['close'].rolling(5, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Return ONLY these 6 features, in this exact order
    return df[['return', 'log_return', 'sma_5', 'sma_20', 'rsi_14', 'vol_change']]