import pandas as pd
import numpy as np


def fetch_data(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data using yfinance or synthetic fallback."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period="1y", progress=False)
        if df.empty:
            raise RuntimeError("yfinance returned empty")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        if len(df) > limit:
            df = df.tail(limit)
        return df
    except Exception:
        rng = pd.date_range(end=pd.Timestamp.today(), periods=limit, freq='B')
        rnd = np.random.RandomState(42)
        returns = rnd.normal(0, 0.01, size=len(rng))
        price = 100 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            'open': price * 0.995,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': rnd.randint(100000, 1000000, size=len(rng))
        }, index=rng)
        return df