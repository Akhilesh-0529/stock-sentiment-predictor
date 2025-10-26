import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.sentiment.processors import analyze_sentiment

NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "fb153bd4858d48e99dc8d0e7ed4b8339")


def fetch_price_history(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """Fetch historical intraday price data."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {symbol}")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_news(query: str, from_days: int = 7, page_size: int = 50) -> pd.DataFrame:
    """Fetch recent news using NewsAPI."""
    if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here" or NEWSAPI_KEY == "":
        return pd.DataFrame(columns=["publishedAt", "title", "description", "url"]).set_index(pd.DatetimeIndex([]))
    
    try:
        params = {
            "q": query,
            "apiKey": NEWSAPI_KEY,
            "pageSize": page_size,
            "from": (datetime.utcnow() - timedelta(days=from_days)).isoformat(),
            "language": "en",
            "sortBy": "publishedAt"
        }
        r = requests.get(NEWSAPI_URL, params=params, timeout=10)
        
        if r.status_code == 401:
            return pd.DataFrame(columns=["publishedAt", "title", "description", "url"]).set_index(pd.DatetimeIndex([]))
        
        r.raise_for_status()
        articles = r.json().get("articles", [])
        
        rows = []
        for a in articles:
            rows.append({
                "publishedAt": pd.to_datetime(a.get("publishedAt")),
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", "")
            })
        
        df = pd.DataFrame(rows)
        if df.empty:
            return df.set_index(pd.DatetimeIndex([]))
        return df.set_index("publishedAt")
    except Exception as e:
        return pd.DataFrame(columns=["publishedAt", "title", "description", "url"]).set_index(pd.DatetimeIndex([]))


def news_to_sentiment_series(news_df: pd.DataFrame, price_index: pd.DatetimeIndex, window: str = "1H") -> pd.Series:
    """Convert news articles to time-indexed sentiment aligned with price data."""
    if news_df.empty:
        return pd.Series(0.0, index=price_index)
    
    texts = (news_df['title'].fillna('') + ". " + news_df['description'].fillna('')).tolist()
    scores = analyze_sentiment(texts)
    
    news_df = news_df.copy()
    news_df['score'] = scores
    
    sentiment_resampled = news_df['score'].resample(window).mean().fillna(0)
    sentiment_aligned = sentiment_resampled.reindex(price_index, method='ffill').fillna(0)
    return sentiment_aligned