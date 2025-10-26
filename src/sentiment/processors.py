import numpy as np
from typing import List, Union

_POSITIVE = {"good", "great", "up", "beat", "positive", "strong", "improved", "outperform"}
_NEGATIVE = {"bad", "down", "miss", "negative", "weak", "decline", "loss", "underperform"}


def _score_text(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in _POSITIVE if w in t)
    neg = sum(1 for w in _NEGATIVE if w in t)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def analyze_sentiment(texts: Union[str, List[str]]) -> Union[float, np.ndarray]:
    """Lightweight sentiment scoring."""
    if isinstance(texts, str):
        return _score_text(texts)
    arr = np.array([_score_text(t) for t in texts], dtype=float)
    return arr