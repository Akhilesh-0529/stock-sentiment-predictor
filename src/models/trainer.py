import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class Trainer:
    def __init__(self, model_path: str = "models/latest.pkl"):
        self.model_path = model_path
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

    def build_pipeline(self):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        return pipeline

    def train(self, X, y):
        """Train time-series aware model."""
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False

        if has_pandas and hasattr(X, 'copy'):
            X_proc = X.copy()
        else:
            X_proc = np.asarray(X)

        y_arr = np.asarray(y)

        nX = len(X_proc)
        ny = len(y_arr)
        if ny != nX:
            min_len = min(nX, ny)
            if has_pandas and hasattr(X, 'iloc'):
                X_proc = X_proc.iloc[-min_len:]
            else:
                X_proc = X_proc[-min_len:]
            y_arr = y_arr[-min_len:]

        pipeline = self.build_pipeline()
        tscv = TimeSeriesSplit(n_splits=3)
        params = {
            "model__max_depth": [5, 10],
            "model__n_estimators": [100]
        }
        search = GridSearchCV(pipeline, params, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        search.fit(X_proc, y_arr)
        joblib.dump(search.best_estimator_, self.model_path)
        return search.best_estimator_