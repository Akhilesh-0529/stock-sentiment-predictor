import joblib
import numpy as np
import pandas as pd
import os
import json


class Predictor:
    def __init__(self, model_path: str = "models/latest.pkl"):
        self.model_path = model_path
        self.model = None
        self.expected_features = None

    def load(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}. Train first.")
            self.model = joblib.load(self.model_path)
            
            # Load metadata to get expected features
            metadata_path = self.model_path.replace('_model.pkl', '_metadata.json').replace('latest.pkl', 'AAPL_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    features = metadata.get('features', [])
                    
                    # Ensure it's a simple list of strings, remove empty strings and duplicates
                    if isinstance(features, list) and len(features) > 0:
                        seen = set()
                        self.expected_features = []
                        for item in features:
                            if isinstance(item, str) and item.strip() and item not in seen:
                                seen.add(item)
                                self.expected_features.append(item)
                            elif isinstance(item, list):
                                for subitem in item:
                                    if isinstance(subitem, str) and subitem.strip() and subitem not in seen:
                                        seen.add(subitem)
                                        self.expected_features.append(subitem)
                    
                    print(f"Loaded {len(self.expected_features)} expected features: {self.expected_features}")
                            
        return self.model

    def predict(self, X):
        model = self.load()
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Make a copy to avoid modifying original
        X = X.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in X.columns]
        
        print(f"Input features: {list(X.columns)}")
        print(f"Expected features: {self.expected_features}")
        
        # If we have expected features, ensure all are present
        if self.expected_features and len(self.expected_features) > 0:
            # Add missing features with 0
            for feature in self.expected_features:
                if feature and feature not in X.columns:  # Skip empty strings
                    print(f"Adding missing feature: {feature}")
                    X[feature] = 0.0
            
            # Reorder to match training order and remove extra columns
            X = X[self.expected_features]
            print(f"Final feature count: {len(X.columns)}")
        
        preds = model.predict(X)
        return np.asarray(preds)