from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.models.predictor import Predictor
from src.data.live_api import fetch_price_history
from src.numerical.features import generate_features
import os

app = FastAPI(title="Stock Sentiment Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    symbol: str
    period: str = "30d"


class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    signal: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Stock Sentiment Predictor API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Get prediction for a stock symbol."""
    model_path = f"models/{request.symbol}_model.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for {request.symbol}")
    
    try:
        # Fetch data and generate features
        price_df = fetch_price_history(request.symbol, period=request.period)
        features = generate_features(price_df)
        
        # Make prediction
        predictor = Predictor(model_path=model_path)
        pred = predictor.predict(features.tail(1))[0]
        
        # Determine signal
        if pred > 0.005:
            signal = "BUY"
        elif pred < -0.005:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        confidence = min(abs(pred) * 100, 100)
        
        return PredictionResponse(
            symbol=request.symbol,
            prediction=float(pred),
            signal=signal,
            confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}