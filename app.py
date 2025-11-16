from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import torch

from run import load_model, classify_sentiment

app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for predicting sentiment of financial statements using FinBERT",
    version="1.0.0"
)

# Global model variables
tokenizer = None
model = None
model_loaded = False

class SentimentRequest(BaseModel):
    text: str

class SentimentBatchRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float

class SentimentBatchResponse(BaseModel):
    results: List[SentimentResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None

@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the API starts"""
    global tokenizer, model, model_loaded
    try:
        # Default checkpoint path - adjust as needed
        project_root = Path(__file__).resolve().parent
        checkpoint_path = project_root / "Model" / "Final_Model"
        
        # Allow override via environment variable
        import os
        if os.getenv("CHECKPOINT_PATH"):
            checkpoint_path = Path(os.getenv("CHECKPOINT_PATH"))
        
        if checkpoint_path.exists():
            print(f"Loading model from {checkpoint_path}...")
            tokenizer, model = load_model(checkpoint_path)
            model_loaded = True
            print(f"Model loaded successfully from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("API will start but model endpoints will return errors")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    project_root = Path(__file__).resolve().parent
    model_path = str(project_root / "Model" / "Final_Model")
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_path=model_path if model_loaded else None
    )

@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a single text statement.
    
    Returns the predicted sentiment label (negative, neutral, or positive) 
    along with the confidence score.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        results = classify_sentiment([request.text], tokenizer, model)
        text, label, confidence = results[0]
        return SentimentResponse(text=text, label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=SentimentBatchResponse, tags=["Prediction"])
async def predict_sentiment_batch(request: SentimentBatchRequest):
    """
    Predict sentiment for multiple text statements in a single request.
    
    Returns predictions for all texts with their sentiment labels and confidence scores.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    # Filter out empty texts
    valid_texts = [text for text in request.texts if text and text.strip()]
    if not valid_texts:
        raise HTTPException(status_code=400, detail="No valid texts provided")
    
    try:
        results = classify_sentiment(valid_texts, tokenizer, model)
        response_results = [
            SentimentResponse(text=text, label=label, confidence=confidence)
            for text, label, confidence in results
        ]
        return SentimentBatchResponse(results=response_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

