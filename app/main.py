# app/main.py

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.predict import BreastCancerPredictor

# ── Lifespan ──────────────────────────────────────────────────────────────────
# Model loads once at startup, not on every request

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global predictor
    print("Starting up — loading production model...")
    predictor = BreastCancerPredictor()
    print("Model ready. Server accepting requests.")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="""
    Production ML API serving the Wisconsin Breast Cancer classifier.

    - Model: Logistic Regression (C=0.1)
    - Trained on: Wisconsin Breast Cancer Dataset (HuggingFace)
    - ROC AUC: 0.9977 | Recall: 0.9524 | Precision: 1.0
    - Tracked with: MLflow 3.x
    - Registry alias: @production
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Breast Cancer Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Returns server and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.is_loaded,
        model_uri=predictor.model_uri if predictor else "not loaded",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Runs breast cancer prediction on 30 input features.

    Returns:
    - prediction: 0 (Benign) or 1 (Malignant)
    - prediction_label: human readable
    - probability_benign: confidence score
    - probability_malignant: confidence score
    - model_version: MLflow URI of the model used
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))