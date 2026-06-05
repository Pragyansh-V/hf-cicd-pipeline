# app/predict.py

import os
import mlflow.sklearn
import pandas as pd
import numpy as np
from app.schemas import PredictionRequest, PredictionResponse

# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/breast-cancer-classifier@production")

# ── Column name mapping ───────────────────────────────────────────────────────
# Pydantic uses underscores (concave_points_mean)
# Dataset uses spaces (concave points_mean)
# This map bridges the two

COLUMN_MAP = {
    "concave_points_mean": "concave points_mean",
    "concave_points_se":   "concave points_se",
    "concave_points_worst": "concave points_worst",
}


class BreastCancerPredictor:
    """
    Loads the @production MLflow model and serves predictions.
    Singleton pattern — model loads once at startup, reused for every request.
    """

    def __init__(self):
        self.model = None
        self.model_uri = MODEL_URI
        self._load_model()

    def _load_model(self):
        """Loads the production model from MLflow Registry."""
        print(f"Loading model from: {self.model_uri}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.sklearn.load_model(self.model_uri)
        print(f"Model loaded: {type(self.model.named_steps['classifier']).__name__}")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Runs inference on a single prediction request.
        Returns prediction, label, and probabilities.
        """
        # Convert request to dict and rename columns to match training data
        data = request.model_dump()
        data = {COLUMN_MAP.get(k, k): v for k, v in data.items()}

        # Build DataFrame with correct column order
        df = pd.DataFrame([data])

        # Run inference
        prediction = int(self.model.predict(df)[0])
        probabilities = self.model.predict_proba(df)[0]

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Malignant" if prediction == 1 else "Benign",
            probability_benign=round(float(probabilities[0]), 4),
            probability_malignant=round(float(probabilities[1]), 4),
            model_version=self.model_uri,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None