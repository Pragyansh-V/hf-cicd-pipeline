# app/schemas.py

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for breast cancer prediction.
    All 30 features from the Wisconsin dataset.
    """
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "radius_mean": 17.99,
                    "texture_mean": 10.38,
                    "perimeter_mean": 122.8,
                    "area_mean": 1001.0,
                    "smoothness_mean": 0.1184,
                    "compactness_mean": 0.2776,
                    "concavity_mean": 0.3001,
                    "concave_points_mean": 0.1471,
                    "symmetry_mean": 0.2419,
                    "fractal_dimension_mean": 0.07871,
                    "radius_se": 1.095,
                    "texture_se": 0.9053,
                    "perimeter_se": 8.589,
                    "area_se": 153.4,
                    "smoothness_se": 0.006399,
                    "compactness_se": 0.04904,
                    "concavity_se": 0.05373,
                    "concave_points_se": 0.01587,
                    "symmetry_se": 0.03003,
                    "fractal_dimension_se": 0.006193,
                    "radius_worst": 25.38,
                    "texture_worst": 17.33,
                    "perimeter_worst": 184.6,
                    "area_worst": 2019.0,
                    "smoothness_worst": 0.1622,
                    "compactness_worst": 0.6656,
                    "concavity_worst": 0.7119,
                    "concave_points_worst": 0.2654,
                    "symmetry_worst": 0.4601,
                    "fractal_dimension_worst": 0.1189
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for breast cancer prediction."""
    prediction: int = Field(..., description="0 = Benign, 1 = Malignant")
    prediction_label: str = Field(..., description="Human readable label")
    probability_benign: float = Field(..., description="Probability of being benign")
    probability_malignant: float = Field(..., description="Probability of being malignant")
    model_version: str = Field(..., description="MLflow model URI used for prediction")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_uri: str