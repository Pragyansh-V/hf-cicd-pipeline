# tests/test_predict.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ── Sample test data ──────────────────────────────────────────────────────────
# Real values from the Wisconsin dataset — first row (malignant case)

SAMPLE_MALIGNANT = {
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

SAMPLE_BENIGN = {
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    "perimeter_mean": 87.46,
    "area_mean": 566.3,
    "smoothness_mean": 0.09779,
    "compactness_mean": 0.08129,
    "concavity_mean": 0.06664,
    "concave_points_mean": 0.04781,
    "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699,
    "texture_se": 0.7886,
    "perimeter_se": 2.058,
    "area_se": 23.56,
    "smoothness_se": 0.008462,
    "compactness_se": 0.0146,
    "concavity_se": 0.02387,
    "concave_points_se": 0.01315,
    "symmetry_se": 0.0198,
    "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11,
    "texture_worst": 19.26,
    "perimeter_worst": 99.7,
    "area_worst": 711.2,
    "smoothness_worst": 0.144,
    "compactness_worst": 0.1773,
    "concavity_worst": 0.239,
    "concave_points_worst": 0.1288,
    "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_predictor():
    """Returns a mocked BreastCancerPredictor — no MLflow needed for tests."""
    with patch("app.main.predictor") as mock:
        mock.is_loaded = True
        mock.model_uri = "models:/breast-cancer-classifier@production"
        yield mock


@pytest.fixture
def client(mock_predictor):
    """Returns a FastAPI test client with mocked predictor."""
    from app.main import app
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_root(client):
    """Root endpoint returns correct keys."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "health" in data
    assert "predict" in data


def test_health_endpoint(client):
    """Health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_malignant(client, mock_predictor):
    """Prediction endpoint returns valid response shape for malignant input."""
    from app.schemas import PredictionResponse

    mock_predictor.predict.return_value = PredictionResponse(
        prediction=1,
        prediction_label="Malignant",
        probability_benign=0.02,
        probability_malignant=0.98,
        model_version="models:/breast-cancer-classifier@production"
    )

    response = client.post("/predict", json=SAMPLE_MALIGNANT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["Benign", "Malignant"]
    assert 0.0 <= data["probability_benign"] <= 1.0
    assert 0.0 <= data["probability_malignant"] <= 1.0
    assert abs(data["probability_benign"] + data["probability_malignant"] - 1.0) < 0.01


def test_predict_benign(client, mock_predictor):
    """Prediction endpoint returns valid response shape for benign input."""
    from app.schemas import PredictionResponse

    mock_predictor.predict.return_value = PredictionResponse(
        prediction=0,
        prediction_label="Benign",
        probability_benign=0.97,
        probability_malignant=0.03,
        model_version="models:/breast-cancer-classifier@production"
    )

    response = client.post("/predict", json=SAMPLE_BENIGN)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["Benign", "Malignant"]


def test_predict_missing_field(client):
    """Prediction endpoint rejects incomplete input."""
    incomplete = SAMPLE_MALIGNANT.copy()
    del incomplete["radius_mean"]

    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_invalid_type(client):
    """Prediction endpoint rejects wrong data types."""
    invalid = SAMPLE_MALIGNANT.copy()
    invalid["radius_mean"] = "not_a_number"

    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_probabilities_sum_to_one(client, mock_predictor):
    """Probabilities must sum to 1.0."""
    from app.schemas import PredictionResponse

    mock_predictor.predict.return_value = PredictionResponse(
        prediction=1,
        prediction_label="Malignant",
        probability_benign=0.0234,
        probability_malignant=0.9766,
        model_version="models:/breast-cancer-classifier@production"
    )

    response = client.post("/predict", json=SAMPLE_MALIGNANT)
    data = response.json()
    total = data["probability_benign"] + data["probability_malignant"]
    assert abs(total - 1.0) < 0.01