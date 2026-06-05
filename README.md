# HF CI/CD Pipeline — Automated ML Deployment to Vertex AI

A production-grade CI/CD pipeline that automatically builds, tests, and deploys a containerised ML model to Google Vertex AI on every `git push`.

> **"Push code. Model deploys. Zero manual steps."**

---

## What This Does

```
git push origin main
        ↓
GitHub Actions triggers automatically
        ↓
Runs 7 tests on prediction code
        ↓
Builds Docker container
        ↓
Pushes to Google Artifact Registry
        ↓
Deploys to Vertex AI endpoint
```

---

## The Model

Serves the Wisconsin Breast Cancer classifier trained and tracked in [Project 7 — MLflow Experiment Tracking](https://github.com/Pragyansh-V/mlflow-experiment-tracking).

| Property | Value |
|---|---|
| Algorithm | Logistic Regression (C=0.1) |
| Dataset | Wisconsin Breast Cancer (HuggingFace) |
| ROC AUC | 0.9977 |
| Recall | 0.9524 |
| Precision | 1.0000 |
| Tracked with | MLflow 3.x |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Root — lists available endpoints |
| `GET` | `/health` | Health check — model load status |
| `POST` | `/predict` | Breast cancer prediction |

### Sample Request

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Sample Response

```json
{
  "prediction": 1,
  "prediction_label": "Malignant",
  "probability_benign": 0.0001,
  "probability_malignant": 0.9999,
  "model_version": "models:/breast-cancer-classifier@production"
}
```

---

## Project Structure

```
hf-cicd-pipeline/
├── app/
│   ├── main.py              # FastAPI server — lifespan, routes
│   ├── predict.py           # Prediction logic — model loading, inference
│   └── schemas.py           # Pydantic schemas — request/response validation
├── tests/
│   └── test_predict.py      # 7 unit tests — runs in CI before every deploy
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions — test → build → push → deploy
├── exported_model/
│   └── model.pkl            # Exported sklearn pipeline
├── Dockerfile               # Container definition
├── .dockerignore
└── requirements.txt
```

---

## CI/CD Pipeline

```yaml
# Three jobs — chained with needs:
test → build-and-push → deploy
```

| Job | Runs on | What it does |
|---|---|---|
| **test** | Every push + PR | Installs deps, runs pytest — deployment gatekeeper |
| **build-and-push** | main branch only, after tests pass | Builds Docker image, tags with commit SHA, pushes to Artifact Registry |
| **deploy** | After build succeeds | Uploads model to Vertex AI with /health and /predict routes |

**Authentication:** Google Workload Identity Federation — no long-lived JSON keys.

---

## Setup

### Prerequisites
- Docker Desktop
- Google Cloud SDK (`gcloud`)
- GitHub repository with Actions enabled

### Local Development

```bash
# Clone
git clone https://github.com/Pragyansh-V/hf-cicd-pipeline.git
cd hf-cicd-pipeline

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run locally
docker build -t breast-cancer-api:latest .
docker run -p 8080:8080 breast-cancer-api:latest

# Open Swagger UI
open http://localhost:8080/docs
```

### GitHub Secrets Required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_SERVICE_ACCOUNT` | Service account email |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload Identity Provider resource name |

### GCP Setup

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create breast-cancer-api \
  --repository-format=docker \
  --location=us-central1

# Create Workload Identity Pool + Provider
gcloud iam workload-identity-pools create "github-pool" \
  --location="global"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --workload-identity-pool="github-pool" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='Pragyansh-V/hf-cicd-pipeline'" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| FastAPI | Prediction API server |
| Pydantic | Request/response validation |
| Docker | Containerisation |
| GitHub Actions | CI/CD automation |
| Google Artifact Registry | Docker image storage |
| Google Vertex AI | Model hosting |
| Workload Identity Federation | Keyless GCP authentication |
| pytest + httpx | Testing |

---

## Part of the HuggingFace Portfolio Series

| # | Project | Key Skills |
|---|---|---|
| 1 | Sentiment Audit | HuggingFace pipelines |
| 2 | Emotion Classifier | LoRA fine-tuning |
| 3 | RAG Pipeline | Retrieval-Augmented Generation |
| 4 | Bias Audit | Fairness evaluation |
| 5 | LangGraph Agent | Agentic workflows |
| 6 | Vertex AI Deployment | Cloud MLOps |
| 7 | MLflow Tracking | Experiment tracking, Model Registry |
| **8** | **CI/CD Pipeline** | **Docker, GitHub Actions, Vertex AI automation** |