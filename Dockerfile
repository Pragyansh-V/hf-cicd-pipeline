# Dockerfile

# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — smaller than full image, compatible with all dependencies
FROM python:3.11-slim

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
# Only re-runs pip install if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app/ ./app/

# ── MLflow model ─────────────────────────────────────────────────────────────
COPY exported_model/ ./exported_model/

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]