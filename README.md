# Air Compressor Predictive Maintenance 🛠️

📚 **Docs:** [README](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Concepts](docs/CONCEPTS.md)

## 📌 Project Overview

Industrial air compressors are the "lungs" of manufacturing plants. Unplanned downtime can cost thousands of dollars per hour. This project implements an **end-to-end MLOps pipeline** to predict multi-component failures (Bearings, Radiators, Pumps) using real-time sensor data (Vibration, Temperature, Pressure).

### Key Features

* **Production Architecture:** Transitioned from experimental Jupyter Notebooks to a modular Python package.
* **Time-Series Engineering:** Implemented rolling window statistics and lag features to capture equipment degradation.
* **Robust Validation:** Utilized `TimeSeriesSplit` to prevent data leakage and ensure temporal reliability.
* **Deployment Ready:** Containerized FastAPI service for real-time inference.

---

## 🏗️ System Architecture

The system is designed as a modular pipeline to ensure scalability and maintainability.

1. **Ingestion:** Robust loading with structured logging and error handling.
2. **Preprocessing:** Scikit-Learn Pipelines with `RobustScaler` for sensor outlier handling.
3. **Feature Engineering:** Generation of rolling mean/std-dev for mechanical vibration (GACC/HACC) and thermal sensors.
4. **Experiment Tracking:** MLflow manages hyperparameters, metrics (F1-Score), and model versioning.
5. **Inference:** FastAPI endpoint serving predictions via a Docker container.

---

## 📁 Repository Structure

```text
├── api/
│   └── main.py         # FastAPI implementation (/health, /predict)
├── data/raw/           # Sensor CSV data
├── src/                # Core Logic
│   ├── config.py       # Shared column/experiment constants
│   ├── ingestion.py    # Data loading & logging
│   ├── preprocessing.py# Sklearn Transformation Pipelines
│   ├── features.py     # Rolling window & lag engineering
│   ├── train.py        # MLflow training logic with TimeSeriesSplit
│   └── predict.py      # Loads latest MLflow models & serves predictions
├── Dockerfile          # Containerization for production
└── pyproject.toml / uv.lock  # Project dependencies (uv)

```

Model artifacts aren't stored as local `.pkl` files — each training run logs a
self-contained preprocessing+model pipeline to MLflow, and `src/predict.py`
loads the latest run per target at serving time.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.13+
* Docker (Optional for containerization)

### 2. Installation

```bash
git clone https://github.com/Prafful-Vyas/Air-Compressor-predictive-maintenance-using-ML.git
cd Air-Compressor-predictive-maintenance-using-ML
uv sync

```

### 3. Training & Tracking

Run the training pipeline to log metrics to MLflow:

```bash
python -m src.train
mlflow ui  # View results at http://localhost:5000

```

Training **registers** a new candidate model version per target in the MLflow
Model Registry, but that version is not automatically served — see step 4.

### 4. Promoting a Trained Model

`src/predict.py` only ever loads the model version holding the `production`
alias (configurable via `ACPDM_MODEL_REGISTRY_ALIAS`) for each target, so a
freshly trained run has zero effect on what the API serves until you
explicitly promote it:

```bash
python -m src.promote list bearings        # see candidate versions + their holdout_f1
python -m src.promote promote bearings 1   # attach the 'production' alias to version 1
python -m src.promote current bearings     # confirm what's currently promoted
```

Promotion refuses a version whose `holdout_f1` is below
`ACPDM_MIN_HOLDOUT_F1_FOR_PROMOTION` (default `0.0`, i.e. no gate) unless you
pass `--force`. Re-running `promote` with an older version number is an
instant rollback. Repeat for every target (`bearings`, `wpump`, `radiator`,
`exvalve`) — the API reports `"degraded"` at `/health/ready` for any target
that hasn't been promoted yet.

### 5. Running the API

```bash
uvicorn api.main:app --reload

```

Navigate to `http://localhost:8000/docs` to test the interactive Swagger API.
`/predict` requires an `X-API-Key` header matching `ACPDM_API_KEY` (see
`.env.example`); `/health` and `/health/ready` are unauthenticated.

### 6. Running via Docker

The container serves the API only; it loads trained models from an MLflow
tracking store at startup, so mount the local `mlruns/` directory produced
by step 3 (or point `MLFLOW_TRACKING_URI` at a remote tracking server):

```bash
docker build -t air-compressor-api .
docker run -p 8000:8000 -v "$(pwd)/mlruns:/app/mlruns" air-compressor-api

```

### 7. Testing & Code Quality

```bash
uv run pytest              # unit + integration tests
uv run ruff check .        # lint
uv run ruff format .       # auto-format

```

`pre-commit install` will run lint/format automatically on each commit
(config in `.pre-commit-config.yaml`). The same checks run in CI on every
push and pull request against `main` (`.github/workflows/ci.yml`).

---

## 📊 Performance Metrics

Instead of simple accuracy, this project prioritizes **F1-Score** and **Precision-Recall** due to the class imbalance inherent in machinery failure data.

| Target      | Mean CV F1 (5-fold) | Holdout F1 |
|-------------|----------------------|------------|
| Bearings    | 0.80                 | 0.63       |
| Water Pump  | 0.84                 | 0.84       |
| Radiator    | 0.83                 | 0.88       |
| Exhaust Valve | 0.67               | 0.79       |

* **Validation Strategy:** 5-Fold `TimeSeriesSplit` for a robustness estimate, plus a chronological 80/20 holdout as the reported test metric. Numbers above are from a single run and will shift as the training data grows — rerun `python -m src.train` and update this table periodically rather than treating it as fixed.

---

## 🛠️ Tech Stack

* **Language:** Python
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **MLOps:** MLflow
* **Backend:** FastAPI, Uvicorn, Pydantic
* **DevOps:** Docker, GitHub Actions (CI)

---

## 🚀 Future Improvements & Scalability

To transition this from a standalone project to an enterprise-grade Industrial IoT (IIoT) platform, the following enhancements are proposed:

### 1. Real-time Data Streaming (Apache Kafka)

Currently, the system processes static CSVs. Integrating **Apache Kafka** would allow the pipeline to ingest high-frequency sensor data streams directly from PLC (Programmable Logic Controller) systems in a factory setting.

### 2. Model Monitoring & Observability (Grafana & Prometheus)

Implementing a monitoring dashboard to track:

* **Model Drift:** Detecting when the compressor's physical characteristics change (e.g., after a major part replacement).
* **Latency:** Monitoring the inference time of the FastAPI endpoint.
* **System Health:** Tracking CPU/Memory usage of the Docker containers.

### 3. Advanced Modeling: RUL Prediction

Transition from binary classification ("Will it fail?") to **Remaining Useful Life (RUL)** estimation using **LSTMs** or **GRUs**. This provides a countdown (e.g., "14 days until bearing failure"), allowing for much better maintenance scheduling.

### 4. Automated Retraining Pipeline (CI/CD/CT)

Setting up **GitHub Actions** or **Airflow** to trigger a "Continuous Training" (CT) pipeline. When model performance drops below a certain F1-score threshold, the system would automatically retrain on the latest 3 months of sensor data and promote the best model to production.

### 5. Edge Deployment

Optimizing the model using **ONNX** or **TensorRT** to deploy the inference engine directly onto "Edge" devices (like an NVIDIA Jetson or Raspberry Pi) located physically on the air compressor, reducing the need for constant cloud connectivity.
