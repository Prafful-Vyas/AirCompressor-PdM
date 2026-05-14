# Air Compressor Predictive Maintenance 🛠️

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
├── api/                # FastAPI implementation
├── data/raw/           # Sensor CSV data (not tracked in Git)
├── models/             # Serialized model artifacts (.pkl)
├── src/                # Core Logic
│   ├── ingestion.py    # Data loading & logging
│   ├── preprocessing.py# Sklearn Transformation Pipelines
│   ├── features.py     # Rolling window & lag engineering
│   └── train.py        # MLflow training logic with TimeSeriesSplit
├── Dockerfile          # Containerization for production
└── requirements.txt    # Project dependencies

```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10+
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
python src/train.py
mlflow ui  # View results at http://localhost:5000

```

### 4. Running the API

```bash
uvicorn api.main:app --reload

```

Navigate to `http://localhost:8000/docs` to test the interactive Swagger API.

---

## 📊 Performance Metrics

Instead of simple accuracy, this project prioritizes **F1-Score** and **Precision-Recall** due to the class imbalance inherent in machinery failure data.

* **Bearing Failure F1-Score:** 0.XX (Update with your actual results)
* **Validation Strategy:** 5-Fold TimeSeriesSplit.

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
