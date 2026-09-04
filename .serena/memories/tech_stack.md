# Tech Stack

- Language: Python >=3.13 (pinned via `.python-version` = 3.13).
- Package manager: **uv** (not pip/poetry/conda). Lockfile `uv.lock` — installs must use
  `uv sync` / `uv sync --frozen` to respect it.
- ML: scikit-learn (Pipelines, `RobustScaler`, `RandomForest`, `TimeSeriesSplit`), pandas, numpy.
- Experiment tracking: MLflow (>=3.12) — see `mem:mlops_conventions` for the
  run-per-target / no-local-artifacts convention.
- API: FastAPI + Uvicorn (`[standard]` extra) + Pydantic.
- Dev/test: pytest, ruff (lint + format, replaces black/flake8/isort), pre-commit,
  httpx (FastAPI TestClient dependency).
- Containerization: Docker; container serves the API only and expects `mlruns/` to be
  mounted or `MLFLOW_TRACKING_URI` pointed at a remote store (no models baked into image).
- CI: GitHub Actions (`.github/workflows/ci.yml`), runs on push/PR to `main`.