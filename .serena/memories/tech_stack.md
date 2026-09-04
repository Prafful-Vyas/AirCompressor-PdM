# Tech Stack

- Language: Python >=3.13 (pinned via `.python-version` = 3.13).
- Package manager: **uv** (not pip/poetry/conda). Lockfile `uv.lock` — installs must use
  `uv sync` / `uv sync --frozen` to respect it.
- ML: scikit-learn (Pipelines, `RobustScaler`, `RandomForest`, `TimeSeriesSplit`), pandas, numpy.
- Experiment tracking + registry: MLflow, pinned `>=3.12.0,<3.13.0` (see
  `mem:mlops_conventions` for why) — run-per-target training, registry-alias-based
  serving, no local model artifacts.
- Config: `pydantic-settings` (`src/settings.py`, env prefix `ACPDM_`, `.env`-backed).
- API: FastAPI + Uvicorn (`[standard]` extra) + Pydantic. API-key auth, explicit CORS
  allow-list, structured JSON logging with request IDs.
- Dev/test: pytest + pytest-cov (`--cov-fail-under=70`, currently ~84%), ruff (lint +
  format), mypy (pragmatic config: untyped defs allowed, missing stubs ignored,
  scoped to `src`/`api`), pre-commit (ruff + gitleaks secret scanning), httpx (FastAPI
  TestClient dependency).
- Containerization: Docker — multi-stage build, runs as non-root (uid 1000), stdlib
  `HEALTHCHECK` against `/health`. `docker-compose.yml` runs the API alongside a real
  MLflow tracking server for local prod-like verification (not a cloud manifest).
- CI: GitHub Actions (`.github/workflows/ci.yml`) — ruff check/format, mypy, pytest
  w/coverage gate, `uvx pip-audit --strict` (with a documented ignore list, see
  `mem:mlops_conventions`), and a Docker build smoke test. Runs on push/PR to `main`.
- Governance: MIT `LICENSE`, `SECURITY.md` with private-disclosure contact.