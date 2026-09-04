# AirCompressor-PdM — Core

Predictive maintenance ML project: predicts 4 independent binary faults
(`bearings`, `wpump`, `radiator`, `exvalve`) for an industrial air compressor from
sensor CSV data. Python 3.13, uv-managed, MLflow-tracked, served via FastAPI. Went
through a production-hardening pass (2026-09): env-based settings, API-key auth,
input validation, an MLflow registry promotion gate, hardened Docker/compose, and
an expanded CI pipeline — see `mem:mlops_conventions` and `mem:conventions`.

## Source map
- `src/config.py` — single source of truth for column lists (`SENSOR_COLS`,
  `RAW_FEATURE_COLS`, `TARGET_COLS`, `DROP_COLS`, `NON_NEGATIVE_COLS`), `WINDOW_SIZES`,
  `EXPERIMENT_NAME`, and `registered_model_name(target)` (MLflow registry naming —
  used by train.py, predict.py, promote.py). train.py/predict.py/ingestion.py all
  import from here — never redefine column lists locally.
- `src/settings.py` — `Settings` (pydantic-settings, env prefix `ACPDM_`, `.env`-backed),
  `get_settings()` (lru_cache). Runtime/infra config (API key, CORS, log level, MLflow
  tracking URI, size limits, registry alias/threshold) — separate from `config.py`'s
  domain constants. Tests call `get_settings.cache_clear()` after `monkeypatch.setenv`.
- `src/logging_utils.py` — `configure_logging()` + JSON formatter + `request_id_var`
  ContextVar, used by `api/main.py` for structured logs with request IDs.
- `src/ingestion.py` — `DataIngestor`: loads CSV, validates required columns present
  and sensor columns numeric (raises `ValueError` naming offenders).
- `src/preprocessing.py` — `build_preprocessing_pipeline()` (sklearn `RobustScaler`-based)
  + `SensorSanityChecker`.
- `src/features.py` — `FeatureEngineer`: rolling mean/std (`WINDOW_SIZES` = [5, 10]) and
  lag features, applied only to `SENSOR_COLS` (one representative sensor per physical
  subsystem, not all 20 numeric columns — see `mem:domain_model`).
- `src/train.py` — `Trainer`: one MLflow run per target in `TARGET_COLS`, trains a
  RandomForest pipeline (preprocessing + model bundled as one sklearn Pipeline),
  `TimeSeriesSplit` CV + chronological 80/20 holdout, **registers** the fitted pipeline
  in the MLflow Model Registry (see `mem:mlops_conventions` — registering ≠ serving).
- `src/predict.py` — `Predictor`: loads, per target, the model version holding the
  `settings.model_registry_alias` alias (default `"production"`); validates input for
  missing columns, non-finite values, and negative values in `NON_NEGATIVE_COLS` before
  scoring. No local `.pkl`/`.joblib` artifacts are ever used.
- `src/promote.py` — CLI (`list` / `promote` / `current`) that attaches the serving
  alias to a registered model version; the only way a trained model becomes servable.
- `api/main.py` — FastAPI app. `/health` = pure liveness (always 200, unauthenticated).
  `/health/ready` = old predictor-loaded check. `/predict` requires `X-API-Key` header
  (`Depends(require_api_key)`), has an explicit CORS allow-list, a body-size cap, and a
  request-ID/access-log middleware. Custom `RequestValidationError` handler avoids
  echoing raw invalid values (NaN/Infinity crash Starlette's default JSON encoder
  otherwise — see `mem:conventions`).
- `tests/` — mirrors `src/` + `api/` 1:1, plus `test_settings.py`, `test_promote.py`.
- `docs/ARCHITECTURE.md` — physical machine + data-dictionary + correlation analysis of
  the 26 raw CSV columns; read before touching feature selection (`mem:domain_model`).
  §5 also documents the promotion-gate workflow.
- `docs/CONCEPTS.md` — ML/software concepts layered on top of the physical model.
- `docker-compose.yml` — local prod-like stack (API + real MLflow server), not a cloud
  manifest. `LICENSE` (MIT), `SECURITY.md` — repo governance.

## Project-wide invariants
- Read/write paths must agree on column order: `RAW_FEATURE_COLS` defines the exact
  order inference input is reindexed to before feature engineering.
- Never store trained models as local files — MLflow registry is the only source of
  truth for a servable pipeline (`mem:mlops_conventions`).
- `data/`, `.venv/`, `mlruns/`, `.ruff_cache/` are excluded from Serena's index
  (`.serena/project.yml` `ignored_paths`) — large/generated, not source.

See `mem:tech_stack`, `mem:suggested_commands`, `mem:conventions`, `mem:task_completion`,
`mem:mlops_conventions`, `mem:domain_model`.