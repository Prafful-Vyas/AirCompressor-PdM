entries
# AirCompressor-PdM — Core

Predictive maintenance ML project: predicts 4 independent binary faults
(`bearings`, `wpump`, `radiator`, `exvalve`) for an industrial air compressor from
sensor CSV data. Python 3.13, uv-managed, MLflow-tracked, served via FastAPI.

## Source map
- `src/config.py` — single source of truth for column lists (`SENSOR_COLS`,
  `RAW_FEATURE_COLS`, `TARGET_COLS`, `DROP_COLS`), `WINDOW_SIZES`, `EXPERIMENT_NAME`.
  train.py and predict.py both import from here — never redefine column lists locally.
- `src/ingestion.py` — `DataIngestor`, CSV loading + logging.
- `src/preprocessing.py` — `build_preprocessing_pipeline()` (sklearn `RobustScaler`-based)
  + `SensorSanityChecker`.
- `src/features.py` — `FeatureEngineer`: rolling mean/std (`WINDOW_SIZES` = [5, 10]) and
  lag features, applied only to `SENSOR_COLS` (one representative sensor per physical
  subsystem, not all 20 numeric columns — see `mem:domain_model`).
- `src/train.py` — `Trainer`: one MLflow run per target in `TARGET_COLS`, trains a
  RandomForest pipeline (preprocessing + model bundled as one sklearn Pipeline),
  `TimeSeriesSplit` CV + chronological 80/20 holdout, logs the fitted pipeline itself
  (see `mem:mlops_conventions`).
- `src/predict.py` — `Predictor`: loads the latest MLflow run per target at serving time
  (see `mem:mlops_conventions`); no local `.pkl`/`.joblib` artifacts are ever used despite
  `.gitignore` allowing for them.
- `api/main.py` — FastAPI app (`/health`, `/predict`), Pydantic request/response models,
  `lifespan` loads `Predictor` once at startup.
- `tests/` — mirrors `src/` + `api/` 1:1 (`test_ingestion.py`, `test_features.py`,
  `test_preprocessing.py`, `test_train.py`, `test_predict.py`, `test_api.py`).
- `docs/ARCHITECTURE.md` — physical machine + data-dictionary + correlation analysis of
  the 26 raw CSV columns; read before touching feature selection (`mem:domain_model`).
- `docs/CONCEPTS.md` — ML/software concepts layered on top of the physical model.

## Project-wide invariants
- Read/write paths must agree on column order: `RAW_FEATURE_COLS` defines the exact
  order inference input is reindexed to before feature engineering.
- Never store trained models as local files — MLflow run artifacts are the only source
  of truth for a trained pipeline (`mem:mlops_conventions`).
- `data/`, `.venv/`, `mlruns/`, `.ruff_cache/` are excluded from Serena's index
  (`.serena/project.yml` `ignored_paths`) — large/generated, not source.

See `mem:tech_stack`, `mem:suggested_commands`, `mem:conventions`, `mem:task_completion`,
`mem:mlops_conventions`, `mem:domain_model`.