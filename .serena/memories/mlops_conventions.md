# MLflow / Serving Conventions

- No local `.pkl`/`.joblib` model files are ever produced or read. The trained
  artifact's only home is the MLflow Model Registry.
- **Registering ≠ serving.** `Trainer.train_target` (`src/train.py`) logs the fitted
  pipeline via `mlflow.sklearn.log_model(final_pipeline, name="model",
  registered_model_name=registered_model_name(target_col))` — this creates a new
  registry *version* with **no alias attached**, so it has zero effect on what's
  currently served. `registered_model_name(target)` (`src/config.py`) →
  `f"{EXPERIMENT_NAME}-{target}"` is the single naming source, shared by train.py,
  predict.py, and promote.py.
- `Predictor.load()` (`src/predict.py`) loads, per target, only the version holding the
  `settings.model_registry_alias` alias (default `"production"`) via
  `client.get_model_version_by_alias(name, alias)` →
  `mlflow.sklearn.load_model(f"models:/{name}@{alias}")`. If no version has been
  promoted, it raises `RuntimeError` naming the exact `promote` command to run — this
  is intentional, not a bug to silence.
- **Promotion is a deliberate, separate step**: `python -m src.promote promote <target>
  <version>` (`src/promote.py`) attaches the alias. It refuses versions whose
  `holdout_f1` metric is below `settings.min_holdout_f1_for_promotion` unless `--force`
  is passed. Re-pointing the alias to an older version is an instant rollback.
  `promote list <target>` shows candidates + their metrics; `promote current <target>`
  shows what's currently promoted. This replaces the old "serve whatever run has the
  newest start_time" behavior, which let a bad retrain go live with no review step.
- Tests must never touch the real `mlruns/` store: use the `isolated_mlflow_tracking`
  fixture (`tests/conftest.py`), which points `MLFLOW_TRACKING_URI` at a `tmp_path`.
  The file-based store *does* support the Model Registry (confirmed empirically), so
  registry/alias tests work fine against it.
- **mlflow is pinned `>=3.12.0,<3.13.0`** (`pyproject.toml`) — empirically verified that
  mlflow>=3.14 hard-disables the file-based tracking backend (`./mlruns`, this
  project's local dev default) by default, breaking both the zero-config local
  workflow and `isolated_mlflow_tracking`. Do not bump mlflow past this range without
  first migrating the default tracking backend to a database (e.g. sqlite) — see the
  comment on the pin in `pyproject.toml` and `SECURITY.md`. This cap also transitively
  blocks patching `starlette` (mlflow-skinny pins `starlette<1`) and `cryptography`
  (mlflow pins `<47`); those 3 packages' advisories are `--ignore-vuln`'d in CI with a
  comment tracing them back to this one root cause.