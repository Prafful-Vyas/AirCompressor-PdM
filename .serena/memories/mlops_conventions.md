# MLflow / Serving Conventions

- No local `.pkl`/`.joblib` model files are ever produced or read, even though
  `.gitignore` has entries for them (legacy/defensive, not actually used). The trained
  artifact's only home is an MLflow run.
- `Trainer.train_target` (`src/train.py`) opens one `mlflow.start_run(run_name=f"Compressor_{target}_Model")`
  per target in `TARGET_COLS`, logs params (`target`, `model_type`, `features_count`,
  `window_sizes`, `n_splits`, `random_state`), per-fold + mean CV F1 (`TimeSeriesSplit`),
  a chronological 80/20 holdout F1, and logs the fitted pipeline via
  `mlflow_sklearn.log_model(final_pipeline, "model")`. All runs share
  `EXPERIMENT_NAME = "air-compressor-predictive-maintenance"` (`src/config.py`).
- `Predictor._latest_run_id` (`src/predict.py`) finds the model to serve for a target by
  querying `client.search_runs(... filter_string=f"params.target = '{target}'", order_by=["start_time DESC"], max_results=1)`
  — i.e. **the target is identified by the `target` run param, not by run name parsing**.
  Retraining a target just means running `python -m src.train` again; the newest run for
  that `target` param is picked up automatically, no manual model promotion step.
- If no run exists yet for a target, `Predictor` raises `RuntimeError` telling the user to
  run `python -m src.train` first — this is intentional, not a bug to silence.
- Tests must never touch the real `mlruns/` store: use the `isolated_mlflow_tracking`
  fixture (`tests/conftest.py`), which points `MLFLOW_TRACKING_URI` at a `tmp_path`.