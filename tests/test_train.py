import mlflow
import numpy as np
import pandas as pd

from src.config import RAW_FEATURE_COLS, SENSOR_COLS, TARGET_COLS
from src.features import FeatureEngineer
from src.train import Trainer


def make_synthetic_df(n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {col: rng.normal(size=n_rows) for col in RAW_FEATURE_COLS}
    data["id"] = range(n_rows)
    data["acmotor"] = "Stable"
    for target in TARGET_COLS:
        data[target] = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    return pd.DataFrame(data)


def test_train_target_logs_a_single_properly_scoped_run(isolated_mlflow_tracking):
    mlflow.set_experiment("test-experiment")

    raw_df = make_synthetic_df()
    fe = FeatureEngineer(window_sizes=[2])
    engineered = fe.create_rolling_features(raw_df, SENSOR_COLS)

    trainer = Trainer(window_sizes=[2], n_splits=2)
    result = trainer.train_target(engineered, "bearings")

    assert result["target"] == "bearings"
    assert 0.0 <= result["mean_cv_f1"] <= 1.0
    assert 0.0 <= result["holdout_f1"] <= 1.0

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])

    # Everything -- CV fold metrics and the final holdout metrics -- must
    # land inside exactly one run, not leak into an ambient/default run.
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["target"] == "bearings"
    assert "fold_0_f1" in run.data.metrics
    assert "mean_cv_f1" in run.data.metrics
    assert "holdout_f1" in run.data.metrics
