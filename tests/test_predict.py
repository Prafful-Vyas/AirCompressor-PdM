import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.config import RAW_FEATURE_COLS, TARGET_COLS, registered_model_name
from src.predict import Predictor


class FakeModel:
    """Stands in for a fitted sklearn Pipeline: always predicts class 1
    with a fixed probability, so tests can assert on Predictor's plumbing
    without needing a real trained model."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


def make_readings(n_rows: int) -> pd.DataFrame:
    data = {col: np.linspace(1.0, 2.0, n_rows) for col in RAW_FEATURE_COLS}
    return pd.DataFrame(data)


def test_predict_returns_prediction_and_probability_per_target():
    predictor = Predictor()
    predictor.models = {"bearings": FakeModel(), "wpump": FakeModel()}

    result = predictor.predict(make_readings(15))

    assert result["bearings"] == {"prediction": 1, "failure_probability": 0.7}
    assert result["wpump"] == {"prediction": 1, "failure_probability": 0.7}


def test_predict_raises_on_missing_columns():
    predictor = Predictor()
    predictor.models = {"bearings": FakeModel()}

    incomplete = make_readings(15).drop(columns=[RAW_FEATURE_COLS[0]])
    with pytest.raises(ValueError, match="Missing required sensor columns"):
        predictor.predict(incomplete)


def test_predict_raises_on_nan_reading():
    predictor = Predictor()
    predictor.models = {"bearings": FakeModel()}

    readings = make_readings(15)
    readings.loc[readings.index[-1], "rpm"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        predictor.predict(readings)


def test_predict_raises_on_negative_rpm():
    predictor = Predictor()
    predictor.models = {"bearings": FakeModel()}

    readings = make_readings(15)
    readings.loc[readings.index[-1], "rpm"] = -1.0
    with pytest.raises(ValueError, match="Negative values"):
        predictor.predict(readings)


def test_predict_raises_if_models_not_loaded():
    predictor = Predictor()
    with pytest.raises(RuntimeError, match="not loaded"):
        predictor.predict(make_readings(15))


def test_predict_works_with_a_single_reading():
    # min_periods=1 rolling features tolerate a single row, even though
    # accuracy in practice benefits from more history.
    predictor = Predictor()
    predictor.models = {"bearings": FakeModel()}

    result = predictor.predict(make_readings(1))
    assert result["bearings"]["prediction"] == 1


def test_load_raises_when_experiment_missing(isolated_mlflow_tracking):
    predictor = Predictor(experiment_name="nonexistent-experiment")
    with pytest.raises(RuntimeError, match="not found"):
        predictor.load()


def _register_and_promote_all_targets() -> None:
    mlflow.set_experiment("test-experiment")
    client = MlflowClient()
    for target in TARGET_COLS:
        with mlflow.start_run():
            model = LogisticRegression().fit(
                np.array([[1], [2], [3], [4]]), [0, 1, 0, 1]
            )
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=registered_model_name(target),
            )
        versions = client.search_model_versions(
            f"name='{registered_model_name(target)}'"
        )
        version = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias(
            registered_model_name(target), "production", version
        )


def test_load_succeeds_once_every_target_is_promoted(isolated_mlflow_tracking):
    _register_and_promote_all_targets()

    predictor = Predictor(experiment_name="test-experiment")
    predictor.load()

    assert set(predictor.models.keys()) == set(TARGET_COLS)


def test_load_raises_when_registered_but_not_promoted(isolated_mlflow_tracking):
    mlflow.set_experiment("test-experiment")
    with mlflow.start_run():
        model = LogisticRegression().fit(np.array([[1], [2], [3], [4]]), [0, 1, 0, 1])
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=registered_model_name("bearings"),
        )

    predictor = Predictor(experiment_name="test-experiment")
    with pytest.raises(RuntimeError, match="No model version has the"):
        predictor.load()
