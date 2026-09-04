import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .config import (
    EXPERIMENT_NAME,
    NON_NEGATIVE_COLS,
    RAW_FEATURE_COLS,
    SENSOR_COLS,
    TARGET_COLS,
    WINDOW_SIZES,
    registered_model_name,
)
from .features import FeatureEngineer
from .settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Loads the most recently trained model per target from MLflow and
    scores the most recent row of a window of raw sensor readings."""

    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self.models: dict = {}

    def load(self) -> None:
        """Loads, for each target, the model version currently holding the
        promotion alias (see src/settings.py's model_registry_alias,
        default "production") in the MLflow Model Registry. A freshly
        trained run is registered as a candidate but is never automatically
        servable -- promoting it is a deliberate step via
        `python -m src.promote promote <target> <version>` (src/promote.py).
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment '{self.experiment_name}' not found. "
                "Run `python -m src.train` first."
            )

        alias = get_settings().model_registry_alias
        for target in TARGET_COLS:
            name = registered_model_name(target)
            try:
                version = client.get_model_version_by_alias(name, alias)
            except MlflowException as e:
                raise RuntimeError(
                    f"No model version has the '{alias}' alias for "
                    f"registered model '{name}'. Run `python -m src.promote "
                    f"promote {target} <version>` after training."
                ) from e

            model_uri = f"models:/{name}@{alias}"
            self.models[target] = mlflow.sklearn.load_model(model_uri)
            logger.info(
                f"Loaded model for '{target}' from {model_uri} "
                f"(registry version {version.version})"
            )

    def predict(self, readings: pd.DataFrame) -> dict:
        """Scores the most recent row of a chronologically ordered window of
        raw sensor readings (oldest first). Earlier rows only provide
        history for rolling features; the prediction is for the last row.
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Call load() first.")

        missing = [c for c in RAW_FEATURE_COLS if c not in readings.columns]
        if missing:
            raise ValueError(f"Missing required sensor columns: {missing}")

        raw = readings[RAW_FEATURE_COLS]

        non_finite = [c for c in RAW_FEATURE_COLS if not np.isfinite(raw[c]).all()]
        if non_finite:
            raise ValueError(
                f"Non-finite (NaN/Infinity) values in columns: {non_finite}"
            )

        negative = [c for c in NON_NEGATIVE_COLS if (raw[c] < 0).any()]
        if negative:
            raise ValueError(
                f"Negative values in columns that must be non-negative: {negative}"
            )

        if len(readings) < max(WINDOW_SIZES):
            logger.warning(
                f"Only {len(readings)} reading(s) provided; rolling features "
                f"for windows up to {max(WINDOW_SIZES)} will reflect less "
                "history than the model was trained with."
            )

        df = raw.reset_index(drop=True)
        fe = FeatureEngineer(window_sizes=WINDOW_SIZES)
        engineered = fe.create_rolling_features(df, SENSOR_COLS)
        latest = engineered.iloc[[-1]]

        results = {}
        for target, model in self.models.items():
            pred = int(model.predict(latest)[0])
            proba = model.predict_proba(latest)[0]
            class_idx = list(model.classes_).index(1)
            results[target] = {
                "prediction": pred,
                "failure_probability": float(proba[class_idx]),
            }
        return results
