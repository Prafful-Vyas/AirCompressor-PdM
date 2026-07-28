import logging

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .config import (
    EXPERIMENT_NAME,
    RAW_FEATURE_COLS,
    SENSOR_COLS,
    TARGET_COLS,
    WINDOW_SIZES,
)
from .features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Loads the most recently trained model per target from MLflow and
    scores the most recent row of a window of raw sensor readings."""

    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self.models: dict = {}

    def _latest_run_id(
        self, client: MlflowClient, experiment_id: str, target: str
    ) -> str:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"params.target = '{target}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                f"No MLflow run found for target '{target}' in experiment "
                f"'{self.experiment_name}'. Run `python -m src.train` first."
            )
        return runs[0].info.run_id

    def load(self) -> None:
        """Loads the most recent model for each target from MLflow."""
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment '{self.experiment_name}' not found. "
                "Run `python -m src.train` first."
            )

        for target in TARGET_COLS:
            run_id = self._latest_run_id(client, experiment.experiment_id, target)
            model_uri = f"runs:/{run_id}/model"
            self.models[target] = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model for '{target}' from run {run_id}")

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

        if len(readings) < max(WINDOW_SIZES):
            logger.warning(
                f"Only {len(readings)} reading(s) provided; rolling features "
                f"for windows up to {max(WINDOW_SIZES)} will reflect less "
                "history than the model was trained with."
            )

        df = readings[RAW_FEATURE_COLS].reset_index(drop=True)
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
