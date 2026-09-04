import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import (
    DROP_COLS,
    EXPERIMENT_NAME,
    SENSOR_COLS,
    TARGET_COLS,
    WINDOW_SIZES,
    registered_model_name,
)
from .features import FeatureEngineer
from .ingestion import DataIngestor
from .preprocessing import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        window_sizes: list = WINDOW_SIZES,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.window_sizes = window_sizes
        self.n_splits = n_splits
        self.random_state = random_state

    def _build_pipeline(self) -> Pipeline:
        """Preprocessing + classifier as a single fitted artifact, so the
        model logged to MLflow can be applied directly to raw feature rows
        at inference time without re-implementing preprocessing elsewhere."""
        preprocessing = build_preprocessing_pipeline([])
        return Pipeline(
            steps=[
                *preprocessing.steps,
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100, random_state=self.random_state
                    ),
                ),
            ]
        )

    def train_target(self, df: pd.DataFrame, target_col: str) -> dict:
        """Trains and logs a model for a single binary target column."""
        y = df[target_col]
        X = df.drop(columns=DROP_COLS)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []

        with mlflow.start_run(run_name=f"Compressor_{target_col}_Model"):
            mlflow.log_param("target", target_col)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("features_count", X.shape[1])
            mlflow.log_param("window_sizes", self.window_sizes)
            mlflow.log_param("n_splits", self.n_splits)
            mlflow.log_param("random_state", self.random_state)

            # Time-Series CV: gives a robustness estimate of how the model
            # performs across different chronological slices of the data.
            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                pipeline = self._build_pipeline()
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                fold_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
                cv_scores.append(fold_f1)

                mlflow.log_metric(f"fold_{fold}_f1", fold_f1, step=fold)
                logger.info(f"[{target_col}] Fold {fold} F1 Score: {fold_f1:.4f}")

            mean_cv_f1 = float(np.mean(cv_scores))
            mlflow.log_metric("mean_cv_f1", mean_cv_f1)
            logger.info(f"[{target_col}] Mean CV F1 Score: {mean_cv_f1:.4f}")

            # Final holdout fit: train on the first 80% chronologically, hold
            # out the last 20% as the reported test set. This becomes the
            # model artifact that gets logged and (later) served.
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            final_pipeline = self._build_pipeline()
            final_pipeline.fit(X_train, y_train)
            preds = final_pipeline.predict(X_test)
            holdout_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            mlflow.log_metric("holdout_f1", holdout_f1)
            # Registers a new candidate version in the Model Registry, but
            # attaches no alias -- it has zero effect on what Predictor
            # serves until a human promotes it (see src/promote.py).
            mlflow_sklearn.log_model(
                final_pipeline,
                name="model",
                registered_model_name=registered_model_name(target_col),
            )

            report = classification_report(y_test, preds, zero_division=0)
            logger.info(f"[{target_col}] Holdout F1 Score: {holdout_f1:.4f}")
            logger.info(f"[{target_col}] Classification Report:\n{report}")

            return {
                "target": target_col,
                "mean_cv_f1": mean_cv_f1,
                "holdout_f1": holdout_f1,
            }

    def train_all(self, raw_df: pd.DataFrame) -> list:
        fe = FeatureEngineer(window_sizes=self.window_sizes)
        df = fe.create_rolling_features(raw_df, SENSOR_COLS)

        results = []
        for target_col in TARGET_COLS:
            results.append(self.train_target(df, target_col))
        return results


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Ensure you have your CSV in data/raw/
    loader = DataIngestor("data/raw/aircompressor.csv")
    raw_df = loader.load_data()

    trainer = Trainer()
    results = trainer.train_all(raw_df)

    for r in results:
        logger.info(
            f"{r['target']}: mean_cv_f1={r['mean_cv_f1']:.4f}, "
            f"holdout_f1={r['holdout_f1']:.4f}"
        )
