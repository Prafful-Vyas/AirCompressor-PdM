import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit

from features import FeatureEngineer
from ingestion import DataIngestor
from preprocessing import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self):
        self.target_cols = [
            "bearings",
            "wpump",
            "radiator",
            "exvalve",
        ]  # Binary targets
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, df: pd.DataFrame):
        # 1. Feature Engineering
        fe = FeatureEngineer(window_sizes=[5, 10])
        # Include all numerical sensor columns for engineering
        sensors = [
            "rpm",
            "motor_power",
            "torque",
            "outlet_pressure_bar",
            "noise_db",
            "outlet_temp",
            "gaccx",
            "haccx",
        ]
        df = fe.create_rolling_features(df, sensors)

        # 2. Define Features and Target
        # For this example, let's predict 'bearings' failure
        y = df["bearings"]
        X = df.drop(
            columns=["id", "bearings", "wpump", "radiator", "exvalve", "acmotor"]
        )

        # 3. Time-Series Split (Essential for Sensor Data)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        # We iterate through the splits to ensure the model is robust across time
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Preprocess and Fit
            pipeline = build_preprocessing_pipeline(X.columns.tolist())
            X_train_transformed = pipeline.fit_transform(X_train)

            self.model.fit(X_train_transformed, y_train)

            # Evaluate Fold
            X_test_transformed = pipeline.transform(X_test)
            preds = self.model.predict(X_test_transformed)
            fold_f1 = f1_score(y_test, preds, average="weighted")

            cv_scores.append(fold_f1)

            # Log each fold's performance to MLflow
            mlflow.log_metric(f"fold_{fold}_f1", fold_f1)
            logger.info(f"Fold {fold} F1 Score: {fold_f1:.4f}")

        # Log the final average performance
        avg_f1 = np.mean(cv_scores)
        mlflow.log_metric("mean_cv_f1", avg_f1.item())
        logger.info(f"Mean CV F1 Score: {avg_f1:.4f}")

        with mlflow.start_run(run_name="Compressor_Bearing_Model"):
            # Log params
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("features_count", X.shape[1])

            # Simple Train/Test split for the demo (keep the last 20% for test)
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # 4. Preprocessing & Fit
            pipeline = build_preprocessing_pipeline(X.columns.tolist())
            X_train_transformed = pipeline.fit_transform(X_train)
            self.model.fit(X_train_transformed, y_train)

            # 5. Evaluate
            X_test_transformed = pipeline.transform(X_test)
            preds = self.model.predict(X_test_transformed)
            score = f1_score(y_test, preds, average="weighted")

            # Log metrics and model
            mlflow.log_metric("f1_score", score)
            mlflow_sklearn.log_model(self.model, "model")

            print(f"Training Complete. F1 Score: {score}")
            print(classification_report(y_test, preds))


if __name__ == "__main__":
    # Ensure you have your CSV in data/raw/
    loader = DataIngestor("data/raw/aircompressor.csv")
    raw_df = loader.load_data()

    trainer = Trainer()
    trainer.train(raw_df)
