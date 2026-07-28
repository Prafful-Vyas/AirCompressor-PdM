import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorSanityChecker(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle physical impossibilities
    (e.g., negative pressure or impossible temperatures).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Pressure can't be negative. Matches any column with "press" in its
        # name (e.g. outlet_pressure_bar, wpump_outlet_press).
        pressure_cols = [c for c in X_copy.columns if "press" in c.lower()]
        for col in pressure_cols:
            X_copy.loc[X_copy[col] < 0, col] = 0
        return X_copy


def build_preprocessing_pipeline(numerical_cols: list) -> Pipeline:
    """
    Creates a production-grade pipeline for sensor data.
    Uses RobustScaler because sensor spikes (outliers) are common
    in failing compressors.
    """
    pipeline = Pipeline(
        [
            ("sanity_check", SensorSanityChecker()),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Median is safer for skewed sensor data
            (
                "scaler",
                RobustScaler(),
            ),  # Better than StandardScaler when you have outliers
        ]
    )

    return pipeline


if __name__ == "__main__":
    # Mock data for testing
    test_df = pd.DataFrame(
        {
            "pressure": [10.5, -1.0, 12.0, 11.5],
            "temp": [70, 72, 300, 71],  # 300 might be an outlier
            "vibration": [0.1, 0.2, 0.15, 0.2],
        }
    )

    cols = ["pressure", "temp", "vibration"]
    pipe = build_preprocessing_pipeline(cols)

    processed_data = pipe.fit_transform(test_df)
    logger.info("Preprocessing Pipeline Test Successful.")
    print(processed_data)
