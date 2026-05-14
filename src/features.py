import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, window_sizes: list = [5, 10, 20]):
        self.window_sizes = window_sizes

    def create_rolling_features(
        self, df: pd.DataFrame, sensor_cols: list
    ) -> pd.DataFrame:
        """
        Creates rolling mean and std dev for sensor columns.
        Essential for capturing degradation trends in machinery.
        """
        df_feat = df.copy()

        for col in sensor_cols:
            for window in self.window_sizes:
                # Rolling Mean: Captures shifts in the baseline
                df_feat[f"{col}_roll_mean_{window}"] = (
                    df[col].rolling(window=window).mean()
                )

                # Rolling Std: Captures increased 'shakiness' or instability
                df_feat[f"{col}_roll_std_{window}"] = (
                    df[col].rolling(window=window).std()
                )

        # Fill the initial NaNs created by the rolling window
        df_feat = df_feat.bfill()
        logger.info(
            f"Generated {len(df_feat.columns) - len(df.columns)} new rolling features."
        )
        return df_feat

    def create_lag_features(
        self, df: pd.DataFrame, sensor_cols: list, lags: list = [1, 2]
    ) -> pd.DataFrame:
        """
        Captures the delta between current and previous states.
        """
        for col in sensor_cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df.bfill()


if __name__ == "__main__":
    # Mock data: 50 timestamps of pressure data
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="T"),
            "pressure": np.random.normal(100, 5, 50),
            "vibration": np.random.normal(0.5, 0.1, 50),
        }
    )

    fe = FeatureEngineer(window_sizes=[5, 10])
    sensors = ["pressure", "vibration"]

    # Apply engineering
    data_with_features = fe.create_rolling_features(data, sensors)
    data_with_features = fe.create_lag_features(data_with_features, sensors)

    print(data_with_features.head(15))
