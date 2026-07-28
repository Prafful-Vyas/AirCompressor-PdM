import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, window_sizes: list | None = None):
        self.window_sizes = window_sizes if window_sizes is not None else [5, 10, 20]

    def create_rolling_features(
        self, df: pd.DataFrame, sensor_cols: list
    ) -> pd.DataFrame:
        """
        Creates rolling mean and std dev for sensor columns.
        Essential for capturing degradation trends in machinery.
        """
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Sensor columns not found in dataframe: {missing}")

        df_feat = df.copy()

        for col in sensor_cols:
            for window in self.window_sizes:
                # min_periods=1 keeps the window causal: each row only ever
                # aggregates its own past/current values, never future ones.
                df_feat[f"{col}_roll_mean_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )

                # Rolling Std: Captures increased 'shakiness' or instability
                df_feat[f"{col}_roll_std_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )

        # std of a single observation is undefined (NaN); treat "no variability
        # observed yet" as 0 rather than back-filling from future rows.
        std_cols = [c for c in df_feat.columns if "_roll_std_" in c]
        df_feat[std_cols] = df_feat[std_cols].fillna(0)

        logger.info(
            f"Generated {len(df_feat.columns) - len(df.columns)} new rolling features."
        )
        return df_feat

    def create_lag_features(
        self, df: pd.DataFrame, sensor_cols: list, lags: list | None = None
    ) -> pd.DataFrame:
        """
        Captures the delta between current and previous states.
        """
        lags = lags if lags is not None else [1, 2]
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Sensor columns not found in dataframe: {missing}")

        df_feat = df.copy()
        for col in sensor_cols:
            for lag in lags:
                df_feat[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Leading NaNs from shift() are left for the preprocessing pipeline's
        # imputer (fit on train data only) to handle, instead of back-filling
        # them with future values.
        return df_feat


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
