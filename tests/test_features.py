import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer


def make_df(values):
    return pd.DataFrame({"sensor": values})


def test_rolling_mean_is_causal_not_leaky():
    # A huge spike at the very end must not influence the rolling mean of
    # earlier rows -- that would mean future data leaked backwards.
    values = [1.0, 1.0, 1.0, 1.0, 1000.0]
    fe = FeatureEngineer(window_sizes=[3])
    result = fe.create_rolling_features(make_df(values), ["sensor"])

    assert result["sensor_roll_mean_3"].iloc[0] == 1.0
    assert result["sensor_roll_mean_3"].iloc[1] == 1.0
    expected_last_mean = (1.0 + 1.0 + 1000.0) / 3
    assert result["sensor_roll_mean_3"].iloc[-1] == pytest.approx(expected_last_mean)


def test_rolling_features_have_no_nans():
    values = [5.0, 6.0, 7.0, 8.0]
    fe = FeatureEngineer(window_sizes=[5, 10])
    result = fe.create_rolling_features(make_df(values), ["sensor"])

    assert not result.isnull().values.any()


def test_rolling_features_missing_column_raises():
    fe = FeatureEngineer(window_sizes=[5])
    with pytest.raises(ValueError, match="not found"):
        fe.create_rolling_features(make_df([1.0, 2.0]), ["does_not_exist"])


def test_lag_features_shift_without_leaking_future_values():
    values = [10.0, 20.0, 30.0, 40.0]
    fe = FeatureEngineer()
    result = fe.create_lag_features(make_df(values), ["sensor"], lags=[1])

    assert np.isnan(result["sensor_lag_1"].iloc[0])
    assert result["sensor_lag_1"].iloc[1] == 10.0
    assert result["sensor_lag_1"].iloc[3] == 30.0


def test_lag_features_missing_column_raises():
    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="not found"):
        fe.create_lag_features(make_df([1.0, 2.0]), ["does_not_exist"])
