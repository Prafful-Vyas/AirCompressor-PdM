import numpy as np
import pandas as pd

from src.preprocessing import SensorSanityChecker, build_preprocessing_pipeline


def test_sensor_sanity_checker_clips_negative_pressure_columns():
    df = pd.DataFrame(
        {
            "outlet_pressure_bar": [-1.0, 2.0],
            "wpump_outlet_press": [-0.5, 3.0],
            "unrelated": [-9.0, -9.0],
        }
    )
    checker = SensorSanityChecker()
    result = checker.transform(df)

    assert (result["outlet_pressure_bar"] >= 0).all()
    assert (result["wpump_outlet_press"] >= 0).all()
    # Columns unrelated to pressure are left untouched.
    assert (result["unrelated"] == -9.0).all()


def test_preprocessing_pipeline_imputes_missing_values():
    df = pd.DataFrame(
        {
            "pressure": [10.5, np.nan, 12.0, 11.5],
            "temp": [70.0, 72.0, np.nan, 71.0],
        }
    )
    pipeline = build_preprocessing_pipeline(list(df.columns))
    result = pipeline.fit_transform(df)

    assert not np.isnan(result).any()
    assert result.shape == df.shape
