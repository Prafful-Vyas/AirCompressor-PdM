"""Shared constants for training and serving. Both paths must agree on
column names/order and MLflow experiment naming, so they're defined once
here rather than duplicated in train.py and predict.py."""

# Sensor columns that get rolling mean/std features engineered from them.
SENSOR_COLS = [
    "rpm",
    "motor_power",
    "torque",
    "outlet_pressure_bar",
    "noise_db",
    "outlet_temp",
    "gaccx",
    "haccx",
]

# All raw feature columns (everything except id, targets, and acmotor), in
# the exact order they appear in the source CSV. Inference input is
# reindexed to this order before feature engineering so the resulting
# feature matrix lines up with what the model was trained on.
RAW_FEATURE_COLS = [
    "rpm",
    "motor_power",
    "torque",
    "outlet_pressure_bar",
    "air_flow",
    "noise_db",
    "outlet_temp",
    "wpump_outlet_press",
    "water_inlet_temp",
    "water_outlet_temp",
    "wpump_power",
    "water_flow",
    "oilpump_power",
    "oil_tank_temp",
    "gaccx",
    "gaccy",
    "gaccz",
    "haccx",
    "haccy",
    "haccz",
]

TARGET_COLS = ["bearings", "wpump", "radiator", "exvalve"]  # Binary targets
DROP_COLS = ["id", "acmotor"] + TARGET_COLS

WINDOW_SIZES = [5, 10]

EXPERIMENT_NAME = "air-compressor-predictive-maintenance"
