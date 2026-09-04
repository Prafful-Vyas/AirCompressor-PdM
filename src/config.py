"""Shared constants for training and serving. Both paths must agree on
column names/order and MLflow experiment naming, so they're defined once
here rather than duplicated in train.py and predict.py.

Runtime/infra config that varies between environments (API keys, CORS,
tracking URIs, ports) lives in src/settings.py, not here."""

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

# Columns that are physically impossible to be negative (speeds, power
# draws, flow rates, noise level). Deliberately excludes temperatures
# (can legitimately read near/below zero at startup or in cold climates),
# torque, and the accelerometer axes (gaccx/y/z, haccx/y/z are signed
# vibration readings that oscillate around zero).
NON_NEGATIVE_COLS = [
    "rpm",
    "motor_power",
    "air_flow",
    "noise_db",
    "outlet_pressure_bar",
    "wpump_outlet_press",
    "wpump_power",
    "water_flow",
    "oilpump_power",
]

WINDOW_SIZES = [5, 10]

EXPERIMENT_NAME = "air-compressor-predictive-maintenance"


def registered_model_name(target: str) -> str:
    """Single naming source for the MLflow Model Registry entry a target's
    runs are registered under. Used by train.py (registers), predict.py
    (loads via the promotion alias), and promote.py (the promotion CLI)."""
    return f"{EXPERIMENT_NAME}-{target}"
