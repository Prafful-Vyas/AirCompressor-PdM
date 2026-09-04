"""Runtime/infra configuration, loaded from environment variables (prefix
`ACPDM_`) and optionally a `.env` file. Domain/ML constants (column names,
window sizes, experiment name) stay in `src/config.py` — this module is only
for things that vary between environments (dev/prod, local/Docker)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACPDM_", env_file=".env")

    env: Literal["development", "production"] = "development"

    # API auth. Must be overridden via ACPDM_API_KEY in any real deployment.
    api_key: str = "dev-local-key"

    # CORS: explicit allow-list only, never a wildcard.
    cors_allow_origins: list[str] = []

    log_level: str = "INFO"
    log_json: bool = True

    # None lets the MLflow client fall back to its own default (local
    # ./mlruns); set explicitly to point at a tracking server.
    mlflow_tracking_uri: str | None = None

    max_readings_per_request: int = 500
    max_body_size_bytes: int = 1_048_576  # 1 MB

    # Model registry promotion gate (see src/promote.py).
    model_registry_alias: str = "production"
    min_holdout_f1_for_promotion: float = 0.0

    host: str = "0.0.0.0"
    port: int = 8000

    data_raw_path: str = "data/raw/aircompressor.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
