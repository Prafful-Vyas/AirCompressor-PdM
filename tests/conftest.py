import mlflow
import pytest


@pytest.fixture
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    """Points MLflow at a throwaway local store so tests never read from or
    write to the project's real mlruns/ directory."""
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
