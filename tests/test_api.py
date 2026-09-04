import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import RAW_FEATURE_COLS
from src.settings import get_settings


class FakePredictor:
    def __init__(self):
        self.models = {"bearings": object(), "wpump": object()}

    def predict(self, df):
        return {t: {"prediction": 0, "failure_probability": 0.1} for t in self.models}


class FailingPredictor:
    models = {"bearings": object()}

    def predict(self, df):
        raise ValueError("bad input")


def sample_reading() -> dict:
    return {col: 1.0 for col in RAW_FEATURE_COLS}


@pytest.fixture
def client():
    """Authenticated by default (matches the dev API key), since most
    tests care about business logic, not auth itself."""
    test_client = TestClient(app)
    test_client.headers.update({"X-API-Key": get_settings().api_key})
    return test_client


@pytest.fixture
def unauthenticated_client():
    return TestClient(app)


def test_health_is_always_ok_regardless_of_model_state(client):
    app.state.predictor = None
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_reports_ok_when_models_loaded(client):
    app.state.predictor = FakePredictor()
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": ["bearings", "wpump"]}


def test_health_ready_reports_degraded_when_models_not_loaded(client):
    app.state.predictor = None
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "models_loaded": []}


def test_predict_returns_predictions(client):
    app.state.predictor = FakePredictor()
    response = client.post("/predict", json={"readings": [sample_reading()]})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"]["bearings"] == {
        "prediction": 0,
        "failure_probability": 0.1,
    }


def test_predict_response_carries_request_id_header(client):
    app.state.predictor = FakePredictor()
    response = client.post("/predict", json={"readings": [sample_reading()]})

    assert "X-Request-ID" in response.headers


def test_predict_returns_503_when_models_not_loaded(client):
    app.state.predictor = None
    response = client.post("/predict", json={"readings": [sample_reading()]})

    assert response.status_code == 503


def test_predict_returns_400_on_predictor_value_error(client):
    app.state.predictor = FailingPredictor()
    response = client.post("/predict", json={"readings": [sample_reading()]})

    assert response.status_code == 400
    assert response.json()["detail"] == "bad input"


def test_predict_returns_422_on_incomplete_reading(client):
    app.state.predictor = FakePredictor()
    response = client.post("/predict", json={"readings": [{"rpm": 500}]})

    assert response.status_code == 422


def test_predict_returns_422_on_empty_readings_list(client):
    app.state.predictor = FakePredictor()
    response = client.post("/predict", json={"readings": []})

    assert response.status_code == 422


def test_predict_returns_422_on_oversized_readings_list(client):
    app.state.predictor = FakePredictor()
    too_many = [sample_reading()] * (get_settings().max_readings_per_request + 1)
    response = client.post("/predict", json={"readings": too_many})

    assert response.status_code == 422


def test_predict_returns_422_on_nan_reading(client):
    """httpx's `json=` kwarg refuses to serialize NaN client-side, so this
    sends a raw body (Python's json.dumps emits a bare `NaN` token by
    default) to actually exercise the server's allow_inf_nan=False check."""
    app.state.predictor = FakePredictor()
    reading = sample_reading()
    reading["rpm"] = float("nan")
    body = json.dumps({"readings": [reading]})
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_predict_returns_422_on_wrong_type(client):
    app.state.predictor = FakePredictor()
    reading = sample_reading()
    reading["rpm"] = "not-a-number"
    response = client.post("/predict", json={"readings": [reading]})

    assert response.status_code == 422


def test_predict_requires_api_key(unauthenticated_client):
    app.state.predictor = FakePredictor()
    response = unauthenticated_client.post(
        "/predict", json={"readings": [sample_reading()]}
    )

    assert response.status_code == 401


def test_predict_rejects_wrong_api_key(unauthenticated_client):
    app.state.predictor = FakePredictor()
    unauthenticated_client.headers.update({"X-API-Key": "wrong-key"})
    response = unauthenticated_client.post(
        "/predict", json={"readings": [sample_reading()]}
    )

    assert response.status_code == 401


def test_health_does_not_require_api_key(unauthenticated_client):
    response = unauthenticated_client.get("/health")

    assert response.status_code == 200


def test_cors_header_absent_for_unconfigured_origin(client):
    """Default settings ship with an empty CORS allow-list, so a
    cross-origin request should not receive an allow-origin header."""
    app.state.predictor = FakePredictor()
    response = client.post(
        "/predict",
        json={"readings": [sample_reading()]},
        headers={"Origin": "https://not-allowed.example"},
    )

    assert "access-control-allow-origin" not in response.headers
