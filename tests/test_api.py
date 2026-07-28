import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import RAW_FEATURE_COLS


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
    return TestClient(app)


def test_health_reports_ok_when_models_loaded(client):
    app.state.predictor = FakePredictor()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": ["bearings", "wpump"]}


def test_health_reports_degraded_when_models_not_loaded(client):
    app.state.predictor = None
    response = client.get("/health")

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
