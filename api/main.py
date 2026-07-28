import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    rpm: float
    motor_power: float
    torque: float
    outlet_pressure_bar: float
    air_flow: float
    noise_db: float
    outlet_temp: float
    wpump_outlet_press: float
    water_inlet_temp: float
    water_outlet_temp: float
    wpump_power: float
    water_flow: float
    oilpump_power: float
    oil_tank_temp: float
    gaccx: float
    gaccy: float
    gaccz: float
    haccx: float
    haccy: float
    haccz: float


class PredictionRequest(BaseModel):
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description=(
            "Consecutive sensor readings, oldest first, ending at the "
            "point in time to score. Provide at least 10 readings so "
            "rolling-window features reflect real recent history."
        ),
    )


class TargetPrediction(BaseModel):
    prediction: int
    failure_probability: float


class PredictionResponse(BaseModel):
    predictions: dict[str, TargetPrediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = Predictor()
    try:
        predictor.load()
        logger.info(f"Models loaded successfully: {list(predictor.models.keys())}")
        app.state.predictor = predictor
    except Exception:
        logger.exception(
            "Failed to load models at startup; /predict will return 503 "
            "until models are available."
        )
        app.state.predictor = None
    yield


app = FastAPI(
    title="Air Compressor Predictive Maintenance API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    predictor = app.state.predictor
    return {
        "status": "ok" if predictor is not None else "degraded",
        "models_loaded": list(predictor.models.keys()) if predictor else [],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Train and log models via "
            "`python -m src.train` first.",
        )

    df = pd.DataFrame([r.model_dump() for r in request.readings])

    try:
        results = predictor.predict(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500, detail="Internal error during prediction"
        ) from e

    return PredictionResponse(
        predictions={
            target: TargetPrediction(**result) for target, result in results.items()
        }
    )
