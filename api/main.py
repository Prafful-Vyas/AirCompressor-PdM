import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.logging_utils import configure_logging, request_id_var
from src.predict import Predictor
from src.settings import get_settings

configure_logging(get_settings().log_level, get_settings().log_json)
logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    rpm: float = Field(allow_inf_nan=False)
    motor_power: float = Field(allow_inf_nan=False)
    torque: float = Field(allow_inf_nan=False)
    outlet_pressure_bar: float = Field(allow_inf_nan=False)
    air_flow: float = Field(allow_inf_nan=False)
    noise_db: float = Field(allow_inf_nan=False)
    outlet_temp: float = Field(allow_inf_nan=False)
    wpump_outlet_press: float = Field(allow_inf_nan=False)
    water_inlet_temp: float = Field(allow_inf_nan=False)
    water_outlet_temp: float = Field(allow_inf_nan=False)
    wpump_power: float = Field(allow_inf_nan=False)
    water_flow: float = Field(allow_inf_nan=False)
    oilpump_power: float = Field(allow_inf_nan=False)
    oil_tank_temp: float = Field(allow_inf_nan=False)
    gaccx: float = Field(allow_inf_nan=False)
    gaccy: float = Field(allow_inf_nan=False)
    gaccz: float = Field(allow_inf_nan=False)
    haccx: float = Field(allow_inf_nan=False)
    haccy: float = Field(allow_inf_nan=False)
    haccz: float = Field(allow_inf_nan=False)


class PredictionRequest(BaseModel):
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        max_length=get_settings().max_readings_per_request,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Rejects oversized bodies, and tags every request/response pair with
    an X-Request-ID that also appears in structured log lines emitted
    while handling it (see src/logging_utils.py's ContextVar)."""
    settings = get_settings()
    content_length = request.headers.get("content-length")
    if (
        content_length is not None
        and int(content_length) > settings.max_body_size_bytes
    ):
        return JSONResponse(
            status_code=413, content={"detail": "Request body too large"}
        )

    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            f"{request.method} {request.url.path} "
            f"{response.status_code} {duration_ms:.1f}ms"
        )
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI's default handler echoes the raw invalid value back in the
    response body (e.g. a rejected NaN reading), which Starlette's
    JSONResponse then fails to serialize (it disallows NaN/Infinity) --
    turning a clean 422 into an unhandled 500. Omit the raw value instead."""
    errors = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Depends(api_key_header)) -> None:
    settings = get_settings()
    if provided_key is None or not secrets.compare_digest(
        provided_key, settings.api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    """Pure liveness probe: reports the process is up, independent of
    whether a model is loaded. Safe for infra to poll unauthenticated."""
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    """Readiness probe: reports whether the API can actually serve
    predictions right now."""
    predictor = app.state.predictor
    return {
        "status": "ok" if predictor is not None else "degraded",
        "models_loaded": list(predictor.models.keys()) if predictor else [],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
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
