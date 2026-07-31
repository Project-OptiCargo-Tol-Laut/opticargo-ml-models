from __future__ import annotations

import importlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import get_settings
from .contracts import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    CargoMatchRequest,
    CargoMatchResponse,
    DemandForecastRequest,
    DemandForecastResponse,
    ModelRegistryStatusResponse,
    ModelStatusEntry,
)
from .manager import MultiModelManager
from .metrics import INFERENCE_DURATION
from .security import require_internal_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
manager = MultiModelManager(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.require_shared_contracts:
        importlib.import_module("opticargo_shared")
    if settings.model_preload:
        manager.preload()
    yield


app = FastAPI(
    title="OptiCargo ML Models",
    version="1.0.0",
    docs_url="/docs" if settings.opticargo_environment == "development" else None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/health/live", tags=["health"])
def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def health_ready() -> dict[str, str | int | bool]:
    registry = manager.status()
    return {
        "status": registry.service_status,
        "loaded_models": registry.loaded_models,
        "total_models": registry.total_models,
        "fallback_available": all(item.fallback_available for item in registry.models),
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/v1/models/status",
    response_model=ModelRegistryStatusResponse,
    dependencies=[Depends(require_internal_token)],
    tags=["models"],
)
def model_status() -> ModelRegistryStatusResponse:
    return manager.status()


@app.get(
    "/v1/models/{model_name}/status",
    response_model=ModelStatusEntry,
    dependencies=[Depends(require_internal_token)],
    tags=["models"],
)
def individual_model_status(model_name: str) -> ModelStatusEntry:
    for key, runtime in manager.models.items():
        if model_name in {key, runtime.name}:
            return manager.status_entry(key)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model tidak ditemukan.")


@app.post(
    "/v1/score/cargo-match",
    response_model=CargoMatchResponse,
    dependencies=[Depends(require_internal_token)],
    tags=["scoring"],
)
def score_cargo_match(payload: CargoMatchRequest) -> CargoMatchResponse:
    started = time.perf_counter()
    try:
        return manager.score_cargo(payload)
    finally:
        INFERENCE_DURATION.labels(model_name="cargo-match-scorer").observe(
            time.perf_counter() - started
        )


@app.post(
    "/v1/forecast/demand",
    response_model=DemandForecastResponse,
    dependencies=[Depends(require_internal_token)],
    tags=["forecasting"],
)
def forecast_demand(payload: DemandForecastRequest) -> DemandForecastResponse:
    started = time.perf_counter()
    try:
        return manager.forecast_demand(payload)
    finally:
        INFERENCE_DURATION.labels(model_name="demand-forecaster").observe(
            time.perf_counter() - started
        )


@app.post(
    "/v1/anomalies/detect",
    response_model=AnomalyDetectionResponse,
    dependencies=[Depends(require_internal_token)],
    tags=["anomalies"],
)
def detect_anomaly(payload: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
    started = time.perf_counter()
    try:
        return manager.detect_anomaly(payload)
    finally:
        INFERENCE_DURATION.labels(model_name="operational-anomaly-detector").observe(
            time.perf_counter() - started
        )
