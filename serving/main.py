from fastapi import FastAPI
from opticargo_shared.models.ml_contracts import (
    ScoringRequest, ScoringResponse,
    ForecastRequest, ForecastResponse,
    AnomalyRequest, AnomalyResponse
)
from serving.model_loader import (
    get_scoring_model,
    get_forecasting_model,
    get_anomaly_model
)

app = FastAPI(title="OptiCargo ML Models Service", version="1.0")

@app.post("/score-cargo-match", response_model=ScoringResponse)
async def score_cargo_match(request: ScoringRequest):
    model, mode = get_scoring_model()
    score, exp = model.predict(
        distance_km=request.distance_km,
        remaining_capacity_ton=float(request.remaining_capacity_ton),
        cargo_weight_ton=float(request.cargo_weight_ton)
    )
    return ScoringResponse(match_score=score, model_mode=mode, explanation=exp)

@app.post("/forecast-demand", response_model=ForecastResponse)
async def forecast_demand(request: ForecastRequest):
    model, mode = get_forecasting_model()
    val, exp = model.predict(historical_volumes_ton=request.historical_volumes_ton)
    return ForecastResponse(forecasted_volume_ton=val, model_mode=mode, explanation=exp)

@app.post("/detect-anomaly", response_model=AnomalyResponse)
async def detect_anomaly(request: AnomalyRequest):
    model, mode = get_anomaly_model()
    is_anom, exp = model.predict(
        unit_price=request.unit_price,
        historical_prices=request.historical_prices
    )
    return AnomalyResponse(is_anomaly=is_anom, model_mode=mode, explanation=exp)