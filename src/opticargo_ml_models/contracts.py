from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelMode(StrEnum):
    HEURISTIC = "heuristic"
    TRAINED = "trained"


class VoyageContext(StrictModel):
    voyage_id: UUID = Field(default_factory=uuid4)
    route_id: UUID = Field(default_factory=uuid4)
    route_distance_km: float = Field(gt=0, le=20_000)
    remaining_weight_ton: float = Field(gt=0, le=100_000)
    remaining_volume_m3: float = Field(gt=0, le=1_000_000)
    operating_cost_per_km_idr: float = Field(gt=0)


class CargoCandidate(StrictModel):
    cargo_listing_id: UUID = Field(default_factory=uuid4)
    supplier_id: UUID = Field(default_factory=uuid4)
    cargo_weight_ton: float = Field(gt=0, le=100_000)
    cargo_volume_m3: float = Field(gt=0, le=1_000_000)
    asking_price_per_ton_idr: float = Field(gt=0)
    market_rate_per_ton_idr: float = Field(gt=0)
    origin_distance_km: float = Field(ge=0, le=10_000)
    destination_distance_km: float = Field(ge=0, le=10_000)
    schedule_gap_hours: float = Field(ge=-720, le=8_760)
    supplier_rating: float = Field(ge=1, le=5)
    supplier_success_rate: float = Field(ge=0, le=1)
    supplier_cancellation_rate: float = Field(ge=0, le=1)
    commodity_compatibility: bool
    certification_match: bool
    temperature_match: bool = True
    weather_risk: float = Field(ge=0, le=1)
    port_congestion: float = Field(ge=0, le=1)
    historical_acceptance_rate: float = Field(ge=0, le=1)


class CargoMatchRequest(StrictModel):
    voyage: VoyageContext
    candidate: CargoCandidate
    trace_id: str | None = None

    @model_validator(mode="after")
    def add_trace_id(self) -> CargoMatchRequest:
        self.trace_id = self.trace_id or str(uuid4())
        return self


class DemandForecastRequest(StrictModel):
    route_id: UUID = Field(default_factory=uuid4)
    forecast_date: datetime
    route_distance_km: float = Field(gt=0, le=20_000)
    historical_volume_7d_ton: float = Field(ge=0, le=1_000_000)
    historical_volume_30d_ton: float = Field(ge=0, le=4_000_000)
    bookings_7d: int = Field(ge=0, le=1_000_000)
    vessel_capacity_ton: float = Field(gt=0, le=100_000)
    commodity_index: float = Field(ge=0.25, le=3.0)
    is_holiday: bool = False
    port_congestion: float = Field(ge=0, le=1)
    weather_risk: float = Field(ge=0, le=1)
    fuel_price_index: float = Field(ge=0.25, le=3.0)
    economic_activity_index: float = Field(ge=0.25, le=3.0)
    lead_time_days: float = Field(ge=0, le=365)
    forecast_horizon_days: int = Field(default=7, ge=1, le=90)
    trace_id: str | None = None

    @model_validator(mode="after")
    def add_trace_id(self) -> DemandForecastRequest:
        self.trace_id = self.trace_id or str(uuid4())
        return self


class OperationalSnapshot(StrictModel):
    route_id: UUID = Field(default_factory=uuid4)
    observed_at: datetime
    booking_count: float = Field(ge=0, le=1_000_000)
    cargo_volume_ton: float = Field(ge=0, le=1_000_000)
    average_price_per_ton_idr: float = Field(gt=0)
    cancellation_rate: float = Field(ge=0, le=1)
    average_delay_hours: float = Field(ge=0, le=10_000)
    utilization_rate: float = Field(ge=0, le=2)
    port_congestion: float = Field(ge=0, le=1)
    weather_risk: float = Field(ge=0, le=1)
    supplier_failure_rate: float = Field(ge=0, le=1)


class AnomalyDetectionRequest(StrictModel):
    snapshot: OperationalSnapshot
    trace_id: str | None = None

    @model_validator(mode="after")
    def add_trace_id(self) -> AnomalyDetectionRequest:
        self.trace_id = self.trace_id or str(uuid4())
        return self


class FeatureExplanation(StrictModel):
    feature: str
    contribution: float
    direction: str
    value: float | bool | str | None = None
    description: str | None = None


class CargoMatchResponse(StrictModel):
    score: float = Field(ge=0, le=1)
    model_mode: ModelMode
    model_version: str
    fallback_used: bool
    hard_constraint_valid: bool
    feature_explanations: list[FeatureExplanation]
    warnings: list[str] = Field(default_factory=list)
    trace_id: str


class DemandForecastResponse(StrictModel):
    predicted_volume_ton: float = Field(ge=0)
    lower_bound_ton: float = Field(ge=0)
    upper_bound_ton: float = Field(ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    forecast_horizon_days: int
    model_mode: ModelMode
    model_version: str
    fallback_used: bool
    feature_explanations: list[FeatureExplanation]
    warnings: list[str] = Field(default_factory=list)
    trace_id: str


class AnomalyDetectionResponse(StrictModel):
    is_anomaly: bool
    anomaly_score: float = Field(ge=0, le=1)
    severity: str
    model_mode: ModelMode
    model_version: str
    fallback_used: bool
    feature_explanations: list[FeatureExplanation]
    warnings: list[str] = Field(default_factory=list)
    trace_id: str


class ModelStatusEntry(StrictModel):
    model_name: str
    model_mode: ModelMode
    model_version: str
    artifact_path: str | None
    loaded: bool
    fallback_available: bool = True
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRegistryStatusResponse(StrictModel):
    service_status: str
    loaded_models: int
    total_models: int
    models: list[ModelStatusEntry]
