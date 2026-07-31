from datetime import UTC, datetime

from opticargo_ml_models.anomalies import build_anomaly_feature_row, heuristic_anomaly
from opticargo_ml_models.contracts import (
    AnomalyDetectionRequest,
    CargoCandidate,
    CargoMatchRequest,
    DemandForecastRequest,
    OperationalSnapshot,
    VoyageContext,
)
from opticargo_ml_models.features import build_feature_row, heuristic_score
from opticargo_ml_models.forecasting import build_forecast_feature_row, heuristic_forecast


def valid_request() -> CargoMatchRequest:
    return CargoMatchRequest(
        voyage=VoyageContext(
            route_distance_km=1200,
            remaining_weight_ton=180,
            remaining_volume_m3=600,
            operating_cost_per_km_idr=35000,
        ),
        candidate=CargoCandidate(
            cargo_weight_ton=90,
            cargo_volume_m3=270,
            asking_price_per_ton_idr=1_500_000,
            market_rate_per_ton_idr=2_100_000,
            origin_distance_km=40,
            destination_distance_km=30,
            schedule_gap_hours=24,
            supplier_rating=4.6,
            supplier_success_rate=0.92,
            supplier_cancellation_rate=0.04,
            commodity_compatibility=True,
            certification_match=True,
            temperature_match=True,
            weather_risk=0.15,
            port_congestion=0.20,
            historical_acceptance_rate=0.75,
        ),
    )


def test_valid_candidate_has_positive_score():
    features = build_feature_row(valid_request())
    assert features["hard_constraint_valid"] == 1.0
    assert 0.5 < heuristic_score(features) <= 1.0


def test_overcapacity_is_hard_rejected():
    request = valid_request()
    request.candidate.cargo_weight_ton = 999
    features = build_feature_row(request)
    assert features["hard_constraint_valid"] == 0.0
    assert heuristic_score(features) == 0.0


def test_forecast_heuristic_returns_bounded_positive_value():
    request = DemandForecastRequest(
        forecast_date=datetime(2026, 8, 7, tzinfo=UTC),
        route_distance_km=1200,
        historical_volume_7d_ton=700,
        historical_volume_30d_ton=2800,
        bookings_7d=80,
        vessel_capacity_ton=1400,
        commodity_index=1.1,
        port_congestion=0.2,
        weather_risk=0.15,
        fuel_price_index=1.0,
        economic_activity_index=1.05,
        lead_time_days=7,
        forecast_horizon_days=7,
    )
    value = heuristic_forecast(build_forecast_feature_row(request))
    assert 0 < value <= request.vessel_capacity_ton * 1.5


def test_anomaly_heuristic_separates_normal_and_extreme():
    normal = AnomalyDetectionRequest(
        snapshot=OperationalSnapshot(
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
            booking_count=50,
            cargo_volume_ton=430,
            average_price_per_ton_idr=2_400_000,
            cancellation_rate=0.04,
            average_delay_hours=4,
            utilization_rate=0.7,
            port_congestion=0.2,
            weather_risk=0.1,
            supplier_failure_rate=0.03,
        )
    )
    extreme = normal.model_copy(deep=True)
    extreme.snapshot.cancellation_rate = 0.75
    extreme.snapshot.average_delay_hours = 120
    extreme.snapshot.supplier_failure_rate = 0.60
    normal_score, _ = heuristic_anomaly(build_anomaly_feature_row(normal))
    extreme_score, _ = heuristic_anomaly(build_anomaly_feature_row(extreme))
    assert normal_score < extreme_score
    assert extreme_score >= 0.5
