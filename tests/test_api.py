from fastapi.testclient import TestClient

from opticargo_ml_models.api import app

CARGO_PAYLOAD = {
    "voyage": {
        "route_distance_km": 1200,
        "remaining_weight_ton": 180,
        "remaining_volume_m3": 600,
        "operating_cost_per_km_idr": 35000,
    },
    "candidate": {
        "cargo_weight_ton": 90,
        "cargo_volume_m3": 270,
        "asking_price_per_ton_idr": 1500000,
        "market_rate_per_ton_idr": 2100000,
        "origin_distance_km": 40,
        "destination_distance_km": 30,
        "schedule_gap_hours": 24,
        "supplier_rating": 4.6,
        "supplier_success_rate": 0.92,
        "supplier_cancellation_rate": 0.04,
        "commodity_compatibility": True,
        "certification_match": True,
        "temperature_match": True,
        "weather_risk": 0.15,
        "port_congestion": 0.20,
        "historical_acceptance_rate": 0.75,
    },
}

FORECAST_PAYLOAD = {
    "forecast_date": "2026-08-07T00:00:00Z",
    "route_distance_km": 1200,
    "historical_volume_7d_ton": 720,
    "historical_volume_30d_ton": 2850,
    "bookings_7d": 88,
    "vessel_capacity_ton": 1400,
    "commodity_index": 1.08,
    "is_holiday": False,
    "port_congestion": 0.22,
    "weather_risk": 0.18,
    "fuel_price_index": 1.03,
    "economic_activity_index": 1.07,
    "lead_time_days": 8,
    "forecast_horizon_days": 7,
}

ANOMALY_PAYLOAD = {
    "snapshot": {
        "observed_at": "2026-08-01T10:00:00Z",
        "booking_count": 52,
        "cargo_volume_ton": 430,
        "average_price_per_ton_idr": 2450000,
        "cancellation_rate": 0.04,
        "average_delay_hours": 4.5,
        "utilization_rate": 0.72,
        "port_congestion": 0.25,
        "weather_risk": 0.15,
        "supplier_failure_rate": 0.03,
    }
}


def test_health_and_model_registry():
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready").json()
        assert ready["status"] == "ready"
        assert ready["total_models"] == 3
        registry = client.get("/v1/models/status")
        assert registry.status_code == 200
        assert len(registry.json()["models"]) == 3


def test_all_inference_endpoints():
    with TestClient(app) as client:
        cargo = client.post("/v1/score/cargo-match", json=CARGO_PAYLOAD)
        assert cargo.status_code == 200
        cargo_body = cargo.json()
        assert 0 <= cargo_body["score"] <= 1
        assert cargo_body["hard_constraint_valid"] is True
        assert cargo_body["feature_explanations"]

        forecast = client.post("/v1/forecast/demand", json=FORECAST_PAYLOAD)
        assert forecast.status_code == 200
        forecast_body = forecast.json()
        assert forecast_body["predicted_volume_ton"] >= 0
        assert forecast_body["lower_bound_ton"] <= forecast_body["predicted_volume_ton"]
        assert forecast_body["upper_bound_ton"] >= forecast_body["predicted_volume_ton"]

        anomaly = client.post("/v1/anomalies/detect", json=ANOMALY_PAYLOAD)
        assert anomaly.status_code == 200
        anomaly_body = anomaly.json()
        assert 0 <= anomaly_body["anomaly_score"] <= 1
        assert anomaly_body["severity"] in {"normal", "low", "medium", "high", "critical"}
