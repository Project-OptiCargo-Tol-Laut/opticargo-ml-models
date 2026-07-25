import pytest
from fastapi.testclient import TestClient
from serving.main import app

client = TestClient(app)

def test_score_cargo_match_endpoint():
    payload = {
        "ship_id": "123e4567-e89b-12d3-a456-426614174000",
        "cargo_listing_id": "123e4567-e89b-12d3-a456-426614174001",
        "distance_km": 200.0,
        "remaining_capacity_ton": 500.0,
        "cargo_weight_ton": 100.0
    }
    response = client.post("/score-cargo-match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_mode"] == "heuristic"
    assert "match_score" in data

def test_forecast_demand_endpoint():
    payload = {
        "commodity_id": "123e4567-e89b-12d3-a456-426614174002",
        "port_id": "123e4567-e89b-12d3-a456-426614174003",
        "historical_volumes_ton": [100.0, 150.0, 200.0]
    }
    response = client.post("/forecast-demand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_mode"] == "heuristic"
    assert data["forecasted_volume_ton"] == 150.0

def test_detect_anomaly_endpoint():
    # Menguji harga ekstrem (anomali)
    payload = {
        "commodity_id": "123e4567-e89b-12d3-a456-426614174002",
        "unit_price": 50000.0,
        "historical_prices": [10000.0, 10500.0, 9800.0]
    }
    response = client.post("/detect-anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_mode"] == "heuristic"
    assert data["is_anomaly"] is True