from training.scoring_model.heuristic import HeuristicScoringModel
from training.forecasting_model.heuristic import HeuristicForecastingModel
from training.anomaly_detection.heuristic import HeuristicAnomalyModel

def test_scoring_heuristic_logic():
    model = HeuristicScoringModel()
    score, exp = model.predict(distance_km=100.0, remaining_capacity_ton=500.0, cargo_weight_ton=100.0)
    assert isinstance(score, float)

def test_forecasting_heuristic_logic():
    model = HeuristicForecastingModel()
    val, exp = model.predict(historical_volumes_ton=[10.0, 20.0, 30.0])
    assert val == 20.0  # Rata-rata dari 10, 20, 30

def test_anomaly_heuristic_logic():
    model = HeuristicAnomalyModel()
    # Harga 100, median 120. Harga uji 500 (jauh di atas 3x median)
    is_anom, exp = model.predict(unit_price=500.0, historical_prices=[100.0, 150.0, 120.0])
    assert is_anom is True