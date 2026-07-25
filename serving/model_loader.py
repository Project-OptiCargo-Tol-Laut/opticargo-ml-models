import os
from training.scoring_model.heuristic import HeuristicScoringModel
from training.forecasting_model.heuristic import HeuristicForecastingModel
from training.anomaly_detection.heuristic import HeuristicAnomalyModel

def get_scoring_model():
    mode = os.getenv("SCORING_MODEL_MODE", "heuristic")
    return HeuristicScoringModel(), "heuristic"

def get_forecasting_model():
    mode = os.getenv("FORECASTING_MODEL_MODE", "heuristic")
    return HeuristicForecastingModel(), "heuristic"

def get_anomaly_model():
    mode = os.getenv("ANOMALY_MODEL_MODE", "heuristic")
    return HeuristicAnomalyModel(), "heuristic"