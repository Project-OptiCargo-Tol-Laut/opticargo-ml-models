import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluate_forecasting_model(y_true: list[float], y_pred: list[float]) -> dict:
    """
    Mengevaluasi performa model Demand Forecasting (Time Series / Regresi).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print("=== Evaluasi Model Forecasting ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape * 100:.2f}%")
    print("==================================\n")
    
    return {"rmse": rmse, "mae": mae, "mape": mape}