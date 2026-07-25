import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_scoring_model(y_true: list[float], y_pred: list[float]) -> dict:
    """
    Mengevaluasi performa model Cargo Scoring (Regresi).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("=== Evaluasi Model Scoring ===")
    print(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print(f"MAE (Mean Absolute Error)      : {mae:.4f}")
    print(f"R-squared (R2)                 : {r2:.4f}")
    print("==============================\n")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}