from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_anomaly_model(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Mengevaluasi performa model Anomaly Detection (Klasifikasi Biner).
    Asumsi: 1 = Anomali, 0 = Normal.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("=== Evaluasi Model Anomaly ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("==============================\n")
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "confusion_matrix": cm.tolist()
    }