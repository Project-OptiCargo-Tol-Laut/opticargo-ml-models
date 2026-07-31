from prometheus_client import Counter, Gauge, Histogram, Info

PREDICTION_TOTAL = Counter(
    "opticargo_ml_prediction_total",
    "Jumlah inference per model.",
    ["model_name", "model_mode", "result"],
)
FALLBACK_TOTAL = Counter(
    "opticargo_ml_fallback_total",
    "Jumlah penggunaan fallback per model.",
    ["model_name", "reason"],
)
INFERENCE_DURATION = Histogram(
    "opticargo_ml_inference_duration_seconds",
    "Durasi inference model.",
    ["model_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
MODEL_READY = Gauge(
    "opticargo_ml_model_ready",
    "Status readiness tiap model.",
    ["model_name"],
)
MODEL_INFO = Info(
    "opticargo_ml_model",
    "Metadata model aktif.",
    ["model_name"],
)
