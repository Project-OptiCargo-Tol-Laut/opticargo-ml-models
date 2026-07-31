# OptiCargo ML Models

Internal service untuk tiga kemampuan machine learning OptiCargo:

1. **Cargo match scoring** — menilai kecocokan cargo terhadap voyage.
2. **Demand forecasting** — memprediksi volume cargo per route dan horizon waktu.
3. **Operational anomaly detection** — mendeteksi pola operasional tidak biasa.

Repository ini juga menyediakan generator data sintetis, training pipeline, promotion guard,
artifact lifecycle, MLflow/MinIO integration, retraining job, drift check, health/readiness, metrics,
dan heuristic fallback.

> Seluruh baseline awal menggunakan data sintetis. Target sekitar 90% adalah kualitas terhadap pola
> generator sintetis, bukan jaminan performa produksi.

## Arsitektur runtime

```text
opticargo-agents / gateway
          |
          v
opticargo-ml-models :8000
  |       |        |
  |       |        +-- anomaly detector
  |       +----------- demand forecaster
  +------------------- cargo match scorer
          |
          +-- MLflow :5000
          +-- MinIO  :9000
          +-- Prometheus /metrics
```

Service ini **internal-only**. Browser tidak memanggil endpoint ML secara langsung.

## Endpoint

```text
GET  /health/live
GET  /health/ready
GET  /metrics
GET  /v1/models/status
GET  /v1/models/{model_name}/status
POST /v1/score/cargo-match
POST /v1/forecast/demand
POST /v1/anomalies/detect
```

Untuk environment non-development, kirim header:

```text
X-Internal-Service-Token: <INTERNAL_SERVICE_TOKEN>
```

## Setup Conda

```powershell
cd C:\MY_FOLDER\CODE\OptiCargo-Tol-Laut-Project\opticargo-ml-models

conda create -n opticargo-ml-models python=3.11 -y
conda activate opticargo-ml-models

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ..\opticargo-shared
python -m pip install -e ".[dev]"
```

Verifikasi interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

Path harus menunjuk ke:

```text
C:\Users\Komputer\miniconda3\envs\opticargo-ml-models\python.exe
```

## Generate ketiga dataset

Paket telah menyertakan dataset sintetis hasil generator. Untuk membuat ulang:

```powershell
python -m opticargo_ml_models.training.generate_all `
  --rows 8000 `
  --output-dir data/synthetic
```

Hasil:

```text
data/synthetic/cargo_match.csv
data/synthetic/demand_forecast.csv
data/synthetic/operational_anomaly.csv
```

Masing-masing mempunyai `.manifest.json` berisi:

- dataset version;
- checksum SHA-256;
- provenance;
- `is_synthetic=true`;
- label/target definition;
- hasil kalibrasi;
- limitations.

### Generator individual

```powershell
python -m opticargo_ml_models.training.generator `
  --rows 8000 --seed 42 --target-accuracy 0.90

python -m opticargo_ml_models.training.forecast_generator `
  --rows 8000 --seed 43 --target-r2 0.90

python -m opticargo_ml_models.training.anomaly_generator `
  --rows 8000 --seed 44 --target-f1 0.90
```

## Train ketiga model

```powershell
python -m opticargo_ml_models.training.train_all `
  --data-dir data/synthetic `
  --artifact-dir artifacts `
  --release synthetic-baseline-v1
```

Hasil:

```text
artifacts/cargo_match_model.joblib
artifacts/cargo_match_model.metadata.json
artifacts/cargo_match_model.evaluation.json

artifacts/demand_forecast_model.joblib
artifacts/demand_forecast_model.metadata.json
artifacts/demand_forecast_model.evaluation.json

artifacts/anomaly_detector.joblib
artifacts/anomaly_detector.metadata.json
artifacts/anomaly_detector.evaluation.json
```

`train_all` keluar dengan kode `2` bila salah satu model gagal promotion guard.

### Training individual

```powershell
python -m opticargo_ml_models.training.train `
  --dataset data/synthetic/cargo_match.csv `
  --artifact artifacts/cargo_match_model.joblib `
  --model-version synthetic-baseline-v1-cargo-match

python -m opticargo_ml_models.training.forecast_train `
  --dataset data/synthetic/demand_forecast.csv `
  --artifact artifacts/demand_forecast_model.joblib `
  --model-version synthetic-baseline-v1-demand-forecast

python -m opticargo_ml_models.training.anomaly_train `
  --dataset data/synthetic/operational_anomaly.csv `
  --artifact artifacts/anomaly_detector.joblib `
  --model-version synthetic-baseline-v1-anomaly
```

## Hasil validasi baseline

| Model | Hasil utama | Promotion |
|---|---:|---|
| Cargo match | Accuracy 90.0625%, F1 83.9556%, ROC-AUC 90.6771% | promoted |
| Demand forecast | R² 92.3577%, WAPE 16.1650% | promoted |
| Operational anomaly | F1 98.8930%, ROC-AUC 99.9825% | promoted |

Anomaly sangat tinggi karena skenario sintetis relatif terpisah. Jangan menganggap angka tersebut
sebagai estimasi performa incident/fraud produksi.

Detail tersedia di `VALIDATION_REPORT.md`, `validation/baseline_metrics.json`, dan `model_cards/`.

## Jalankan test

```powershell
python -m pytest -q
python -m compileall -q src tests
```

Target saat validasi:

```text
12 passed
```

## Jalankan API

Salin environment:

```powershell
Copy-Item .env.example .env
```

Pastikan ketiga artifact sudah dibuat, lalu:

```powershell
$env:MODEL_MODE = "auto"
$env:MODEL_PRELOAD = "true"
uvicorn opticargo_ml_models.api:app --host 0.0.0.0 --port 8000
```

Dokumentasi development:

```text
http://127.0.0.1:8000/docs
```

### Cargo match request

```json
{
  "voyage": {
    "route_distance_km": 1200,
    "remaining_weight_ton": 180,
    "remaining_volume_m3": 600,
    "operating_cost_per_km_idr": 35000
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
    "commodity_compatibility": true,
    "certification_match": true,
    "temperature_match": true,
    "weather_risk": 0.15,
    "port_congestion": 0.2,
    "historical_acceptance_rate": 0.75
  }
}
```

### Demand forecast request

```json
{
  "forecast_date": "2026-08-07T00:00:00Z",
  "route_distance_km": 1200,
  "historical_volume_7d_ton": 720,
  "historical_volume_30d_ton": 2850,
  "bookings_7d": 88,
  "vessel_capacity_ton": 1400,
  "commodity_index": 1.08,
  "is_holiday": false,
  "port_congestion": 0.22,
  "weather_risk": 0.18,
  "fuel_price_index": 1.03,
  "economic_activity_index": 1.07,
  "lead_time_days": 8,
  "forecast_horizon_days": 7
}
```

### Anomaly request

```json
{
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
    "supplier_failure_rate": 0.03
  }
}
```

## Fallback

`MODEL_MODE` menerima:

```text
heuristic  -> selalu menggunakan formula fallback
trained    -> mencoba trained model; fallback bila gagal

auto       -> mencoba trained model; fallback bila artifact belum tersedia
```

Setiap model memiliki fallback terpisah. Service tetap `ready` selama fallback tersedia.

## Artifact compatibility

Scikit-learn/joblib tidak menjamin kompatibilitas lintas versi. Metadata sidecar menyimpan versi:

```text
Python
scikit-learn
NumPy
Pandas
Joblib
```

Sebelum unpickle, loader memeriksa checksum dan versi scikit-learn. Bila berbeda, artifact ditolak
dan service memakai fallback. Solusinya adalah training ulang menggunakan environment serving.

## Retraining job

Semua model:

```powershell
python -m opticargo_ml_models.jobs.retrain `
  --model all `
  --release synthetic-baseline-v2 `
  --lock artifacts/retraining.lock
```

Satu model:

```powershell
python -m opticargo_ml_models.jobs.retrain --model demand-forecast
```

Lock mencegah dua retraining berjalan bersamaan.

## Drift check

```powershell
python -m opticargo_ml_models.jobs.drift `
  --model cargo-match `
  --reference data/reference/cargo_match.csv `
  --current data/current/cargo_match.csv `
  --output artifacts/cargo_match_drift.json `
  --fail-on-alert
```

Pilihan model:

```text
cargo-match
demand-forecast
anomaly
```

Metric drift memakai Population Stability Index. Default alert threshold `0.20`.

## MLflow dan MinIO

Aktifkan dependency MLflow:

```powershell
python -m pip install -e ".[mlflow]"
```

Konfigurasi:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_PREFIX=opticargo
MINIO_ENDPOINT=minio:9000
MINIO_MODELS_BUCKET=opticargo-model-artifacts
UPLOAD_MODEL_ARTIFACT=true
```

Training mencatat parameter, metrics, artifact, dataset version, Git SHA, dan synthetic flag.

## Docker

Artifact harus dibuat sebelum build image:

```powershell
python -m opticargo_ml_models.training.train_all

docker build `
  -t ghcr.io/opticargo-ai/opticargo-ml-models:dev `
  .
```

Infra menjalankan image sebagai internal service pada port `8000` dan meneruskan endpoint MLflow,
MinIO, token internal, serta path artifact melalui environment variable.

## Batasan dan langkah produksi

Sebelum production:

- ganti synthetic dataset dengan export outcome nyata;
- tetapkan observation window dan negative sampling;
- kalibrasi forecast per route/commodity;
- kalibrasi anomaly threshold dari incident dan operator feedback;
- jalankan shadow evaluation;
- gunakan MLflow registry alias champion/candidate;
- tambah fairness/risk review dan audit trail;
- monitor drift, fallback, latency, dan conversion impact.
