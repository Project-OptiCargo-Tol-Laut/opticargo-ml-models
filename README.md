# opticargo-ml-models

Model machine learning terlatih (bukan LLM) yang mendukung Optimization Agent
dan Graph Analysis Agent, sesuai Bagian 8.3 dokumen desain: Cargo Scoring Model,
Demand Forecasting Model, dan Anomaly Detection Model.

> Repo ini terpisah dari `opticargo-agents` karena punya siklus hidup berbeda:
> training pipeline, dataset historis, evaluasi metrik, dan retraining berkala —
> bukan orkestrasi/pemanggilan LLM.

## Model yang Dikelola

| Model | Tipe | Dipakai oleh | Fungsi |
|---|---|---|---|
| Cargo Scoring Model | Supervised (klasifikasi/regresi) | Optimization Agent | Belajar pola pasangan kapal-kargo yang berhasil vs gagal, tingkatkan akurasi matching dari waktu ke waktu |
| Demand Forecasting Model | Time-series/regresi | Graph Analysis Agent | Prediksi volume komoditas yang akan tersedia per daerah berdasarkan musim & tren historis |
| Anomaly Detection Model | Unsupervised | Data Ingestion Agent | Deteksi ketidakwajaran data (harga abnormal, jadwal tidak konsisten) |

## Tech Stack
- Python: scikit-learn / XGBoost (scoring & forecasting), Isolation Forest atau
  sejenis (anomaly detection)
- MLflow (atau alternatif ringan) untuk experiment tracking & model registry
- Model diserving lewat endpoint HTTP ringan (FastAPI) yang dipanggil oleh
  `opticargo-agents`

## Struktur Direktori
    /training/scoring_model/         → notebook & script training
    /training/forecasting_model/
    /training/anomaly_detection/
    /models/                          → model artifact terversi (atau pointer ke model registry)
    /serving/                         → API serving untuk inference
    /evaluation/                      → skrip evaluasi & metrik

## Dependensi Repo Lain
- Data training bersumber dari `opticargo-data` (untuk MVP) dan data transaksional
  di `opticargo-gateway-api` (untuk produksi).
- Schema fitur/prediksi mengikuti `opticargo-shared`.
- Dipanggil oleh `opticargo-agents` (Optimization & Graph Analysis Agent) via HTTP call.

## Catatan MVP
Untuk MVP lomba (2-3 minggu), 3 model ini boleh disederhanakan jadi model
sederhana/rule-based dengan sedikit heuristic + threshold, karena dataset
historis nyata belum tersedia. Prioritaskan struktur pipeline (training →
artifact → serving) berjalan, akurasi model bisa ditingkatkan pasca-MVP.