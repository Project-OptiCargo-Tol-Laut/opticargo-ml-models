# Validation Report — OptiCargo ML Models 1.0.0

## Cakupan

Repository telah divalidasi untuk tiga model:

1. Cargo match scoring.
2. Demand forecasting.
3. Operational anomaly detection.

## Hasil

| Model | Promotion | Metrik utama |
|---|---|---|
| Cargo match | promoted | accuracy 90.0625%, F1 83.9556%, ROC-AUC 90.6771% |
| Demand forecast | promoted | R² 92.3577%, WAPE 16.1650%, interval coverage 94.5% |
| Operational anomaly | promoted | F1 98.8930%, ROC-AUC 99.9825% |

Hard-constraint violation cargo match adalah `0.0`. Seluruh dataset memiliki
`is_synthetic=true`, provenance, dataset version, dan target/label definition version.

## Pemeriksaan teknis

- `python -m pytest -q`: 12 passed.
- `python -m compileall -q src tests`: lulus.
- Generate-all 8.000 baris per model: lulus.
- Train-all: ketiga model promoted.
- Artifact loading dan checksum sidecar: diuji pada environment validasi.
- API heuristic fallback untuk tiga endpoint: diuji.

Ruff tidak tersedia dalam environment pembuatan artifact ini. Workflow CI tetap menjalankan Ruff
pada Python 3.11 setelah dependency development diinstal.

## Catatan portabilitas artifact

Joblib/pickle scikit-learn tidak portable lintas versi. Karena environment pengguna menggunakan
scikit-learn 1.9.0 sedangkan environment validasi memakai 1.8.0, paket final tidak menyertakan
artifact `.joblib`. Jalankan `train_all` pada environment pengguna agar ketiga artifact kompatibel
dengan runtime lokal.
