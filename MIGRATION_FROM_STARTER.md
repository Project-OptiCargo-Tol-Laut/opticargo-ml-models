# Migrasi dari starter 0.1.0

1. Salin source versi 1.0.0 ke repository `opticargo-ml-models`.
2. Pertahankan `.git`, `.env`, dan konfigurasi rahasia lokal.
3. Hapus artifact cargo lama yang dibuat oleh versi scikit-learn berbeda.
4. Aktifkan Conda environment `opticargo-ml-models`.
5. Instal ulang package.
6. Generate dan train seluruh model.

```powershell
conda activate opticargo-ml-models
python -m pip install -e ".[dev]"
python -m opticargo_ml_models.training.generate_all --rows 8000
python -m opticargo_ml_models.training.train_all --release synthetic-baseline-v1
python -m pytest -q
```

File artifact yang harus muncul:

```text
artifacts/cargo_match_model.joblib
artifacts/demand_forecast_model.joblib
artifacts/anomaly_detector.joblib
```
