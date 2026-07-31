# Artifacts

Folder ini sengaja tidak menyertakan file `joblib` bawaan. Artifact scikit-learn berbasis
pickle/joblib harus dibuat menggunakan versi Python, NumPy, Joblib, dan scikit-learn yang sama
dengan runtime serving.

Buat ketiga artifact pada environment aktif:

```powershell
python -m opticargo_ml_models.training.train_all `
  --data-dir data/synthetic `
  --artifact-dir artifacts `
  --release synthetic-baseline-v1
```

Hasil:

- `cargo_match_model.joblib`
- `demand_forecast_model.joblib`
- `anomaly_detector.joblib`
- sidecar `.metadata.json`
- laporan `.evaluation.json`

Metadata sidecar menyimpan checksum dan versi dependency. Service akan memakai heuristic fallback
bila artifact hilang, rusak, atau tidak kompatibel.
