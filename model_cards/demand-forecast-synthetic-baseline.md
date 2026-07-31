# Model Card — Demand Forecaster

## Tujuan

Memperkirakan volume cargo per route dan horizon 1–90 hari.

## Model

`HistGradientBoostingRegressor` menggunakan histori volume, booking, kapasitas vessel, kalender,
commodity index, economic activity, fuel price, congestion, weather, route distance, dan lead time.

## Metrik validasi sintetis

- R²: 0.923577
- MAE: 202.3868 ton
- RMSE: 360.5118 ton
- WAPE: 0.161650
- 95% interval coverage: 0.945

## Batasan

Interval prediksi berasal dari residual validation sintetis, bukan interval probabilistik yang telah
dikalibrasi pada data produksi. Seasonality dan route mix harus dikalibrasi ulang dengan data nyata.
