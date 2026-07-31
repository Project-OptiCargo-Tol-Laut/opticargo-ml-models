# Model Card — Operational Anomaly Detector

## Tujuan

Mendeteksi pola operasional tidak biasa pada booking, volume, harga, pembatalan, delay, utilization,
congestion, weather, dan supplier failure.

## Model

Baseline supervised `HistGradientBoostingClassifier`. Label sintetis dibentuk dari enam skenario
anomali. Desain ini dipilih agar baseline dapat dievaluasi; versi produksi dapat diganti menjadi
semi-supervised/unsupervised setelah incident label tersedia.

## Metrik validasi sintetis

- Precision: 1.0
- Recall: 0.978102
- F1: 0.988930
- ROC-AUC: 0.999825
- False-positive rate: 0.0

## Batasan

Nilai tinggi terjadi karena skenario anomali sintetis relatif terpisah. Ini bukan bukti bahwa model
akan mendeteksi fraud, incident, atau concept drift produksi dengan kualitas yang sama.
