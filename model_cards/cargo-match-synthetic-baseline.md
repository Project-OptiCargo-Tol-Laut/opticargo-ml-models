# Model Card — Cargo Match Scorer

## Tujuan

Memberikan skor kecocokan satu kandidat cargo terhadap voyage setelah hard constraint diterapkan.

## Model

`HistGradientBoostingClassifier` dengan feature domain ekonomi, jadwal, kapasitas, kualitas supplier,
jarak, risiko, historical acceptance, dan hard-constraint validity.

## Target sintetis

`match_label=1` merepresentasikan rekomendasi yang diperkirakan diterima berdasarkan teacher
heuristic, kemudian diberi label noise terkalibrasi.

## Metrik validasi sintetis

- Accuracy: 0.900625
- F1: 0.839556
- ROC-AUC: 0.906771
- Hard-constraint violation: 0.0

## Batasan

Metrik hanya berlaku terhadap pola generator sintetis. Model harus dilatih ulang menggunakan
outcome recommendation, booking, cancellation, dispute, dan completion nyata sebelum dipakai
untuk keputusan operasional bernilai tinggi.
