from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

from .contracts import AnomalyDetectionRequest, FeatureExplanation

ANOMALY_FEATURE_COLUMNS = [
    "booking_count",
    "cargo_volume_ton",
    "average_price_per_ton_idr",
    "cancellation_rate",
    "average_delay_hours",
    "utilization_rate",
    "port_congestion",
    "weather_risk",
    "supplier_failure_rate",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _time_features(value: datetime) -> dict[str, float]:
    return {
        "hour_sin": float(np.sin(2 * np.pi * value.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * value.hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * value.weekday() / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * value.weekday() / 7.0)),
    }


def build_anomaly_feature_row(request: AnomalyDetectionRequest) -> dict[str, float]:
    snapshot = request.snapshot
    return {
        "booking_count": snapshot.booking_count,
        "cargo_volume_ton": snapshot.cargo_volume_ton,
        "average_price_per_ton_idr": snapshot.average_price_per_ton_idr,
        "cancellation_rate": snapshot.cancellation_rate,
        "average_delay_hours": snapshot.average_delay_hours,
        "utilization_rate": snapshot.utilization_rate,
        "port_congestion": snapshot.port_congestion,
        "weather_risk": snapshot.weather_risk,
        "supplier_failure_rate": snapshot.supplier_failure_rate,
        **_time_features(snapshot.observed_at),
    }


def build_anomaly_features_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observed_at",
        "booking_count",
        "cargo_volume_ton",
        "average_price_per_ton_idr",
        "cancellation_rate",
        "average_delay_hours",
        "utilization_rate",
        "port_congestion",
        "weather_risk",
        "supplier_failure_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom anomaly tidak lengkap: {sorted(missing)}")
    observed = pd.to_datetime(frame["observed_at"], utc=True)
    hour = observed.dt.hour.to_numpy()
    dow = observed.dt.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "booking_count": frame["booking_count"].astype(float),
            "cargo_volume_ton": frame["cargo_volume_ton"].astype(float),
            "average_price_per_ton_idr": frame["average_price_per_ton_idr"].astype(float),
            "cancellation_rate": frame["cancellation_rate"].astype(float),
            "average_delay_hours": frame["average_delay_hours"].astype(float),
            "utilization_rate": frame["utilization_rate"].astype(float),
            "port_congestion": frame["port_congestion"].astype(float),
            "weather_risk": frame["weather_risk"].astype(float),
            "supplier_failure_rate": frame["supplier_failure_rate"].astype(float),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=frame.index,
    )


def heuristic_anomaly(
    features: Mapping[str, float],
) -> tuple[float, list[tuple[str, float]]]:
    rules = {
        "cancellation_rate": max(0.0, (features["cancellation_rate"] - 0.18) / 0.35),
        "average_delay_hours": max(0.0, (features["average_delay_hours"] - 24.0) / 72.0),
        "utilization_rate": max(0.0, (features["utilization_rate"] - 1.0) / 0.6),
        "port_congestion": max(0.0, (features["port_congestion"] - 0.75) / 0.25),
        "weather_risk": max(0.0, (features["weather_risk"] - 0.70) / 0.30),
        "supplier_failure_rate": max(0.0, (features["supplier_failure_rate"] - 0.15) / 0.35),
    }
    price = features["average_price_per_ton_idr"]
    if price > 4_500_000:
        rules["average_price_per_ton_idr"] = min(1.5, (price - 4_500_000) / 4_000_000)
    if features["cargo_volume_ton"] > 5_000:
        rules["cargo_volume_ton"] = min(1.5, (features["cargo_volume_ton"] - 5_000) / 8_000)
    ranked = sorted(rules.items(), key=lambda item: item[1], reverse=True)
    top = [value for _, value in ranked[:3]]
    score = float(np.clip(1.0 - np.exp(-sum(top)), 0.0, 1.0))
    return score, ranked


def anomaly_severity(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "medium"
    if score >= 0.25:
        return "low"
    return "normal"


def build_anomaly_explanations(
    features: Mapping[str, float],
    metadata: Mapping[str, object],
    score: float,
) -> list[FeatureExplanation]:
    baselines = metadata.get("feature_baselines", {}) if metadata else {}
    rows: list[FeatureExplanation] = []
    if isinstance(baselines, dict) and baselines:
        deviations: list[tuple[str, float]] = []
        for name in ANOMALY_FEATURE_COLUMNS[:9]:
            baseline = baselines.get(name, {})
            if not isinstance(baseline, dict):
                continue
            mean = float(baseline.get("mean", 0.0))
            std = max(float(baseline.get("std", 1.0)), 1e-9)
            deviations.append((name, (float(features[name]) - mean) / std))
        for name, z_score in sorted(deviations, key=lambda item: abs(item[1]), reverse=True)[:5]:
            rows.append(
                FeatureExplanation(
                    feature=name,
                    contribution=round(abs(z_score), 6),
                    direction="positive" if z_score > 0 else "negative",
                    value=round(float(features[name]), 6),
                    description=f"Deviasi {z_score:.2f} standar deviasi dari baseline training.",
                )
            )
    if not rows:
        _, ranked = heuristic_anomaly(features)
        for name, value in ranked[:5]:
            rows.append(
                FeatureExplanation(
                    feature=name,
                    contribution=round(float(value), 6),
                    direction="positive",
                    value=round(float(features[name]), 6),
                    description="Kontribusi rule-based terhadap indikasi anomali.",
                )
            )
    rows.append(
        FeatureExplanation(
            feature="anomaly_score",
            contribution=round(score, 6),
            direction="positive" if score >= 0.5 else "neutral",
            value=round(score, 6),
            description="Skor anomali terkalibrasi pada rentang 0 sampai 1.",
        )
    )
    return rows
