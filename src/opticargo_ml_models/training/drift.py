from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..anomalies import ANOMALY_FEATURE_COLUMNS, build_anomaly_features_from_dataframe
from ..features import MODEL_FEATURE_COLUMNS, build_features_from_dataframe
from ..forecasting import FORECAST_FEATURE_COLUMNS, build_forecast_features_from_dataframe

ModelKind = Literal["cargo-match", "demand-forecast", "anomaly"]


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return 0.0
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    ref_hist, _ = np.histogram(reference, bins=quantiles)
    cur_hist, _ = np.histogram(current, bins=quantiles)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_frame(frame: pd.DataFrame, model: ModelKind) -> tuple[pd.DataFrame, list[str]]:
    if model == "cargo-match":
        return build_features_from_dataframe(frame), MODEL_FEATURE_COLUMNS
    if model == "demand-forecast":
        return build_forecast_features_from_dataframe(frame), FORECAST_FEATURE_COLUMNS
    if model == "anomaly":
        return build_anomaly_features_from_dataframe(frame), ANOMALY_FEATURE_COLUMNS
    raise ValueError(f"Model tidak dikenal: {model}")


def compare(reference: pd.DataFrame, current: pd.DataFrame, model: ModelKind = "cargo-match") -> dict[str, float]:
    ref_features, columns = feature_frame(reference, model)
    cur_features, _ = feature_frame(current, model)
    return {
        column: population_stability_index(
            ref_features[column].to_numpy(), cur_features[column].to_numpy()
        )
        for column in columns
    }
