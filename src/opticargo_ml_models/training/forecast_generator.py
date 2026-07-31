from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..forecasting import FORECAST_FEATURE_COLUMNS, build_forecast_features_from_dataframe
from .common import write_manifest


@dataclass(frozen=True)
class ForecastCalibrationResult:
    target_r2: float
    achieved_r2: float
    achieved_mae: float
    achieved_rmse: float
    noise_ratio: float
    train_rows: int
    validation_rows: int


def _stable_uuid(kind: str, index: int, seed: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"opticargo:forecast:{kind}:{seed}:{index}"))


def _base_frame(rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_time = datetime(2023, 1, 1, tzinfo=UTC)
    route_count = max(30, min(150, rows // 80))
    route_ids = np.arange(rows) % route_count
    forecast_dates = np.array(
        [base_time + timedelta(hours=6 * i + int(rng.integers(0, 6))) for i in range(rows)]
    )
    months = np.array([value.month for value in forecast_dates])
    day_of_week = np.array([value.weekday() for value in forecast_dates])

    route_distance = rng.uniform(120, 3_000, rows)
    vessel_capacity = rng.uniform(120, 2_500, rows)
    commodity_index = np.clip(rng.lognormal(0.0, 0.16, rows), 0.55, 1.65)
    economic_index = np.clip(rng.normal(1.0, 0.10, rows), 0.70, 1.35)
    fuel_index = np.clip(rng.normal(1.0, 0.15, rows), 0.60, 1.60)
    congestion = np.clip(rng.beta(2.2, 4.5, rows), 0, 1)
    weather = np.clip(rng.beta(1.8, 6.0, rows), 0, 1)
    holiday = ((day_of_week >= 5) | (rng.random(rows) < 0.05)).astype(int)
    lead_time = np.clip(rng.gamma(2.0, 4.0, rows), 0, 45)
    horizon = rng.choice(np.array([1, 3, 7, 14, 30]), size=rows, p=[0.08, 0.12, 0.52, 0.18, 0.10])

    route_base = 35 + route_ids * (240.0 / max(route_count - 1, 1))
    annual = 1.0 + 0.16 * np.sin(2 * np.pi * months / 12.0) + 0.07 * np.cos(4 * np.pi * months / 12.0)
    weekly = 1.0 + 0.05 * np.cos(2 * np.pi * day_of_week / 7.0)
    operational = np.clip(1.0 - 0.22 * congestion - 0.15 * weather, 0.45, 1.10)
    underlying_daily = (
        route_base
        * annual
        * weekly
        * commodity_index
        * economic_index
        * operational
        * (1.0 + 0.06 * holiday)
    )
    underlying_daily = np.minimum(underlying_daily, vessel_capacity * 0.72)

    historical_30 = np.clip(underlying_daily * 30 * rng.normal(1.0, 0.055, rows), 1, None)
    recent_shock = rng.normal(1.0, 0.075, rows)
    historical_7 = np.clip(underlying_daily * 7 * recent_shock, 1, None)
    average_shipment = rng.uniform(4.0, 14.0, rows)
    bookings_7 = np.maximum(0, np.rint(historical_7 / average_shipment + rng.normal(0, 2.5, rows))).astype(int)

    clean_target = (
        (0.58 * historical_7 / 7.0 + 0.42 * historical_30 / 30.0)
        * horizon
        * (0.82 + 0.18 * commodity_index)
        * (0.85 + 0.15 * economic_index)
        * (1.0 + 0.04 * holiday)
        * np.clip(1.0 - 0.12 * congestion - 0.09 * weather, 0.62, 1.10)
        + bookings_7 * 0.45
        + np.sqrt(route_distance) * 0.9
        - np.maximum(fuel_index - 1.0, 0) * 12.0
        + np.minimum(lead_time, 30) * 0.35
    )
    capacity_ceiling = vessel_capacity * np.maximum(1.0, horizon / 7.0) * 1.6
    clean_target = np.clip(clean_target, 0, capacity_ceiling)

    frame = pd.DataFrame(
        {
            "record_id": [_stable_uuid("record", i, seed) for i in range(rows)],
            "route_id": [_stable_uuid("route", int(route_ids[i]), seed) for i in range(rows)],
            "observed_at": [value.isoformat() for value in forecast_dates],
            "forecast_date": [(value + timedelta(days=int(horizon[i]))).isoformat() for i, value in enumerate(forecast_dates)],
            "route_distance_km": route_distance.round(3),
            "historical_volume_7d_ton": historical_7.round(3),
            "historical_volume_30d_ton": historical_30.round(3),
            "bookings_7d": bookings_7,
            "vessel_capacity_ton": vessel_capacity.round(3),
            "commodity_index": commodity_index.round(6),
            "is_holiday": holiday,
            "port_congestion": congestion.round(6),
            "weather_risk": weather.round(6),
            "fuel_price_index": fuel_index.round(6),
            "economic_activity_index": economic_index.round(6),
            "lead_time_days": lead_time.round(3),
            "forecast_horizon_days": horizon,
            "is_synthetic": True,
            "provenance": "opticargo-synthetic-demand-forecast-generator-v1",
        }
    )
    return frame, clean_target


def _temporal_split(frame: pd.DataFrame, features: pd.DataFrame, target: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], target[train_idx], target[valid_idx]


def _probe(frame: pd.DataFrame, target: np.ndarray, seed: int) -> tuple[float, float, float, int, int]:
    features = build_forecast_features_from_dataframe(frame)[FORECAST_FEATURE_COLUMNS]
    x_train, x_valid, y_train, y_valid = _temporal_split(frame, features, target)
    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.07,
        max_leaf_nodes=23,
        min_samples_leaf=24,
        l2_regularization=1.2,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    prediction = np.maximum(0.0, model.predict(x_valid))
    return (
        float(r2_score(y_valid, prediction)),
        float(mean_absolute_error(y_valid, prediction)),
        float(np.sqrt(mean_squared_error(y_valid, prediction))),
        len(x_train),
        len(x_valid),
    )


def generate_calibrated_forecast_dataset(
    rows: int = 10_000,
    seed: int = 43,
    target_r2: float = 0.90,
    tolerance: float = 0.025,
) -> tuple[pd.DataFrame, ForecastCalibrationResult]:
    if rows < 1_500:
        raise ValueError("Kalibrasi forecast membutuhkan minimal 1.500 baris.")
    if not 0.75 <= target_r2 <= 0.98:
        raise ValueError("target R2 harus berada pada rentang 0.75..0.98")
    frame, clean_target = _base_frame(rows, seed)
    rng = np.random.default_rng(seed + 20260730)
    base_noise = rng.normal(0.0, 1.0, rows)
    candidates = np.linspace(0.02, 0.24, 12)
    best = None
    for noise_ratio in candidates:
        target = np.clip(clean_target * (1.0 + base_noise * noise_ratio), 0, None)
        r2, mae, rmse, train_rows, valid_rows = _probe(frame, target, seed)
        candidate = (abs(r2 - target_r2), r2, mae, rmse, float(noise_ratio), train_rows, valid_rows, target)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if abs(r2 - target_r2) <= tolerance:
            break
    assert best is not None
    _, r2, mae, rmse, noise_ratio, train_rows, valid_rows, target = best
    frame["demand_volume_ton"] = np.round(target, 3)
    frame["target_definition_version"] = "route-demand-volume-v1"
    frame["dataset_version"] = f"synthetic-demand-forecast-{datetime.now(UTC).date().isoformat()}-seed{seed}"
    calibration = ForecastCalibrationResult(
        target_r2=target_r2,
        achieved_r2=r2,
        achieved_mae=mae,
        achieved_rmse=rmse,
        noise_ratio=noise_ratio,
        train_rows=train_rows,
        validation_rows=valid_rows,
    )
    return frame, calibration


def write_dataset(frame: pd.DataFrame, calibration: ForecastCalibrationResult, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return write_manifest(
        output,
        {
            "dataset_name": "opticargo-synthetic-demand-forecast",
            "dataset_version": str(frame["dataset_version"].iloc[0]),
            "is_synthetic": True,
            "provenance": "opticargo-synthetic-demand-forecast-generator-v1",
            "rows": len(frame),
            "target_column": "demand_volume_ton",
            "target_definition_version": "route-demand-volume-v1",
            "calibration": asdict(calibration),
            "limitations": [
                "R2 hanya mengukur pola sintetis, bukan akurasi demand operasional nyata.",
                "Kalibrasi ulang wajib dilakukan ketika outcome route dan booking tersedia.",
                "Data sintetis dan nyata harus tetap dipisahkan dengan provenance.",
            ],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset demand forecasting sintetis.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--target-r2", type=float, default=0.90)
    parser.add_argument("--tolerance", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/demand_forecast.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame, calibration = generate_calibrated_forecast_dataset(
        rows=args.rows,
        seed=args.seed,
        target_r2=args.target_r2,
        tolerance=args.tolerance,
    )
    outputs = write_dataset(frame, calibration, args.output)
    print(json.dumps({**outputs, "calibration": asdict(calibration)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
