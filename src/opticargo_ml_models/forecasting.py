from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import DemandForecastRequest, FeatureExplanation

FORECAST_FEATURE_COLUMNS = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "route_distance_km",
    "historical_volume_7d_ton",
    "historical_volume_30d_ton",
    "bookings_7d",
    "vessel_capacity_ton",
    "commodity_index",
    "port_congestion",
    "weather_risk",
    "fuel_price_index",
    "economic_activity_index",
    "lead_time_days",
    "forecast_horizon_days",
]


def _calendar_features(month: int, day_of_week: int) -> dict[str, float]:
    return {
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
    }


def build_forecast_feature_row(request: DemandForecastRequest) -> dict[str, float]:
    calendar = _calendar_features(request.forecast_date.month, request.forecast_date.weekday())
    return {
        **calendar,
        "is_holiday": float(request.is_holiday),
        "route_distance_km": request.route_distance_km,
        "historical_volume_7d_ton": request.historical_volume_7d_ton,
        "historical_volume_30d_ton": request.historical_volume_30d_ton,
        "bookings_7d": float(request.bookings_7d),
        "vessel_capacity_ton": request.vessel_capacity_ton,
        "commodity_index": request.commodity_index,
        "port_congestion": request.port_congestion,
        "weather_risk": request.weather_risk,
        "fuel_price_index": request.fuel_price_index,
        "economic_activity_index": request.economic_activity_index,
        "lead_time_days": request.lead_time_days,
        "forecast_horizon_days": float(request.forecast_horizon_days),
    }


def build_forecast_features_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_date",
        "is_holiday",
        "route_distance_km",
        "historical_volume_7d_ton",
        "historical_volume_30d_ton",
        "bookings_7d",
        "vessel_capacity_ton",
        "commodity_index",
        "port_congestion",
        "weather_risk",
        "fuel_price_index",
        "economic_activity_index",
        "lead_time_days",
        "forecast_horizon_days",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom forecast tidak lengkap: {sorted(missing)}")
    dates = pd.to_datetime(frame["forecast_date"], utc=True)
    month = dates.dt.month.to_numpy()
    dow = dates.dt.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_holiday": frame["is_holiday"].astype(float),
            "route_distance_km": frame["route_distance_km"].astype(float),
            "historical_volume_7d_ton": frame["historical_volume_7d_ton"].astype(float),
            "historical_volume_30d_ton": frame["historical_volume_30d_ton"].astype(float),
            "bookings_7d": frame["bookings_7d"].astype(float),
            "vessel_capacity_ton": frame["vessel_capacity_ton"].astype(float),
            "commodity_index": frame["commodity_index"].astype(float),
            "port_congestion": frame["port_congestion"].astype(float),
            "weather_risk": frame["weather_risk"].astype(float),
            "fuel_price_index": frame["fuel_price_index"].astype(float),
            "economic_activity_index": frame["economic_activity_index"].astype(float),
            "lead_time_days": frame["lead_time_days"].astype(float),
            "forecast_horizon_days": frame["forecast_horizon_days"].astype(float),
        },
        index=frame.index,
    )


def heuristic_forecast(features: Mapping[str, float]) -> float:
    weekly_average = features["historical_volume_7d_ton"] / 7.0
    monthly_average = features["historical_volume_30d_ton"] / 30.0
    daily_baseline = 0.62 * weekly_average + 0.38 * monthly_average
    seasonality = 1.0 + 0.10 * features["month_sin"] + 0.04 * features["dow_cos"]
    demand_factor = (
        features["commodity_index"]
        * features["economic_activity_index"]
        * (1.0 + 0.08 * features["is_holiday"])
    )
    operational_factor = np.clip(
        1.0
        - 0.24 * features["port_congestion"]
        - 0.18 * features["weather_risk"]
        - 0.06 * max(0.0, features["fuel_price_index"] - 1.0),
        0.35,
        1.25,
    )
    booking_signal = 0.22 * features["bookings_7d"] * 4.5
    predicted = (
        daily_baseline
        * features["forecast_horizon_days"]
        * seasonality
        * demand_factor
        * operational_factor
        + booking_signal
    )
    capacity_ceiling = features["vessel_capacity_ton"] * max(1.0, features["forecast_horizon_days"] / 7.0) * 1.5
    return float(np.clip(predicted, 0.0, capacity_ceiling))


def build_forecast_explanations(
    features: Mapping[str, float],
    prediction: float,
) -> list[FeatureExplanation]:
    weekly_daily = features["historical_volume_7d_ton"] / 7.0
    monthly_daily = features["historical_volume_30d_ton"] / 30.0
    contributions = [
        ("historical_volume_7d_ton", weekly_daily, "positive", "Sinyal volume tujuh hari terakhir."),
        ("historical_volume_30d_ton", monthly_daily, "positive", "Baseline volume tiga puluh hari."),
        ("bookings_7d", features["bookings_7d"], "positive", "Jumlah booking terbaru sebagai leading indicator."),
        ("economic_activity_index", features["economic_activity_index"] - 1.0, "positive", "Indeks aktivitas ekonomi."),
        ("commodity_index", features["commodity_index"] - 1.0, "positive", "Indeks permintaan komoditas."),
        ("port_congestion", -features["port_congestion"], "negative", "Kepadatan pelabuhan menekan demand efektif."),
        ("weather_risk", -features["weather_risk"], "negative", "Risiko cuaca menekan volume terlayani."),
    ]
    rows = [
        FeatureExplanation(
            feature=name,
            contribution=round(float(value), 6),
            direction=direction if value != 0 else "neutral",
            value=round(float(features.get(name, value)), 6),
            description=description,
        )
        for name, value, direction, description in contributions
    ]
    rows.sort(key=lambda item: abs(item.contribution), reverse=True)
    rows.append(
        FeatureExplanation(
            feature="predicted_volume_ton",
            contribution=round(prediction, 6),
            direction="positive",
            value=round(prediction, 6),
            description="Prediksi volume untuk horizon yang diminta.",
        )
    )
    return rows
