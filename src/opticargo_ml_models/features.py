from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import CargoMatchRequest, FeatureExplanation

MODEL_FEATURE_COLUMNS = [
    "economic_score",
    "schedule_fit_score",
    "capacity_fit_score",
    "supplier_quality_score",
    "distance_efficiency_score",
    "risk_safety_score",
    "historical_acceptance_rate",
    "hard_constraint_valid",
]


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))


def build_feature_row(request: CargoMatchRequest) -> dict[str, float]:
    voyage = request.voyage
    cargo = request.candidate

    weight_ok = cargo.cargo_weight_ton <= voyage.remaining_weight_ton
    volume_ok = cargo.cargo_volume_m3 <= voyage.remaining_volume_m3
    time_ok = 0 <= cargo.schedule_gap_hours <= 168
    hard_constraint_valid = all(
        [
            weight_ok,
            volume_ok,
            time_ok,
            cargo.commodity_compatibility,
            cargo.certification_match,
            cargo.temperature_match,
        ]
    )

    estimated_margin_idr = (
        (cargo.market_rate_per_ton_idr - cargo.asking_price_per_ton_idr)
        * cargo.cargo_weight_ton
        - (cargo.origin_distance_km + cargo.destination_distance_km)
        * voyage.operating_cost_per_km_idr
    )
    economic_score = float(_sigmoid(estimated_margin_idr / 80_000_000.0))
    schedule_fit_score = (
        float(np.exp(-abs(cargo.schedule_gap_hours - 24.0) / 80.0)) if time_ok else 0.0
    )
    utilization = min(
        cargo.cargo_weight_ton / voyage.remaining_weight_ton,
        cargo.cargo_volume_m3 / voyage.remaining_volume_m3,
    )
    capacity_fit_score = float(np.clip(utilization / 0.70, 0.0, 1.0)) if weight_ok and volume_ok else 0.0
    supplier_quality_score = float(
        np.clip(
            0.45 * ((cargo.supplier_rating - 1.0) / 4.0)
            + 0.40 * cargo.supplier_success_rate
            + 0.15 * (1.0 - cargo.supplier_cancellation_rate),
            0.0,
            1.0,
        )
    )
    distance_efficiency_score = float(
        np.exp(
            -(cargo.origin_distance_km + cargo.destination_distance_km)
            / (voyage.route_distance_km * 0.18 + 50.0)
        )
    )
    risk_safety_score = float(
        np.clip(1.0 - (0.55 * cargo.weather_risk + 0.45 * cargo.port_congestion), 0.0, 1.0)
    )

    return {
        "economic_score": economic_score,
        "schedule_fit_score": schedule_fit_score,
        "capacity_fit_score": capacity_fit_score,
        "supplier_quality_score": supplier_quality_score,
        "distance_efficiency_score": distance_efficiency_score,
        "risk_safety_score": risk_safety_score,
        "historical_acceptance_rate": cargo.historical_acceptance_rate,
        "hard_constraint_valid": float(hard_constraint_valid),
        "estimated_margin_idr": estimated_margin_idr,
        "weight_capacity_ok": float(weight_ok),
        "volume_capacity_ok": float(volume_ok),
        "route_time_valid": float(time_ok),
    }


def build_features_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "route_distance_km",
        "remaining_weight_ton",
        "remaining_volume_m3",
        "operating_cost_per_km_idr",
        "cargo_weight_ton",
        "cargo_volume_m3",
        "asking_price_per_ton_idr",
        "market_rate_per_ton_idr",
        "origin_distance_km",
        "destination_distance_km",
        "schedule_gap_hours",
        "supplier_rating",
        "supplier_success_rate",
        "supplier_cancellation_rate",
        "commodity_compatibility",
        "certification_match",
        "temperature_match",
        "weather_risk",
        "port_congestion",
        "historical_acceptance_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")

    weight_ok = frame["cargo_weight_ton"] <= frame["remaining_weight_ton"]
    volume_ok = frame["cargo_volume_m3"] <= frame["remaining_volume_m3"]
    time_ok = frame["schedule_gap_hours"].between(0, 168)
    hard = (
        weight_ok
        & volume_ok
        & time_ok
        & frame["commodity_compatibility"].astype(bool)
        & frame["certification_match"].astype(bool)
        & frame["temperature_match"].astype(bool)
    )

    margin = (
        (frame["market_rate_per_ton_idr"] - frame["asking_price_per_ton_idr"])
        * frame["cargo_weight_ton"]
        - (frame["origin_distance_km"] + frame["destination_distance_km"])
        * frame["operating_cost_per_km_idr"]
    )
    economic = _sigmoid(margin.to_numpy() / 80_000_000.0)
    schedule = np.where(
        time_ok,
        np.exp(-np.abs(frame["schedule_gap_hours"].to_numpy() - 24.0) / 80.0),
        0.0,
    )
    utilization = np.minimum(
        frame["cargo_weight_ton"] / frame["remaining_weight_ton"],
        frame["cargo_volume_m3"] / frame["remaining_volume_m3"],
    )
    capacity = np.where(weight_ok & volume_ok, np.clip(utilization / 0.70, 0.0, 1.0), 0.0)
    supplier = np.clip(
        0.45 * ((frame["supplier_rating"] - 1.0) / 4.0)
        + 0.40 * frame["supplier_success_rate"]
        + 0.15 * (1.0 - frame["supplier_cancellation_rate"]),
        0.0,
        1.0,
    )
    distance = np.exp(
        -(frame["origin_distance_km"] + frame["destination_distance_km"])
        / (frame["route_distance_km"] * 0.18 + 50.0)
    )
    risk = np.clip(1.0 - (0.55 * frame["weather_risk"] + 0.45 * frame["port_congestion"]), 0.0, 1.0)

    return pd.DataFrame(
        {
            "economic_score": economic,
            "schedule_fit_score": schedule,
            "capacity_fit_score": capacity,
            "supplier_quality_score": supplier,
            "distance_efficiency_score": distance,
            "risk_safety_score": risk,
            "historical_acceptance_rate": frame["historical_acceptance_rate"].astype(float),
            "hard_constraint_valid": hard.astype(float),
        },
        index=frame.index,
    )


def heuristic_score(features: Mapping[str, float]) -> float:
    score = (
        0.28 * features["economic_score"]
        + 0.20 * features["schedule_fit_score"]
        + 0.18 * features["capacity_fit_score"]
        + 0.12 * features["supplier_quality_score"]
        + 0.10 * features["distance_efficiency_score"]
        + 0.07 * features["risk_safety_score"]
        + 0.05 * features["historical_acceptance_rate"]
    )
    if not bool(features["hard_constraint_valid"]):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def build_explanations(features: Mapping[str, float], score: float) -> list[FeatureExplanation]:
    weights = {
        "economic_score": 0.28,
        "schedule_fit_score": 0.20,
        "capacity_fit_score": 0.18,
        "supplier_quality_score": 0.12,
        "distance_efficiency_score": 0.10,
        "risk_safety_score": 0.07,
        "historical_acceptance_rate": 0.05,
    }
    descriptions = {
        "economic_score": "Estimasi margin setelah harga muatan dan biaya deviasi jarak.",
        "schedule_fit_score": "Kesesuaian jendela ketersediaan dengan keberangkatan voyage.",
        "capacity_fit_score": "Kontribusi muatan terhadap pemanfaatan kapasitas tersisa.",
        "supplier_quality_score": "Kualitas pemasok berdasarkan rating, keberhasilan, dan pembatalan.",
        "distance_efficiency_score": "Efisiensi jarak pengambilan dan pengantaran terhadap panjang rute.",
        "risk_safety_score": "Skor risiko cuaca dan kepadatan pelabuhan.",
        "historical_acceptance_rate": "Riwayat penerimaan rekomendasi serupa.",
    }
    if not bool(features["hard_constraint_valid"]):
        return [
            FeatureExplanation(
                feature="hard_constraint_valid",
                contribution=-1.0,
                direction="negative",
                value=False,
                description="Kandidat melanggar kapasitas, waktu, kompatibilitas, sertifikasi, atau suhu.",
            )
        ]

    rows = []
    for name, weight in weights.items():
        contribution = weight * float(features[name])
        rows.append(
            FeatureExplanation(
                feature=name,
                contribution=round(contribution, 6),
                direction="positive" if contribution >= 0 else "negative",
                value=round(float(features[name]), 6),
                description=descriptions[name],
            )
        )
    rows.sort(key=lambda item: abs(item.contribution), reverse=True)
    rows.append(
        FeatureExplanation(
            feature="final_score",
            contribution=round(score, 6),
            direction="positive" if score >= 0.5 else "neutral",
            value=round(score, 6),
            description="Skor akhir setelah hard constraint diterapkan.",
        )
    )
    return rows
