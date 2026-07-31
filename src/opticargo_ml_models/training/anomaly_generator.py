from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ..anomalies import ANOMALY_FEATURE_COLUMNS, build_anomaly_features_from_dataframe
from .common import write_manifest


@dataclass(frozen=True)
class AnomalyCalibrationResult:
    target_f1: float
    achieved_f1: float
    achieved_precision: float
    achieved_recall: float
    achieved_roc_auc: float
    anomaly_rate: float
    severity_multiplier: float
    train_rows: int
    validation_rows: int


def _stable_uuid(kind: str, index: int, seed: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"opticargo:anomaly:{kind}:{seed}:{index}"))


def _base_frame(rows: int, seed: int, anomaly_rate: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    route_count = max(30, min(120, rows // 80))
    route_ids = np.arange(rows) % route_count
    observed = np.array([base_time + timedelta(hours=3 * i) for i in range(rows)])
    hours = np.array([value.hour for value in observed])
    weekdays = np.array([value.weekday() for value in observed])
    route_scale = 0.75 + route_ids / max(route_count - 1, 1) * 1.05
    daily = 1.0 + 0.12 * np.sin(2 * np.pi * hours / 24.0)
    weekly = 1.0 + 0.06 * np.cos(2 * np.pi * weekdays / 7.0)

    booking_count = np.clip(rng.normal(42 * route_scale * daily, 7.0, rows), 1, None)
    cargo_volume = np.clip(booking_count * rng.normal(8.5, 0.8, rows), 1, None)
    price = np.clip(rng.normal(2_100_000 + route_scale * 350_000, 190_000, rows), 500_000, None)
    cancellation = np.clip(rng.beta(1.5, 20, rows), 0, 0.35)
    delay = np.clip(rng.gamma(1.6, 3.8, rows), 0, 36)
    utilization = np.clip(rng.normal(0.68 * weekly, 0.11, rows), 0.18, 1.05)
    congestion = np.clip(rng.beta(2.0, 5.0, rows), 0, 0.90)
    weather = np.clip(rng.beta(1.6, 7.5, rows), 0, 0.85)
    supplier_failure = np.clip(rng.beta(1.2, 22, rows), 0, 0.28)

    anomaly_count = int(round(rows * anomaly_rate))
    anomaly_idx = rng.choice(rows, anomaly_count, replace=False)
    labels = np.zeros(rows, dtype=int)
    labels[anomaly_idx] = 1
    anomaly_type = rng.integers(0, 6, anomaly_count)
    base_shift = rng.uniform(0.85, 1.15, anomaly_count)

    frame = pd.DataFrame(
        {
            "record_id": [_stable_uuid("record", i, seed) for i in range(rows)],
            "route_id": [_stable_uuid("route", int(route_ids[i]), seed) for i in range(rows)],
            "observed_at": [value.isoformat() for value in observed],
            "booking_count": booking_count,
            "cargo_volume_ton": cargo_volume,
            "average_price_per_ton_idr": price,
            "cancellation_rate": cancellation,
            "average_delay_hours": delay,
            "utilization_rate": utilization,
            "port_congestion": congestion,
            "weather_risk": weather,
            "supplier_failure_rate": supplier_failure,
            "is_synthetic": True,
            "provenance": "opticargo-synthetic-operational-anomaly-generator-v1",
        }
    )
    return frame, labels, np.column_stack((anomaly_idx, anomaly_type, base_shift))


def _inject_anomalies(frame: pd.DataFrame, anomaly_spec: np.ndarray, severity: float) -> pd.DataFrame:
    result = frame.copy()
    for row in anomaly_spec:
        index, anomaly_type, base_shift = int(row[0]), int(row[1]), float(row[2])
        s = severity * base_shift
        if anomaly_type == 0:
            result.loc[index, "average_price_per_ton_idr"] *= 1.0 + 0.65 * s
        elif anomaly_type == 1:
            result.loc[index, "cancellation_rate"] = min(1.0, 0.18 + 0.12 * s)
            result.loc[index, "supplier_failure_rate"] = min(1.0, 0.12 + 0.10 * s)
        elif anomaly_type == 2:
            result.loc[index, "average_delay_hours"] += 18.0 * s
            result.loc[index, "port_congestion"] = min(1.0, 0.60 + 0.09 * s)
        elif anomaly_type == 3:
            result.loc[index, "cargo_volume_ton"] *= 1.0 + 0.70 * s
            result.loc[index, "utilization_rate"] = min(2.0, 0.95 + 0.15 * s)
        elif anomaly_type == 4:
            result.loc[index, "booking_count"] *= 1.0 + 0.75 * s
            result.loc[index, "cargo_volume_ton"] *= 0.55
        else:
            result.loc[index, "weather_risk"] = min(1.0, 0.55 + 0.10 * s)
            result.loc[index, "average_delay_hours"] += 10.0 * s
            result.loc[index, "utilization_rate"] *= max(0.1, 1.0 - 0.12 * s)
    return result


def _temporal_split(frame: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], labels[train_idx], labels[valid_idx]


def _probe(frame: pd.DataFrame, labels: np.ndarray, seed: int, anomaly_rate: float):
    features = build_anomaly_features_from_dataframe(frame)[ANOMALY_FEATURE_COLUMNS]
    x_train, x_valid, y_train, y_valid = _temporal_split(frame, features, labels)
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.065,
        max_leaf_nodes=19,
        min_samples_leaf=24,
        l2_regularization=1.2,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    return (
        float(f1_score(y_valid, prediction, zero_division=0)),
        float(precision_score(y_valid, prediction, zero_division=0)),
        float(recall_score(y_valid, prediction, zero_division=0)),
        float(roc_auc_score(y_valid, probabilities)),
        len(x_train),
        len(x_valid),
    )


def generate_calibrated_anomaly_dataset(
    rows: int = 10_000,
    seed: int = 44,
    target_f1: float = 0.90,
    tolerance: float = 0.06,
    anomaly_rate: float = 0.08,
) -> tuple[pd.DataFrame, AnomalyCalibrationResult]:
    if rows < 2_000:
        raise ValueError("Kalibrasi anomaly membutuhkan minimal 2.000 baris.")
    if not 0.70 <= target_f1 <= 0.98:
        raise ValueError("target F1 anomaly harus berada pada rentang 0.70..0.98")
    base, labels, spec = _base_frame(rows, seed, anomaly_rate)
    best = None
    for severity in np.linspace(1.4, 5.5, 10):
        candidate_frame = _inject_anomalies(base, spec, float(severity))
        f1, precision, recall, auc, train_rows, valid_rows = _probe(
            candidate_frame, labels, seed, anomaly_rate
        )
        candidate = (
            abs(f1 - target_f1),
            f1,
            precision,
            recall,
            auc,
            float(severity),
            train_rows,
            valid_rows,
            candidate_frame,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
        if abs(f1 - target_f1) <= tolerance and auc >= 0.90:
            break
    assert best is not None
    _, f1, precision, recall, auc, severity, train_rows, valid_rows, frame = best
    frame["anomaly_label"] = labels
    frame["label_definition_version"] = "operational-anomaly-v1"
    frame["dataset_version"] = f"synthetic-operational-anomaly-{datetime.now(UTC).date().isoformat()}-seed{seed}"
    calibration = AnomalyCalibrationResult(
        target_f1=target_f1,
        achieved_f1=f1,
        achieved_precision=precision,
        achieved_recall=recall,
        achieved_roc_auc=auc,
        anomaly_rate=anomaly_rate,
        severity_multiplier=severity,
        train_rows=train_rows,
        validation_rows=valid_rows,
    )
    return frame, calibration


def write_dataset(frame: pd.DataFrame, calibration: AnomalyCalibrationResult, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return write_manifest(
        output,
        {
            "dataset_name": "opticargo-synthetic-operational-anomaly",
            "dataset_version": str(frame["dataset_version"].iloc[0]),
            "is_synthetic": True,
            "provenance": "opticargo-synthetic-operational-anomaly-generator-v1",
            "rows": len(frame),
            "label_column": "anomaly_label",
            "label_definition_version": "operational-anomaly-v1",
            "calibration": asdict(calibration),
            "limitations": [
                "Anomali dibentuk dari skenario sintetis dan belum merepresentasikan seluruh fraud/incident produksi.",
                "Threshold wajib dikalibrasi ulang dengan incident dan feedback operasional nyata.",
                "Label sintetis tidak boleh dicampur dengan label aktual tanpa provenance.",
            ],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset anomaly operasional sintetis.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/operational_anomaly.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame, calibration = generate_calibrated_anomaly_dataset(
        rows=args.rows,
        seed=args.seed,
        target_f1=args.target_f1,
        tolerance=args.tolerance,
        anomaly_rate=args.anomaly_rate,
    )
    outputs = write_dataset(frame, calibration, args.output)
    print(json.dumps({**outputs, "calibration": asdict(calibration)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
