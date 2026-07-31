from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..features import MODEL_FEATURE_COLUMNS, build_features_from_dataframe, heuristic_score


@dataclass(frozen=True)
class CalibrationResult:
    target_accuracy: float
    achieved_accuracy: float
    achieved_f1: float
    achieved_roc_auc: float
    teacher_agreement: float
    label_noise_rate: float
    train_rows: int
    validation_rows: int


def _stable_uuid(kind: str, index: int, seed: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"opticargo:{kind}:{seed}:{index}"))


def _base_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    route_distance = rng.uniform(150, 2_500, rows)
    origin_distance = np.clip(rng.gamma(2.0, 35.0, rows), 0, 350)
    destination_distance = np.clip(rng.gamma(2.0, 30.0, rows), 0, 350)
    schedule_gap = np.clip(rng.normal(36, 48, rows), -96, 240)
    remaining_weight = rng.uniform(40, 600, rows)
    remaining_volume = rng.uniform(100, 1_800, rows)
    cargo_weight = np.clip(rng.gamma(2.2, 45.0, rows), 3, 450)
    cargo_volume = np.clip(cargo_weight * rng.uniform(1.5, 4.0, rows) + rng.normal(0, 30, rows), 10, 1_600)
    asking_price = rng.uniform(800_000, 4_500_000, rows)
    market_rate = asking_price * rng.uniform(0.90, 1.55, rows)
    operating_cost = rng.uniform(12_000, 75_000, rows)

    frame = pd.DataFrame(
        {
            "record_id": [_stable_uuid("record", i, seed) for i in range(rows)],
            "voyage_id": [_stable_uuid("voyage", i % max(40, rows // 20), seed) for i in range(rows)],
            "cargo_listing_id": [_stable_uuid("cargo", i, seed) for i in range(rows)],
            "route_id": [_stable_uuid("route", i % 25, seed) for i in range(rows)],
            "supplier_id": [_stable_uuid("supplier", i % 120, seed) for i in range(rows)],
            "observed_at": [
                (base_time + timedelta(hours=int(i * 8 + rng.integers(0, 8)))).isoformat()
                for i in range(rows)
            ],
            "route_distance_km": route_distance.round(3),
            "origin_distance_km": origin_distance.round(3),
            "destination_distance_km": destination_distance.round(3),
            "schedule_gap_hours": schedule_gap.round(3),
            "cargo_weight_ton": cargo_weight.round(3),
            "cargo_volume_m3": cargo_volume.round(3),
            "remaining_weight_ton": remaining_weight.round(3),
            "remaining_volume_m3": remaining_volume.round(3),
            "asking_price_per_ton_idr": asking_price.round(2),
            "market_rate_per_ton_idr": market_rate.round(2),
            "operating_cost_per_km_idr": operating_cost.round(2),
            "supplier_rating": np.clip(rng.normal(4.2, 0.55, rows), 1.0, 5.0).round(4),
            "supplier_success_rate": np.clip(rng.beta(8, 2, rows), 0.30, 0.995).round(6),
            "supplier_cancellation_rate": np.clip(rng.beta(1.5, 8, rows) * 0.55, 0, 0.60).round(6),
            "commodity_compatibility": (rng.random(rows) < 0.94).astype(int),
            "certification_match": (rng.random(rows) < 0.96).astype(int),
            "temperature_match": (rng.random(rows) < 0.97).astype(int),
            "weather_risk": np.clip(rng.beta(2, 6, rows), 0, 1).round(6),
            "port_congestion": np.clip(rng.beta(2.3, 4.5, rows), 0, 1).round(6),
            "historical_acceptance_rate": np.clip(rng.beta(5, 3, rows), 0.05, 0.98).round(6),
            "is_synthetic": True,
            "provenance": "opticargo-synthetic-cargo-match-generator-v1",
        }
    )
    return frame


def _clean_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = build_features_from_dataframe(frame)
    scores = features.apply(lambda row: heuristic_score(row.to_dict()), axis=1).to_numpy()
    labels = (scores >= 0.58).astype(int)
    return features, scores, labels


def _apply_label_noise(clean_labels: np.ndarray, agreement: float, seed: int) -> np.ndarray:
    if not 0.5 <= agreement <= 1.0:
        raise ValueError("teacher agreement harus berada pada rentang 0.5..1.0")
    noisy = clean_labels.copy()
    count = int(round(len(noisy) * (1.0 - agreement)))
    rng = np.random.default_rng(seed + 20_260_730)
    if count > 0:
        indices = rng.choice(len(noisy), count, replace=False)
        noisy[indices] = 1 - noisy[indices]
    return noisy


def _temporal_split(frame: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], labels[train_idx], labels[valid_idx]


def _probe(frame: pd.DataFrame, labels: np.ndarray, seed: int) -> tuple[float, float, float, int, int]:
    features = build_features_from_dataframe(frame)[MODEL_FEATURE_COLUMNS]
    x_train, x_valid, y_train, y_valid = _temporal_split(frame, features, labels)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.07,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    hard_valid = x_valid["hard_constraint_valid"].to_numpy().astype(bool)
    predictions[~hard_valid] = 0
    probabilities[~hard_valid] = 0.0
    accuracy = float(accuracy_score(y_valid, predictions))
    f1 = float(f1_score(y_valid, predictions, zero_division=0))
    roc_auc = float(roc_auc_score(y_valid, probabilities))
    return accuracy, f1, roc_auc, len(x_train), len(x_valid)


def generate_calibrated_dataset(
    rows: int = 8_000,
    seed: int = 42,
    target_accuracy: float = 0.90,
    tolerance: float = 0.005,
) -> tuple[pd.DataFrame, CalibrationResult]:
    if rows < 1_000:
        raise ValueError("Kalibrasi membutuhkan minimal 1.000 baris.")
    if not 0.75 <= target_accuracy <= 0.97:
        raise ValueError("target accuracy generator harus berada pada rentang 0.75..0.97")

    frame = _base_frame(rows, seed)
    _, teacher_score, clean_labels = _clean_labels(frame)
    candidates = np.round(np.arange(max(0.84, target_accuracy - 0.01), min(0.995, target_accuracy + 0.09), 0.01), 3)
    best: tuple[float, float, float, float, float, int, int, np.ndarray] | None = None

    for agreement in candidates:
        labels = _apply_label_noise(clean_labels, float(agreement), seed)
        accuracy, f1, roc_auc, train_rows, valid_rows = _probe(frame, labels, seed)
        distance = abs(accuracy - target_accuracy)
        candidate = (distance, accuracy, f1, roc_auc, float(agreement), train_rows, valid_rows, labels)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if distance <= tolerance:
            break

    assert best is not None
    _, accuracy, f1, roc_auc, agreement, train_rows, valid_rows, labels = best
    frame["teacher_match_score"] = teacher_score.round(6)
    frame["match_label"] = labels.astype(int)
    frame["label_definition_version"] = "cargo-match-acceptance-v1"
    frame["dataset_version"] = f"synthetic-cargo-match-{datetime.now(UTC).date().isoformat()}-seed{seed}"

    result = CalibrationResult(
        target_accuracy=target_accuracy,
        achieved_accuracy=accuracy,
        achieved_f1=f1,
        achieved_roc_auc=roc_auc,
        teacher_agreement=agreement,
        label_noise_rate=1.0 - agreement,
        train_rows=train_rows,
        validation_rows=valid_rows,
    )
    return frame, result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(frame: pd.DataFrame, calibration: CalibrationResult, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    checksum = _sha256(output)
    manifest = {
        "dataset_name": "opticargo-synthetic-cargo-match",
        "dataset_version": str(frame["dataset_version"].iloc[0]),
        "created_at": datetime.now(UTC).isoformat(),
        "is_synthetic": True,
        "provenance": "opticargo-synthetic-cargo-match-generator-v1",
        "rows": len(frame),
        "label_column": "match_label",
        "label_definition_version": "cargo-match-acceptance-v1",
        "sha256": checksum,
        "calibration": calibration.__dict__,
        "limitations": [
            "Kemiripan/akurasi hanya terhadap aturan pembangkit sintetis, bukan bukti performa pada data nyata.",
            "Distribusi harus diganti atau dikalibrasi ulang saat data operasional tersedia.",
            "Data sintetis tidak boleh digabung dengan data nyata tanpa flag dan provenance.",
        ],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": str(output), "manifest": str(manifest_path), "sha256": checksum}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset cargo matching sintetis terkalibrasi.")
    parser.add_argument("--rows", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/cargo_match.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame, calibration = generate_calibrated_dataset(
        rows=args.rows,
        seed=args.seed,
        target_accuracy=args.target_accuracy,
        tolerance=args.tolerance,
    )
    outputs = write_dataset(frame, calibration, args.output)
    print(json.dumps({**outputs, "calibration": calibration.__dict__}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
