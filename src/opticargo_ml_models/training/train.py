from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..artifacts import save_bundle, upload_to_minio
from ..config import get_settings
from ..features import MODEL_FEATURE_COLUMNS, build_features_from_dataframe
from .tracking import log_mlflow


def temporal_split(frame: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], labels[train_idx], labels[valid_idx]


def evaluate_model(model: Any, x_valid: pd.DataFrame, y_valid: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    hard_valid = x_valid["hard_constraint_valid"].to_numpy().astype(bool)
    predictions[~hard_valid] = 0
    probabilities[~hard_valid] = 0.0
    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "brier_score": float(brier_score_loss(y_valid, probabilities)),
        "hard_constraint_violation_rate": float(np.mean(predictions[~hard_valid])) if (~hard_valid).any() else 0.0,
    }


def train(dataset: Path, artifact: Path, model_version: str) -> dict[str, Any]:
    settings = get_settings()
    frame = pd.read_csv(dataset)
    required_metadata = {"is_synthetic", "provenance", "dataset_version", "label_definition_version", "match_label", "observed_at"}
    missing = required_metadata.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset tidak memiliki provenance/label wajib: {sorted(missing)}")
    if not frame["is_synthetic"].astype(bool).all():
        raise ValueError("Starter ini hanya menerima dataset sintetis terpisah. Jangan campur data nyata.")

    features = build_features_from_dataframe(frame)[MODEL_FEATURE_COLUMNS]
    labels = frame["match_label"].astype(int).to_numpy()
    x_train, x_valid, y_train, y_valid = temporal_split(frame, features, labels)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.07,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=42,
    )
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_valid, y_valid)

    promoted = (
        metrics["accuracy"] >= settings.promotion_min_accuracy
        and metrics["f1"] >= settings.promotion_min_f1
        and metrics["hard_constraint_violation_rate"] == 0.0
    )
    metadata = {
        "model_name": settings.cargo_match_model_name,
        "model_version": model_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": settings.opticargo_git_sha,
        "dataset_version": str(frame["dataset_version"].iloc[0]),
        "dataset_provenance": str(frame["provenance"].iloc[0]),
        "label_definition_version": str(frame["label_definition_version"].iloc[0]),
        "feature_schema_version": "cargo-match-features-v1",
        "is_synthetic": True,
        "train_rows": len(x_train),
        "validation_rows": len(x_valid),
        "metrics": metrics,
        "promotion_decision": "promoted" if promoted else "rejected",
        "promotion_reason": "metrics_passed" if promoted else "metrics_below_guard",
    }
    saved = save_bundle(artifact, model, MODEL_FEATURE_COLUMNS, metadata)
    run_id = log_mlflow(
        experiment_suffix="cargo-match",
        metrics=metrics,
        metadata=metadata,
        artifact=artifact,
        params={"model_type": "HistGradientBoostingClassifier"},
    )
    artifact_uri = None
    if settings.upload_model_artifact:
        artifact_uri = upload_to_minio(
            artifact,
            endpoint=settings.minio_endpoint,
            secure=settings.minio_secure,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            bucket=settings.minio_models_bucket,
            object_key=settings.cargo_match_artifact_object_key,
        )
    report = {**saved["metadata"], **{k: v for k, v in saved.items() if k != "metadata"}, "mlflow_run_id": run_id, "artifact_uri": artifact_uri}
    report_path = artifact.with_suffix(".evaluation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline cargo match model.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/cargo_match.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/cargo_match_model.joblib"))
    parser.add_argument("--model-version", default="synthetic-baseline-v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = train(args.dataset, args.artifact, args.model_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["promotion_decision"] != "promoted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
