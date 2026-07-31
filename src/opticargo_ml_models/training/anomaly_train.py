from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ..anomalies import ANOMALY_FEATURE_COLUMNS, build_anomaly_features_from_dataframe
from ..artifacts import save_bundle, upload_to_minio
from ..config import get_settings
from .tracking import log_mlflow


def temporal_split(frame: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], labels[train_idx], labels[valid_idx]


def evaluate_model(
    model: Any,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    probabilities = model.predict_proba(x_valid)[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    normal_mask = y_valid == 0
    metrics = {
        "precision": float(precision_score(y_valid, prediction, zero_division=0)),
        "recall": float(recall_score(y_valid, prediction, zero_division=0)),
        "f1": float(f1_score(y_valid, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "false_positive_rate": float(np.mean(prediction[normal_mask])) if normal_mask.any() else 0.0,
        "predicted_anomaly_rate": float(np.mean(prediction)),
        "validation_anomaly_rate": float(np.mean(y_valid)),
    }
    return metrics, 0.5, 1.0


def train(dataset: Path, artifact: Path, model_version: str) -> dict[str, Any]:
    settings = get_settings()
    frame = pd.read_csv(dataset)
    required = {
        "is_synthetic",
        "provenance",
        "dataset_version",
        "label_definition_version",
        "anomaly_label",
        "observed_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset anomaly tidak memiliki metadata wajib: {sorted(missing)}")
    if not frame["is_synthetic"].astype(bool).all():
        raise ValueError("Pipeline sintetis tidak boleh menerima campuran data nyata.")

    features = build_anomaly_features_from_dataframe(frame)[ANOMALY_FEATURE_COLUMNS]
    labels = frame["anomaly_label"].astype(int).to_numpy()
    x_train, x_valid, y_train, y_valid = temporal_split(frame, features, labels)
    normal_train = x_train[y_train == 0]
    anomaly_rate = float(np.clip(np.mean(y_train), 0.01, 0.25))

    model = HistGradientBoostingClassifier(
        max_iter=340,
        learning_rate=0.06,
        max_leaf_nodes=21,
        min_samples_leaf=24,
        l2_regularization=1.4,
        random_state=44,
    )
    model.fit(x_train, y_train)
    metrics, threshold, score_scale = evaluate_model(model, x_valid, y_valid)
    promoted = (
        metrics["roc_auc"] >= settings.promotion_min_anomaly_roc_auc
        and metrics["f1"] >= settings.promotion_min_anomaly_f1
    )
    baselines = {
        column: {
            "mean": float(normal_train[column].mean()),
            "std": float(max(normal_train[column].std(), 1e-9)),
        }
        for column in ANOMALY_FEATURE_COLUMNS[:9]
    }
    metadata = {
        "model_name": settings.anomaly_model_name,
        "model_version": model_version,
        "model_family": "operational_anomaly_detection",
        "model_type": "HistGradientBoostingClassifier",
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": settings.opticargo_git_sha,
        "dataset_version": str(frame["dataset_version"].iloc[0]),
        "dataset_provenance": str(frame["provenance"].iloc[0]),
        "label_definition_version": str(frame["label_definition_version"].iloc[0]),
        "feature_schema_version": "operational-anomaly-features-v1",
        "is_synthetic": True,
        "train_rows": len(x_train),
        "normal_train_rows": len(normal_train),
        "validation_rows": len(x_valid),
        "anomaly_threshold": threshold,
        "anomaly_score_scale": score_scale,
        "training_anomaly_rate": anomaly_rate,
        "feature_baselines": baselines,
        "metrics": metrics,
        "promotion_decision": "promoted" if promoted else "rejected",
        "promotion_reason": "metrics_passed" if promoted else "metrics_below_guard",
    }
    saved = save_bundle(artifact, model, ANOMALY_FEATURE_COLUMNS, metadata)
    run_id = log_mlflow(
        experiment_suffix="operational-anomaly",
        metrics=metrics,
        metadata=metadata,
        artifact=artifact,
        params={"model_type": "HistGradientBoostingClassifier", "training_anomaly_rate": anomaly_rate},
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
            object_key=settings.anomaly_artifact_object_key,
        )
    report = {
        **saved["metadata"],
        **{key: value for key, value in saved.items() if key != "metadata"},
        "mlflow_run_id": run_id,
        "artifact_uri": artifact_uri,
    }
    artifact.with_suffix(".evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train operational anomaly detector.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/operational_anomaly.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/anomaly_detector.joblib"))
    parser.add_argument("--model-version", default="synthetic-anomaly-v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = train(args.dataset, args.artifact, args.model_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["promotion_decision"] != "promoted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
