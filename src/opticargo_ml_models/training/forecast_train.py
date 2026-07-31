from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..artifacts import save_bundle, upload_to_minio
from ..config import get_settings
from ..forecasting import FORECAST_FEATURE_COLUMNS, build_forecast_features_from_dataframe
from .tracking import log_mlflow


def temporal_split(frame: pd.DataFrame, features: pd.DataFrame, target: np.ndarray, ratio: float = 0.80):
    order = pd.to_datetime(frame["observed_at"], utc=True).sort_values().index
    cut = int(len(frame) * ratio)
    train_idx, valid_idx = order[:cut], order[cut:]
    return features.loc[train_idx], features.loc[valid_idx], target[train_idx], target[valid_idx]


def evaluate_model(model: Any, x_valid: pd.DataFrame, y_valid: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    prediction = np.maximum(0.0, model.predict(x_valid))
    denominator = np.maximum(np.abs(y_valid), 1.0)
    residual = y_valid - prediction
    metrics = {
        "mae_ton": float(mean_absolute_error(y_valid, prediction)),
        "rmse_ton": float(np.sqrt(mean_squared_error(y_valid, prediction))),
        "mape": float(np.mean(np.abs(residual) / denominator)),
        "wape": float(np.sum(np.abs(residual)) / max(np.sum(np.abs(y_valid)), 1.0)),
        "r2": float(r2_score(y_valid, prediction)),
        "bias_ton": float(np.mean(prediction - y_valid)),
    }
    residual_std = max(float(np.std(residual)), 1.0)
    lower = np.maximum(0.0, prediction - 1.96 * residual_std)
    upper = prediction + 1.96 * residual_std
    metrics["interval_95_coverage"] = float(np.mean((y_valid >= lower) & (y_valid <= upper)))
    return metrics, residual


def train(dataset: Path, artifact: Path, model_version: str) -> dict[str, Any]:
    settings = get_settings()
    frame = pd.read_csv(dataset)
    required = {
        "is_synthetic",
        "provenance",
        "dataset_version",
        "target_definition_version",
        "demand_volume_ton",
        "observed_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset forecast tidak memiliki metadata wajib: {sorted(missing)}")
    if not frame["is_synthetic"].astype(bool).all():
        raise ValueError("Pipeline sintetis tidak boleh menerima campuran data nyata.")

    features = build_forecast_features_from_dataframe(frame)[FORECAST_FEATURE_COLUMNS]
    target = frame["demand_volume_ton"].astype(float).to_numpy()
    x_train, x_valid, y_train, y_valid = temporal_split(frame, features, target)

    model = HistGradientBoostingRegressor(
        max_iter=340,
        learning_rate=0.065,
        max_leaf_nodes=25,
        min_samples_leaf=24,
        l2_regularization=1.4,
        random_state=43,
    )
    model.fit(x_train, y_train)
    metrics, residual = evaluate_model(model, x_valid, y_valid)
    promoted = (
        metrics["r2"] >= settings.promotion_min_forecast_r2
        and metrics["wape"] <= settings.promotion_max_forecast_wape
    )
    metadata = {
        "model_name": settings.demand_forecast_model_name,
        "model_version": model_version,
        "model_family": "demand_forecasting",
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": settings.opticargo_git_sha,
        "dataset_version": str(frame["dataset_version"].iloc[0]),
        "dataset_provenance": str(frame["provenance"].iloc[0]),
        "target_definition_version": str(frame["target_definition_version"].iloc[0]),
        "feature_schema_version": "demand-forecast-features-v1",
        "is_synthetic": True,
        "train_rows": len(x_train),
        "validation_rows": len(x_valid),
        "residual_std_ton": float(max(np.std(residual), 1.0)),
        "metrics": metrics,
        "promotion_decision": "promoted" if promoted else "rejected",
        "promotion_reason": "metrics_passed" if promoted else "metrics_below_guard",
    }
    saved = save_bundle(artifact, model, FORECAST_FEATURE_COLUMNS, metadata)
    run_id = log_mlflow(
        experiment_suffix="demand-forecast",
        metrics=metrics,
        metadata=metadata,
        artifact=artifact,
        params={"model_type": "HistGradientBoostingRegressor"},
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
            object_key=settings.demand_forecast_artifact_object_key,
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
    parser = argparse.ArgumentParser(description="Train demand forecasting baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/demand_forecast.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/demand_forecast_model.joblib"))
    parser.add_argument("--model-version", default="synthetic-forecast-v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = train(args.dataset, args.artifact, args.model_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["promotion_decision"] != "promoted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
