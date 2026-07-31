from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import get_settings


def log_mlflow(
    *,
    experiment_suffix: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    artifact: Path,
    params: dict[str, Any],
) -> str | None:
    settings = get_settings()
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment_prefix}-{experiment_suffix}")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **params,
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "feature_schema_version": metadata["feature_schema_version"],
                "dataset_version": metadata["dataset_version"],
                "is_synthetic": metadata["is_synthetic"],
                "git_sha": metadata["git_sha"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact))
        metadata_path = artifact.with_suffix(".metadata.json")
        if metadata_path.exists():
            mlflow.log_artifact(str(metadata_path))
        return run.info.run_id
