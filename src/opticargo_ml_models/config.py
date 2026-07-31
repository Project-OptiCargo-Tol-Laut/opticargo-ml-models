from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    opticargo_environment: str = "development"
    opticargo_release: str = "dev"
    opticargo_git_sha: str = "local"
    opticargo_shared_version: str = "1.0.0"

    model_mode: Literal["heuristic", "trained", "auto"] = "auto"
    model_preload: bool = True
    model_load_timeout_seconds: int = Field(default=30, ge=1, le=300)
    model_inference_timeout_seconds: int = Field(default=5, ge=1, le=60)
    require_shared_contracts: bool = False

    cargo_match_model_name: str = "cargo-match-scorer"
    cargo_match_model_version: str = "heuristic-v1"
    cargo_match_artifact_path: Path = Path("artifacts/cargo_match_model.joblib")
    cargo_match_artifact_object_key: str = "cargo-match/champion/cargo_match_model.joblib"

    demand_forecast_model_name: str = "demand-forecaster"
    demand_forecast_model_version: str = "heuristic-v1"
    demand_forecast_artifact_path: Path = Path("artifacts/demand_forecast_model.joblib")
    demand_forecast_artifact_object_key: str = "demand-forecast/champion/demand_forecast_model.joblib"

    anomaly_model_name: str = "operational-anomaly-detector"
    anomaly_model_version: str = "heuristic-v1"
    anomaly_artifact_path: Path = Path("artifacts/anomaly_detector.joblib")
    anomaly_artifact_object_key: str = "anomaly/champion/anomaly_detector.joblib"

    internal_service_token: str = ""

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_prefix: str = "opticargo"

    minio_endpoint: str = "minio:9000"
    minio_secure: bool = False
    minio_root_user: str = ""
    minio_root_password: str = ""
    minio_models_bucket: str = "opticargo-model-artifacts"
    upload_model_artifact: bool = False

    promotion_min_accuracy: float = Field(default=0.88, ge=0.0, le=1.0)
    promotion_min_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    promotion_min_forecast_r2: float = Field(default=0.90, ge=-1.0, le=1.0)
    promotion_max_forecast_wape: float = Field(default=0.20, ge=0.0, le=10.0)
    promotion_min_anomaly_roc_auc: float = Field(default=0.90, ge=0.0, le=1.0)
    promotion_min_anomaly_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    drift_psi_alert_threshold: float = Field(default=0.20, ge=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
