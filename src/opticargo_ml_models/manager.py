from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from .anomalies import (
    ANOMALY_FEATURE_COLUMNS,
    anomaly_severity,
    build_anomaly_explanations,
    build_anomaly_feature_row,
    heuristic_anomaly,
)
from .artifacts import download_from_minio, load_bundle
from .config import Settings
from .contracts import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    CargoMatchRequest,
    CargoMatchResponse,
    DemandForecastRequest,
    DemandForecastResponse,
    ModelMode,
    ModelRegistryStatusResponse,
    ModelStatusEntry,
)
from .features import MODEL_FEATURE_COLUMNS, build_explanations, build_feature_row, heuristic_score
from .forecasting import (
    FORECAST_FEATURE_COLUMNS,
    build_forecast_explanations,
    build_forecast_feature_row,
    heuristic_forecast,
)
from .metrics import FALLBACK_TOTAL, MODEL_INFO, MODEL_READY, PREDICTION_TOTAL

LOGGER = logging.getLogger(__name__)

CARGO_MATCH = "cargo-match-scorer"
DEMAND_FORECAST = "demand-forecaster"
ANOMALY_DETECTOR = "operational-anomaly-detector"


@dataclass
class RuntimeModel:
    name: str
    default_version: str
    artifact_path: Path
    object_key: str
    expected_features: list[str]
    estimator: Any | None = None
    feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    last_error: str | None = None

    @property
    def mode(self) -> ModelMode:
        return ModelMode.TRAINED if self.loaded else ModelMode.HEURISTIC

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", self.default_version)) if self.loaded else self.default_version


class MultiModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self.models: dict[str, RuntimeModel] = {
            CARGO_MATCH: RuntimeModel(
                name=settings.cargo_match_model_name,
                default_version="heuristic-v1",
                artifact_path=settings.cargo_match_artifact_path,
                object_key=settings.cargo_match_artifact_object_key,
                expected_features=MODEL_FEATURE_COLUMNS.copy(),
            ),
            DEMAND_FORECAST: RuntimeModel(
                name=settings.demand_forecast_model_name,
                default_version="heuristic-v1",
                artifact_path=settings.demand_forecast_artifact_path,
                object_key=settings.demand_forecast_artifact_object_key,
                expected_features=FORECAST_FEATURE_COLUMNS.copy(),
            ),
            ANOMALY_DETECTOR: RuntimeModel(
                name=settings.anomaly_model_name,
                default_version="heuristic-v1",
                artifact_path=settings.anomaly_artifact_path,
                object_key=settings.anomaly_artifact_object_key,
                expected_features=ANOMALY_FEATURE_COLUMNS.copy(),
            ),
        }

    def preload(self) -> None:
        for key in self.models:
            self.preload_model(key)

    def preload_model(self, key: str) -> None:
        runtime = self.models[key]
        if self.settings.model_mode == "heuristic":
            self._mark_ready(runtime)
            return
        try:
            if (
                not runtime.artifact_path.exists()
                and self.settings.minio_root_user
                and self.settings.minio_root_password
            ):
                download_from_minio(
                    runtime.artifact_path,
                    endpoint=self.settings.minio_endpoint,
                    secure=self.settings.minio_secure,
                    access_key=self.settings.minio_root_user,
                    secret_key=self.settings.minio_root_password,
                    bucket=self.settings.minio_models_bucket,
                    object_key=runtime.object_key,
                )
            self.load_model(key, runtime.artifact_path)
        except Exception as exc:  # fallback wajib menjaga service tetap tersedia
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            runtime.loaded = False
            runtime.estimator = None
            FALLBACK_TOTAL.labels(model_name=runtime.name, reason="model_load_failed").inc()
            LOGGER.warning("%s gagal dimuat; memakai fallback: %s", runtime.name, runtime.last_error)
            self._mark_ready(runtime)

    def load_model(self, key: str, path: Path) -> None:
        runtime = self.models[key]
        with self._lock:
            bundle = load_bundle(path)
            columns = list(bundle["feature_columns"])
            if columns != runtime.expected_features:
                raise ValueError(
                    "Feature schema artifact tidak kompatibel. "
                    f"expected={runtime.expected_features}, actual={columns}"
                )
            metadata = dict(bundle.get("metadata", {}))
            artifact_name = metadata.get("model_name")
            if artifact_name and artifact_name != runtime.name:
                raise ValueError(f"Model artifact salah: expected={runtime.name}, actual={artifact_name}")
            runtime.estimator = bundle["estimator"]
            runtime.feature_columns = columns
            runtime.metadata = metadata
            runtime.loaded = True
            runtime.last_error = None
            self._mark_ready(runtime)

    def _mark_ready(self, runtime: RuntimeModel) -> None:
        MODEL_READY.labels(model_name=runtime.name).set(1)
        MODEL_INFO.labels(model_name=runtime.name).info(
            {
                "version": runtime.version,
                "mode": runtime.mode.value,
                "release": self.settings.opticargo_release,
                "git_sha": self.settings.opticargo_git_sha,
            }
        )

    def score_cargo(self, request: CargoMatchRequest) -> CargoMatchResponse:
        runtime = self.models[CARGO_MATCH]
        features = build_feature_row(request)
        hard_valid = bool(features["hard_constraint_valid"])
        warnings: list[str] = []

        if not hard_valid:
            score = 0.0
            warnings.append("Kandidat ditolak oleh hard constraint sebelum ranking model.")
            result = "hard_constraint_rejected"
        elif runtime.loaded and runtime.estimator is not None:
            frame = pd.DataFrame([{key: features[key] for key in runtime.feature_columns}])
            score = float(runtime.estimator.predict_proba(frame)[0, 1])
            warnings.append("Penjelasan fitur bersifat domain-directed, bukan estimasi kausal/SHAP.")
            result = "scored"
        else:
            score = heuristic_score(features)
            if runtime.last_error:
                warnings.append(f"Trained model tidak tersedia: {runtime.last_error}")
            FALLBACK_TOTAL.labels(model_name=runtime.name, reason="trained_model_unavailable").inc()
            result = "scored"

        PREDICTION_TOTAL.labels(
            model_name=runtime.name,
            model_mode=runtime.mode.value,
            result=result,
        ).inc()
        return CargoMatchResponse(
            score=round(float(np.clip(score, 0.0, 1.0)), 6),
            model_mode=runtime.mode,
            model_version=runtime.version,
            fallback_used=not runtime.loaded,
            hard_constraint_valid=hard_valid,
            feature_explanations=build_explanations(features, score),
            warnings=warnings,
            trace_id=request.trace_id or "",
        )

    def forecast_demand(self, request: DemandForecastRequest) -> DemandForecastResponse:
        runtime = self.models[DEMAND_FORECAST]
        features = build_forecast_feature_row(request)
        warnings: list[str] = []
        if runtime.loaded and runtime.estimator is not None:
            frame = pd.DataFrame([{key: features[key] for key in runtime.feature_columns}])
            prediction = max(0.0, float(runtime.estimator.predict(frame)[0]))
            residual_std = max(float(runtime.metadata.get("residual_std_ton", prediction * 0.10)), 1.0)
            warnings.append("Interval memakai residual validation sintetis, bukan jaminan probabilistik produksi.")
        else:
            prediction = heuristic_forecast(features)
            residual_std = max(prediction * 0.18, 1.0)
            if runtime.last_error:
                warnings.append(f"Trained model tidak tersedia: {runtime.last_error}")
            FALLBACK_TOTAL.labels(model_name=runtime.name, reason="trained_model_unavailable").inc()
        lower = max(0.0, prediction - 1.96 * residual_std)
        upper = max(lower, prediction + 1.96 * residual_std)
        PREDICTION_TOTAL.labels(
            model_name=runtime.name,
            model_mode=runtime.mode.value,
            result="forecasted",
        ).inc()
        return DemandForecastResponse(
            predicted_volume_ton=round(prediction, 6),
            lower_bound_ton=round(lower, 6),
            upper_bound_ton=round(upper, 6),
            forecast_horizon_days=request.forecast_horizon_days,
            model_mode=runtime.mode,
            model_version=runtime.version,
            fallback_used=not runtime.loaded,
            feature_explanations=build_forecast_explanations(features, prediction),
            warnings=warnings,
            trace_id=request.trace_id or "",
        )

    def detect_anomaly(self, request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
        runtime = self.models[ANOMALY_DETECTOR]
        features = build_anomaly_feature_row(request)
        warnings: list[str] = []
        if runtime.loaded and runtime.estimator is not None:
            frame = pd.DataFrame([{key: features[key] for key in runtime.feature_columns}])
            if hasattr(runtime.estimator, "predict_proba"):
                score = float(runtime.estimator.predict_proba(frame)[0, 1])
                threshold = float(runtime.metadata.get("anomaly_threshold", 0.5))
                is_anomaly = score >= threshold
                warnings.append("Skor adalah probabilitas classifier yang dilatih pada skenario sintetis.")
            else:
                raw_score = float(-runtime.estimator.decision_function(frame)[0])
                threshold = float(runtime.metadata.get("anomaly_threshold", 0.0))
                scale = max(float(runtime.metadata.get("anomaly_score_scale", 0.05)), 1e-6)
                score = float(1.0 / (1.0 + np.exp(-np.clip((raw_score - threshold) / scale, -60, 60))))
                is_anomaly = raw_score >= threshold
                warnings.append("Skor adalah kalibrasi decision function model anomaly.")
        else:
            score, _ = heuristic_anomaly(features)
            is_anomaly = score >= 0.50
            if runtime.last_error:
                warnings.append(f"Trained model tidak tersedia: {runtime.last_error}")
            FALLBACK_TOTAL.labels(model_name=runtime.name, reason="trained_model_unavailable").inc()
        PREDICTION_TOTAL.labels(
            model_name=runtime.name,
            model_mode=runtime.mode.value,
            result="anomaly" if is_anomaly else "normal",
        ).inc()
        return AnomalyDetectionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(float(np.clip(score, 0.0, 1.0)), 6),
            severity=anomaly_severity(score),
            model_mode=runtime.mode,
            model_version=runtime.version,
            fallback_used=not runtime.loaded,
            feature_explanations=build_anomaly_explanations(features, runtime.metadata, score),
            warnings=warnings,
            trace_id=request.trace_id or "",
        )

    def status_entry(self, key: str) -> ModelStatusEntry:
        runtime = self.models[key]
        return ModelStatusEntry(
            model_name=runtime.name,
            model_mode=runtime.mode,
            model_version=runtime.version,
            artifact_path=str(runtime.artifact_path),
            loaded=runtime.loaded,
            fallback_available=True,
            last_error=runtime.last_error,
            metadata=runtime.metadata,
        )

    def status(self) -> ModelRegistryStatusResponse:
        entries = [self.status_entry(key) for key in self.models]
        loaded = sum(item.loaded for item in entries)
        return ModelRegistryStatusResponse(
            service_status="ready",
            loaded_models=loaded,
            total_models=len(entries),
            models=entries,
        )
