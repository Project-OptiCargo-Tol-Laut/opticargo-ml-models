from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anomaly_train import train as train_anomaly
from .forecast_train import train as train_forecast
from .train import train as train_cargo


def train_all(data_dir: Path, artifact_dir: Path, release: str) -> dict[str, object]:
    reports = {
        "cargo_match": train_cargo(
            data_dir / "cargo_match.csv",
            artifact_dir / "cargo_match_model.joblib",
            f"{release}-cargo-match",
        ),
        "demand_forecast": train_forecast(
            data_dir / "demand_forecast.csv",
            artifact_dir / "demand_forecast_model.joblib",
            f"{release}-demand-forecast",
        ),
        "operational_anomaly": train_anomaly(
            data_dir / "operational_anomaly.csv",
            artifact_dir / "anomaly_detector.joblib",
            f"{release}-anomaly",
        ),
    }
    reports["all_promoted"] = all(
        report["promotion_decision"] == "promoted"
        for key, report in reports.items()
        if isinstance(report, dict)
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seluruh model OptiCargo.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--release", default="synthetic-baseline-v1")
    args = parser.parse_args()
    reports = train_all(args.data_dir, args.artifact_dir, args.release)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if not reports["all_promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
