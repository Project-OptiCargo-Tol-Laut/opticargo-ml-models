from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .anomaly_generator import generate_calibrated_anomaly_dataset
from .anomaly_generator import write_dataset as write_anomaly
from .forecast_generator import generate_calibrated_forecast_dataset
from .forecast_generator import write_dataset as write_forecast
from .generator import generate_calibrated_dataset
from .generator import write_dataset as write_cargo


def generate_all(output_dir: Path, rows: int = 8_000) -> dict[str, object]:
    cargo, cargo_cal = generate_calibrated_dataset(rows=rows, seed=42, target_accuracy=0.90)
    forecast, forecast_cal = generate_calibrated_forecast_dataset(
        rows=max(rows, 1_500), seed=43, target_r2=0.90
    )
    anomaly, anomaly_cal = generate_calibrated_anomaly_dataset(
        rows=max(rows, 2_000), seed=44, target_f1=0.90
    )
    return {
        "cargo_match": {
            **write_cargo(cargo, cargo_cal, output_dir / "cargo_match.csv"),
            "calibration": cargo_cal.__dict__,
        },
        "demand_forecast": {
            **write_forecast(forecast, forecast_cal, output_dir / "demand_forecast.csv"),
            "calibration": asdict(forecast_cal),
        },
        "operational_anomaly": {
            **write_anomaly(anomaly, anomaly_cal, output_dir / "operational_anomaly.csv"),
            "calibration": asdict(anomaly_cal),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seluruh dataset sintetis OptiCargo ML.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--rows", type=int, default=8_000)
    args = parser.parse_args()
    print(json.dumps(generate_all(args.output_dir, args.rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
