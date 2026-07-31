from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from ..training.anomaly_train import train as train_anomaly
from ..training.forecast_train import train as train_forecast
from ..training.train import train as train_cargo
from ..training.train_all import train_all


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Retraining job sedang berjalan; lock ditemukan: {path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-off guarded retraining job.")
    parser.add_argument(
        "--model",
        choices=["all", "cargo-match", "demand-forecast", "anomaly"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--release", default="synthetic-baseline-v1")
    parser.add_argument("--lock", type=Path, default=Path("artifacts/retraining.lock"))
    args = parser.parse_args()

    with exclusive_lock(args.lock):
        if args.model == "all":
            report = train_all(args.data_dir, args.artifact_dir, args.release)
            promoted = bool(report["all_promoted"])
        elif args.model == "cargo-match":
            report = train_cargo(
                args.data_dir / "cargo_match.csv",
                args.artifact_dir / "cargo_match_model.joblib",
                f"{args.release}-cargo-match",
            )
            promoted = report["promotion_decision"] == "promoted"
        elif args.model == "demand-forecast":
            report = train_forecast(
                args.data_dir / "demand_forecast.csv",
                args.artifact_dir / "demand_forecast_model.joblib",
                f"{args.release}-demand-forecast",
            )
            promoted = report["promotion_decision"] == "promoted"
        else:
            report = train_anomaly(
                args.data_dir / "operational_anomaly.csv",
                args.artifact_dir / "anomaly_detector.joblib",
                f"{args.release}-anomaly",
            )
            promoted = report["promotion_decision"] == "promoted"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
