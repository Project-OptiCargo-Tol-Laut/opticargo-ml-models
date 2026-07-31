from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..training.drift import compare


def main() -> None:
    parser = argparse.ArgumentParser(description="One-off feature drift check.")
    parser.add_argument("--model", choices=["cargo-match", "demand-forecast", "anomaly"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift_report.json"))
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    scores = compare(pd.read_csv(args.reference), pd.read_csv(args.current), args.model)
    alerts = {
        name: score
        for name, score in scores.items()
        if score >= settings.drift_psi_alert_threshold
    }
    report = {
        "model": args.model,
        "status": "alert" if alerts else "ok",
        "threshold": settings.drift_psi_alert_threshold,
        "scores": scores,
        "alerts": alerts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if alerts and args.fail_on_alert:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
