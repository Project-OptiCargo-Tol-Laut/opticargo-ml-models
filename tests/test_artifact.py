from pathlib import Path

from opticargo_ml_models.artifacts import load_bundle

EXPECTED = {
    "cargo_match_model.joblib": "cargo-match-scorer",
    "demand_forecast_model.joblib": "demand-forecaster",
    "anomaly_detector.joblib": "operational-anomaly-detector",
}


def test_prebuilt_artifacts_have_required_metadata():
    for filename, model_name in EXPECTED.items():
        path = Path("artifacts") / filename
        if not path.exists():
            continue
        bundle = load_bundle(path)
        metadata = bundle["metadata"]
        assert metadata["model_name"] == model_name
        assert metadata["is_synthetic"] is True
        assert metadata["promotion_decision"] == "promoted"
        assert "dependencies" in metadata
