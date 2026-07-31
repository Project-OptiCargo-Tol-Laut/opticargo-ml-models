import numpy as np

from opticargo_ml_models.training.anomaly_generator import _base_frame as anomaly_base
from opticargo_ml_models.training.forecast_generator import _base_frame as forecast_base
from opticargo_ml_models.training.generator import _apply_label_noise, _base_frame, _clean_labels


def test_cargo_generator_is_deterministic_and_marks_provenance():
    first = _base_frame(1200, 42)
    second = _base_frame(1200, 42)
    assert first.equals(second)
    assert first["is_synthetic"].all()
    assert first["provenance"].nunique() == 1


def test_label_teacher_agreement_is_exact():
    frame = _base_frame(2000, 42)
    _, _, clean = _clean_labels(frame)
    noisy = _apply_label_noise(clean, 0.90, 42)
    agreement = float(np.mean(clean == noisy))
    assert agreement == 0.90


def test_forecast_generator_base_is_deterministic():
    first, first_target = forecast_base(300, 43)
    second, second_target = forecast_base(300, 43)
    assert first.equals(second)
    assert np.array_equal(first_target, second_target)
    assert first["is_synthetic"].all()


def test_anomaly_generator_marks_expected_anomaly_rate():
    frame, labels, _ = anomaly_base(1000, 44, 0.08)
    assert frame["is_synthetic"].all()
    assert np.isclose(labels.mean(), 0.08)
