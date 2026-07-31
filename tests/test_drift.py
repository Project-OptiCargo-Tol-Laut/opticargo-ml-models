import numpy as np

from opticargo_ml_models.training.drift import population_stability_index


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 5000)
    similar = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert population_stability_index(reference, similar) < 0.1
    assert population_stability_index(reference, shifted) > 0.2
