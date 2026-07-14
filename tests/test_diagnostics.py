from __future__ import annotations

import numpy as np

from uff.diagnostics import (
    covariance_invariants,
    normalized_shannon_entropy,
    phase_fingerprint,
)


def test_entropy_endpoints():
    assert np.isclose(normalized_shannon_entropy(np.array([1.0, 0.0, 0.0])), 0.0)
    assert np.isclose(normalized_shannon_entropy(np.ones(3)), 1.0)


def test_covariance_invariants_survive_orthogonal_rotation():
    covariance = np.array([[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.0]])
    q, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
    rotated = q @ covariance @ q.T
    first = covariance_invariants(covariance)
    second = covariance_invariants(rotated)
    assert np.allclose(first["eigenvalues"], second["eigenvalues"])
    assert np.isclose(first["frobenius_norm"], second["frobenius_norm"])


def test_phase_fingerprint_has_unit_energy_per_element():
    values = np.array([-1.0, 0.0, 2.0, 5.0])
    fingerprint = phase_fingerprint(values)
    assert np.isclose(fingerprint["energy"], len(values))
