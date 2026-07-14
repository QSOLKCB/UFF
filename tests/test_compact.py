from __future__ import annotations

import numpy as np

from uff.compact import (
    kerr_characteristic_radii,
    lqg_area_suppression,
    phenomenological_lqg_velocity_kms,
    smbh_velocity_kms,
)


def test_schwarzschild_characteristic_radii():
    radii = kerr_characteristic_radii(4.0e6, 0.0)
    assert np.isclose(radii.horizon_rg, 2.0)
    assert np.isclose(radii.photon_orbit_rg, 3.0)
    assert np.isclose(radii.isco_rg, 6.0)


def test_prograde_spin_moves_isco_inward():
    zero = kerr_characteristic_radii(1.0e8, 0.0)
    prograde = kerr_characteristic_radii(1.0e8, 0.9)
    retrograde = kerr_characteristic_radii(1.0e8, -0.9)
    assert prograde.isco_rg < zero.isco_rg < retrograde.isco_rg


def test_smbh_kepler_scaling_and_zero_lqg_correction():
    radius = np.array([0.1, 0.4])
    velocity = smbh_velocity_kms(radius, 1.0e8)
    assert np.isclose(velocity[0] / velocity[1], 2.0)
    corrected = phenomenological_lqg_velocity_kms(radius, 1.0e8, alpha=0.0)
    assert np.array_equal(corrected, velocity)
    assert lqg_area_suppression(1.0) < 1.0e-100
