from __future__ import annotations

import numpy as np
import pytest

from uff.constants import DEFAULT_A0_M_S2
from uff.data import GalaxyData
from uff.models import (
    ModelOptions,
    build_model,
    burkert_enclosed_mass_msun,
    mond_acceleration_m_s2,
    nfw_enclosed_mass_msun,
    nfw_r200_kpc,
    nfw_velocity_kms,
    uff_empirical_velocity_kms,
)


def test_nfw_mass_definition_is_exact_at_r200():
    mass = 1.0e12
    concentration = 9.0
    r200 = nfw_r200_kpc(mass, 70.0)
    enclosed = nfw_enclosed_mass_msun(np.array([r200]), mass, concentration, 70.0)
    assert np.isclose(enclosed[0], mass, rtol=1e-12)
    assert nfw_velocity_kms(np.array([0.01, 1.0, 10.0]), mass, concentration).min() > 0


def test_burkert_small_radius_matches_constant_density_limit():
    density = 2.0e7
    core = 5.0
    radius = np.array([1.0e-4])
    enclosed = burkert_enclosed_mass_msun(radius, density, core)[0]
    expected = 4.0 * np.pi * density * radius[0] ** 3 / 3.0
    assert np.isclose(enclosed, expected, rtol=1e-6)


def test_mond_has_newtonian_and_deep_mond_limits():
    high = mond_acceleration_m_s2(np.array([1.0e-6]), relation="rar")[0]
    assert np.isclose(high, 1.0e-6, rtol=0.02)
    low_newtonian = 1.0e-14
    low = mond_acceleration_m_s2(np.array([low_newtonian]), relation="simple")[0]
    assert np.isclose(low, np.sqrt(low_newtonian * DEFAULT_A0_M_S2), rtol=0.01)


def test_uff_empirical_is_finite_bounded_and_zero_at_origin_limit():
    radius = np.geomspace(1.0e-8, 1.0e5, 200)
    velocity = uff_empirical_velocity_kms(radius, 150.0, 3.0, 0.2)
    assert np.all(np.isfinite(velocity))
    assert np.all(velocity >= 0)
    assert velocity[0] < 0.01
    assert velocity[-1] < 150.0 * np.exp(0.2) * 1.001


def test_model_builder_separates_efe_proxy():
    data = GalaxyData(
        radius_kpc=np.array([1.0, 2.0, 3.0]),
        velocity_obs_kms=np.array([30.0, 40.0, 50.0]),
        velocity_err_kms=np.ones(3),
        velocity_gas_kms=np.array([5.0, 6.0, 7.0]),
        velocity_disk_kms=np.array([20.0, 25.0, 30.0]),
        velocity_bulge_kms=np.zeros(3),
    )
    with pytest.raises(ValueError, match="external_field"):
        build_model("mond-efe", data, ModelOptions(external_field_a0=0.0))
    model = build_model(
        "mond-efe",
        data,
        ModelOptions(external_field_a0=0.03, fit_stellar_mass_to_light=False),
    )
    prediction = model.predict(data.radius_kpc, np.empty(0))
    assert np.all(np.isfinite(prediction))


def test_distance_and_inclination_nuisance_scaling():
    data = GalaxyData(
        radius_kpc=np.array([1.0, 2.0, 3.0]),
        velocity_obs_kms=np.array([30.0, 40.0, 50.0]),
        velocity_err_kms=np.ones(3),
        velocity_gas_kms=np.zeros(3),
        velocity_disk_kms=np.array([20.0, 25.0, 30.0]),
        velocity_bulge_kms=np.zeros(3),
        metadata={"INC_deg": 60.0},
    )
    fixed = build_model("baryons", data, ModelOptions(fit_stellar_mass_to_light=False))
    scaled = build_model(
        "baryons",
        data,
        ModelOptions(
            fit_stellar_mass_to_light=False,
            fit_distance_scale=True,
            fit_inclination=True,
        ),
    )
    baseline = fixed.predict(data.radius_kpc, np.empty(0))
    changed = scaled.predict(
        data.radius_kpc,
        {"distance_scale": 1.21, "inclination_deg": 30.0},
    )
    expected_factor = (
        np.sqrt(1.21) * np.sin(np.deg2rad(30.0)) / np.sin(np.deg2rad(60.0))
    )
    assert np.allclose(changed, baseline * expected_factor)
