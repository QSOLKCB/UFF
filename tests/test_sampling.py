from __future__ import annotations

import numpy as np

from uff.data import GalaxyData
from uff.fitting import fit_model
from uff.models import ModelOptions, build_model, nfw_velocity_kms
from uff.sampling import sample_posterior


def _data_and_model():
    radius = np.geomspace(0.8, 20.0, 16)
    disk = 55.0 * np.sqrt(radius / 2.0) * np.exp(-radius / 7.0)
    gas = 12.0 * (1.0 - np.exp(-radius / 2.0))
    halo = nfw_velocity_kms(radius, 1.5e11, 9.0)
    observed = np.sqrt(gas**2 + 0.5 * disk**2 + halo**2)
    data = GalaxyData(
        radius_kpc=radius,
        velocity_obs_kms=observed,
        velocity_err_kms=np.full_like(radius, 2.0),
        velocity_gas_kms=gas,
        velocity_disk_kms=disk,
        velocity_bulge_kms=np.zeros_like(radius),
    )
    model = build_model("nfw", data, ModelOptions(fit_stellar_mass_to_light=False))
    return data, model


def test_sampler_is_deterministic_and_reports_diagnostics():
    data, model = _data_and_model()
    fit = fit_model(model, data, restarts=4, random_state=5)
    first = sample_posterior(
        model, data, fit, steps=300, burn=150, thin=5, n_chains=2, seed=9
    )
    second = sample_posterior(
        model, data, fit, steps=300, burn=150, thin=5, n_chains=2, seed=9
    )
    assert first.samples.shape == (2, 30, 2)
    assert np.array_equal(first.samples, second.samples)
    assert np.all(np.isfinite(first.samples))
    assert np.all(np.isfinite(first.rhat))
    assert np.all(first.effective_sample_size > 0)
    assert np.all((first.acceptance_rates >= 0) & (first.acceptance_rates <= 1))
