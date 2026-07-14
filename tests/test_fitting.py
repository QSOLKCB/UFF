from __future__ import annotations

import numpy as np

from uff.data import GalaxyData
from uff.fitting import fit_model, information_criteria, model_weights
from uff.models import ModelOptions, build_model, nfw_velocity_kms


def _synthetic_nfw_data() -> GalaxyData:
    radius = np.geomspace(0.5, 30.0, 28)
    gas = 18.0 * (1.0 - np.exp(-radius / 3.0))
    disk = 80.0 * np.sqrt(radius / 2.5) * np.exp(-radius / 8.0)
    bulge = 35.0 * np.exp(-radius / 1.5)
    baryon_v2 = gas**2 + 0.5 * disk**2 + 0.7 * bulge**2
    halo = nfw_velocity_kms(radius, 2.5e11, 11.0)
    observed = np.sqrt(baryon_v2 + halo**2)
    return GalaxyData(
        radius_kpc=radius,
        velocity_obs_kms=observed,
        velocity_err_kms=np.full_like(radius, 2.0),
        velocity_gas_kms=gas,
        velocity_disk_kms=disk,
        velocity_bulge_kms=bulge,
        name="SYNTHETIC_NFW",
    )


def test_nfw_fit_recovers_noise_free_synthetic_curve():
    data = _synthetic_nfw_data()
    options = ModelOptions(fit_stellar_mass_to_light=False)
    model = build_model("nfw", data, options)
    result = fit_model(model, data, restarts=8, random_state=7)
    assert result.success
    assert result.rmse_kms < 0.05
    assert abs(result.parameters["log10_m200"] - np.log10(2.5e11)) < 0.01
    assert abs(result.parameters["c200"] - 11.0) < 0.1


def test_information_criteria_and_weights():
    criteria = information_criteria(-10.0, 2, 20)
    assert criteria["aic"] == 24.0
    assert criteria["aicc"] > criteria["aic"]
    data = _synthetic_nfw_data()
    options = ModelOptions(fit_stellar_mass_to_light=False)
    nfw = fit_model(build_model("nfw", data, options), data, restarts=4)
    burkert = fit_model(build_model("burkert", data, options), data, restarts=4)
    weights = model_weights([nfw, burkert])
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["nfw"] > weights["burkert"]
