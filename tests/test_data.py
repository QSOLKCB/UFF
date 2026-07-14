from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uff.data import GalaxyData


def test_loader_accepts_sparc_aliases_and_sorts(tmp_path):
    path = tmp_path / "alias.csv"
    pd.DataFrame(
        {
            "Rad": [2.0, 0.5, 1.0],
            "Vobs": [80.0, 30.0, 55.0],
            "errV": [3.0, 2.0, 2.5],
            "Vgas": [8.0, -3.0, 5.0],
            "Vdisk": [40.0, 20.0, 35.0],
            "Vbul": [0.0, 0.0, 0.0],
            "Galaxy": ["TEST"] * 3,
        }
    ).to_csv(path, index=False)

    data = GalaxyData.from_csv(path)
    assert data.name == "TEST"
    assert np.allclose(data.radius_kpc, [0.5, 1.0, 2.0])
    assert np.allclose(data.velocity_gas_kms, [-3.0, 5.0, 8.0])


def test_baryonic_v2_preserves_negative_gas_sign():
    data = GalaxyData(
        radius_kpc=np.array([1.0, 2.0, 3.0]),
        velocity_obs_kms=np.array([10.0, 10.0, 10.0]),
        velocity_err_kms=np.ones(3),
        velocity_gas_kms=np.array([-3.0, 4.0, 5.0]),
        velocity_disk_kms=np.array([5.0, 5.0, 5.0]),
        velocity_bulge_kms=np.zeros(3),
    )
    v2 = data.baryonic_velocity_squared(disk_mass_to_light=1.0)
    assert np.allclose(v2, [16.0, 41.0, 50.0])


def test_missing_optional_components_are_zero(tmp_path):
    path = tmp_path / "minimal.csv"
    pd.DataFrame(
        {"R_kpc": [1, 2, 3], "V_obs_kms": [10, 20, 30], "e_V_kms": [1, 1, 1]}
    ).to_csv(path, index=False)
    data = GalaxyData.from_csv(path)
    assert not data.has_gas
    assert not data.has_disk
    assert not data.has_bulge
    assert len(data.metadata["missing_components"]) == 3


@pytest.mark.parametrize(
    "radius,error",
    [([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]), ([1.0, 2.0, 3.0], [1.0, 0.0, 1.0])],
)
def test_validation_rejects_nonphysical_values(radius, error):
    with pytest.raises(ValueError):
        GalaxyData(
            radius_kpc=np.array(radius),
            velocity_obs_kms=np.ones(3),
            velocity_err_kms=np.array(error),
            velocity_gas_kms=np.zeros(3),
            velocity_disk_kms=np.zeros(3),
            velocity_bulge_kms=np.zeros(3),
        )
