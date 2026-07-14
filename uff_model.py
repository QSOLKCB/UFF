"""Compatibility API for the historical ``v_circ_uff`` function."""

from __future__ import annotations

import numpy as np

from uff.models import uff_empirical_velocity_kms


def v_circ_uff(R_kpc: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the v4 bounded empirical UFF law.

    ``theta`` remains ``(V_inf [km/s], R_core [kpc], beta)``.  Version 4
    intentionally changes the old placeholder expression; use a v3 tag when
    exact reproduction of historical placeholder outputs is required.
    """

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim != 1 or parameters.size < 3:
        raise ValueError("UFF parameter vector must contain V_inf, R_core, and beta")
    return uff_empirical_velocity_kms(R_kpc, *parameters[:3])


__all__ = ["v_circ_uff"]
