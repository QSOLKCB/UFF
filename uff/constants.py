"""Physical constants and unit conversions used by :mod:`uff`.

The rotation-curve code works in kpc, km/s, and solar masses.  SI values are
kept here for acceleration scales and compact-object diagnostics.  Values are
explicit rather than imported from a large astronomy dependency so that model
outputs remain easy to reproduce.
"""

from __future__ import annotations

import math

# CODATA-compatible values (the precision shown is more than sufficient for
# galactic rotation-curve work).
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
C_M_S = 299_792_458.0
C_KM_S = C_M_S / 1_000.0
M_SUN_KG = 1.98847e30
KPC_TO_M = 3.085677581491367e19
MPC_TO_M = KPC_TO_M * 1_000.0
YEAR_TO_S = 31_557_600.0
PLANCK_LENGTH_M = 1.616255e-35

# G in the natural units used by the galaxy models.
G_KPC_KMS2_MSUN = G_SI * M_SUN_KG / (KPC_TO_M * 1.0e6)

# Common defaults.  Both are configurable at the public API and CLI.
DEFAULT_A0_M_S2 = 1.2e-10
DEFAULT_H0_KM_S_MPC = 70.0
DEFAULT_BARBERO_IMMIRZI = 0.2375


def critical_density_msun_kpc3(
    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC,
) -> float:
    """Return the redshift-zero critical density in ``M_sun / kpc^3``.

    Parameters
    ----------
    h0_km_s_mpc:
        Hubble constant in km/s/Mpc.  It must be finite and positive.
    """

    if not math.isfinite(h0_km_s_mpc) or h0_km_s_mpc <= 0:
        raise ValueError("H0 must be finite and positive")
    h0_km_s_kpc = h0_km_s_mpc / 1_000.0
    return 3.0 * h0_km_s_kpc**2 / (8.0 * math.pi * G_KPC_KMS2_MSUN)


__all__ = [
    "C_KM_S",
    "C_M_S",
    "DEFAULT_A0_M_S2",
    "DEFAULT_BARBERO_IMMIRZI",
    "DEFAULT_H0_KM_S_MPC",
    "G_KPC_KMS2_MSUN",
    "G_SI",
    "KPC_TO_M",
    "MPC_TO_M",
    "M_SUN_KG",
    "PLANCK_LENGTH_M",
    "YEAR_TO_S",
    "critical_density_msun_kpc3",
]
