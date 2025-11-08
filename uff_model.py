
# Implement your UFF circular velocity model here.
# Keep it pure: given radius R (kpc) and parameter vector theta, return V_circ (km/s).
# You can also add helper functions for potentials, mass profiles, etc.

import numpy as np

def v_circ_uff(R_kpc, theta):
    """
    Compute circular velocity (km/s) at radius R_kpc (array-like) under your UFF.

    Parameters
    ----------
    R_kpc : array-like
        Radii in kiloparsecs.
    theta : array-like
        Model parameters. Define them below and document thoroughly.
        Example placeholder: theta = [V0, Rc, beta]

    Returns
    -------
    v_kms : ndarray
        Circular velocity in km/s for each R.

    Notes
    -----
    Replace the placeholder below with **your** physics.
    The current placeholder is a pseudo-isothermal-like profile plus a mild power-law tweak.
    """
    R = np.asarray(R_kpc, dtype=float)
    # --- PLACEHOLDER MODEL (replace with UFF) ---
    V0, Rc, beta = theta  # interpret as you wish in UFF
    eps = 1e-9
    base = V0 * np.sqrt(1.0 - (Rc / np.sqrt(R**2 + Rc**2 + eps)))
    tweak = (R / (R + Rc + eps))**beta
    v = base * (1.0 + 0.2 * tweak)
    # Clip to non-negative
    return np.clip(v, 0, None)
