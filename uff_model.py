"""
Unified Field Framework circular velocity model.

This module defines the base circular velocity law used in the UFF analysis.
It intentionally keeps the core physics separated from observational
extensions such as baryonic scaling and dark‑field contributions, which
are implemented in `analyze_sparc.py`.

The default implementation below provides a pseudo‑isothermal-like profile
with a mild power‑law tweak. Users are encouraged to replace it with
their own analytic expressions appropriate to their variant of the
Unified Field Framework. The function must accept a radius array and
a parameter vector and return a velocity array of matching shape.
"""

import numpy as np

def v_circ_uff(R_kpc: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute the base circular velocity under the UFF model.

    Parameters
    ----------
    R_kpc : array-like
        Radii in kiloparsecs.
    theta : array-like
        Model parameters (e.g. V0, Rc, beta). Only the first three
        elements are used by this placeholder implementation.

    Returns
    -------
    ndarray
        Circular velocity in km/s for each radius.

    Notes
    -----
    The current implementation is a placeholder combining a core curve and
    a mild power‑law tweak. It is clipped to be non‑negative. Feel free
    to modify or replace this function with your own physics.
    """
    R = np.asarray(R_kpc, dtype=float)
    if len(theta) < 3:
        raise ValueError("UFF parameter vector must have at least 3 entries")
    V0, Rc, beta = theta[:3]
    eps = 1e-9
    base = V0 * np.sqrt(1.0 - (Rc / np.sqrt(R**2 + Rc**2 + eps)))
    tweak = (R / (R + Rc + eps)) ** beta
    v = base * (1.0 + 0.2 * tweak)
    return np.clip(v, 0, None)