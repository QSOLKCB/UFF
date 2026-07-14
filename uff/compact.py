"""SMBH and loop-quantum-gravity scale diagnostics.

Galaxy rotation curves normally probe radii many orders of magnitude beyond a
black-hole horizon.  This module therefore keeps two regimes separate:

* a central point-mass contribution for weak-field galaxy mass models; and
* Kerr characteristic radii and LQG scale bookkeeping for compact-object work.

The LQG helper does **not** claim a unique LQG effective metric.  It exposes the
standard area-gap scale and a clearly labelled phenomenological suppression
factor so that speculative corrections cannot silently masquerade as a
galaxy-scale prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .constants import (
    C_M_S,
    DEFAULT_BARBERO_IMMIRZI,
    G_KPC_KMS2_MSUN,
    G_SI,
    KPC_TO_M,
    M_SUN_KG,
    PLANCK_LENGTH_M,
)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def smbh_velocity_kms(radius_kpc: np.ndarray, mass_msun: float) -> np.ndarray:
    """Return the weak-field point-mass circular speed ``sqrt(G M / r)``."""

    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("radius_kpc must contain finite positive values")
    mass = float(mass_msun)
    if not math.isfinite(mass) or mass < 0:
        raise ValueError("mass_msun must be finite and non-negative")
    if mass == 0:
        return np.zeros_like(radius)
    return np.sqrt(G_KPC_KMS2_MSUN * mass / radius)


def gravitational_radius_kpc(mass_msun: float) -> float:
    """Return ``r_g = GM/c^2`` in kpc (half the Schwarzschild radius)."""

    mass = _positive_finite(mass_msun, "mass_msun")
    return G_SI * mass * M_SUN_KG / C_M_S**2 / KPC_TO_M


def sphere_of_influence_kpc(mass_msun: float, velocity_dispersion_kms: float) -> float:
    """Return the conventional SMBH influence radius ``G M / sigma^2``."""

    mass = _positive_finite(mass_msun, "mass_msun")
    sigma = _positive_finite(velocity_dispersion_kms, "velocity_dispersion_kms")
    return G_KPC_KMS2_MSUN * mass / sigma**2


@dataclass(frozen=True)
class KerrRadii:
    """Equatorial Kerr radii in gravitational-radius and kpc units."""

    mass_msun: float
    dimensionless_spin: float
    gravitational_radius_kpc: float
    horizon_rg: float
    photon_orbit_rg: float
    isco_rg: float
    horizon_kpc: float
    photon_orbit_kpc: float
    isco_kpc: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def kerr_characteristic_radii(
    mass_msun: float, dimensionless_spin: float = 0.0
) -> KerrRadii:
    """Calculate the outer horizon, equatorial photon orbit, and ISCO.

    ``dimensionless_spin`` is signed relative to the orbit: positive values
    describe prograde motion and negative values retrograde motion.  The
    conservative Thorne limit ``|a*| <= 0.998`` is used by the CLI, while this
    low-level function permits the mathematical Kerr range ``|a*| <= 1``.
    """

    mass = _positive_finite(mass_msun, "mass_msun")
    spin = float(dimensionless_spin)
    if not math.isfinite(spin) or abs(spin) > 1.0:
        raise ValueError("dimensionless_spin must lie in [-1, 1]")

    rg = gravitational_radius_kpc(mass)
    horizon = 1.0 + math.sqrt(max(0.0, 1.0 - spin**2))
    photon = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(-spin)))

    z1 = 1.0 + np.cbrt(1.0 - spin**2) * (np.cbrt(1.0 + spin) + np.cbrt(1.0 - spin))
    z2 = math.sqrt(3.0 * spin**2 + z1**2)
    sign = 0.0 if spin == 0 else math.copysign(1.0, spin)
    isco = 3.0 + z2 - sign * math.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)))

    return KerrRadii(
        mass_msun=mass,
        dimensionless_spin=spin,
        gravitational_radius_kpc=rg,
        horizon_rg=horizon,
        photon_orbit_rg=photon,
        isco_rg=isco,
        horizon_kpc=horizon * rg,
        photon_orbit_kpc=photon * rg,
        isco_kpc=isco * rg,
    )


def lqg_area_gap_m2(
    barbero_immirzi: float = DEFAULT_BARBERO_IMMIRZI,
) -> float:
    """Return ``Delta = 4 sqrt(3) pi gamma l_P^2`` in square metres.

    This is a widely used convention.  Effective LQG papers can use different
    normalizations, so the convention and ``gamma`` are always reported.
    """

    gamma = _positive_finite(barbero_immirzi, "barbero_immirzi")
    return 4.0 * math.sqrt(3.0) * math.pi * gamma * PLANCK_LENGTH_M**2


def lqg_area_suppression(
    radius_kpc: float,
    *,
    barbero_immirzi: float = DEFAULT_BARBERO_IMMIRZI,
) -> float:
    """Return the dimensionless scale ratio ``Delta / r^2``.

    This is a scale diagnostic, not an LQG field equation.  At galactic radii
    it makes the enormous separation between the area gap and the observation
    scale explicit.
    """

    radius = _positive_finite(radius_kpc, "radius_kpc") * KPC_TO_M
    return lqg_area_gap_m2(barbero_immirzi) / radius**2


def phenomenological_lqg_velocity_kms(
    radius_kpc: np.ndarray,
    mass_msun: float,
    *,
    alpha: float = 0.0,
    power: float = 1.0,
    barbero_immirzi: float = DEFAULT_BARBERO_IMMIRZI,
) -> np.ndarray:
    """Apply an opt-in area-gap-suppressed correction to a point mass.

    The bookkeeping ansatz is

    ``V^2 = (GM/r) [1 + alpha (Delta/r^2)^power]``.

    It is deliberately *not* included in any default galaxy model.  Users who
    need a published effective LQG metric should implement that metric as a
    separately named model with its own citation and tests.
    """

    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    exponent = _positive_finite(power, "power")
    radius = np.asarray(radius_kpc, dtype=float)
    base = smbh_velocity_kms(radius, mass_msun)
    ratio = lqg_area_gap_m2(barbero_immirzi) / np.square(radius * KPC_TO_M)
    correction = 1.0 + alpha * np.power(ratio, exponent)
    if np.any(correction < 0):
        raise ValueError("the requested phenomenological correction makes V^2 negative")
    return base * np.sqrt(correction)


def compact_object_report(
    mass_msun: float,
    *,
    dimensionless_spin: float = 0.0,
    probe_radius_kpc: float | None = None,
    velocity_dispersion_kms: float | None = None,
    barbero_immirzi: float = DEFAULT_BARBERO_IMMIRZI,
) -> dict[str, object]:
    """Build a serializable, explicitly scoped compact-object report."""

    radii = kerr_characteristic_radii(mass_msun, dimensionless_spin)
    probe = probe_radius_kpc if probe_radius_kpc is not None else radii.isco_kpc
    report: dict[str, object] = {
        "regime": "Kerr baseline plus LQG scale diagnostic",
        "kerr": radii.to_dict(),
        "lqg": {
            "barbero_immirzi": float(barbero_immirzi),
            "area_gap_m2": lqg_area_gap_m2(barbero_immirzi),
            "probe_radius_kpc": float(probe),
            "area_gap_over_radius_squared": lqg_area_suppression(
                probe, barbero_immirzi=barbero_immirzi
            ),
            "status": "scale diagnostic; not a unique LQG effective metric",
        },
    }
    if velocity_dispersion_kms is not None:
        report["sphere_of_influence_kpc"] = sphere_of_influence_kpc(
            mass_msun, velocity_dispersion_kms
        )
    return report


__all__ = [
    "KerrRadii",
    "compact_object_report",
    "gravitational_radius_kpc",
    "kerr_characteristic_radii",
    "lqg_area_gap_m2",
    "lqg_area_suppression",
    "phenomenological_lqg_velocity_kms",
    "smbh_velocity_kms",
    "sphere_of_influence_kpc",
]
