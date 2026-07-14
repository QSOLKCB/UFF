"""Dimensionally explicit galaxy rotation-curve models.

The module distinguishes observational inputs, established baselines,
phenomenological MOND relations, and the repository's empirical UFF extension.
No model is presented as a proof of a fundamental theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np

from .compact import smbh_velocity_kms
from .constants import (
    DEFAULT_A0_M_S2,
    DEFAULT_H0_KM_S_MPC,
    G_KPC_KMS2_MSUN,
    KPC_TO_M,
    critical_density_msun_kpc3,
)
from .data import GalaxyData


Array = np.ndarray


def _radius_array(radius_kpc: Array) -> Array:
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("radius_kpc must contain finite positive values")
    return radius


def _nfw_shape(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1.0e-4
    result = np.empty_like(x)
    xs = x[small]
    result[small] = 0.5 * xs**2 - (2.0 / 3.0) * xs**3 + 0.75 * xs**4
    result[~small] = np.log1p(x[~small]) - x[~small] / (1.0 + x[~small])
    return result


def nfw_r200_kpc(
    mass_200_msun: float,
    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC,
) -> float:
    """Return ``r200`` for a halo defined at 200 times critical density."""

    mass = float(mass_200_msun)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("mass_200_msun must be finite and positive")
    rho_critical = critical_density_msun_kpc3(h0_km_s_mpc)
    return (3.0 * mass / (4.0 * math.pi * 200.0 * rho_critical)) ** (1.0 / 3.0)


def nfw_enclosed_mass_msun(
    radius_kpc: Array,
    mass_200_msun: float,
    concentration_200: float,
    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC,
) -> Array:
    """Return the enclosed mass of an NFW halo parameterized by ``M200,c200``."""

    radius = _radius_array(radius_kpc)
    concentration = float(concentration_200)
    if not math.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration_200 must be finite and positive")
    r200 = nfw_r200_kpc(mass_200_msun, h0_km_s_mpc)
    x = concentration * radius / r200
    normalization = _nfw_shape(np.array(concentration)).item()
    return float(mass_200_msun) * _nfw_shape(x) / normalization


def nfw_velocity_kms(
    radius_kpc: Array,
    mass_200_msun: float,
    concentration_200: float,
    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC,
) -> Array:
    """Return the physical NFW circular speed in km/s."""

    radius = _radius_array(radius_kpc)
    enclosed_mass = nfw_enclosed_mass_msun(
        radius, mass_200_msun, concentration_200, h0_km_s_mpc
    )
    return np.sqrt(G_KPC_KMS2_MSUN * enclosed_mass / radius)


def burkert_enclosed_mass_msun(
    radius_kpc: Array,
    central_density_msun_kpc3: float,
    core_radius_kpc: float,
) -> Array:
    """Return enclosed mass for ``rho=rho0/[(1+x)(1+x^2)]``."""

    radius = _radius_array(radius_kpc)
    density = float(central_density_msun_kpc3)
    core = float(core_radius_kpc)
    if not math.isfinite(density) or density <= 0:
        raise ValueError("central_density_msun_kpc3 must be finite and positive")
    if not math.isfinite(core) or core <= 0:
        raise ValueError("core_radius_kpc must be finite and positive")
    x = radius / core
    shape = np.empty_like(x)
    small = x < 1.0e-3
    shape[small] = (4.0 / 3.0) * x[small] ** 3
    shape[~small] = np.log(
        np.square(1.0 + x[~small]) * (1.0 + np.square(x[~small]))
    ) - 2.0 * np.arctan(x[~small])
    return math.pi * density * core**3 * shape


def burkert_velocity_kms(
    radius_kpc: Array,
    central_density_msun_kpc3: float,
    core_radius_kpc: float,
) -> Array:
    """Return the Burkert-halo circular speed in km/s."""

    radius = _radius_array(radius_kpc)
    mass = burkert_enclosed_mass_msun(
        radius, central_density_msun_kpc3, core_radius_kpc
    )
    return np.sqrt(G_KPC_KMS2_MSUN * mass / radius)


def uff_empirical_velocity_kms(
    radius_kpc: Array,
    asymptotic_velocity_kms: float,
    core_radius_kpc: float,
    shape_beta: float = 0.0,
) -> Array:
    """Return the bounded v4 empirical UFF extra-field velocity.

    The base is the circular-speed law of a cored pseudo-isothermal profile.
    ``shape_beta`` applies a bounded exponential deformation.  This is an
    empirical test law; it is not derived from a covariant UFF action.
    """

    radius = _radius_array(radius_kpc)
    velocity = float(asymptotic_velocity_kms)
    core = float(core_radius_kpc)
    beta = float(shape_beta)
    if not math.isfinite(velocity) or velocity < 0:
        raise ValueError("asymptotic_velocity_kms must be finite and non-negative")
    if not math.isfinite(core) or core <= 0:
        raise ValueError("core_radius_kpc must be finite and positive")
    if not math.isfinite(beta):
        raise ValueError("shape_beta must be finite")
    x = radius / core
    base_squared = np.maximum(1.0 - np.arctan(x) / x, 0.0)
    bounded_coordinate = x / (1.0 + x)
    return velocity * np.sqrt(base_squared) * np.exp(beta * bounded_coordinate)


def mond_nu(y: Array, relation: str = "rar") -> Array:
    """Return a MOND/RAR boost ``nu(y)`` for ``y=g_N/a0``.

    Supported relations are ``simple``, ``standard``, and ``rar`` (the
    empirical McGaugh-Lelli-Schombert exponential relation).
    """

    y = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("y must be finite and non-negative")
    safe = np.maximum(y, np.finfo(float).tiny)
    key = relation.casefold().replace("mond-", "")
    if key == "simple":
        result = 0.5 + np.sqrt(0.25 + 1.0 / safe)
    elif key == "standard":
        result = np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.square(safe)))
    elif key in {"rar", "exponential"}:
        result = 1.0 / (-np.expm1(-np.sqrt(safe)))
    else:
        raise ValueError(f"unknown MOND relation: {relation}")
    return result


def mond_acceleration_m_s2(
    newtonian_acceleration_m_s2: Array,
    *,
    a0_m_s2: float = DEFAULT_A0_M_S2,
    relation: str = "rar",
    external_field_a0: float = 0.0,
    external_field_angle_deg: float = 0.0,
) -> Array:
    """Map Newtonian baryonic acceleration to a MOND/RAR acceleration.

    With ``external_field_a0 > 0`` this uses a scalar/vector-aligned algebraic
    QUMOND proxy.  It is useful for sensitivity tests but is not a substitute
    for solving AQUAL/QUMOND for a non-spherical galaxy and its environment.
    """

    g_newton = np.asarray(newtonian_acceleration_m_s2, dtype=float)
    if np.any(~np.isfinite(g_newton)) or np.any(g_newton < 0):
        raise ValueError("newtonian acceleration must be finite and non-negative")
    a0 = float(a0_m_s2)
    external = float(external_field_a0)
    angle = float(external_field_angle_deg)
    if not math.isfinite(a0) or a0 <= 0:
        raise ValueError("a0_m_s2 must be finite and positive")
    if not math.isfinite(external) or external < 0:
        raise ValueError("external_field_a0 must be finite and non-negative")
    if not math.isfinite(angle):
        raise ValueError("external_field_angle_deg must be finite")

    if external == 0:
        return g_newton * mond_nu(g_newton / a0, relation)

    g_external = external * a0
    cosine = math.cos(math.radians(angle))
    total_magnitude = np.sqrt(
        np.maximum(
            g_newton**2 + g_external**2 + 2.0 * g_newton * g_external * cosine, 0.0
        )
    )
    nu_total = mond_nu(total_magnitude / a0, relation)
    nu_external = float(mond_nu(np.array([external]), relation)[0])
    radial = nu_total * g_newton + (nu_total - nu_external) * g_external * cosine
    return np.maximum(radial, 0.0)


def mond_velocity_kms(
    radius_kpc: Array,
    newtonian_velocity_squared_kms2: Array,
    *,
    a0_m_s2: float = DEFAULT_A0_M_S2,
    relation: str = "rar",
    external_field_a0: float = 0.0,
    external_field_angle_deg: float = 0.0,
) -> Array:
    """Return a MOND/RAR circular speed from a Newtonian ``V^2`` profile."""

    radius = _radius_array(radius_kpc)
    velocity_squared = np.asarray(newtonian_velocity_squared_kms2, dtype=float)
    if velocity_squared.shape != radius.shape:
        raise ValueError("newtonian_velocity_squared_kms2 must match radius_kpc")
    if np.any(~np.isfinite(velocity_squared)) or np.any(velocity_squared < 0):
        raise ValueError("newtonian V^2 must be finite and non-negative")
    g_newton = velocity_squared * 1.0e6 / (radius * KPC_TO_M)
    g_mond = mond_acceleration_m_s2(
        g_newton,
        a0_m_s2=a0_m_s2,
        relation=relation,
        external_field_a0=external_field_a0,
        external_field_angle_deg=external_field_angle_deg,
    )
    return np.sqrt(g_mond * radius * KPC_TO_M) / 1.0e3


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        values = (self.initial, self.lower, self.upper)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(
                f"parameter {self.name} has a non-finite bound or initial value"
            )
        if not self.lower < self.upper:
            raise ValueError(f"parameter {self.name} requires lower < upper")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError(
                f"parameter {self.name} initial value lies outside its bounds"
            )


@dataclass(frozen=True)
class ModelOptions:
    """Configuration shared by model builders and the CLI."""

    h0_km_s_mpc: float = DEFAULT_H0_KM_S_MPC
    a0_m_s2: float = DEFAULT_A0_M_S2
    fit_a0: bool = False
    disk_mass_to_light: float = 0.5
    bulge_mass_to_light: float = 0.7
    fit_stellar_mass_to_light: bool = True
    distance_scale: float = 1.0
    fit_distance_scale: bool = False
    reference_inclination_deg: float | None = None
    fit_inclination: bool = False
    smbh_mass_msun: float = 0.0
    fit_smbh: bool = False
    external_field_a0: float = 0.0
    external_field_angle_deg: float = 0.0


@dataclass
class RotationCurveModel:
    """A named prediction function and its bounded free parameters."""

    name: str
    label: str
    family: str
    parameters: tuple[ParameterSpec, ...]
    prediction: Callable[[Array, Mapping[str, float]], Array] = field(repr=False)
    status: str = "baseline"
    notes: str = ""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def initial(self) -> Array:
        return np.array(
            [parameter.initial for parameter in self.parameters], dtype=float
        )

    @property
    def lower_bounds(self) -> Array:
        return np.array([parameter.lower for parameter in self.parameters], dtype=float)

    @property
    def upper_bounds(self) -> Array:
        return np.array([parameter.upper for parameter in self.parameters], dtype=float)

    def parameter_dict(self, theta: Array | Mapping[str, float]) -> dict[str, float]:
        if isinstance(theta, Mapping):
            missing = [name for name in self.parameter_names if name not in theta]
            if missing:
                raise ValueError(
                    f"missing parameter(s) for {self.name}: {', '.join(missing)}"
                )
            return {name: float(theta[name]) for name in self.parameter_names}
        vector = np.asarray(theta, dtype=float)
        if vector.shape != (len(self.parameters),):
            raise ValueError(
                f"{self.name} expects {len(self.parameters)} parameters; got shape {vector.shape}"
            )
        return dict(zip(self.parameter_names, map(float, vector)))

    def predict(self, radius_kpc: Array, theta: Array | Mapping[str, float]) -> Array:
        radius = _radius_array(radius_kpc)
        values = self.prediction(radius, self.parameter_dict(theta))
        prediction = np.asarray(values, dtype=float)
        if prediction.shape != radius.shape:
            raise ValueError(f"{self.name} returned the wrong prediction shape")
        if np.any(~np.isfinite(prediction)) or np.any(prediction < 0):
            raise ValueError(f"{self.name} produced an invalid velocity")
        return prediction


MODEL_ALIASES = {
    "newtonian": "baryons",
    "baryon": "baryons",
    "lcdm": "nfw",
    "lambda-cdm": "nfw",
    "mond": "mond-rar",
    "rar": "mond-rar",
    "simple": "mond-simple",
    "standard": "mond-standard",
    "uff": "uff-empirical",
}


def available_models() -> tuple[str, ...]:
    return (
        "baryons",
        "nfw",
        "burkert",
        "mond-rar",
        "mond-simple",
        "mond-standard",
        "mond-efe",
        "uff-empirical",
    )


def build_model(
    name: str,
    data: GalaxyData,
    options: ModelOptions | None = None,
) -> RotationCurveModel:
    """Build a model bound to one galaxy's baryonic component curves."""

    options = options or ModelOptions()
    key = name.casefold().strip()
    key = MODEL_ALIASES.get(key, key)
    if key not in available_models():
        raise ValueError(
            f"unknown model {name!r}; choose from {', '.join(available_models())}"
        )

    parameters: list[ParameterSpec] = []
    if not math.isfinite(options.distance_scale) or options.distance_scale <= 0:
        raise ValueError("distance_scale must be finite and positive")
    if options.fit_distance_scale:
        parameters.append(
            ParameterSpec("distance_scale", options.distance_scale, 0.5, 1.5, "D/D_ref")
        )

    reference_inclination = options.reference_inclination_deg
    if reference_inclination is None:
        for metadata_name in ("INC_deg", "inclination_deg", "Inc"):
            if metadata_name in data.metadata:
                try:
                    reference_inclination = float(data.metadata[metadata_name])
                except (TypeError, ValueError):
                    pass
                break
    if reference_inclination is not None and (
        not math.isfinite(reference_inclination) or not 0 < reference_inclination <= 90
    ):
        raise ValueError("reference inclination must lie in (0, 90] degrees")
    if options.fit_inclination:
        if reference_inclination is None:
            raise ValueError(
                "fit_inclination requires reference_inclination_deg or INC_deg metadata"
            )
        lower_inclination = max(10.0, reference_inclination - 15.0)
        upper_inclination = min(89.5, reference_inclination + 15.0)
        if lower_inclination >= upper_inclination:
            raise ValueError(
                "inclination bounds collapsed; provide a reference away from 0/90 degrees"
            )
        parameters.append(
            ParameterSpec(
                "inclination_deg",
                float(
                    np.clip(reference_inclination, lower_inclination, upper_inclination)
                ),
                lower_inclination,
                upper_inclination,
                "deg",
            )
        )
    if options.fit_stellar_mass_to_light and data.has_disk:
        parameters.append(
            ParameterSpec(
                "disk_ml", options.disk_mass_to_light, 0.05, 1.5, "M_sun/L_sun"
            )
        )
    if options.fit_stellar_mass_to_light and data.has_bulge:
        parameters.append(
            ParameterSpec(
                "bulge_ml", options.bulge_mass_to_light, 0.05, 2.0, "M_sun/L_sun"
            )
        )
    if options.fit_smbh:
        parameters.append(ParameterSpec("log10_mbh", 7.0, 3.0, 11.5, "log10(M_sun)"))

    def distance_scale(values: Mapping[str, float]) -> float:
        return values.get("distance_scale", options.distance_scale)

    def physical_radius(radius: Array, values: Mapping[str, float]) -> Array:
        return radius * distance_scale(values)

    def projection_factor(values: Mapping[str, float]) -> float:
        if not options.fit_inclination:
            return 1.0
        assert reference_inclination is not None
        fitted = values["inclination_deg"]
        return math.sin(math.radians(fitted)) / math.sin(
            math.radians(reference_inclination)
        )

    def project_velocity(velocity: Array, values: Mapping[str, float]) -> Array:
        return velocity * projection_factor(values)

    def baryonic_v2(radius: Array, values: Mapping[str, float]) -> Array:
        # SPARC component speeds scale as sqrt(D/D_ref), hence V^2 scales
        # linearly with distance.  ``radius`` remains the catalog coordinate;
        # halo and SMBH laws are evaluated at the rescaled physical radius.
        return distance_scale(values) * data.baryonic_velocity_squared(
            radius,
            disk_mass_to_light=values.get("disk_ml", options.disk_mass_to_light),
            bulge_mass_to_light=values.get("bulge_ml", options.bulge_mass_to_light),
        )

    def central_v2(radius: Array, values: Mapping[str, float]) -> Array:
        mass = (
            10.0 ** values["log10_mbh"] if options.fit_smbh else options.smbh_mass_msun
        )
        return np.square(smbh_velocity_kms(physical_radius(radius, values), mass))

    if key == "baryons":

        def predict(radius: Array, values: Mapping[str, float]) -> Array:
            physical = np.sqrt(baryonic_v2(radius, values) + central_v2(radius, values))
            return project_velocity(physical, values)

        return RotationCurveModel(
            key,
            "Newtonian baryons",
            "Newtonian",
            tuple(parameters),
            predict,
            notes="SPARC baryons plus an optional central point mass",
        )

    if key == "nfw":
        parameters.extend(
            [
                ParameterSpec("log10_m200", 11.5, 8.0, 14.5, "log10(M_sun)"),
                ParameterSpec("c200", 10.0, 1.0, 40.0, ""),
            ]
        )

        def predict(radius: Array, values: Mapping[str, float]) -> Array:
            halo = nfw_velocity_kms(
                physical_radius(radius, values),
                10.0 ** values["log10_m200"],
                values["c200"],
                options.h0_km_s_mpc,
            )
            physical = np.sqrt(
                baryonic_v2(radius, values) + central_v2(radius, values) + halo**2
            )
            return project_velocity(physical, values)

        return RotationCurveModel(
            key,
            "NFW + baryons",
            "Lambda-CDM halo",
            tuple(parameters),
            predict,
            notes="M200 is defined at 200 times the critical density",
        )

    if key == "burkert":
        parameters.extend(
            [
                ParameterSpec("log10_rho0", 7.5, 4.0, 11.0, "log10(M_sun/kpc^3)"),
                ParameterSpec("core_radius_kpc", 5.0, 0.05, 100.0, "kpc"),
            ]
        )

        def predict(radius: Array, values: Mapping[str, float]) -> Array:
            halo = burkert_velocity_kms(
                physical_radius(radius, values),
                10.0 ** values["log10_rho0"],
                values["core_radius_kpc"],
            )
            physical = np.sqrt(
                baryonic_v2(radius, values) + central_v2(radius, values) + halo**2
            )
            return project_velocity(physical, values)

        return RotationCurveModel(
            key,
            "Burkert + baryons",
            "Cored dark-matter halo",
            tuple(parameters),
            predict,
            notes="Empirical cored halo baseline",
        )

    if key.startswith("mond-"):
        relation = key.removeprefix("mond-")
        if relation == "efe":
            relation = "rar"
            if options.external_field_a0 <= 0:
                raise ValueError("mond-efe requires external_field_a0 > 0")
        if options.fit_a0:
            parameters.append(
                ParameterSpec(
                    "log10_a0", math.log10(options.a0_m_s2), -11.5, -9.2, "log10(m/s^2)"
                )
            )

        def predict(radius: Array, values: Mapping[str, float]) -> Array:
            a0 = 10.0 ** values["log10_a0"] if options.fit_a0 else options.a0_m_s2
            newtonian_v2 = baryonic_v2(radius, values) + central_v2(radius, values)
            external = options.external_field_a0 if key == "mond-efe" else 0.0
            return mond_velocity_kms(
                physical_radius(radius, values),
                newtonian_v2,
                a0_m_s2=a0,
                relation=relation,
                external_field_a0=external,
                external_field_angle_deg=options.external_field_angle_deg,
            ) * projection_factor(values)

        labels = {
            "mond-rar": "MOND/RAR exponential",
            "mond-simple": "MOND simple mu",
            "mond-standard": "MOND standard mu",
            "mond-efe": "MOND/RAR + EFE proxy",
        }
        return RotationCurveModel(
            key,
            labels[key],
            "MOND phenomenology",
            tuple(parameters),
            predict,
            status="approximate" if key == "mond-efe" else "phenomenological baseline",
            notes=(
                "Algebraic external-field sensitivity proxy; not a full AQUAL/QUMOND solution"
                if key == "mond-efe"
                else "Algebraic acceleration relation"
            ),
        )

    # Repository-specific empirical model.  It is intentionally last and
    # explicitly labelled, so users do not confuse it with a standard theory.
    parameters.extend(
        [
            ParameterSpec("v_inf_kms", 120.0, 1.0, 500.0, "km/s"),
            ParameterSpec("uff_core_kpc", 3.0, 0.02, 100.0, "kpc"),
            ParameterSpec("uff_beta", 0.0, -1.0, 1.0, ""),
        ]
    )

    def predict(radius: Array, values: Mapping[str, float]) -> Array:
        extra = uff_empirical_velocity_kms(
            physical_radius(radius, values),
            values["v_inf_kms"],
            values["uff_core_kpc"],
            values["uff_beta"],
        )
        physical = np.sqrt(
            baryonic_v2(radius, values) + central_v2(radius, values) + extra**2
        )
        return project_velocity(physical, values)

    return RotationCurveModel(
        key,
        "UFF empirical v4 + baryons",
        "UFF empirical extension",
        tuple(parameters),
        predict,
        status="research model",
        notes="Bounded cored velocity law; no claim of a covariant field derivation",
    )


__all__ = [
    "MODEL_ALIASES",
    "ModelOptions",
    "ParameterSpec",
    "RotationCurveModel",
    "available_models",
    "build_model",
    "burkert_enclosed_mass_msun",
    "burkert_velocity_kms",
    "mond_acceleration_m_s2",
    "mond_nu",
    "mond_velocity_kms",
    "nfw_enclosed_mass_msun",
    "nfw_r200_kpc",
    "nfw_velocity_kms",
    "uff_empirical_velocity_kms",
]
