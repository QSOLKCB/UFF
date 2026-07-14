"""Validated galaxy rotation-curve data structures.

The loader accepts the repository's readable column names as well as the short
names used by SPARC rotation-curve tables.  It preserves the sign of ``Vgas``:
some SPARC radii contain a negative gas contribution, so gas must enter the
mass model as ``Vgas * abs(Vgas)`` rather than ``Vgas**2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = {
    "radius_kpc": ("R_kpc", "Rad", "radius_kpc", "radius", "R"),
    "velocity_obs_kms": ("V_obs_kms", "Vobs", "velocity_obs_kms", "V_obs"),
    "velocity_err_kms": ("e_V_kms", "errV", "velocity_err_kms", "eV"),
    "velocity_gas_kms": ("V_gas_kms", "Vgas", "velocity_gas_kms", "V_gas"),
    "velocity_disk_kms": ("V_disk_kms", "Vdisk", "velocity_disk_kms", "V_disk"),
    "velocity_bulge_kms": ("V_bul_kms", "Vbul", "velocity_bulge_kms", "V_bul"),
}

REQUIRED_FIELDS = ("radius_kpc", "velocity_obs_kms", "velocity_err_kms")


def _resolve_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    exact = {column: column for column in columns}
    folded = {column.casefold(): column for column in columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _as_float_vector(values: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} has {array.size} values; expected {length}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class GalaxyData:
    """A validated, radius-sorted rotation curve and its mass components."""

    radius_kpc: np.ndarray
    velocity_obs_kms: np.ndarray
    velocity_err_kms: np.ndarray
    velocity_gas_kms: np.ndarray
    velocity_disk_kms: np.ndarray
    velocity_bulge_kms: np.ndarray
    name: str = "galaxy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        radius = _as_float_vector(self.radius_kpc, "radius_kpc")
        n_points = radius.size
        if n_points < 3:
            raise ValueError("a rotation curve needs at least three points")

        vectors = {
            "velocity_obs_kms": _as_float_vector(
                self.velocity_obs_kms, "velocity_obs_kms", n_points
            ),
            "velocity_err_kms": _as_float_vector(
                self.velocity_err_kms, "velocity_err_kms", n_points
            ),
            "velocity_gas_kms": _as_float_vector(
                self.velocity_gas_kms, "velocity_gas_kms", n_points
            ),
            "velocity_disk_kms": _as_float_vector(
                self.velocity_disk_kms, "velocity_disk_kms", n_points
            ),
            "velocity_bulge_kms": _as_float_vector(
                self.velocity_bulge_kms, "velocity_bulge_kms", n_points
            ),
        }

        if np.any(radius <= 0):
            raise ValueError("all radii must be greater than zero")
        if np.any(vectors["velocity_obs_kms"] < 0):
            raise ValueError("observed circular velocities cannot be negative")
        if np.any(vectors["velocity_err_kms"] <= 0):
            raise ValueError("all velocity uncertainties must be greater than zero")

        order = np.argsort(radius, kind="stable")
        radius = radius[order]
        if np.any(np.diff(radius) <= 0):
            raise ValueError("radii must be unique")

        object.__setattr__(self, "radius_kpc", radius)
        for key, vector in vectors.items():
            object.__setattr__(self, key, vector[order])
        object.__setattr__(self, "name", str(self.name).strip() or "galaxy")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> "GalaxyData":
        """Load a canonical or SPARC-style CSV file.

        Missing gas, disk, or bulge columns are represented by zeros.  Missing
        baryonic data are recorded in ``metadata['missing_components']`` so a
        caller can decide whether a baryon-dependent model is meaningful.
        """

        csv_path = Path(path)
        frame = pd.read_csv(csv_path, comment="#")
        if frame.empty:
            raise ValueError(f"{csv_path} contains no data rows")

        resolved: dict[str, str | None] = {
            field_name: _resolve_column(list(frame.columns), aliases)
            for field_name, aliases in CANONICAL_COLUMNS.items()
        }
        missing_required = [
            field for field in REQUIRED_FIELDS if resolved[field] is None
        ]
        if missing_required:
            expected = ", ".join(
                CANONICAL_COLUMNS[field][0] for field in missing_required
            )
            raise ValueError(f"missing required column(s): {expected}")

        n_points = len(frame)
        values: dict[str, np.ndarray] = {}
        missing_components: list[str] = []
        for field_name in CANONICAL_COLUMNS:
            source = resolved[field_name]
            if source is None:
                values[field_name] = np.zeros(n_points, dtype=float)
                missing_components.append(field_name)
            else:
                try:
                    values[field_name] = pd.to_numeric(
                        frame[source], errors="raise"
                    ).to_numpy(dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"column {source!r} contains non-numeric values"
                    ) from exc

        inferred_name = name
        if inferred_name is None:
            for candidate in ("GALNAME", "galaxy", "Galaxy", "name"):
                if candidate in frame.columns and frame[candidate].notna().any():
                    inferred_name = str(frame[candidate].dropna().iloc[0])
                    break
        inferred_name = inferred_name or csv_path.stem

        metadata: dict[str, Any] = {
            "source_path": str(csv_path),
            "source_columns": list(frame.columns),
            "missing_components": missing_components,
        }
        for column in frame.columns:
            if column not in {
                value for value in resolved.values() if value is not None
            }:
                unique = frame[column].dropna().unique()
                if len(unique) == 1:
                    scalar = unique[0]
                    metadata[column] = (
                        scalar.item() if hasattr(scalar, "item") else scalar
                    )

        return cls(name=inferred_name, metadata=metadata, **values)

    @property
    def n_points(self) -> int:
        return int(self.radius_kpc.size)

    @property
    def has_disk(self) -> bool:
        return bool(np.any(np.abs(self.velocity_disk_kms) > 0))

    @property
    def has_bulge(self) -> bool:
        return bool(np.any(np.abs(self.velocity_bulge_kms) > 0))

    @property
    def has_gas(self) -> bool:
        return bool(np.any(np.abs(self.velocity_gas_kms) > 0))

    def components_at(
        self, radius_kpc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate gas, disk, and bulge velocities onto ``radius_kpc``."""

        radius = np.asarray(radius_kpc, dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("evaluation radii must be finite and positive")
        gas = np.interp(radius, self.radius_kpc, self.velocity_gas_kms)
        disk = np.interp(radius, self.radius_kpc, self.velocity_disk_kms)
        bulge = np.interp(radius, self.radius_kpc, self.velocity_bulge_kms)
        return gas, disk, bulge

    def baryonic_velocity_squared(
        self,
        radius_kpc: np.ndarray | None = None,
        *,
        disk_mass_to_light: float = 0.5,
        bulge_mass_to_light: float = 0.7,
        gas_scale: float = 1.0,
    ) -> np.ndarray:
        """Return the SPARC-convention baryonic contribution ``V_bar^2``.

        Stellar reference curves are scaled linearly in ``V^2`` by their
        mass-to-light ratios.  The gas term is sign preserving.
        """

        parameters = (disk_mass_to_light, bulge_mass_to_light, gas_scale)
        if any(not np.isfinite(value) or value < 0 for value in parameters):
            raise ValueError(
                "mass-to-light ratios and gas_scale must be finite and non-negative"
            )
        radius = (
            self.radius_kpc
            if radius_kpc is None
            else np.asarray(radius_kpc, dtype=float)
        )
        gas, disk, bulge = self.components_at(radius)
        velocity_squared = (
            gas_scale * gas * np.abs(gas)
            + disk_mass_to_light * np.square(disk)
            + bulge_mass_to_light * np.square(bulge)
        )
        # Negative gas terms can dominate at a small number of inner radii.
        # A negative total V^2 cannot be passed to an algebraic acceleration
        # relation, so the physically admissible floor is explicit.
        return np.maximum(velocity_squared, 0.0)


def load_galaxy_csv(path: str | Path, name: str | None = None) -> GalaxyData:
    """Convenience wrapper around :meth:`GalaxyData.from_csv`."""

    return GalaxyData.from_csv(path, name=name)


__all__ = ["CANONICAL_COLUMNS", "GalaxyData", "load_galaxy_csv"]
