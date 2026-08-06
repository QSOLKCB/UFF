"""Spherical geometry and TFT-inspired invariance checks for UFF-SLFA."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

ROTATION_TOLERANCE = 1.0e-10
GRAM_TOLERANCE = 1.0e-10
ANGLE_TOLERANCE_RAD = 1.0e-10


class GeometryError(RuntimeError):
    """Raised when a proposed null transform deforms the frozen lattice."""


@dataclass(frozen=True, slots=True)
class InvarianceResiduals:
    orthogonality_frobenius: float
    determinant_abs_error: float
    gram_max_abs: float
    pairwise_angle_max_abs_rad: float

    def to_dict(self) -> dict[str, float]:
        return {
            "orthogonality_frobenius": self.orthogonality_frobenius,
            "determinant_abs_error": self.determinant_abs_error,
            "gram_max_abs": self.gram_max_abs,
            "pairwise_angle_max_abs_rad": self.pairwise_angle_max_abs_rad,
        }


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert ICRS right ascension/declination in degrees to unit vectors."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    vectors = np.column_stack(
        (
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        )
    )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1.0, rtol=0.0, atol=1e-12):
        raise GeometryError("RA/Dec conversion did not produce finite unit vectors")
    return vectors


def unit_to_radec(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert finite vectors to ICRS right ascension/declination in degrees."""
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise GeometryError("vectors must be a finite array with shape (n, 3)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0):
        raise GeometryError("zero vectors cannot be converted to sky coordinates")
    unit = array / norms[:, None]
    ra = np.rad2deg(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    return ra, dec


def cap_membership(
    catalogue_vectors: np.ndarray,
    node_vectors: np.ndarray,
    radius_deg: float,
) -> np.ndarray:
    """Return a boolean catalogue-by-node spherical-cap membership matrix."""
    catalogue = np.asarray(catalogue_vectors, dtype=float)
    nodes = np.asarray(node_vectors, dtype=float)
    if catalogue.ndim != 2 or catalogue.shape[1] != 3:
        raise GeometryError("catalogue vectors must have shape (n, 3)")
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 1:
        raise GeometryError("node vectors must have shape (m, 3) with m >= 1")
    if not 0.0 < float(radius_deg) <= 45.0:
        raise GeometryError("spherical-cap radius must be in (0, 45] degrees")
    return catalogue @ nodes.T >= math.cos(math.radians(float(radius_deg)))


def random_so3(rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-uniform proper rotation using a random unit quaternion."""
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def ra_shift_rotation(angle_rad: float) -> np.ndarray:
    """Return a proper rotation about the ICRS z axis."""
    angle = float(angle_rad)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def pairwise_angles(vectors: np.ndarray) -> np.ndarray:
    """Return stable upper-triangular angular separations in radians."""
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise GeometryError("vectors must have shape (n, 3)")
    values: list[float] = []
    for left in range(array.shape[0]):
        for right in range(left + 1, array.shape[0]):
            dot = float(np.clip(np.dot(array[left], array[right]), -1.0, 1.0))
            cross_norm = float(np.linalg.norm(np.cross(array[left], array[right])))
            values.append(math.atan2(cross_norm, dot))
    return np.asarray(values, dtype=float)


def validate_proper_rotation(
    rotation: np.ndarray,
    *,
    tolerance: float = ROTATION_TOLERANCE,
) -> tuple[float, float]:
    """Validate R^T R = I and det(R) = +1; return residuals."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise GeometryError("rotation must be a finite 3x3 matrix")
    orthogonality = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant_error = abs(float(np.linalg.det(matrix)) - 1.0)
    if orthogonality > tolerance or determinant_error > tolerance:
        raise GeometryError(
            "null transform is not a proper rotation: "
            f"orthogonality={orthogonality:.3e}, det_error={determinant_error:.3e}"
        )
    return orthogonality, determinant_error


def validate_lattice_invariance(
    original_nodes: np.ndarray,
    transformed_nodes: np.ndarray,
    rotation: np.ndarray,
    *,
    rotation_tolerance: float = ROTATION_TOLERANCE,
    gram_tolerance: float = GRAM_TOLERANCE,
    angle_tolerance_rad: float = ANGLE_TOLERANCE_RAD,
) -> InvarianceResiduals:
    """Verify that a proper rotation preserves the frozen lattice geometry."""
    original = np.asarray(original_nodes, dtype=float)
    transformed = np.asarray(transformed_nodes, dtype=float)
    if original.shape != transformed.shape or original.ndim != 2 or original.shape[1] != 3:
        raise GeometryError("original and transformed nodes must have matching shape (n, 3)")
    orthogonality, determinant_error = validate_proper_rotation(
        rotation, tolerance=rotation_tolerance
    )
    expected = original @ np.asarray(rotation, dtype=float).T
    mapping_error = float(np.max(np.abs(expected - transformed)))
    if mapping_error > gram_tolerance:
        raise GeometryError(f"transformed nodes do not equal the declared rotation: {mapping_error:.3e}")
    gram_error = float(np.max(np.abs(original @ original.T - transformed @ transformed.T)))
    angle_error = float(
        np.max(np.abs(pairwise_angles(original) - pairwise_angles(transformed)))
        if original.shape[0] > 1
        else 0.0
    )
    if gram_error > gram_tolerance or angle_error > angle_tolerance_rad:
        raise GeometryError(
            "null transform deformed the frozen lattice: "
            f"gram={gram_error:.3e}, angle={angle_error:.3e} rad"
        )
    return InvarianceResiduals(
        orthogonality_frobenius=orthogonality,
        determinant_abs_error=determinant_error,
        gram_max_abs=gram_error,
        pairwise_angle_max_abs_rad=angle_error,
    )
