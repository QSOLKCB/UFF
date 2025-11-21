"""
E₈ root system visualization utilities.

This module provides a simple function to generate and plot a 3D
projection of the 8‑dimensional E₈ root lattice. The visualization is
intended for exploratory purposes and as a playful hook; it does not
affect the UFF fitting results. Two classes of roots are coloured
differently: those with integer entries and those with half‑integer
entries.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_e8_roots() -> Tuple[np.ndarray, np.ndarray]:
    """Generate the 240 roots of the E₈ lattice.

    Returns
    -------
    roots_int : ndarray, shape (112, 8)
        Roots with integer coordinates of the form (±1, ±1, 0, 0, …, 0)
        with all sign combinations and permutations.
    roots_half : ndarray, shape (128, 8)
        Roots with half‑integer coordinates of the form (±½, ±½, …, ±½)
        such that the number of + signs is even.
    """
    roots_int: List[np.ndarray] = []
    # Integer roots: choose two indices i<j and assign ±1 to those entries,
    # zeros elsewhere. There are 8 choose 2 = 28 pairs, and for each pair
    # four sign combinations.
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    vec = np.zeros(8)
                    vec[i] = si
                    vec[j] = sj
                    roots_int.append(vec)
    roots_int_arr = np.array(roots_int)
    # Half-integer roots: all 8-vectors with entries ±1/2 and an even
    # number of positive signs (equivalently, sum of entries an integer).
    roots_half: List[np.ndarray] = []
    for bits in range(1 << 8):
        # Determine signs for each of 8 bits
        signs = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(8)])
        # Count positive signs
        pos_count = int(np.sum(signs > 0))
        if pos_count % 2 == 0:
            roots_half.append(0.5 * signs)
    roots_half_arr = np.array(roots_half)
    assert roots_int_arr.shape[0] == 112, f"Expected 112 integer roots, got {roots_int_arr.shape[0]}"
    assert roots_half_arr.shape[0] == 128, f"Expected 128 half roots, got {roots_half_arr.shape[0]}"
    return roots_int_arr, roots_half_arr


def random_projection_matrix(seed: int = 42) -> np.ndarray:
    """Generate a random orthonormal 3×8 projection matrix.

    Uses a fixed random seed by default for deterministic output. The
    resulting matrix ``P`` has shape (3, 8) and orthonormal rows (i.e.
    ``P @ P.T = I``). Points ``x`` in ℝ⁸ can be projected to ℝ³ via
    ``P @ x``.
    """
    rng = np.random.default_rng(seed)
    # Generate a random 8×3 matrix and perform QR decomposition
    A = rng.normal(size=(8, 3))
    Q, _ = np.linalg.qr(A)
    # Q has shape (8,3) with orthonormal columns; we need (3,8)
    return Q.T


def project_roots(roots: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project a set of roots to 3D using projection matrix P.

    Parameters
    ----------
    roots : ndarray, shape (n, 8)
        Input roots to project.
    P : ndarray, shape (3, 8)
        Projection matrix with orthonormal rows.

    Returns
    -------
    ndarray, shape (n, 3)
        3D coordinates of the projected roots.
    """
    return (P @ roots.T).T


def plot_e8_projection(save_path: str) -> None:
    """Generate and save a 3D scatter plot of the E₈ root system.

    Two colours distinguish the integer and half‑integer classes. A
    deterministic random projection is used for reproducibility.

    Parameters
    ----------
    save_path : str
        File path where the figure will be saved. The directory must
        exist.
    """
    roots_int, roots_half = generate_e8_roots()
    P = random_projection_matrix(seed=42)
    coords_int = project_roots(roots_int, P)
    coords_half = project_roots(roots_half, P)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords_int[:, 0], coords_int[:, 1], coords_int[:, 2],
        s=8, alpha=0.6, color="C0", label="integer roots"
    )
    ax.scatter(
        coords_half[:, 0], coords_half[:, 1], coords_half[:, 2],
        s=8, alpha=0.6, color="C1", label="half roots"
    )
    ax.set_title("E₈ root system — random 3D projection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize="small")
    # Improve spacing
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)