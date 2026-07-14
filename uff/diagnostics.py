"""Plots, coordinate-invariant diagnostics, and deterministic fingerprints."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import wave

# Some minimal/container environments expose a read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "uff-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import GalaxyData
from .fitting import FitResult, model_weights
from .models import RotationCurveModel


def normalized_shannon_entropy(probabilities: np.ndarray) -> float:
    """Return Shannon entropy normalized to ``[0,1]`` for a finite simplex.

    This diagnostic bridges UFF model comparison with the entropy telemetry in
    QSOLKCB/QNTOY.  It measures ambiguity among models, not quantum entropy.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("at least one probability must be positive")
    values = values / total
    positive = values[values > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return 0.0 if values.size == 1 else entropy / math.log(values.size)


def covariance_invariants(covariance: np.ndarray) -> dict[str, object]:
    """Return basis-independent invariants of a parameter covariance matrix.

    The eigenspectrum/trace diagnostic is an interoperability bridge to the
    invariant-preserving tensor work in QSOLKCB/TFT.  Parameter units still
    matter, so comparisons are meaningful only within one parameterization.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape == (0, 0):
        return {"rank": 0, "trace": 0.0, "frobenius_norm": 0.0, "eigenvalues": []}
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if np.any(~np.isfinite(matrix)):
        return {
            "rank": None,
            "trace": None,
            "frobenius_norm": None,
            "eigenvalues": None,
        }
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return {
        "rank": int(np.linalg.matrix_rank(symmetric)),
        "trace": float(np.trace(symmetric)),
        "frobenius_norm": float(np.linalg.norm(symmetric, "fro")),
        "eigenvalues": eigenvalues.tolist(),
    }


def phase_fingerprint(
    values: np.ndarray, phase: float = math.pi / 2.0
) -> dict[str, object]:
    """Encode normalized values as a deterministic complex phase fingerprint.

    This is inspired by the phi-locked Tensor Phase Cube in QSOLKCB/QAI-UFT.
    It is a provenance/visualization transform and does not alter the fit.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite non-empty vector")
    scale = float(np.ptp(array))
    normalized = np.zeros_like(array) if scale == 0 else (array - np.min(array)) / scale
    encoded = np.exp(1j * phase * normalized)
    return {
        "phase_radians": float(phase),
        "real": np.real(encoded).tolist(),
        "imaginary": np.imag(encoded).tolist(),
        "energy": float(np.sum(np.abs(encoded) ** 2)),
        "status": "diagnostic fingerprint; not a physical field observable",
    }


def comparison_diagnostics(results: list[FitResult]) -> dict[str, object]:
    """Build entropy, covariance-invariant, and phase diagnostics."""

    weights = model_weights(results, "bic")
    ordered_weights = np.array([weights[result.model_name] for result in results])
    return {
        "bic_model_weights": weights,
        "normalized_model_weight_entropy": normalized_shannon_entropy(ordered_weights),
        "interpretation": (
            "0 means one model dominates these candidates; 1 means equal BIC weights. "
            "Weights are relative to the candidate set and are not posterior theory probabilities."
        ),
        "models": {
            result.model_name: {
                "covariance_invariants": covariance_invariants(result.covariance),
                "residual_phase_fingerprint": phase_fingerprint(
                    result.standardized_residuals
                ),
            }
            for result in results
        },
        "project_bridges": {
            "QAI-UFT": "phi-locked diagnostic fingerprint",
            "QNTOY": "normalized entropy telemetry",
            "TFT": "basis-invariant covariance eigenspectrum",
        },
    }


def plot_model_comparison(
    data: GalaxyData,
    models: list[RotationCurveModel],
    results: list[FitResult],
    save_path: str | Path,
) -> None:
    """Plot fitted curves, baryonic reference, and standardized residuals."""

    model_by_name = {model.name: model for model in models}
    minimum_bic = min(result.bic for result in results)
    radius_grid = np.geomspace(data.radius_kpc.min(), data.radius_kpc.max(), 400)
    figure, (curve_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    curve_axis.errorbar(
        data.radius_kpc,
        data.velocity_obs_kms,
        yerr=data.velocity_err_kms,
        fmt="o",
        color="black",
        markersize=4.5,
        capsize=2,
        label="Observed",
        zorder=10,
    )
    baryonic = np.sqrt(
        data.baryonic_velocity_squared(
            radius_grid, disk_mass_to_light=0.5, bulge_mass_to_light=0.7
        )
    )
    curve_axis.plot(
        radius_grid, baryonic, "--", color="0.55", label="Baryons (0.5/0.7)"
    )

    color_map = plt.get_cmap("tab10")
    for index, result in enumerate(sorted(results, key=lambda item: item.bic)):
        model = model_by_name[result.model_name]
        color = color_map(index % 10)
        prediction_grid = model.predict(radius_grid, result.theta)
        curve_axis.plot(
            radius_grid,
            prediction_grid,
            color=color,
            linewidth=2.0,
            label=f"{result.model_label} (DeltaBIC={result.bic - minimum_bic:.1f})",
        )
        residual_axis.plot(
            data.radius_kpc,
            result.standardized_residuals,
            marker="o",
            linewidth=1.0,
            markersize=3.5,
            color=color,
            label=result.model_name,
        )

    curve_axis.set_ylabel("Circular velocity [km/s]")
    curve_axis.set_title(f"{data.name} — rotation-curve model comparison")
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(fontsize=8, ncol=2)
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.axhspan(-1.0, 1.0, color="0.9", zorder=-10)
    residual_axis.set_xlabel("Radius [kpc]")
    residual_axis.set_ylabel("Residual / sigma")
    residual_axis.grid(alpha=0.2)
    figure.align_ylabels()
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_rotation_sonification(
    radius_kpc: np.ndarray,
    velocity_kms: np.ndarray,
    save_path: str | Path,
    *,
    duration_s: float = 6.0,
    sample_rate: int = 48_000,
) -> None:
    """Save a deterministic stereo phase-quadrature rotation-curve glyph."""

    radius = np.asarray(radius_kpc, dtype=float)
    velocity = np.asarray(velocity_kms, dtype=float)
    if radius.shape != velocity.shape or radius.ndim != 1 or radius.size < 2:
        raise ValueError("radius and velocity must be matching one-dimensional vectors")
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(velocity)):
        raise ValueError("radius and velocity must be finite")
    if duration_s <= 0 or sample_rate < 8_000:
        raise ValueError("duration_s must be positive and sample_rate at least 8000")

    order = np.argsort(radius)
    radius = radius[order]
    velocity = velocity[order]
    if np.any(np.diff(radius) <= 0):
        raise ValueError("radius_kpc must be strictly increasing")

    sample_count = int(round(duration_s * sample_rate))
    normalized_time = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    radial_coordinate = (radius - radius.min()) / max(float(np.ptp(radius)), 1.0e-12)
    interpolated = np.interp(normalized_time, radial_coordinate, velocity)
    v_min = float(np.min(interpolated))
    v_range = max(float(np.ptp(interpolated)), 1.0e-12)
    # Log-frequency mapping keeps velocity ratios perceptually useful.
    frequency = 110.0 * np.power(8.0, (interpolated - v_min) / v_range)
    phase = 2.0 * math.pi * np.cumsum(frequency) / sample_rate
    envelope = np.sin(math.pi * np.linspace(0.0, 1.0, sample_count)) ** 2
    left = np.sin(phase) * envelope
    right = np.cos(phase) * envelope
    stereo = np.column_stack((left, right))
    peak = max(float(np.max(np.abs(stereo))), 1.0e-12)
    pcm = np.int16(np.clip(stereo / peak * 0.85, -1.0, 1.0) * 32767)
    with wave.open(str(save_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def plot_posterior_corner(
    samples: np.ndarray,
    parameter_names: tuple[str, ...],
    save_path: str | Path,
) -> None:
    """Plot retained posterior draws without an external corner-plot package."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != len(parameter_names):
        raise ValueError("sample shape does not match parameter_names")
    dimension = values.shape[1]
    figure, axes = plt.subplots(
        dimension,
        dimension,
        figsize=(2.25 * dimension, 2.25 * dimension),
        squeeze=False,
    )
    max_points = min(values.shape[0], 5_000)
    stride = max(1, values.shape[0] // max_points)
    plotted = values[::stride]
    for row in range(dimension):
        for column in range(dimension):
            axis = axes[row, column]
            if row == column:
                axis.hist(values[:, row], bins=35, color="0.35", alpha=0.8)
                axis.axvline(np.median(values[:, row]), color="C1", linewidth=1.0)
                axis.set_yticks([])
            elif column < row:
                axis.scatter(
                    plotted[:, column],
                    plotted[:, row],
                    s=2,
                    alpha=0.12,
                    rasterized=True,
                )
            else:
                axis.set_visible(False)
            if row == dimension - 1 and axis.get_visible():
                axis.set_xlabel(parameter_names[column], fontsize=8)
            if column == 0 and row > 0:
                axis.set_ylabel(parameter_names[row], fontsize=8)
            axis.tick_params(labelsize=7)
    figure.suptitle("Retained posterior draws", y=1.01)
    figure.savefig(Path(save_path), dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_posterior_predictive(
    data: GalaxyData,
    model: RotationCurveModel,
    samples: np.ndarray,
    save_path: str | Path,
    *,
    seed: int = 42,
    max_draws: int = 1_000,
) -> None:
    """Plot 16/50/84% model-curve bands from retained parameter draws."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != len(model.parameters):
        raise ValueError("sample shape does not match model")
    rng = np.random.default_rng(seed)
    if values.shape[0] > max_draws:
        values = values[rng.choice(values.shape[0], max_draws, replace=False)]
    radius_grid = np.geomspace(data.radius_kpc.min(), data.radius_kpc.max(), 300)
    predictions = np.array([model.predict(radius_grid, theta) for theta in values])
    low, median, high = np.percentile(predictions, [16, 50, 84], axis=0)
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    axis.errorbar(
        data.radius_kpc,
        data.velocity_obs_kms,
        yerr=data.velocity_err_kms,
        fmt="o",
        color="black",
        capsize=2,
        label="Observed",
    )
    axis.fill_between(
        radius_grid, low, high, color="C0", alpha=0.25, label="16–84% curve band"
    )
    axis.plot(radius_grid, median, color="C0", linewidth=2.0, label="Posterior median")
    axis.set_xlabel("Radius [kpc]")
    axis.set_ylabel("Circular velocity [km/s]")
    axis.set_title(f"{data.name} — {model.label} posterior curves")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "comparison_diagnostics",
    "covariance_invariants",
    "normalized_shannon_entropy",
    "phase_fingerprint",
    "plot_model_comparison",
    "plot_posterior_corner",
    "plot_posterior_predictive",
    "save_rotation_sonification",
]
