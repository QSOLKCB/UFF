"""Survey-aware spherical density reconstruction for the Sheridan Crucible.

The implementation follows the methodological architecture used in modern
COSMOS-Web density reconstruction: a von Mises--Fisher kernel on the sphere,
leave-one-out bandwidth selection, adaptive smoothing, explicit mask/coverage
quadrature, and edge renormalization.  It is an independent implementation;
no external project code is copied.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import pandas as pd

from .sheridan_contract import DensityConfig, SheridanDecision
from .sky_contract import AuditContract, AuditError
from .sky_geometry import (
    radec_to_unit,
    random_so3,
    validate_lattice_invariance,
)
from .sky_statistics import empirical_p, holm_adjust

DENSITY_SCHEMA = "uff.sheridan-density.v1"
SUPPORT_BLOCK_SIZE = 4096
CENTRE_BLOCK_SIZE = 512


def _log_sinh(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    result = np.empty_like(array)
    small = array < 30.0
    result[small] = np.log(np.sinh(array[small]))
    large = ~small
    result[large] = array[large] - math.log(2.0) + np.log1p(-np.exp(-2.0 * array[large]))
    return result


def _bandwidth_to_kappa(bandwidth_rad: np.ndarray) -> np.ndarray:
    bandwidth = np.asarray(bandwidth_rad, dtype=float)
    if np.any(~np.isfinite(bandwidth)) or np.any(bandwidth <= 0.0):
        raise AuditError("vMF bandwidths must be finite and positive")
    return 1.0 / np.square(bandwidth)


def vmf_kernel(
    evaluation_vectors: np.ndarray,
    centre_vectors: np.ndarray,
    bandwidth_rad: float | np.ndarray,
) -> np.ndarray:
    """Evaluate normalized S2 von Mises--Fisher kernels.

    Columns correspond to kernel centres, allowing each centre to have its own
    adaptive bandwidth.
    """
    evaluation = np.asarray(evaluation_vectors, dtype=float)
    centres = np.asarray(centre_vectors, dtype=float)
    if evaluation.ndim != 2 or evaluation.shape[1] != 3:
        raise AuditError("evaluation vectors must have shape (n, 3)")
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 1:
        raise AuditError("centre vectors must have shape (m, 3) with m >= 1")
    bandwidth = np.asarray(bandwidth_rad, dtype=float)
    if bandwidth.ndim == 0:
        bandwidth = np.full(len(centres), float(bandwidth))
    if bandwidth.shape != (len(centres),):
        raise AuditError("adaptive bandwidth array must have one value per centre")
    kappa = _bandwidth_to_kappa(bandwidth)
    log_normalizer = np.log(kappa) - math.log(4.0 * math.pi) - _log_sinh(kappa)
    dots = np.clip(evaluation @ centres.T, -1.0, 1.0)
    log_kernel = dots * kappa[None, :] + log_normalizer[None, :]
    return np.exp(np.clip(log_kernel, -745.0, 700.0))


def support_kernel_mass(
    centre_vectors: np.ndarray,
    bandwidth_rad: float | np.ndarray,
    support_vectors: np.ndarray,
    area_weight_sr: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """Integrate each spherical kernel over the usable survey support.

    Both support points and kernel centres are streamed in bounded blocks, so
    peak memory does not scale as ``n_support * n_centres``.
    """
    centres = np.asarray(centre_vectors, dtype=float)
    support = np.asarray(support_vectors, dtype=float)
    area = np.asarray(area_weight_sr, dtype=float)
    available = np.asarray(coverage, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 1:
        raise AuditError("support-mass centres must have shape (m, 3) with m >= 1")
    if support.ndim != 2 or support.shape[1] != 3 or len(support) < 1:
        raise AuditError("survey support vectors must have shape (n, 3) with n >= 1")
    if area.shape != (len(support),) or available.shape != (len(support),):
        raise AuditError("survey support vectors, area weights, and coverage must have equal length")
    if (
        np.any(~np.isfinite(support))
        or np.any(~np.isfinite(area))
        or np.any(~np.isfinite(available))
        or np.any(area <= 0.0)
        or np.any((available < 0.0) | (available > 1.0))
    ):
        raise AuditError(
            "survey support vectors and weights must be finite; area must be positive "
            "and coverage must be in [0, 1]"
        )

    bandwidth = np.asarray(bandwidth_rad, dtype=float)
    if bandwidth.ndim == 0:
        bandwidth = np.full(len(centres), float(bandwidth))
    if bandwidth.shape != (len(centres),):
        raise AuditError("support-mass bandwidth array must match centre count")

    measure = area * available
    mass = np.zeros(len(centres), dtype=float)
    for centre_start in range(0, len(centres), CENTRE_BLOCK_SIZE):
        centre_stop = min(centre_start + CENTRE_BLOCK_SIZE, len(centres))
        partial = np.zeros(centre_stop - centre_start, dtype=float)
        for support_start in range(0, len(support), SUPPORT_BLOCK_SIZE):
            support_stop = min(support_start + SUPPORT_BLOCK_SIZE, len(support))
            kernel = vmf_kernel(
                support[support_start:support_stop],
                centres[centre_start:centre_stop],
                bandwidth[centre_start:centre_stop],
            )
            partial += measure[support_start:support_stop] @ kernel
        mass[centre_start:centre_stop] = partial
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise AuditError("one or more kernels have zero usable survey support")
    return mass


@dataclass(frozen=True, slots=True)
class DensityFit:
    source_vectors: np.ndarray
    source_weights: np.ndarray
    source_bandwidth_rad: np.ndarray
    source_edge_mass: np.ndarray
    support_vectors: np.ndarray
    support_area_weight_sr: np.ndarray
    support_coverage: np.ndarray
    usable_area_sr: float
    global_bandwidth_deg: float
    lcv_scores: dict[str, float]
    adaptive_alpha: float
    mean_density: float

    def evaluate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        evaluation = np.asarray(vectors, dtype=float)
        result = np.empty(len(evaluation), dtype=float)
        denominator = float(self.source_weights.sum())
        for start in range(0, len(evaluation), 256):
            stop = min(start + 256, len(evaluation))
            kernel = vmf_kernel(
                evaluation[start:stop],
                self.source_vectors,
                self.source_bandwidth_rad,
            )
            corrected = kernel / self.source_edge_mass[None, :]
            result[start:stop] = corrected @ self.source_weights / denominator
        return result

    def evaluate_radec(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        return self.evaluate_vectors(radec_to_unit(ra_deg, dec_deg))

    def overdensity_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return self.evaluate_vectors(vectors) / self.mean_density - 1.0

    def availability_vectors(self, vectors: np.ndarray) -> np.ndarray:
        reference = math.radians(self.global_bandwidth_deg)
        return support_kernel_mass(
            vectors,
            reference,
            self.support_vectors,
            self.support_area_weight_sr,
            self.support_coverage,
        )


def _loo_score(
    source_vectors: np.ndarray,
    weights: np.ndarray,
    bandwidth_rad: float,
    support_vectors: np.ndarray,
    support_area: np.ndarray,
    support_coverage: np.ndarray,
) -> tuple[float, np.ndarray]:
    edge_mass = support_kernel_mass(
        source_vectors,
        bandwidth_rad,
        support_vectors,
        support_area,
        support_coverage,
    )
    kernel = vmf_kernel(source_vectors, source_vectors, bandwidth_rad)
    corrected = kernel / edge_mass[None, :]
    np.fill_diagonal(corrected, 0.0)
    denominator = float(weights.sum()) - weights
    if np.any(denominator <= 0.0):
        raise AuditError("leave-one-out density requires at least two weighted sources")
    density = corrected @ weights / denominator
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        return -math.inf, edge_mass
    score = float(np.dot(weights, np.log(density)) / weights.sum())
    return score, edge_mass


def fit_density(
    catalogue: pd.DataFrame,
    support: pd.DataFrame,
    config: DensityConfig,
) -> DensityFit:
    if len(catalogue) < 3:
        raise AuditError("density reconstruction requires at least three catalogue rows")
    if len(catalogue) > config.maximum_exact_sources:
        raise AuditError(
            f"exact adaptive KDE is capped at {config.maximum_exact_sources} sources; "
            "freeze a representative weighted sample or raise the explicit contract limit "
            "only after confirming memory and runtime requirements"
        )
    source_vectors = radec_to_unit(
        catalogue["ra_deg"].to_numpy(float),
        catalogue["dec_deg"].to_numpy(float),
    )
    support_vectors = radec_to_unit(
        support["ra_deg"].to_numpy(float),
        support["dec_deg"].to_numpy(float),
    )
    source_weights = catalogue["_analysis_weight"].to_numpy(float)
    support_area = support["_area_weight_sr"].to_numpy(float)
    support_coverage = support["_coverage"].to_numpy(float)
    usable_area = float(np.dot(support_area, support_coverage))

    scores: dict[str, float] = {}
    best_bandwidth: float | None = None
    best_score = -math.inf
    best_edge_mass: np.ndarray | None = None
    for candidate_deg in config.bandwidth_candidates_deg:
        score, edge_mass = _loo_score(
            source_vectors,
            source_weights,
            math.radians(candidate_deg),
            support_vectors,
            support_area,
            support_coverage,
        )
        scores[f"{candidate_deg:.12g}"] = score
        if score > best_score:
            best_score = score
            best_bandwidth = candidate_deg
            best_edge_mass = edge_mass
    if best_bandwidth is None or best_edge_mass is None or not math.isfinite(best_score):
        raise AuditError("no bandwidth candidate produced a finite leave-one-out score")

    global_bandwidth_rad = math.radians(best_bandwidth)
    pilot_kernel = vmf_kernel(source_vectors, source_vectors, global_bandwidth_rad)
    pilot_density = (
        pilot_kernel / best_edge_mass[None, :]
    ) @ source_weights / float(source_weights.sum())
    floor = np.finfo(float).tiny
    pilot_density = np.maximum(pilot_density, floor)
    geometric_mean = math.exp(float(np.dot(source_weights, np.log(pilot_density)) / source_weights.sum()))
    factors = np.power(geometric_mean / pilot_density, config.adaptive_alpha)
    factors = np.clip(
        factors,
        config.minimum_bandwidth_factor,
        config.maximum_bandwidth_factor,
    )
    adaptive_bandwidth = global_bandwidth_rad * factors
    adaptive_edge_mass = support_kernel_mass(
        source_vectors,
        adaptive_bandwidth,
        support_vectors,
        support_area,
        support_coverage,
    )

    provisional = DensityFit(
        source_vectors=source_vectors,
        source_weights=source_weights,
        source_bandwidth_rad=adaptive_bandwidth,
        source_edge_mass=adaptive_edge_mass,
        support_vectors=support_vectors,
        support_area_weight_sr=support_area,
        support_coverage=support_coverage,
        usable_area_sr=usable_area,
        global_bandwidth_deg=best_bandwidth,
        lcv_scores=scores,
        adaptive_alpha=config.adaptive_alpha,
        mean_density=1.0,
    )
    support_density = provisional.evaluate_vectors(support_vectors)
    mean_density = float(
        np.dot(support_area * support_coverage, support_density) / usable_area
    )
    if not math.isfinite(mean_density) or mean_density <= 0.0:
        raise AuditError("survey-weighted mean density is not finite and positive")
    return replace(provisional, mean_density=mean_density)


def _availability_distance(observed: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(observed, dtype=float)
    right = np.asarray(candidate, dtype=float)
    scale = max(float(np.linalg.norm(left)), 1.0e-12)
    return float(np.linalg.norm(right - left) / scale)


def score_nodes(
    fit: DensityFit,
    claim: AuditContract,
    config: DensityConfig,
    decision: SheridanDecision,
) -> tuple[dict[str, Any], pd.DataFrame]:
    node_vectors = radec_to_unit(
        np.array([node.ra_deg for node in claim.nodes]),
        np.array([node.dec_deg for node in claim.nodes]),
    )
    density = fit.evaluate_vectors(node_vectors)
    overdensity = density / fit.mean_density - 1.0
    availability = fit.availability_vectors(node_vectors)
    testable = availability >= config.minimum_availability
    if not np.any(testable):
        raise AuditError("no frozen node has enough survey availability for density testing")
    observed_mean = float(np.mean(overdensity[testable]))

    rng = np.random.default_rng(config.seed)
    global_null: list[float] = []
    per_node_null: list[list[float]] = [[] for _ in claim.nodes]
    max_residuals: dict[str, float] = {}
    accepted_distances: list[float] = []
    maximum_attempts = config.permutations * config.maximum_rotation_attempt_multiplier
    attempts = 0
    while len(global_null) < config.permutations and attempts < maximum_attempts:
        attempts += 1
        rotation = random_so3(rng)
        rotated = node_vectors @ rotation.T
        residuals = validate_lattice_invariance(node_vectors, rotated, rotation)
        for key, value in residuals.to_dict().items():
            max_residuals[key] = max(max_residuals.get(key, 0.0), float(value))
        candidate_availability = fit.availability_vectors(rotated)
        candidate_testable = candidate_availability >= config.minimum_availability
        if not np.array_equal(candidate_testable, testable):
            continue
        distance = _availability_distance(availability, candidate_availability)
        if distance > config.availability_tolerance:
            continue
        candidate_overdensity = fit.overdensity_vectors(rotated)
        global_null.append(float(np.mean(candidate_overdensity[candidate_testable])))
        accepted_distances.append(distance)
        for index in range(len(claim.nodes)):
            if testable[index] and candidate_testable[index]:
                per_node_null[index].append(float(candidate_overdensity[index]))
    if len(global_null) != config.permutations:
        raise AuditError(
            "unable to generate enough survey-matched SO(3) null rotations; "
            "increase availability tolerance or provide a denser support grid"
        )

    global_null_array = np.asarray(global_null, dtype=float)
    global_p = empirical_p(observed_mean, global_null_array)
    raw_p = np.ones(len(claim.nodes), dtype=float)
    for index in range(len(claim.nodes)):
        if testable[index]:
            if len(per_node_null[index]) != config.permutations:
                raise AuditError("per-node null distribution is incomplete")
            raw_p[index] = empirical_p(overdensity[index], np.asarray(per_node_null[index]))
    holm = holm_adjust(raw_p)

    rows: list[dict[str, Any]] = []
    for index, node in enumerate(claim.nodes):
        supported = bool(
            testable[index]
            and holm[index] <= decision.alpha
            and overdensity[index] >= decision.minimum_mean_overdensity
        )
        rows.append(
            {
                "node_id": node.node_id,
                "ra_deg": node.ra_deg,
                "dec_deg": node.dec_deg,
                "density_sr_inv": float(density[index]),
                "overdensity": float(overdensity[index]),
                "survey_availability": float(availability[index]),
                "testable": bool(testable[index]),
                "empirical_p": float(raw_p[index]),
                "holm_p": float(holm[index]),
                "supported": supported,
            }
        )
    table = pd.DataFrame.from_records(rows)
    supported_nodes = int(table["supported"].sum())
    global_pass = bool(
        global_p <= decision.alpha
        and observed_mean >= decision.minimum_mean_overdensity
    )
    passed = global_pass and supported_nodes >= decision.minimum_supported_nodes
    observations = {
        "schema": DENSITY_SCHEMA,
        "decision": "SURVEY_AWARE_DENSITY_CRITERIA_MET" if passed else "SURVEY_AWARE_DENSITY_CRITERIA_NOT_MET",
        "bandwidth": {
            "selected_global_deg": fit.global_bandwidth_deg,
            "adaptive_alpha": fit.adaptive_alpha,
            "candidate_lcv_scores": fit.lcv_scores,
            "adaptive_min_deg": float(np.rad2deg(fit.source_bandwidth_rad.min())),
            "adaptive_max_deg": float(np.rad2deg(fit.source_bandwidth_rad.max())),
        },
        "survey": {
            "usable_area_sr": fit.usable_area_sr,
            "mean_density_sr_inv": fit.mean_density,
            "edge_mass_min": float(fit.source_edge_mass.min()),
            "edge_mass_median": float(np.median(fit.source_edge_mass)),
            "edge_mass_max": float(fit.source_edge_mass.max()),
        },
        "global_test": {
            "testable_nodes": int(testable.sum()),
            "mean_overdensity": observed_mean,
            "empirical_p": global_p,
            "null_mean": float(global_null_array.mean()),
            "null_std": float(global_null_array.std(ddof=1)),
            "passes": global_pass,
        },
        "node_test": {
            "supported_nodes": supported_nodes,
            "required_supported_nodes": decision.minimum_supported_nodes,
            "holm_familywise_alpha": decision.alpha,
        },
        "null_matching": {
            "accepted_rotations": len(global_null),
            "attempted_rotations": attempts,
            "availability_tolerance": config.availability_tolerance,
            "accepted_distance_max": float(max(accepted_distances)),
            "maximum_invariance_residuals": max_residuals,
        },
        "claim_boundary": (
            "This result tests a survey-corrected source-density association at frozen nodes. "
            "It does not identify the physical cause of any surviving density contrast."
        ),
    }
    return observations, table
