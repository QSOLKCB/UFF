"""Permutation statistics for UFF-SLFA."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .sky_contract import AuditContract, AuditError
from .sky_geometry import (
    InvarianceResiduals,
    cap_membership,
    ra_shift_rotation,
    radec_to_unit,
    random_so3,
    validate_lattice_invariance,
)

OBSERVATIONS_SCHEMA = "uff.sky-lattice-observations.v1"
ALGORITHM_ID = "uff-slfa-v1"


@dataclass(frozen=True, slots=True)
class RegionSummary:
    anomaly_inside: float
    normal_inside: float
    anomaly_outside: float
    normal_outside: float

    @property
    def inside_total(self) -> float:
        return self.anomaly_inside + self.normal_inside

    @property
    def outside_total(self) -> float:
        return self.anomaly_outside + self.normal_outside

    @property
    def inside_rate(self) -> float:
        return self.anomaly_inside / self.inside_total if self.inside_total else math.nan

    @property
    def outside_rate(self) -> float:
        return self.anomaly_outside / self.outside_total if self.outside_total else math.nan

    @property
    def rate_contrast(self) -> float:
        return self.inside_rate - self.outside_rate

    @property
    def odds_ratio(self) -> float:
        ai, ni = self.anomaly_inside + 0.5, self.normal_inside + 0.5
        ao, no = self.anomaly_outside + 0.5, self.normal_outside + 0.5
        return ai * no / (ni * ao)

    def to_dict(self) -> dict[str, float]:
        return {
            "anomaly_inside": self.anomaly_inside,
            "normal_inside": self.normal_inside,
            "anomaly_outside": self.anomaly_outside,
            "normal_outside": self.normal_outside,
            "inside_total": self.inside_total,
            "outside_total": self.outside_total,
            "inside_rate": self.inside_rate,
            "outside_rate": self.outside_rate,
            "rate_contrast": self.rate_contrast,
            "odds_ratio": self.odds_ratio,
        }


def summarize_region(mask: np.ndarray, anomaly: np.ndarray, weights: np.ndarray) -> RegionSummary:
    region = np.asarray(mask, dtype=bool)
    labels = np.asarray(anomaly, dtype=bool)
    weight = np.asarray(weights, dtype=float)
    return RegionSummary(
        float(weight[region & labels].sum()),
        float(weight[region & ~labels].sum()),
        float(weight[~region & labels].sum()),
        float(weight[~region & ~labels].sum()),
    )


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise AuditError("Holm adjustment requires finite p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def empirical_p(observed: float, null: np.ndarray) -> float:
    if not math.isfinite(float(observed)):
        raise AuditError("empirical p-value requires a finite observed statistic")
    values = np.asarray(null, dtype=float)
    if values.ndim != 1 or values.size < 1 or np.any(~np.isfinite(values)):
        raise AuditError("empirical p-value requires a finite null distribution")
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def _shuffle_labels(anomaly: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(anomaly, dtype=bool).copy()
    stratum_values = np.asarray(strata)
    if pd.isna(stratum_values).any():
        raise AuditError("stratified-label null requires non-missing stratum values")
    for value in pd.unique(stratum_values):
        indexes = np.flatnonzero(stratum_values == value)
        result[indexes] = rng.permutation(result[indexes])
    return result


def _merge_maxima(maxima: dict[str, float], residuals: InvarianceResiduals) -> None:
    for key, value in residuals.to_dict().items():
        maxima[key] = max(maxima.get(key, 0.0), float(value))


def run_audit(catalogue: pd.DataFrame, contract: AuditContract) -> tuple[dict[str, Any], pd.DataFrame]:
    catalogue_vectors = radec_to_unit(
        catalogue["ra_deg"].to_numpy(float), catalogue["dec_deg"].to_numpy(float)
    )
    node_vectors = radec_to_unit(
        np.array([node.ra_deg for node in contract.nodes]),
        np.array([node.dec_deg for node in contract.nodes]),
    )
    membership = cap_membership(catalogue_vectors, node_vectors, contract.radius_deg)
    anomaly = catalogue["_anomaly"].to_numpy(bool)
    weights = catalogue["_weight"].to_numpy(float)
    global_summary = summarize_region(membership.any(axis=1), anomaly, weights)
    if not global_summary.inside_total or not global_summary.outside_total:
        raise AuditError("node-cap union and complement must both contain rows")
    node_summaries = [
        summarize_region(membership[:, index], anomaly, weights)
        for index in range(len(contract.nodes))
    ]
    testable = np.array(
        [summary.inside_total > 0.0 and summary.outside_total > 0.0 for summary in node_summaries]
    )
    observed = np.array(
        [summary.rate_contrast if valid else 0.0 for summary, valid in zip(node_summaries, testable)]
    )
    rng = np.random.default_rng(contract.seed)
    global_null: list[float] = []
    node_null: list[list[float]] = [[] for _ in contract.nodes]
    maxima = {
        "orthogonality_frobenius": 0.0,
        "determinant_abs_error": 0.0,
        "gram_max_abs": 0.0,
        "pairwise_angle_max_abs_rad": 0.0,
    }
    if contract.null_model in {"ra-shift", "so3"}:
        def nulls_complete() -> bool:
            return len(global_null) == contract.permutations and all(
                not testable[index] or len(node_null[index]) == contract.permutations
                for index in range(len(contract.nodes))
            )

        for _ in range(contract.permutations * 100):
            if nulls_complete():
                break
            rotation = (
                random_so3(rng)
                if contract.null_model == "so3"
                else ra_shift_rotation(rng.uniform(0.0, 2.0 * math.pi))
            )
            transformed = node_vectors @ rotation.T
            _merge_maxima(maxima, validate_lattice_invariance(node_vectors, transformed, rotation))
            rotated = cap_membership(catalogue_vectors, transformed, contract.radius_deg)
            candidate = summarize_region(rotated.any(axis=1), anomaly, weights)
            if (
                len(global_null) < contract.permutations
                and candidate.inside_total > 0.0
                and candidate.outside_total > 0.0
            ):
                global_null.append(candidate.rate_contrast)
            for index in range(len(contract.nodes)):
                if not testable[index] or len(node_null[index]) == contract.permutations:
                    continue
                candidate = summarize_region(rotated[:, index], anomaly, weights)
                if candidate.inside_total > 0.0 and candidate.outside_total > 0.0:
                    node_null[index].append(candidate.rate_contrast)
        if not nulls_complete():
            raise AuditError("unable to generate enough non-empty rotated null regions")
    else:
        strata = catalogue[contract.stratum_column].to_numpy()
        for _ in range(contract.permutations):
            shuffled = _shuffle_labels(anomaly, strata, rng)
            global_null.append(
                summarize_region(membership.any(axis=1), shuffled, weights).rate_contrast
            )
            for index in range(len(contract.nodes)):
                if testable[index]:
                    node_null[index].append(
                        summarize_region(membership[:, index], shuffled, weights).rate_contrast
                    )
    if len(global_null) != contract.permutations:
        raise AuditError("global null distribution is incomplete")
    global_null_array = np.asarray(global_null)
    global_p = empirical_p(global_summary.rate_contrast, global_null_array)
    raw_p = np.ones(len(contract.nodes))
    for index in range(len(contract.nodes)):
        if testable[index]:
            raw_p[index] = empirical_p(observed[index], np.asarray(node_null[index]))
    holm_p = holm_adjust(raw_p)
    rows: list[dict[str, Any]] = []
    for index, (node, summary) in enumerate(zip(contract.nodes, node_summaries)):
        rows.append(
            {
                "node_id": node.node_id,
                "ra_deg": node.ra_deg,
                "dec_deg": node.dec_deg,
                **summary.to_dict(),
                "testable": bool(testable[index]),
                "empirical_p": float(raw_p[index]),
                "holm_p": float(holm_p[index]),
                "survives_holm": bool(
                    testable[index]
                    and holm_p[index] <= contract.alpha
                    and summary.rate_contrast >= contract.minimum_effect
                ),
            }
        )
    node_table = pd.DataFrame.from_records(rows)
    supported = int(node_table["survives_holm"].sum())
    global_pass = bool(
        global_p <= contract.alpha and global_summary.rate_contrast >= contract.minimum_effect
    )
    passed = global_pass and supported >= contract.minimum_supported_nodes
    observations = {
        "schema": OBSERVATIONS_SCHEMA,
        "algorithm": ALGORITHM_ID,
        "claim_id": contract.claim_id,
        "decision": "EMPIRICAL_CRITERIA_MET" if passed else "EMPIRICAL_CRITERIA_NOT_MET",
        "global_test": {
            **global_summary.to_dict(),
            "empirical_p": global_p,
            "null_mean": float(global_null_array.mean()),
            "null_std": float(global_null_array.std(ddof=1)),
            "passes": global_pass,
        },
        "node_test": {
            "supported_nodes": supported,
            "required_supported_nodes": contract.minimum_supported_nodes,
            "holm_familywise_alpha": contract.alpha,
        },
        "null_transform_invariance": {
            "checked": contract.null_model in {"ra-shift", "so3"},
            "maximum_residuals": maxima,
            "meaning": (
                "Proper rotations preserve the node Gram matrix and pairwise angles; "
                "this validates the audit transform, not the physical lattice claim."
            ),
        },
        "claim_boundary": (
            "The decision applies only to the frozen catalogue-level association claim. "
            "It does not prove or disprove a causal ontology or a later modified model."
        ),
    }
    return observations, node_table
