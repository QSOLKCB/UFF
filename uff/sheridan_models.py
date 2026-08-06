"""Competing anomaly models, predictive checks, and injection recovery."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .sheridan_contract import InjectionConfig, ModelConfig
from .sky_contract import AuditContract, AuditError
from .sky_geometry import cap_membership, radec_to_unit
from .sky_statistics import summarize_region

MODEL_SCHEMA = "uff.sheridan-model-comparison.v1"
INJECTION_SCHEMA = "uff.sheridan-injection.v1"


@dataclass(frozen=True, slots=True)
class LogisticFit:
    name: str
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    pseudo_bic: float
    converged: bool
    bound_hits: tuple[str, ...]

    def probabilities(self, design: np.ndarray) -> np.ndarray:
        return expit(np.asarray(design, dtype=float) @ self.coefficients)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "coefficients": {
                name: float(value)
                for name, value in zip(self.feature_names, self.coefficients, strict=True)
            },
            "standard_errors": {
                name: float(math.sqrt(max(self.covariance[index, index], 0.0)))
                for index, name in enumerate(self.feature_names)
            },
            "log_likelihood": self.log_likelihood,
            "pseudo_bic": self.pseudo_bic,
            "converged": self.converged,
            "bound_hits": list(self.bound_hits),
            "weighting_note": (
                "Completeness-adjusted weights are normalized to the catalogue row count; "
                "BIC is therefore a transparent pseudo-BIC, not an exact marginal likelihood."
            ),
        }


def node_membership(catalogue: pd.DataFrame, claim: AuditContract) -> np.ndarray:
    catalogue_vectors = radec_to_unit(
        catalogue["ra_deg"].to_numpy(float),
        catalogue["dec_deg"].to_numpy(float),
    )
    node_vectors = radec_to_unit(
        np.array([node.ra_deg for node in claim.nodes]),
        np.array([node.dec_deg for node in claim.nodes]),
    )
    return cap_membership(catalogue_vectors, node_vectors, claim.radius_deg)


def _standardize(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    scale = float(array.std(ddof=0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise AuditError(f"model covariate {label!r} has zero variance")
    return (array - mean) / scale


def build_design(
    catalogue: pd.DataFrame,
    membership: np.ndarray,
    config: ModelConfig,
    *,
    include_node: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = [np.ones(len(catalogue), dtype=float)]
    names = ["intercept"]
    for column in config.covariate_columns:
        columns.append(_standardize(catalogue[column].to_numpy(float), column))
        names.append(f"z:{column}")
    if config.stratum_column:
        values = catalogue[config.stratum_column].astype(str)
        levels = sorted(values.unique())
        for level in levels[1:]:
            columns.append((values == level).to_numpy(float))
            names.append(f"stratum:{level}")
    if include_node:
        columns.append(np.asarray(membership, dtype=bool).any(axis=1).astype(float))
        names.append("inside_frozen_node")
    return np.column_stack(columns), tuple(names)


def _fit_logistic(
    name: str,
    design: np.ndarray,
    feature_names: tuple[str, ...],
    labels: np.ndarray,
    weights: np.ndarray,
) -> LogisticFit:
    x = np.asarray(design, dtype=float)
    y = np.asarray(labels, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape[0] != len(y) or len(y) != len(w):
        raise AuditError("logistic model arrays have incompatible shapes")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise AuditError("logistic model design and weights must be finite")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        value = float(np.sum(w * (np.logaddexp(0.0, linear) - y * linear)))
        gradient = x.T @ (w * (expit(linear) - y))
        return value, gradient

    bounds = [(-30.0, 30.0)] * x.shape[1]
    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    beta = np.asarray(result.x, dtype=float)
    probability = expit(x @ beta)
    information = x.T @ ((w * probability * (1.0 - probability))[:, None] * x)
    covariance = np.linalg.pinv(information + np.eye(x.shape[1]) * 1.0e-10)
    log_likelihood = -float(result.fun)
    pseudo_bic = x.shape[1] * math.log(len(y)) - 2.0 * log_likelihood
    bound_hits = tuple(
        feature_names[index]
        for index, value in enumerate(beta)
        if abs(value) >= 29.999
    )
    return LogisticFit(
        name=name,
        feature_names=feature_names,
        coefficients=beta,
        covariance=covariance,
        log_likelihood=log_likelihood,
        pseudo_bic=pseudo_bic,
        converged=bool(result.success),
        bound_hits=bound_hits,
    )


def _draw_coefficients(
    fit: LogisticFit,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = (fit.covariance + fit.covariance.T) / 2.0
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.0, None)
    transform = vectors @ np.diag(np.sqrt(values))
    normal = rng.normal(size=(draws, len(fit.coefficients)))
    return fit.coefficients[None, :] + normal @ transform.T


def _rate_contrast(labels: np.ndarray, membership: np.ndarray, weights: np.ndarray) -> float:
    summary = summarize_region(
        np.asarray(membership, dtype=bool).any(axis=1),
        np.asarray(labels, dtype=bool),
        np.asarray(weights, dtype=float),
    )
    if not summary.inside_total or not summary.outside_total:
        raise AuditError("node union and complement must both contain model rows")
    return float(summary.rate_contrast)


def _predictive_check(
    fit: LogisticFit,
    design: np.ndarray,
    observed_labels: np.ndarray,
    membership: np.ndarray,
    weights: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    observed = _rate_contrast(observed_labels, membership, weights)
    coefficient_draws = _draw_coefficients(fit, draws, rng)
    simulated = np.empty(draws, dtype=float)
    for index, beta in enumerate(coefficient_draws):
        probability = expit(design @ beta)
        labels = rng.random(len(probability)) < probability
        simulated[index] = _rate_contrast(labels, membership, weights)
    upper = float((1 + np.count_nonzero(simulated >= observed)) / (draws + 1))
    lower = float((1 + np.count_nonzero(simulated <= observed)) / (draws + 1))
    return {
        "observed_rate_contrast": observed,
        "predictive_mean": float(simulated.mean()),
        "predictive_std": float(simulated.std(ddof=1)),
        "upper_tail_probability": upper,
        "two_sided_tail_probability": min(1.0, 2.0 * min(upper, lower)),
    }


def run_model_comparison(
    catalogue: pd.DataFrame,
    claim: AuditContract,
    config: ModelConfig,
    *,
    labels_override: np.ndarray | None = None,
    seed: int = 0,
    predictive: bool = True,
) -> dict[str, Any]:
    membership = node_membership(catalogue, claim)
    labels = (
        catalogue["_anomaly"].to_numpy(bool)
        if labels_override is None
        else np.asarray(labels_override, dtype=bool)
    )
    if labels.shape != (len(catalogue),) or np.unique(labels).size < 2:
        raise AuditError("model labels must contain both classes")
    weights = catalogue["_analysis_weight"].to_numpy(float)
    null_design, null_names = build_design(catalogue, membership, config, include_node=False)
    node_design, node_names = build_design(catalogue, membership, config, include_node=True)
    null_fit = _fit_logistic("survey_nuisance_null", null_design, null_names, labels, weights)
    node_fit = _fit_logistic("survey_nuisance_plus_node", node_design, node_names, labels, weights)
    if predictive:
        rng = np.random.default_rng(seed)
        null_check = _predictive_check(
            null_fit, null_design, labels, membership, weights, config.predictive_draws, rng
        )
        node_check = _predictive_check(
            node_fit, node_design, labels, membership, weights, config.predictive_draws, rng
        )
    else:
        null_check = None
        node_check = None
    delta = float(null_fit.pseudo_bic - node_fit.pseudo_bic)
    node_coefficient = float(node_fit.coefficients[-1])
    preferred = bool(
        null_fit.converged
        and node_fit.converged
        and "inside_frozen_node" not in node_fit.bound_hits
        and delta >= config.delta_pseudo_bic_threshold
        and node_coefficient > 0.0
    )
    return {
        "schema": MODEL_SCHEMA,
        "decision": "NODE_TERM_PREFERRED" if preferred else "NODE_TERM_NOT_PREFERRED",
        "delta_pseudo_bic_null_minus_node": delta,
        "threshold": config.delta_pseudo_bic_threshold,
        "node_log_odds_coefficient": node_coefficient,
        "node_odds_ratio": float(math.exp(min(node_coefficient, 700.0))),
        "models": {
            "null": null_fit.to_dict(),
            "node": node_fit.to_dict(),
        },
        "laplace_predictive_checks": (
            {"null": null_check, "node": node_check} if predictive else None
        ),
        "claim_boundary": (
            "The comparison asks whether a frozen node-membership term improves a "
            "declared nuisance model. It does not identify a physical mechanism."
        ),
    }


def _permute_within_strata(
    labels: np.ndarray,
    catalogue: pd.DataFrame,
    config: ModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(labels, dtype=bool).copy()
    if config.stratum_column:
        strata = catalogue[config.stratum_column].astype(str).to_numpy()
        for value in np.unique(strata):
            indexes = np.flatnonzero(strata == value)
            result[indexes] = rng.permutation(result[indexes])
    else:
        result = rng.permutation(result)
    return result


def run_injection_recovery(
    catalogue: pd.DataFrame,
    claim: AuditContract,
    model_config: ModelConfig,
    injection: InjectionConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    if not injection.enabled:
        return {
            "schema": INJECTION_SCHEMA,
            "enabled": False,
            "decision": "INJECTION_DISABLED",
        }
    membership = node_membership(catalogue, claim)
    original = catalogue["_anomaly"].to_numpy(bool)
    weights = catalogue["_analysis_weight"].to_numpy(float)
    null_design, null_names = build_design(
        catalogue, membership, model_config, include_node=False
    )
    fitted_null = _fit_logistic(
        "injection_nuisance_null",
        null_design,
        null_names,
        original,
        weights,
    )
    if not fitted_null.converged:
        raise AuditError("nuisance model did not converge for injection calibration")

    rng = np.random.default_rng(seed)
    coefficient_draws = _draw_coefficients(fitted_null, injection.runs, rng)
    recovered = 0
    false_positives = 0
    injected_counts: list[int] = []
    recovery_deltas: list[float] = []
    null_deltas: list[float] = []

    for run_index, beta in enumerate(coefficient_draws):
        probability = expit(null_design @ beta)
        baseline = rng.random(len(probability)) < probability
        # Rare tiny samples can draw only one class. Retry from the same frozen
        # probability model rather than silently changing the prevalence rule.
        for _ in range(20):
            if np.unique(baseline).size == 2:
                break
            baseline = rng.random(len(probability)) < probability
        else:
            raise AuditError("predictive injection baseline repeatedly lost one class")

        null_result = run_model_comparison(
            catalogue,
            claim,
            model_config,
            labels_override=baseline,
            seed=seed + 10_000 + run_index,
            predictive=False,
        )
        null_delta = float(null_result["delta_pseudo_bic_null_minus_node"])
        null_deltas.append(null_delta)
        if (
            null_delta >= injection.delta_pseudo_bic_threshold
            and float(null_result["node_log_odds_coefficient"]) > 0.0
            and not null_result["models"]["node"]["bound_hits"]
        ):
            false_positives += 1

        injected = baseline.copy()
        count = 0
        for node_index in range(membership.shape[1]):
            candidates = np.flatnonzero(membership[:, node_index] & ~injected)
            if len(candidates) == 0:
                continue
            selected = rng.choice(
                candidates,
                size=min(injection.injected_per_node, len(candidates)),
                replace=False,
            )
            injected[selected] = True
            count += len(selected)
        if count == 0:
            raise AuditError("no non-anomalous catalogue rows are available inside frozen nodes")
        injected_counts.append(count)
        signal_result = run_model_comparison(
            catalogue,
            claim,
            model_config,
            labels_override=injected,
            seed=seed + 20_000 + run_index,
            predictive=False,
        )
        signal_delta = float(signal_result["delta_pseudo_bic_null_minus_node"])
        recovery_deltas.append(signal_delta)
        if (
            signal_delta >= injection.delta_pseudo_bic_threshold
            and float(signal_result["node_log_odds_coefficient"]) > 0.0
            and not signal_result["models"]["node"]["bound_hits"]
        ):
            recovered += 1

    recovery_rate = recovered / injection.runs
    false_positive_rate = false_positives / injection.runs
    passed = bool(
        recovery_rate >= injection.minimum_recovery_rate
        and false_positive_rate <= injection.maximum_false_positive_rate
    )
    return {
        "schema": INJECTION_SCHEMA,
        "enabled": True,
        "decision": "INJECTION_CALIBRATION_PASSED" if passed else "INJECTION_CALIBRATION_FAILED",
        "runs": injection.runs,
        "recovered_runs": recovered,
        "false_positive_runs": false_positives,
        "recovery_rate": recovery_rate,
        "minimum_recovery_rate": injection.minimum_recovery_rate,
        "false_positive_rate": false_positive_rate,
        "maximum_false_positive_rate": injection.maximum_false_positive_rate,
        "injected_rows_min": int(min(injected_counts)),
        "injected_rows_max": int(max(injected_counts)),
        "recovery_delta_pseudo_bic_mean": float(np.mean(recovery_deltas)),
        "null_delta_pseudo_bic_mean": float(np.mean(null_deltas)),
        "calibration_boundary": (
            "Synthetic anomaly labels are drawn from the fitted nuisance-only model, "
            "then injected into actual in-footprint node rows. This preserves declared "
            "stratum and covariate effects while calibrating detection power and false "
            "positives; it is not an image-level telescope simulator."
        ),
    }
