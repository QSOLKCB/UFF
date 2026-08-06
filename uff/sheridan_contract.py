"""Frozen survey-aware contract validation for the Sheridan Crucible.

The contract nests an ordinary UFF-SLFA claim and adds explicit survey,
density, model-comparison, injection, and decision rules.  No value is inferred
from the observed result after the run starts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sky_contract import (
    AuditContract,
    AuditError,
    load_contract_payload,
    load_json,
)

SHERIDAN_CONTRACT_SCHEMA = "uff.sheridan-crucible.v1"


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    area_weight_column: str
    coverage_column: str
    completeness_column: str | None
    minimum_completeness: float


@dataclass(frozen=True, slots=True)
class DensityConfig:
    bandwidth_candidates_deg: tuple[float, ...]
    adaptive_alpha: float
    minimum_bandwidth_factor: float
    maximum_bandwidth_factor: float
    permutations: int
    seed: int
    availability_tolerance: float
    minimum_availability: float
    maximum_rotation_attempt_multiplier: int
    maximum_exact_sources: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    stratum_column: str | None
    covariate_columns: tuple[str, ...]
    predictive_draws: int
    delta_pseudo_bic_threshold: float


@dataclass(frozen=True, slots=True)
class InjectionConfig:
    enabled: bool
    runs: int
    injected_per_node: int
    minimum_recovery_rate: float
    maximum_false_positive_rate: float
    delta_pseudo_bic_threshold: float


@dataclass(frozen=True, slots=True)
class SheridanDecision:
    alpha: float
    minimum_mean_overdensity: float
    minimum_supported_nodes: int
    required_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SheridanContract:
    title: str
    claim: AuditContract
    claim_payload: dict[str, Any]
    survey: SurveyConfig
    density: DensityConfig
    models: ModelConfig
    injection: InjectionConfig
    decision: SheridanDecision


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite")
    return result


def _column(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = str(value or "").strip()
    if not result:
        raise AuditError(f"{label} must be a non-empty column name")
    return result


def load_sheridan_contract_payload(payload: dict[str, Any]) -> SheridanContract:
    if payload.get("schema") != SHERIDAN_CONTRACT_SCHEMA:
        raise AuditError(f"contract schema must be {SHERIDAN_CONTRACT_SCHEMA!r}")
    title = str(payload.get("title", "")).strip()
    if not title:
        raise AuditError("title is required")

    claim_payload = _mapping(payload.get("claim"), "claim")
    claim = load_contract_payload(claim_payload)

    survey_raw = _mapping(payload.get("survey", {}), "survey")
    area_column = _column(
        survey_raw.get("area_weight_column", "area_weight_sr"),
        "survey.area_weight_column",
    )
    coverage_column = _column(
        survey_raw.get("coverage_column", "coverage"),
        "survey.coverage_column",
    )
    completeness_column = _column(
        survey_raw.get("completeness_column"),
        "survey.completeness_column",
        optional=True,
    )
    if area_column == coverage_column:
        raise AuditError("survey area-weight and coverage columns must be different")
    if completeness_column and completeness_column == claim.weight_column:
        raise AuditError(
            "the completeness column cannot also be the SLFA weight column; "
            "that would apply the same correction twice"
        )
    minimum_completeness = _number(
        survey_raw.get("minimum_completeness", 0.2),
        "survey.minimum_completeness",
    )
    if not 0.0 < minimum_completeness <= 1.0:
        raise AuditError("survey.minimum_completeness must be in (0, 1]")
    survey = SurveyConfig(
        area_weight_column=str(area_column),
        coverage_column=str(coverage_column),
        completeness_column=completeness_column,
        minimum_completeness=minimum_completeness,
    )

    density_raw = _mapping(payload.get("density", {}), "density")
    candidates_raw = density_raw.get("bandwidth_candidates_deg", [1.0, 2.0, 4.0, 8.0])
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise AuditError("density.bandwidth_candidates_deg must be a non-empty list")
    candidates = tuple(sorted({_number(item, "density bandwidth") for item in candidates_raw}))
    if any(not 0.05 <= item <= 45.0 for item in candidates):
        raise AuditError("density bandwidth candidates must be in [0.05, 45] degrees")
    alpha = _number(density_raw.get("adaptive_alpha", 0.5), "density.adaptive_alpha")
    if not 0.0 <= alpha <= 1.0:
        raise AuditError("density.adaptive_alpha must be in [0, 1]")
    minimum_factor = _number(
        density_raw.get("minimum_bandwidth_factor", 0.5),
        "density.minimum_bandwidth_factor",
    )
    maximum_factor = _number(
        density_raw.get("maximum_bandwidth_factor", 2.0),
        "density.maximum_bandwidth_factor",
    )
    if not 0.0 < minimum_factor <= 1.0 <= maximum_factor:
        raise AuditError(
            "density bandwidth factors must satisfy 0 < minimum <= 1 <= maximum"
        )
    permutations = int(density_raw.get("permutations", 999))
    if permutations < 99:
        raise AuditError("density.permutations must be at least 99")
    availability_tolerance = _number(
        density_raw.get("availability_tolerance", 0.20),
        "density.availability_tolerance",
    )
    if availability_tolerance < 0.0:
        raise AuditError("density.availability_tolerance must be non-negative")
    minimum_availability = _number(
        density_raw.get("minimum_availability", 0.02),
        "density.minimum_availability",
    )
    if not 0.0 <= minimum_availability <= 1.0:
        raise AuditError("density.minimum_availability must be in [0, 1]")
    attempt_multiplier = int(density_raw.get("maximum_rotation_attempt_multiplier", 200))
    if attempt_multiplier < 1:
        raise AuditError("density.maximum_rotation_attempt_multiplier must be positive")
    maximum_exact_sources = int(density_raw.get("maximum_exact_sources", 3000))
    if maximum_exact_sources < 32:
        raise AuditError("density.maximum_exact_sources must be at least 32")
    density = DensityConfig(
        bandwidth_candidates_deg=candidates,
        adaptive_alpha=alpha,
        minimum_bandwidth_factor=minimum_factor,
        maximum_bandwidth_factor=maximum_factor,
        permutations=permutations,
        seed=int(density_raw.get("seed", claim.seed)),
        availability_tolerance=availability_tolerance,
        minimum_availability=minimum_availability,
        maximum_rotation_attempt_multiplier=attempt_multiplier,
        maximum_exact_sources=maximum_exact_sources,
    )

    models_raw = _mapping(payload.get("models", {}), "models")
    stratum_column = _column(
        models_raw.get("stratum_column"),
        "models.stratum_column",
        optional=True,
    )
    covariates_raw = models_raw.get("covariate_columns", [])
    if not isinstance(covariates_raw, list):
        raise AuditError("models.covariate_columns must be a list")
    covariates = tuple(str(item).strip() for item in covariates_raw)
    if any(not item for item in covariates) or len(set(covariates)) != len(covariates):
        raise AuditError("models.covariate_columns must be unique non-empty names")
    if stratum_column in covariates:
        raise AuditError("models.stratum_column cannot also be a numeric covariate")
    predictive_draws = int(models_raw.get("predictive_draws", 999))
    if predictive_draws < 99:
        raise AuditError("models.predictive_draws must be at least 99")
    models = ModelConfig(
        stratum_column=stratum_column,
        covariate_columns=covariates,
        predictive_draws=predictive_draws,
        delta_pseudo_bic_threshold=_number(
            models_raw.get("delta_pseudo_bic_threshold", 6.0),
            "models.delta_pseudo_bic_threshold",
        ),
    )

    injection_raw = _mapping(payload.get("injection", {}), "injection")
    injection = InjectionConfig(
        enabled=injection_raw.get("enabled", True) is True,
        runs=int(injection_raw.get("runs", 50)),
        injected_per_node=int(injection_raw.get("injected_per_node", 5)),
        minimum_recovery_rate=_number(
            injection_raw.get("minimum_recovery_rate", 0.80),
            "injection.minimum_recovery_rate",
        ),
        maximum_false_positive_rate=_number(
            injection_raw.get("maximum_false_positive_rate", 0.10),
            "injection.maximum_false_positive_rate",
        ),
        delta_pseudo_bic_threshold=_number(
            injection_raw.get(
                "delta_pseudo_bic_threshold",
                models.delta_pseudo_bic_threshold,
            ),
            "injection.delta_pseudo_bic_threshold",
        ),
    )
    if injection.runs < 1 or injection.injected_per_node < 1:
        raise AuditError("injection runs and injected_per_node must be positive")
    if not 0.0 <= injection.minimum_recovery_rate <= 1.0:
        raise AuditError("injection.minimum_recovery_rate must be in [0, 1]")
    if not 0.0 <= injection.maximum_false_positive_rate <= 1.0:
        raise AuditError("injection.maximum_false_positive_rate must be in [0, 1]")

    decision_raw = _mapping(payload.get("decision_rule", {}), "decision_rule")
    required_raw = decision_raw.get(
        "required_components", ["density", "model", "injection"]
    )
    if not isinstance(required_raw, list):
        raise AuditError("decision_rule.required_components must be a list")
    required_components = tuple(str(item).strip() for item in required_raw)
    allowed_components = {"density", "model", "injection"}
    if (
        not required_components
        or len(set(required_components)) != len(required_components)
        or any(item not in allowed_components for item in required_components)
    ):
        raise AuditError(
            "decision_rule.required_components must be a unique non-empty subset of "
            "density, model, injection"
        )
    if "injection" in required_components and not injection.enabled:
        raise AuditError("injection cannot be required when injection.enabled is false")
    decision = SheridanDecision(
        alpha=_number(decision_raw.get("alpha", claim.alpha), "decision_rule.alpha"),
        minimum_mean_overdensity=_number(
            decision_raw.get("minimum_mean_overdensity", 0.10),
            "decision_rule.minimum_mean_overdensity",
        ),
        minimum_supported_nodes=int(
            decision_raw.get("minimum_supported_nodes", claim.minimum_supported_nodes)
        ),
        required_components=required_components,
    )
    if not 0.0 < decision.alpha < 1.0:
        raise AuditError("decision_rule.alpha must be in (0, 1)")
    if not 0 <= decision.minimum_supported_nodes <= len(claim.nodes):
        raise AuditError("decision_rule.minimum_supported_nodes is outside the node count")

    return SheridanContract(
        title=title,
        claim=claim,
        claim_payload=claim_payload,
        survey=survey,
        density=density,
        models=models,
        injection=injection,
        decision=decision,
    )


def load_sheridan_contract(path: Path) -> tuple[SheridanContract, dict[str, Any]]:
    payload = load_json(path)
    return load_sheridan_contract_payload(payload), payload


def load_support(path: Path, config: SurveyConfig) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path).copy()
    except Exception as exc:
        raise AuditError(f"cannot read survey-support CSV {path}: {exc}") from exc
    required = {"ra_deg", "dec_deg", config.area_weight_column, config.coverage_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditError(f"survey support is missing columns: {', '.join(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[list(required)].to_numpy(float)
    if len(frame) < 32 or not np.all(np.isfinite(values)):
        raise AuditError("survey support requires at least 32 finite quadrature points")
    if not frame["ra_deg"].between(0.0, 360.0, inclusive="left").all():
        raise AuditError("survey-support ra_deg must be in [0, 360)")
    if not frame["dec_deg"].between(-90.0, 90.0).all():
        raise AuditError("survey-support dec_deg must be in [-90, 90]")
    area = frame[config.area_weight_column].to_numpy(float)
    coverage = frame[config.coverage_column].to_numpy(float)
    if np.any(area <= 0.0):
        raise AuditError("survey-support area weights must be strictly positive")
    if np.any((coverage < 0.0) | (coverage > 1.0)):
        raise AuditError("survey-support coverage must be in [0, 1]")
    total_area = float(area.sum())
    if total_area > 4.0 * math.pi * (1.0 + 1.0e-6):
        raise AuditError("survey-support area weights exceed the full-sky solid angle")
    usable_area = float(np.dot(area, coverage))
    if usable_area <= 0.0:
        raise AuditError("survey support has zero usable area")
    frame["_area_weight_sr"] = area
    frame["_coverage"] = coverage
    return frame.reset_index(drop=True)


def prepare_catalogue(
    frame: pd.DataFrame,
    config: SurveyConfig,
    models: ModelConfig,
) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    required = set(models.covariate_columns)
    if models.stratum_column:
        required.add(models.stratum_column)
    if config.completeness_column:
        required.add(config.completeness_column)
    missing = sorted(required - set(result.columns))
    if missing:
        raise AuditError(f"catalogue is missing Sheridan columns: {', '.join(missing)}")

    removed = 0
    if config.completeness_column:
        column = config.completeness_column
        result[column] = pd.to_numeric(result[column], errors="raise")
        completeness = result[column].to_numpy(float)
        if np.any(~np.isfinite(completeness)) or np.any((completeness <= 0.0) | (completeness > 1.0)):
            raise AuditError("catalogue completeness values must be finite and in (0, 1]")
        keep = completeness >= config.minimum_completeness
        removed = int(np.count_nonzero(~keep))
        result = result.loc[keep].copy()
        if result.empty:
            raise AuditError("minimum completeness cut removed the entire catalogue")
        completeness = result[column].to_numpy(float)
    else:
        completeness = np.ones(len(result), dtype=float)

    for column in models.covariate_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.all(np.isfinite(result[column].to_numpy(float))):
            raise AuditError(f"model covariate {column!r} must be finite")
    if models.stratum_column and result[models.stratum_column].isna().any():
        raise AuditError("model stratum values must be non-missing")

    weights = result["_weight"].to_numpy(float) / completeness
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise AuditError("combined analysis weights must be finite and positive")
    # Normalize pseudo-likelihood weights to one row-equivalent per record.
    result["_analysis_weight"] = weights * len(weights) / float(weights.sum())
    if result["_anomaly"].nunique() < 2:
        raise AuditError("completeness filtering removed one anomaly class")
    return result.reset_index(drop=True), removed


def fibonacci_support(points: int) -> pd.DataFrame:
    """Return deterministic equal-area full-sky quadrature points."""
    if points < 32:
        raise AuditError("a Fibonacci support grid requires at least 32 points")
    indexes = np.arange(points, dtype=float)
    z = 1.0 - 2.0 * (indexes + 0.5) / points
    phi = (math.pi * (3.0 - math.sqrt(5.0)) * indexes) % (2.0 * math.pi)
    dec = np.rad2deg(np.arcsin(z))
    ra = np.rad2deg(phi)
    return pd.DataFrame(
        {
            "ra_deg": ra,
            "dec_deg": dec,
            "area_weight_sr": np.full(points, 4.0 * math.pi / points),
            "coverage": np.ones(points),
        }
    )
