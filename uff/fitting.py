"""Deterministic bounded fitting and model-comparison statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .data import GalaxyData
from .models import RotationCurveModel


@dataclass
class FitResult:
    """Serializable result of one model fit."""

    model_name: str
    model_label: str
    model_family: str
    model_status: str
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    parameter_errors: np.ndarray
    covariance: np.ndarray
    predictions_kms: np.ndarray
    residuals_kms: np.ndarray
    standardized_residuals: np.ndarray
    success: bool
    message: str
    n_points: int
    n_parameters: int
    chi_squared: float
    reduced_chi_squared: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    rmse_kms: float
    bound_hits: tuple[str, ...]
    restarts: int

    @property
    def parameters(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, map(float, self.theta)))

    @property
    def parameter_uncertainties(self) -> dict[str, float | None]:
        return {
            name: (float(error) if np.isfinite(error) else None)
            for name, error in zip(self.parameter_names, self.parameter_errors)
        }

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "model": self.model_name,
            "label": self.model_label,
            "family": self.model_family,
            "status": self.model_status,
            "success": self.success,
            "message": self.message,
            "n_points": self.n_points,
            "n_parameters": self.n_parameters,
            "parameters": self.parameters,
            "parameter_uncertainties": self.parameter_uncertainties,
            "bound_hits": list(self.bound_hits),
            "chi_squared": self.chi_squared,
            "reduced_chi_squared": self.reduced_chi_squared,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "aicc": self.aicc if np.isfinite(self.aicc) else None,
            "bic": self.bic,
            "rmse_kms": self.rmse_kms,
            "restarts": self.restarts,
        }
        if include_arrays:
            result.update(
                {
                    "predictions_kms": self.predictions_kms.tolist(),
                    "residuals_kms": self.residuals_kms.tolist(),
                    "standardized_residuals": self.standardized_residuals.tolist(),
                    "covariance": self.covariance.tolist(),
                }
            )
        return result


def gaussian_log_likelihood(residuals: np.ndarray, errors: np.ndarray) -> float:
    """Return the normalized independent-Gaussian log likelihood."""

    residual = np.asarray(residuals, dtype=float)
    uncertainty = np.asarray(errors, dtype=float)
    if residual.shape != uncertainty.shape:
        raise ValueError("residuals and errors must have the same shape")
    if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(uncertainty)):
        return -math.inf
    if np.any(uncertainty <= 0):
        raise ValueError("errors must be positive")
    return float(
        -0.5
        * np.sum(
            np.square(residual / uncertainty) + np.log(2.0 * math.pi * uncertainty**2)
        )
    )


def information_criteria(
    log_likelihood: float, n_parameters: int, n_points: int
) -> dict[str, float]:
    """Return AIC, finite-sample AICc, and BIC for one likelihood."""

    k = int(n_parameters)
    n = int(n_points)
    if k < 0 or n <= 0:
        raise ValueError("n_parameters must be non-negative and n_points positive")
    aic = 2.0 * k - 2.0 * log_likelihood
    denominator = n - k - 1
    aicc = aic + 2.0 * k * (k + 1) / denominator if denominator > 0 else math.inf
    bic = k * math.log(n) - 2.0 * log_likelihood
    return {"aic": float(aic), "aicc": float(aicc), "bic": float(bic)}


def _covariance_from_jacobian(
    jacobian: np.ndarray, chi_squared: float, dof: int
) -> np.ndarray:
    if jacobian.size == 0:
        return np.empty((0, 0), dtype=float)
    try:
        information = jacobian.T @ jacobian
        scale = chi_squared / dof if dof > 0 else 1.0
        covariance = np.linalg.pinv(information, hermitian=True) * scale
        covariance = 0.5 * (covariance + covariance.T)
        if np.any(~np.isfinite(covariance)):
            raise np.linalg.LinAlgError
        return covariance
    except np.linalg.LinAlgError:
        return np.full((jacobian.shape[1], jacobian.shape[1]), np.nan)


def _make_starts(
    model: RotationCurveModel,
    restarts: int,
    random_state: int,
) -> list[np.ndarray]:
    if not model.parameters:
        return [np.empty(0, dtype=float)]
    count = max(1, int(restarts))
    lower = model.lower_bounds
    upper = model.upper_bounds
    initial = model.initial
    starts = [initial]
    rng = np.random.default_rng(random_state)
    for index in range(1, count):
        if index % 2:
            # Explore broadly across the declared prior/bound box.
            candidate = rng.uniform(lower, upper)
        else:
            # Also examine the scientifically motivated initial neighbourhood.
            width = 0.15 * (upper - lower)
            candidate = np.clip(initial + rng.normal(0.0, width), lower, upper)
        epsilon = np.maximum(1.0e-10 * (upper - lower), 1.0e-12)
        starts.append(np.clip(candidate, lower + epsilon, upper - epsilon))
    return starts


def fit_model(
    model: RotationCurveModel,
    data: GalaxyData,
    *,
    restarts: int = 12,
    random_state: int = 42,
    systematic_kms: float = 0.0,
    max_nfev: int = 20_000,
) -> FitResult:
    """Fit one model with deterministic bounded multi-start least squares."""

    systematic = float(systematic_kms)
    if not math.isfinite(systematic) or systematic < 0:
        raise ValueError("systematic_kms must be finite and non-negative")
    errors = np.sqrt(data.velocity_err_kms**2 + systematic**2)

    def residual_function(theta: np.ndarray) -> np.ndarray:
        try:
            prediction = model.predict(data.radius_kpc, theta)
            residual = (data.velocity_obs_kms - prediction) / errors
            if np.any(~np.isfinite(residual)):
                raise ValueError
            return residual
        except (FloatingPointError, OverflowError, ValueError):
            return np.full(data.n_points, 1.0e12)

    starts = _make_starts(model, restarts, random_state)
    best = None
    best_cost = math.inf
    messages: list[str] = []

    if not model.parameters:
        theta = np.empty(0, dtype=float)
        standardized = residual_function(theta)
        best_cost = 0.5 * float(np.dot(standardized, standardized))
        best = type(
            "FixedResult",
            (),
            {
                "x": theta,
                "jac": np.empty((data.n_points, 0)),
                "success": True,
                "message": "fixed model",
            },
        )()
    else:
        for start in starts:
            result = least_squares(
                residual_function,
                start,
                bounds=(model.lower_bounds, model.upper_bounds),
                method="trf",
                x_scale="jac",
                loss="linear",
                max_nfev=max_nfev,
            )
            messages.append(str(result.message))
            if np.isfinite(result.cost) and result.cost < best_cost:
                best = result
                best_cost = float(result.cost)

    if best is None:
        raise RuntimeError(f"all optimization starts failed for model {model.name}")

    theta = np.asarray(best.x, dtype=float)
    prediction = model.predict(data.radius_kpc, theta)
    residual = data.velocity_obs_kms - prediction
    standardized = residual / errors
    chi_squared = float(np.dot(standardized, standardized))
    k = len(model.parameters)
    dof = data.n_points - k
    covariance = _covariance_from_jacobian(np.asarray(best.jac), chi_squared, dof)
    parameter_errors = (
        np.sqrt(np.maximum(np.diag(covariance), 0.0)) if k else np.empty(0, dtype=float)
    )
    log_likelihood = gaussian_log_likelihood(residual, errors)
    criteria = information_criteria(log_likelihood, k, data.n_points)

    bound_hits: list[str] = []
    if k:
        fraction = (theta - model.lower_bounds) / (
            model.upper_bounds - model.lower_bounds
        )
        for parameter, value in zip(model.parameters, fraction):
            if value <= 1.0e-4:
                bound_hits.append(f"{parameter.name}:lower")
            elif value >= 1.0 - 1.0e-4:
                bound_hits.append(f"{parameter.name}:upper")

    return FitResult(
        model_name=model.name,
        model_label=model.label,
        model_family=model.family,
        model_status=model.status,
        parameter_names=model.parameter_names,
        theta=theta,
        parameter_errors=parameter_errors,
        covariance=covariance,
        predictions_kms=prediction,
        residuals_kms=residual,
        standardized_residuals=standardized,
        success=bool(best.success) and np.all(np.isfinite(prediction)),
        message=str(best.message),
        n_points=data.n_points,
        n_parameters=k,
        chi_squared=chi_squared,
        reduced_chi_squared=chi_squared / dof if dof > 0 else math.nan,
        log_likelihood=log_likelihood,
        aic=criteria["aic"],
        aicc=criteria["aicc"],
        bic=criteria["bic"],
        rmse_kms=float(np.sqrt(np.mean(np.square(residual)))),
        bound_hits=tuple(bound_hits),
        restarts=len(starts),
    )


def fit_models(
    models: Iterable[RotationCurveModel],
    data: GalaxyData,
    **fit_kwargs: object,
) -> list[FitResult]:
    """Fit several models and return them ordered by BIC."""

    results = [fit_model(model, data, **fit_kwargs) for model in models]
    return sorted(results, key=lambda result: (not result.success, result.bic))


def model_weights(
    results: Iterable[FitResult],
    criterion: str = "bic",
) -> dict[str, float]:
    """Return normalized relative weights from AIC, AICc, or BIC."""

    items = list(results)
    key = criterion.casefold()
    if key not in {"aic", "aicc", "bic"}:
        raise ValueError("criterion must be one of: aic, aicc, bic")
    values = np.array([float(getattr(result, key)) for result in items])
    finite = np.isfinite(values)
    if not np.any(finite):
        return {result.model_name: math.nan for result in items}
    minimum = float(np.min(values[finite]))
    relative = np.zeros_like(values)
    relative[finite] = np.exp(-0.5 * (values[finite] - minimum))
    total = float(np.sum(relative))
    return {
        result.model_name: float(weight / total)
        for result, weight in zip(items, relative)
    }


def comparison_records(results: Iterable[FitResult]) -> list[dict[str, object]]:
    """Create flat records with delta-BIC and normalized BIC weights."""

    items = sorted(results, key=lambda result: result.bic)
    if not items:
        return []
    weights = model_weights(items, "bic")
    minimum_bic = min(item.bic for item in items)
    return [
        {
            "model": item.model_name,
            "label": item.model_label,
            "family": item.model_family,
            "status": item.model_status,
            "success": item.success,
            "n_parameters": item.n_parameters,
            "chi_squared": item.chi_squared,
            "reduced_chi_squared": item.reduced_chi_squared,
            "rmse_kms": item.rmse_kms,
            "log_likelihood": item.log_likelihood,
            "aic": item.aic,
            "aicc": item.aicc,
            "bic": item.bic,
            "delta_bic": item.bic - minimum_bic,
            "bic_weight": weights[item.model_name],
            "bound_hits": ";".join(item.bound_hits),
        }
        for item in items
    ]


__all__ = [
    "FitResult",
    "comparison_records",
    "fit_model",
    "fit_models",
    "gaussian_log_likelihood",
    "information_criteria",
    "model_weights",
]
