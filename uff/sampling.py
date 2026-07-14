"""Opt-in bounded posterior sampling with burn-in-only adaptation.

The optimizer is UFF's fast default.  This module is for uncertainty work where
users want retained posterior draws.  Proposal adaptation stops at the end of
burn-in so the retained samples use a fixed transition kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .data import GalaxyData
from .fitting import FitResult, gaussian_log_likelihood
from .models import RotationCurveModel


@dataclass
class PosteriorResult:
    model_name: str
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    log_likelihood: np.ndarray
    acceptance_rates: np.ndarray
    rhat: np.ndarray
    effective_sample_size: np.ndarray
    steps: int
    burn: int
    thin: int
    seed: int

    @property
    def combined_samples(self) -> np.ndarray:
        return self.samples.reshape(-1, self.samples.shape[-1])

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "parameter_names": list(self.parameter_names),
            "n_chains": int(self.samples.shape[0]),
            "draws_per_chain": int(self.samples.shape[1]),
            "steps": self.steps,
            "burn": self.burn,
            "thin": self.thin,
            "seed": self.seed,
            "acceptance_rates": self.acceptance_rates.tolist(),
            "rhat": dict(zip(self.parameter_names, map(float, self.rhat))),
            "effective_sample_size": dict(
                zip(self.parameter_names, map(float, self.effective_sample_size))
            ),
            "parameter_quantiles": {
                name: {
                    "p16": float(np.percentile(self.combined_samples[:, index], 16)),
                    "median": float(np.percentile(self.combined_samples[:, index], 50)),
                    "p84": float(np.percentile(self.combined_samples[:, index], 84)),
                }
                for index, name in enumerate(self.parameter_names)
            },
            "adaptation": "full covariance during burn-in only; fixed kernel for retained draws",
        }


def _regularized_covariance(fit: FitResult, model: RotationCurveModel) -> np.ndarray:
    dimension = len(model.parameters)
    span = model.upper_bounds - model.lower_bounds
    minimum_variance = np.square(np.maximum(span * 1.0e-6, 1.0e-10))
    fallback = np.diag(np.square(np.maximum(span * 0.02, 1.0e-8)))
    covariance = np.asarray(fit.covariance, dtype=float)
    if covariance.shape != (dimension, dimension) or np.any(~np.isfinite(covariance)):
        return fallback
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum_variance = np.square(np.maximum(span * 0.25, 1.0e-7))
    eigenvalues = np.clip(
        eigenvalues, np.min(minimum_variance), np.max(maximum_variance)
    )
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    regularized += np.diag(minimum_variance)
    return regularized


def _gelman_rubin(samples: np.ndarray) -> np.ndarray:
    chains, draws, dimension = samples.shape
    if chains < 2 or draws < 2:
        return np.full(dimension, np.nan)
    chain_means = np.mean(samples, axis=1)
    within = np.mean(np.var(samples, axis=1, ddof=1), axis=0)
    between = draws * np.var(chain_means, axis=0, ddof=1)
    variance = ((draws - 1.0) / draws) * within + between / draws
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(variance / within)
    return np.maximum(np.where(within == 0, 1.0, rhat), 1.0)


def _effective_sample_size(samples: np.ndarray) -> np.ndarray:
    chains, draws, dimension = samples.shape
    total = chains * draws
    if draws < 4:
        return np.full(dimension, float(total))
    result = np.empty(dimension, dtype=float)
    for parameter in range(dimension):
        values = samples[:, :, parameter]
        centered = values - np.mean(values, axis=1, keepdims=True)
        variance = float(np.mean(np.square(centered)))
        if variance <= 0 or not math.isfinite(variance):
            result[parameter] = float(total)
            continue
        autocorrelation_sum = 0.0
        # Initial-positive-sequence approximation, capped for predictable cost.
        maximum_lag = min(draws - 1, 1_000)
        previous_pair = math.inf
        for lag in range(1, maximum_lag, 2):
            pair = 0.0
            for offset in (lag, lag + 1):
                if offset >= draws:
                    continue
                covariance = float(
                    np.mean(centered[:, :-offset] * centered[:, offset:])
                )
                pair += covariance / variance
            if pair <= 0:
                break
            pair = min(pair, previous_pair)
            autocorrelation_sum += pair
            previous_pair = pair
        result[parameter] = min(
            float(total), total / max(1.0 + 2.0 * autocorrelation_sum, 1.0)
        )
    return result


def sample_posterior(
    model: RotationCurveModel,
    data: GalaxyData,
    fit: FitResult,
    *,
    steps: int = 8_000,
    burn: int = 2_500,
    thin: int = 5,
    n_chains: int = 4,
    seed: int = 42,
    systematic_kms: float = 0.0,
) -> PosteriorResult:
    """Sample a bounded, uniform-prior posterior around a fitted solution."""

    dimension = len(model.parameters)
    if dimension == 0:
        raise ValueError("posterior sampling requires at least one free parameter")
    if steps <= 0 or burn < 0 or burn >= steps:
        raise ValueError("require steps > burn >= 0")
    if thin <= 0 or n_chains < 2:
        raise ValueError("thin must be positive and n_chains at least two")
    retained = (steps - burn + thin - 1) // thin
    if retained < 20:
        raise ValueError("sampling configuration retains fewer than 20 draws per chain")
    if systematic_kms < 0 or not math.isfinite(systematic_kms):
        raise ValueError("systematic_kms must be finite and non-negative")

    errors = np.sqrt(data.velocity_err_kms**2 + systematic_kms**2)
    lower = model.lower_bounds
    upper = model.upper_bounds
    span = upper - lower
    base_covariance = _regularized_covariance(fit, model)
    base_covariance *= (2.38**2 / dimension) * 0.25

    def log_probability(theta: np.ndarray) -> float:
        if (
            np.any(theta < lower)
            or np.any(theta > upper)
            or np.any(~np.isfinite(theta))
        ):
            return -math.inf
        try:
            prediction = model.predict(data.radius_kpc, theta)
        except (FloatingPointError, OverflowError, ValueError):
            return -math.inf
        return gaussian_log_likelihood(data.velocity_obs_kms - prediction, errors)

    all_samples = np.empty((n_chains, retained, dimension), dtype=float)
    all_log_likelihood = np.empty((n_chains, retained), dtype=float)
    acceptance_rates = np.empty(n_chains, dtype=float)
    master = np.random.SeedSequence(seed)

    for chain_index, child_seed in enumerate(master.spawn(n_chains)):
        rng = np.random.default_rng(child_seed)
        covariance = base_covariance.copy()
        # Disperse chains locally while respecting the declared parameter box.
        for _ in range(1_000):
            start = fit.theta + rng.multivariate_normal(
                np.zeros(dimension), covariance * 0.1
            )
            if np.all(start > lower) and np.all(start < upper):
                break
        else:
            start = np.clip(fit.theta, lower + 1.0e-8 * span, upper - 1.0e-8 * span)
        theta = start
        current = log_probability(theta)
        history: list[np.ndarray] = []
        accepted = 0
        window_accepted = 0
        write_index = 0

        for step in range(steps):
            proposal = theta + rng.multivariate_normal(np.zeros(dimension), covariance)
            proposed = log_probability(proposal)
            if math.log(rng.random()) < proposed - current:
                theta = proposal
                current = proposed
                accepted += 1
                window_accepted += 1

            if step < burn:
                history.append(theta.copy())
                if (step + 1) % 100 == 0:
                    local_rate = window_accepted / 100.0
                    window_accepted = 0
                    multiplier = math.exp(np.clip(local_rate - 0.234, -0.5, 0.5))
                    if len(history) >= max(200, 10 * dimension):
                        empirical = np.cov(np.asarray(history).T, ddof=1)
                        empirical = np.atleast_2d(empirical)
                        jitter = np.diag(np.square(np.maximum(span * 1.0e-6, 1.0e-10)))
                        covariance = (2.38**2 / dimension) * empirical + jitter
                    covariance *= multiplier
            elif (step - burn) % thin == 0:
                all_samples[chain_index, write_index] = theta
                all_log_likelihood[chain_index, write_index] = current
                write_index += 1

        acceptance_rates[chain_index] = accepted / steps

    return PosteriorResult(
        model_name=model.name,
        parameter_names=model.parameter_names,
        samples=all_samples,
        log_likelihood=all_log_likelihood,
        acceptance_rates=acceptance_rates,
        rhat=_gelman_rubin(all_samples),
        effective_sample_size=_effective_sample_size(all_samples),
        steps=int(steps),
        burn=int(burn),
        thin=int(thin),
        seed=int(seed),
    )


__all__ = ["PosteriorResult", "sample_posterior"]
