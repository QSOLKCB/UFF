import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Local imports
from uff_model import v_circ_uff

"""
Analyze SPARC-like rotation curve data using a Metropolis–Hastings sampler.

This script provides both single-galaxy and batch modes. In batch mode it
accumulates summary statistics across many CSV files and writes a merged
benchmark table. A number of optional model extensions are available:

 - Adaptive proposal covariance: automatically tunes the proposal width
   based on the recent acceptance rate in the MCMC sampler. This helps
   improve mixing without manual tuning of `step_scale`.
 - Baryonic feedback parameters: scale factors for the gas, disk and
   bulge rotation curves, allowing the model to incorporate baryonic
   dynamics rather than fitting purely to the observed velocities.
 - Dark‐field interactions: an additive power‑law term added to the
   theoretical UFF circular velocity. This can mimic the effect of a
   long‑range scalar field or other exotic physics.
 - 3D E₈ visualization: if enabled via a command-line flag, an
   additional figure is written showing a 3D projection of the E₈ root
   lattice. This is provided as a playful hook for exploring
   high‑dimensional symmetry structures using Matplotlib.

To fit a single galaxy:
    python analyze_sparc.py --csv DEMO_GALAXY.csv --gal DEMO_GALAXY

To run a batch benchmark on a directory of CSVs:
    python analyze_sparc.py --batch data/ --out outputs
"""


# ---------- Utilities ----------

def load_sparc_csv(path: str) -> pd.DataFrame:
    """Load a SPARC-style CSV file.

    Required columns: R_kpc, V_obs_kms, e_V_kms.
    Optional baryonic columns (gas, disk, bulge) will be filled with zeros
    if missing.
    """
    df = pd.read_csv(path)
    for col in ["R_kpc", "V_obs_kms", "e_V_kms"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Fill optional baryonic contributions if not provided
    for col in ["V_gas_kms", "V_disk_kms", "V_bul_kms"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def model_v(R: np.ndarray, theta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Compute the total model velocity for radius array `R`.

    The parameter vector theta is interpreted as follows:

    - theta[0], theta[1], theta[2] = V0, Rc, beta for the base UFF law
    - theta[3], theta[4], theta[5] = scale factors for gas, disk and bulge
      baryonic velocities. If omitted, these default to zero (no baryons).
    - theta[6] = dark‑field amplitude
    - theta[7] = dark‑field exponent

    The returned velocity combines baryons in quadrature and adds a power‑law
    correction to the UFF velocity to mimic dark‑field interactions.
    """
    # Base UFF parameters
    V0, Rc, beta = theta[:3]
    V_uff = v_circ_uff(R, np.array([V0, Rc, beta]))
    # Default values for extensions
    f_gas = f_disk = f_bul = 0.0
    df_amp = 0.0
    df_exp = 0.0
    # Override if provided
    if len(theta) >= 6:
        f_gas, f_disk, f_bul = theta[3:6]
    if len(theta) >= 7:
        df_amp = theta[6]
    if len(theta) >= 8:
        df_exp = theta[7]
    # Baryonic contributions
    baryon2 = (
        (f_gas * df["V_gas_kms"].values)**2
        + (f_disk * df["V_disk_kms"].values)**2
        + (f_bul * df["V_bul_kms"].values)**2
    )
    # Dark‑field additive term
    dark_field = df_amp * (np.asarray(R, dtype=float) ** df_exp)
    # Combine UFF and dark field, then quadrature sum with baryons
    total = np.sqrt(baryon2 + (V_uff + dark_field)**2)
    return total


def neg_loglike(theta: np.ndarray, R: np.ndarray, V: np.ndarray, eV: np.ndarray, df: pd.DataFrame) -> float:
    """Return the negative log-likelihood under Gaussian errors.

    Parameters
    ----------
    theta : array-like
        Parameter vector. See ``model_v`` for ordering and interpretation.
    R : ndarray
        Radii in kiloparsecs.
    V : ndarray
        Observed velocities in km/s.
    eV : ndarray
        Measurement uncertainties in km/s.
    df : DataFrame
        Full SPARC dataset, used for baryonic contributions.
    """
    Vmod = model_v(R, theta, df)
    chi2 = np.sum(((V - Vmod) / eV) ** 2)
    return 0.5 * chi2 + np.sum(np.log(eV + 1e-12))


def prior_logprob(theta: np.ndarray, priors: Dict[str, Tuple[float, float]]) -> float:
    """Compute the log prior probability for a parameter vector.

    Uniform priors are assumed on each parameter, defined by the
    corresponding (lo, hi) tuple in ``priors``. If any parameter falls
    outside its bounds the log prior is −∞.
    """
    names = list(priors.keys())
    for i, name in enumerate(names):
        lo, hi = priors[name]
        if not (lo <= theta[i] <= hi):
            return -np.inf
    # All parameters within bounds → constant offset ignored
    return 0.0


def posterior_logprob(theta: np.ndarray, R: np.ndarray, V: np.ndarray, eV: np.ndarray,
                       df: pd.DataFrame, priors: Dict[str, Tuple[float, float]]) -> float:
    """Log posterior = log prior − negative log likelihood."""
    lp = prior_logprob(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp - neg_loglike(theta, R, V, eV, df)


def mh_sampler(
    theta0: np.ndarray,
    logprob_fn,
    steps: int = 20000,
    step_scale: float = 0.05,
    burn: int = 5000,
    thin: int = 5,
    random_state: int = 0,
    adapt: bool = True,
    full_cov: bool = False,
) -> Tuple[np.ndarray, float]:
    """Run a Metropolis–Hastings chain with optional adaptive proposal widths.

    Parameters
    ----------
    theta0 : array-like
        Initial parameter vector.
    logprob_fn : callable
        Function mapping a parameter vector to log probability.
    steps : int
        Total number of MCMC steps to run.
    step_scale : float
        Baseline proposal scale as a fraction of |theta| + 1. Ignored when
        adaptation is disabled.
    burn : int
        Number of steps to discard before thinning.
    thin : int
        Record every ``thin``-th sample after burn-in.
    random_state : int
        Seed for the random number generator.
    adapt : bool
        If True, adapt the proposal widths based on recent acceptance
        statistics. A simple Robbins–Monro scheme is used to tune the
        acceptance rate towards ~0.25.

    Returns
    -------
    chain : ndarray
        Recorded samples with shape (n_samples, ndim+1). The last column is
        the log posterior value.
    accept_rate : float
        Overall acceptance fraction.
    """
    rng = np.random.default_rng(random_state)
    theta = np.array(theta0, dtype=float)
    ndim = theta.size
    # Initialize proposal distribution parameters
    if full_cov:
        # full covariance adaptation: start with diagonal covariance
        cov = np.diag((step_scale * (np.abs(theta) + 1.0)) ** 2)
        cov += 1e-12 * np.eye(ndim)
        chol = np.linalg.cholesky(cov)
        recent_accepts: list = []
    else:
        # diagonal adaptation: per-parameter scales
        scale = step_scale * (np.abs(theta) + 1.0)
    current_lp = logprob_fn(theta)
    chain: list = []
    accepts = 0
    # Variables for adaptation
    adapt_count = 0
    acc_count = 0
    target = 0.25  # desired acceptance rate
    for s in range(steps):
        # Propose a new point
        if full_cov:
            # Draw from multivariate normal using current Cholesky
            eta = rng.normal(size=ndim)
            prop = theta + chol @ eta
        else:
            # Draw from diagonal normal
            prop = theta + rng.normal(0.0, scale, size=ndim)
        prop_lp = logprob_fn(prop)
        if np.log(rng.random()) < (prop_lp - current_lp):
            theta = prop
            current_lp = prop_lp
            accepts += 1
            acc_count += 1
            if full_cov:
                # record accepted state for covariance estimation
                recent_accepts.append(theta.copy())
        # Adaptation
        if adapt:
            adapt_count += 1
            if adapt_count % 50 == 0:
                acc_rate_local = acc_count / 50.0
                delta = acc_rate_local - target
                acc_count = 0
                if full_cov:
                    # scale covariance by exponential of delta
                    cov *= np.exp(delta * 0.1)
                    # update orientation using recent accepted samples if enough samples
                    if len(recent_accepts) >= ndim + 1:
                        X = np.array(recent_accepts)
                        # center
                        Xc = X - X.mean(axis=0)
                        cov_est = (Xc.T @ Xc) / max(1, (len(recent_accepts) - 1))
                        # blend orientation
                        cov = 0.9 * cov + 0.1 * (cov_est + 1e-12 * np.eye(ndim))
                        recent_accepts.clear()
                    # jitter for numerical stability
                    cov += 1e-12 * np.eye(ndim)
                    # recompute Cholesky
                    try:
                        chol = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        # fall back to diagonal if covariance is not PD
                        cov = np.diag(np.maximum(np.diag(cov), 1e-12))
                        chol = np.linalg.cholesky(cov)
                else:
                    # diagonal adaptation: update scale vector
                    scale *= np.exp(delta * 0.1)
        # Record after burn-in and thinning
        if s >= burn and (s - burn) % thin == 0:
            chain.append(np.concatenate([theta, [current_lp]]))
    chain = np.array(chain)
    accept_rate = accepts / steps
    return chain, accept_rate


def aic_bic(loglike_max: float, k: int, n: int) -> Tuple[float, float]:
    """Compute AIC and BIC for a model."""
    AIC = 2 * k - 2 * loglike_max
    BIC = k * np.log(n) - 2 * loglike_max
    return AIC, BIC

# ---------- New diagnostic and utility functions ----------

def plot_corner(chain: np.ndarray, param_names: list, save_path: str) -> None:
    """Create a simple corner plot from MCMC samples.

    Uses only NumPy and Matplotlib (no seaborn or external libs). The
    diagonal shows histograms of each parameter; the off-diagonal plots
    scatter points for pairs of parameters. This is intended for quick
    diagnostics, not publication-quality figures.

    Parameters
    ----------
    chain : ndarray
        MCMC samples with last column = log posterior. Only the
        parameter columns are used.
    param_names : list of str
        Names of the parameters corresponding to the sample columns.
    save_path : str
        File path where the corner plot will be saved.
    """
    samples = chain[:, : len(param_names)]
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i == j:
                # 1D histogram on diagonal
                data = samples[:, i]
                bins = max(20, int(np.sqrt(len(data)) / 2))
                ax.hist(data, bins=bins, color="gray", alpha=0.7)
                ax.axvline(np.median(data), color="C1", linestyle="--")
                ax.set_yticks([])
            elif j < i:
                # Lower triangle: scatter plot
                ax.scatter(samples[:, j], samples[:, i], s=2, alpha=0.3, color="C0")
            else:
                # Upper triangle empty
                ax.axis("off")
            # Label axes on outer edges
            if i == ndim - 1:
                ax.set_xlabel(param_names[j], rotation=45, ha="right")
            if j == 0 and i != 0:
                ax.set_ylabel(param_names[i])
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def posterior_predictive_band(
    chain: np.ndarray,
    R: np.ndarray,
    df: pd.DataFrame,
    priors: Dict[str, Tuple[float, float]],
    nsamples: int = 500,
    quantiles: Tuple[float, float] = (16, 84),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior predictive bands on a common R grid.

    Parameters
    ----------
    chain : ndarray
        MCMC samples with last column = log posterior.
    R : ndarray
        Radii used in the original data. A dense grid will be generated
        spanning this range.
    df : DataFrame
        Full SPARC dataset, used for baryonic contributions.
    priors : dict
        Prior bounds for parameters; used only for ordering names.
    nsamples : int
        Number of posterior samples to use when computing predictive bands.
    quantiles : tuple(float, float)
        Quantiles to compute (e.g. (16, 84) for 68% interval).

    Returns
    -------
    Rgrid : ndarray
        Grid of radii.
    lower : ndarray
        Lower quantile of predicted velocities at each radius.
    upper : ndarray
        Upper quantile of predicted velocities at each radius.
    """
    param_names = list(priors.keys())
    thetas = chain[:, : len(param_names)]
    # Dense grid across observed range
    Rgrid = np.linspace(max(1e-3, float(np.min(R))), float(np.max(R)), 150)
    # Preallocate predictions: shape (nsamples, len(Rgrid))
    preds = np.empty((nsamples, len(Rgrid)))
    # Randomly select samples from chain
    rng = np.random.default_rng(42)
    idxs = rng.choice(thetas.shape[0], size=min(nsamples, thetas.shape[0]), replace=False)
    # Pre-sort original radii for interpolation
    order = np.argsort(R)
    R_sorted = R[order]
    Vgas_sorted = df["V_gas_kms"].values[order]
    Vdisk_sorted = df["V_disk_kms"].values[order]
    Vbul_sorted = df["V_bul_kms"].values[order]
    for k, idx in enumerate(idxs):
        theta = thetas[idx]
        # Base UFF parameters
        V0, Rc_param, beta_param = theta[:3]
        f_gas = theta[3] if len(theta) > 3 else 0.0
        f_disk = theta[4] if len(theta) > 4 else 0.0
        f_bul = theta[5] if len(theta) > 5 else 0.0
        df_amp_param = theta[6] if len(theta) > 6 else 0.0
        df_exp_param = theta[7] if len(theta) > 7 else 0.0
        # Base UFF and dark field contributions at grid points
        V_uff_grid = v_circ_uff(Rgrid, np.array([V0, Rc_param, beta_param]))
        dark_field_grid = df_amp_param * (Rgrid ** df_exp_param)
        # Baryonic interpolation
        baryon2_grid = np.zeros_like(Rgrid)
        if f_gas != 0.0 or f_disk != 0.0 or f_bul != 0.0:
            Vgas_grid = np.interp(Rgrid, R_sorted, Vgas_sorted)
            Vdisk_grid = np.interp(Rgrid, R_sorted, Vdisk_sorted)
            Vbul_grid = np.interp(Rgrid, R_sorted, Vbul_sorted)
            baryon2_grid = (
                (f_gas * Vgas_grid) ** 2
                + (f_disk * Vdisk_grid) ** 2
                + (f_bul * Vbul_grid) ** 2
            )
        preds[k] = np.sqrt(baryon2_grid + (V_uff_grid + dark_field_grid) ** 2)
    # Compute quantile bounds along axis 0 (samples)
    lower = np.percentile(preds, quantiles[0], axis=0)
    upper = np.percentile(preds, quantiles[1], axis=0)
    return Rgrid, lower, upper


def save_sonification(
    R: np.ndarray,
    V: np.ndarray,
    out_path: str,
    duration: float = 4.0,
    sample_rate: int = 44100,
) -> None:
    """Create a simple sonification of a rotation curve and write to a WAV file.

    The radius is mapped linearly to time across ``duration`` seconds and
    the velocity is mapped to pitch in a linear range between two
    frequencies. A pure sine wave is generated at each time sample. This
    function uses only the Python standard library and NumPy, so the
    resulting audio is fairly basic but serves as a playful illustration.

    Parameters
    ----------
    R : ndarray
        Radii grid.
    V : ndarray
        Corresponding velocities.
    out_path : str
        Path of the WAV file to write.
    duration : float
        Duration of the audio in seconds.
    sample_rate : int
        Sample rate in samples per second.
    """
    import wave
    import struct

    # Normalise the radius to time indices
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    # Interpolate velocity onto the time axis
    V_interp = np.interp(t, np.linspace(0.0, duration, len(R)), V)
    # Map velocity range to frequency range (200–800 Hz)
    v_min, v_max = np.min(V), np.max(V)
    if v_max - v_min <= 0:
        freqs = 440.0 * np.ones_like(V_interp)
    else:
        freqs = 200.0 + 600.0 * ((V_interp - v_min) / (v_max - v_min))
    # Generate sine waveform
    phase = 2.0 * np.pi * np.cumsum(freqs) / sample_rate
    audio = 0.5 * np.sin(phase)
    # Convert to 16-bit integers
    audio_int16 = np.int16(audio * 32767)
    with wave.open(out_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit audio
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())


def plot_chain_e8_walk(
    chain: np.ndarray,
    save_path: str,
    seed: int = 42,
    subsample: int = 100,
) -> None:
    """Plot a random 3D projection of the MCMC chain alongside the E₈ roots.

    A random orthonormal projection matrix is generated using the same
    utility as the default E₈ plot. The chain samples are projected
    and plotted as a continuous line in 3D. E₈ roots are plotted as
    faint points for context.

    Parameters
    ----------
    chain : ndarray
        MCMC samples with last column = log posterior.
    save_path : str
        File path where the figure will be saved.
    seed : int
        Seed for the random projection matrix.
    subsample : int
        Subsample factor for the chain; every ``subsample``-th sample
        will be plotted to avoid overcrowding.
    """
    try:
        from e8_visualization import generate_e8_roots, random_projection_matrix, project_roots
    except ImportError:
        print("[WARN] E8 utilities not available; cannot plot parameter walk.")
        return
    # Generate roots and projection
    roots_int, roots_half = generate_e8_roots()
    P = random_projection_matrix(seed=seed)
    coords_int = project_roots(roots_int, P)
    coords_half = project_roots(roots_half, P)
    # Project chain (parameter columns)
    thetas = chain[:, :-1]
    proj_chain = (P @ thetas.T).T
    # Subsample chain for plotting
    if subsample > 1:
        proj_chain = proj_chain[::subsample]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    # Plot E8 roots
    ax.scatter(coords_int[:, 0], coords_int[:, 1], coords_int[:, 2], s=4, alpha=0.2, color="C0")
    ax.scatter(coords_half[:, 0], coords_half[:, 1], coords_half[:, 2], s=4, alpha=0.2, color="C1")
    # Plot chain walk
    ax.plot(proj_chain[:, 0], proj_chain[:, 1], proj_chain[:, 2], color="C3", linewidth=1.0)
    ax.set_title("E₈-projected parameter walk")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def mon_curve(R: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Compute a simple MOND rotation curve based on baryonic contributions.

    This implementation uses the 'simple' interpolating function:
        g = g_N / 2 + sqrt(g_N^2 / 4 + g_N * a0)
    where a0 is the MOND acceleration scale. The baryonic contribution
    to g_N is computed from the provided gas, disk and bulge velocities.
    The result is converted to a circular velocity.

    Parameters
    ----------
    R : ndarray
        Radii in kiloparsecs.
    df : DataFrame
        SPARC dataset containing baryonic velocities (gas, disk, bulge).

    Returns
    -------
    V_mond : ndarray
        MOND circular velocities in km/s.
    """
    # Physical constants
    G = 4.30091e-6  # kpc (km/s)^2 / Msun
    a0 = 1.2e-10  # m/s^2
    # Convert kpc to m
    kpc_to_m = 3.08567758e19
    # Compute baryonic mass enclosed from velocities squared times R / G
    Vgas = df["V_gas_kms"].values
    Vdisk = df["V_disk_kms"].values
    Vbul = df["V_bul_kms"].values
    # Combine baryon velocities quadratically (approx mass distribution)
    Vbar = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
    # Avoid zeros
    Vbar = np.where(Vbar < 1e-3, 1e-3, Vbar)
    # Interpolate baryonic velocities onto input radii for mass calculation
    # convert to m/s and R to m
    R_m = (R * kpc_to_m)
    Vbar_m_s = Vbar[np.argsort(df["R_kpc"].values)] * 1000.0
    # estimate gravitational acceleration g_N = V^2 / R
    # For each R, approximate using baryonic velocity at that radius
    # We'll create interpolation of baryonic velocity vs radius
    R_data = df["R_kpc"].values * kpc_to_m
    Vbar_interp = np.interp(R_m, R_data, df["V_gas_kms"].values + df["V_disk_kms"].values + df["V_bul_kms"].values)
    # g_N
    gN = (Vbar_interp * 1000.0) ** 2 / R_m
    # MOND interpolating function
    gM = 0.5 * gN + np.sqrt(0.25 * gN**2 + gN * a0)
    # MOND circular velocity
    V_mond = np.sqrt(gM * R_m) / 1000.0
    return V_mond


def nfw_curve(R: np.ndarray, Vmax: float = 150.0, Rs: float = 5.0) -> np.ndarray:
    """Compute a simple NFW rotation curve.

    The parameters Vmax and Rs can be tuned to roughly match the
    amplitude and shape of observed rotation curves. The expression used
    here is not derived from first principles but is a convenient
    approximation:
        V(r) = Vmax * sqrt(ln(1 + r/Rs) - (r/Rs)/(1 + r/Rs)) / sqrt(r/Rs)

    Parameters
    ----------
    R : ndarray
        Radii in kiloparsecs.
    Vmax : float
        Maximum circular velocity in km/s.
    Rs : float
        Scale radius in kiloparsecs.

    Returns
    -------
    ndarray
        NFW circular velocity profile.
    """
    x = R / Rs
    # Avoid division by zero near r=0
    x = np.where(x < 1e-5, 1e-5, x)
    f = np.log(1 + x) - x / (1 + x)
    V = Vmax * np.sqrt(f / x)
    return V


def summarize_chain(chain: np.ndarray, names: list) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for an MCMC chain.

    Parameters
    ----------
    chain : ndarray
        MCMC samples with last column = log posterior.
    names : list
        Names of the parameters corresponding to the chain columns.

    Returns
    -------
    stats : dict
        For each parameter: mean, median, and 16/84 percentiles.
    """
    thetas = chain[:, : len(names)]
    stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        col = thetas[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "p16": float(np.percentile(col, 16)),
            "p84": float(np.percentile(col, 84)),
        }
    stats["logpost_median"] = float(np.median(chain[:, -1]))
    return stats


def fit_one(
    csv_path: str,
    galname: str,
    outdir: str,
    priors: Dict[str, Tuple[float, float]],
    e8_hook: bool = False,
    corner: bool = False,
    postpred: bool = False,
    sonify: bool = False,
    walk_e8: bool = False,
    compare: bool = False,
    full_cov: bool = False,
) -> Dict[str, object]:
    """Fit a single galaxy and write diagnostic outputs.

    Returns the summary dictionary so that batch mode can accumulate
    results.
    """
    df = load_sparc_csv(csv_path)
    R = df["R_kpc"].values
    V = df["V_obs_kms"].values
    eV = df["e_V_kms"].values

    param_names = list(priors.keys())
    theta0 = np.array([(priors[n][0] + priors[n][1]) / 2 for n in param_names])

    def lp(theta: np.ndarray) -> float:
        return posterior_logprob(theta, R, V, eV, df, priors)

    chain, acc_rate = mh_sampler(
        theta0,
        lp,
        steps=30000,
        step_scale=0.10,
        burn=8000,
        thin=10,
        random_state=42,
        adapt=True,
        full_cov=full_cov,
    )
    stats = summarize_chain(chain, param_names)
    idx_best = int(np.argmax(chain[:, -1]))
    theta_map = chain[idx_best, : len(param_names)]
    logpost_max = float(chain[idx_best, -1])
    loglike_max = logpost_max  # flat priors ignored
    AIC, BIC = aic_bic(loglike_max, k=len(param_names), n=len(R))

    # Generate rotation curve plot
    Rgrid = np.linspace(max(1e-3, np.min(R)), np.max(R), 200)
    # Compute model on a dense grid. For baryonic terms we interpolate
    # the observed baryonic velocities onto the grid. This avoids
    # broadcasting errors when combining arrays of different length.
    V0, Rc, beta = theta_map[:3]
    f_gas = theta_map[3] if len(theta_map) > 3 else 0.0
    f_disk = theta_map[4] if len(theta_map) > 4 else 0.0
    f_bul = theta_map[5] if len(theta_map) > 5 else 0.0
    df_amp = theta_map[6] if len(theta_map) > 6 else 0.0
    df_exp = theta_map[7] if len(theta_map) > 7 else 0.0
    # Base UFF and dark field contributions at grid points
    V_uff_grid = v_circ_uff(Rgrid, np.array([V0, Rc, beta]))
    dark_field_grid = df_amp * (Rgrid ** df_exp)
    # Interpolate baryonic velocities onto the grid
    baryon2_grid = np.zeros_like(Rgrid)
    if f_gas != 0.0 or f_disk != 0.0 or f_bul != 0.0:
        # Ensure radii are sorted for interpolation
        order = np.argsort(R)
        R_sorted = R[order]
        Vgas_sorted = df["V_gas_kms"].values[order]
        Vdisk_sorted = df["V_disk_kms"].values[order]
        Vbul_sorted = df["V_bul_kms"].values[order]
        Vgas_grid = np.interp(Rgrid, R_sorted, Vgas_sorted)
        Vdisk_grid = np.interp(Rgrid, R_sorted, Vdisk_sorted)
        Vbul_grid = np.interp(Rgrid, R_sorted, Vbul_sorted)
        baryon2_grid = (
            (f_gas * Vgas_grid) ** 2
            + (f_disk * Vdisk_grid) ** 2
            + (f_bul * Vbul_grid) ** 2
        )
    Vmap = np.sqrt(baryon2_grid + (V_uff_grid + dark_field_grid) ** 2)
    # Primary rotation curve figure
    fig_path = os.path.join(outdir, f"{galname}_fit.png")
    plt.figure()
    plt.errorbar(R, V, yerr=eV, fmt="o", label="Observed")
    plt.plot(Rgrid, Vmap, label="Model (MAP)")
    # Posterior predictive bands
    postpred_fig_path = None
    if postpred:
        try:
            Rpred, lower_band, upper_band = posterior_predictive_band(chain, R, df, priors)
            plt.fill_between(Rpred, lower_band, upper_band, color="C2", alpha=0.3, label="Posterior 16–84%")
        except Exception as exc:
            print(f"[WARN] Failed to compute posterior predictive bands: {exc}")
    plt.xlabel("R [kpc]")
    plt.ylabel("V_circ [km/s]")
    plt.title(f"{galname} — UFF fit with extensions")
    plt.legend()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    # Save separate posterior predictive figure if requested
    if postpred:
        # Use same data but highlight predictive band; separate figure
        postpred_fig_path = os.path.join(outdir, f"{galname}_postpred.png")
        plt.figure()
        plt.errorbar(R, V, yerr=eV, fmt="o", label="Observed")
        plt.plot(Rgrid, Vmap, label="Model (MAP)")
        try:
            plt.fill_between(Rpred, lower_band, upper_band, color="C2", alpha=0.3, label="Posterior 16–84%")
        except Exception:
            pass
        plt.xlabel("R [kpc]")
        plt.ylabel("V_circ [km/s]")
        plt.title(f"{galname} — Posterior predictive")
        plt.legend()
        plt.savefig(postpred_fig_path, dpi=160)
        plt.close()

    # Optional E8 visualization
    e8_fig_path = None
    if e8_hook:
        try:
            from e8_visualization import plot_e8_projection

            e8_fig_path = os.path.join(outdir, f"{galname}_e8.png")
            plot_e8_projection(e8_fig_path)
        except Exception as exc:
            print(f"[WARN] E8 visualization failed: {exc}")
            e8_fig_path = None

    # Optional corner plot
    corner_fig_path = None
    if corner:
        try:
            corner_fig_path = os.path.join(outdir, f"{galname}_corner.png")
            plot_corner(chain, param_names, corner_fig_path)
        except Exception as exc:
            print(f"[WARN] Corner plot failed: {exc}")
            corner_fig_path = None

    # Optional sonification
    audio_path = None
    if sonify:
        try:
            audio_path = os.path.join(outdir, f"{galname}.wav")
            save_sonification(Rgrid, Vmap, audio_path)
        except Exception as exc:
            print(f"[WARN] Sonification failed: {exc}")
            audio_path = None

    # Optional E8-projected parameter walk
    walk_fig_path = None
    if walk_e8:
        try:
            walk_fig_path = os.path.join(outdir, f"{galname}_walk_e8.png")
            plot_chain_e8_walk(chain, walk_fig_path)
        except Exception as exc:
            print(f"[WARN] E8 parameter walk failed: {exc}")
            walk_fig_path = None

    # Optional model comparison
    compare_fig_path = None
    if compare:
        try:
            compare_fig_path = os.path.join(outdir, f"{galname}_compare.png")
            V_mond = mon_curve(Rgrid, df)
            V_nfw = nfw_curve(Rgrid)
            plt.figure()
            plt.errorbar(R, V, yerr=eV, fmt="o", label="Observed")
            plt.plot(Rgrid, Vmap, label="UFF (MAP)")
            plt.plot(Rgrid, V_mond, label="MOND")
            plt.plot(Rgrid, V_nfw, label="NFW")
            plt.xlabel("R [kpc]")
            plt.ylabel("V_circ [km/s]")
            plt.title(f"{galname} — Model comparison")
            plt.legend()
            plt.savefig(compare_fig_path, dpi=160)
            plt.close()
        except Exception as exc:
            print(f"[WARN] Model comparison failed: {exc}")
            compare_fig_path = None

    # Write summary JSON
    summary = {
        "galaxy": galname,
        "accept_rate": acc_rate,
        "params": stats,
        "theta_map": {n: float(v) for n, v in zip(param_names, theta_map)},
        "loglike_max": loglike_max,
        "AIC": float(AIC),
        "BIC": float(BIC),
        "n_points": int(len(R)),
        "figure": fig_path,
    }
    if e8_fig_path:
        summary["e8_figure"] = e8_fig_path
    if corner_fig_path:
        summary["corner_figure"] = corner_fig_path
    if postpred and postpred_fig_path:
        summary["posterior_predictive_figure"] = postpred_fig_path
    if audio_path:
        summary["audio"] = audio_path
    if walk_fig_path:
        summary["walk_figure"] = walk_fig_path
    if compare_fig_path:
        summary["compare_figure"] = compare_fig_path
    # Save summary to disk
    out_json = os.path.join(outdir, f"{galname}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[OK] {galname}: acc={acc_rate:.2f}, AIC={AIC:.2f}, BIC={BIC:.2f}, loglike_max={loglike_max:.2f}"
    )
    print(f" -> Plot: {fig_path}")
    print(f" -> Summary: {out_json}")
    if e8_fig_path:
        print(f" -> E8 figure: {e8_fig_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to SPARC-like CSV for a single galaxy")
    ap.add_argument("--gal", type=str, help="Galaxy name (for labeling outputs)")
    ap.add_argument("--batch", type=str, help="Directory containing multiple CSVs")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    # Prior ranges for base UFF parameters
    ap.add_argument("--V0", type=float, nargs=2, default=[50.0, 350.0], help="Prior range for V0 [km/s]")
    ap.add_argument("--Rc", type=float, nargs=2, default=[0.1, 30.0], help="Prior range for Rc [kpc]")
    ap.add_argument("--beta", type=float, nargs=2, default=[-1.0, 2.0], help="Prior range for beta")
    # Prior ranges for baryonic scale factors
    ap.add_argument("--f_gas", type=float, nargs=2, default=[0.0, 2.0], help="Prior range for gas scaling")
    ap.add_argument("--f_disk", type=float, nargs=2, default=[0.0, 2.0], help="Prior range for disk scaling")
    ap.add_argument("--f_bul", type=float, nargs=2, default=[0.0, 2.0], help="Prior range for bulge scaling")
    # Prior ranges for dark field terms
    ap.add_argument("--df_amp", type=float, nargs=2, default=[0.0, 50.0], help="Prior range for dark-field amplitude")
    ap.add_argument("--df_exp", type=float, nargs=2, default=[-2.0, 2.0], help="Prior range for dark-field exponent")
    # Optional E8 visualization hook
    ap.add_argument("--e8", action="store_true", help="Generate a 3D E8 root lattice projection")

    # Additional diagnostic and feature flags
    ap.add_argument("--corner", action="store_true", help="Generate a corner plot of posterior samples")
    ap.add_argument("--postpred", action="store_true", help="Add posterior predictive bands and save separate figure")
    ap.add_argument("--sonify", action="store_true", help="Generate a WAV file sonifying the rotation curve")
    ap.add_argument("--walk-e8", action="store_true", help="Plot the MCMC chain projected into E8")
    ap.add_argument("--compare", action="store_true", help="Plot UFF vs MOND vs NFW curves")
    ap.add_argument("--full-cov", action="store_true", help="Use a full multivariate covariance adaptation in the MCMC sampler")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Construct priors dict based on provided arguments
    priors = {
        "V0": tuple(args.V0),
        "Rc": tuple(args.Rc),
        "beta": tuple(args.beta),
        "f_gas": tuple(args.f_gas),
        "f_disk": tuple(args.f_disk),
        "f_bul": tuple(args.f_bul),
        "df_amp": tuple(args.df_amp),
        "df_exp": tuple(args.df_exp),
    }

    if args.csv and args.gal:
        # Single galaxy mode
        fit_one(
            args.csv,
            args.gal,
            args.out,
            priors,
            e8_hook=args.e8,
            corner=args.corner,
            postpred=args.postpred,
            sonify=args.sonify,
            walk_e8=args.walk_e8,
            compare=args.compare,
            full_cov=args.full_cov,
        )
    elif args.batch:
        # Batch mode: fit all CSVs in the directory and aggregate results
        summaries = []
        for fname in sorted(os.listdir(args.batch)):
            if not fname.lower().endswith(".csv"):
                continue
            gal = os.path.splitext(fname)[0]
            fpath = os.path.join(args.batch, fname)
            summary = fit_one(
                fpath,
                gal,
                args.out,
                priors,
                e8_hook=args.e8,
                corner=args.corner,
                postpred=args.postpred,
                sonify=args.sonify,
                walk_e8=args.walk_e8,
                compare=args.compare,
                full_cov=args.full_cov,
            )
            summaries.append(summary)
        # Write aggregated benchmark table
        bench_table_path = os.path.join(args.out, "batch_benchmark.csv")
        if summaries:
            import csv

            with open(bench_table_path, "w", newline="") as csvfile:
                fieldnames = [
                    "galaxy",
                    "accept_rate",
                    "loglike_max",
                    "AIC",
                    "BIC",
                    "figure",
                ]
                if args.e8:
                    fieldnames.append("e8_figure")
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for s in summaries:
                    row = {k: s.get(k) for k in fieldnames if k in s}
                    writer.writerow(row)
            print(f"[OK] Batch benchmark saved to {bench_table_path}")
    else:
        print("Provide either --csv + --gal, or --batch DIR")


if __name__ == "__main__":
    main()