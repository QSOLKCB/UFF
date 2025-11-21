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


def mh_sampler(theta0: np.ndarray, logprob_fn, steps: int = 20000, step_scale: float = 0.05,
               burn: int = 5000, thin: int = 5, random_state: int = 0,
               adapt: bool = True) -> Tuple[np.ndarray, float]:
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
    # Initial diagonal proposal widths
    scale = step_scale * (np.abs(theta) + 1.0)
    current_lp = logprob_fn(theta)
    chain = []
    accepts = 0
    # Variables for adaptation
    adapt_count = 0
    acc_count = 0
    target = 0.25  # desired acceptance rate
    for s in range(steps):
        # Propose a new point using a multivariate normal with diagonal covariance
        prop = theta + rng.normal(0.0, scale, size=ndim)
        prop_lp = logprob_fn(prop)
        if np.log(rng.random()) < (prop_lp - current_lp):
            theta = prop
            current_lp = prop_lp
            accepts += 1
            acc_count += 1
        # Adaptation of proposal widths
        if adapt:
            adapt_count += 1
            # Update every 50 proposals to avoid over‑reacting to noise
            if adapt_count % 50 == 0:
                acc_rate_local = acc_count / 50.0
                # Robbins–Monro update: scale *= exp(delta)
                delta = (acc_rate_local - target)
                # Use a small damping factor for stability
                scale *= np.exp(delta * 0.1)
                # Reset local counters
                acc_count = 0
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


def fit_one(csv_path: str, galname: str, outdir: str, priors: Dict[str, Tuple[float, float]],
            e8_hook: bool = False) -> Dict[str, object]:
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
    plt.figure()
    plt.errorbar(R, V, yerr=eV, fmt="o", label="Observed")
    plt.plot(Rgrid, Vmap, label="Model (MAP)")
    plt.xlabel("R [kpc]")
    plt.ylabel("V_circ [km/s]")
    plt.title(f"{galname} — UFF fit with extensions")
    plt.legend()
    fig_path = os.path.join(outdir, f"{galname}_fit.png")
    plt.savefig(fig_path, dpi=160)
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
        fit_one(args.csv, args.gal, args.out, priors, e8_hook=args.e8)
    elif args.batch:
        # Batch mode: fit all CSVs in the directory and aggregate results
        summaries = []
        for fname in sorted(os.listdir(args.batch)):
            if not fname.lower().endswith(".csv"):
                continue
            gal = os.path.splitext(fname)[0]
            fpath = os.path.join(args.batch, fname)
            summary = fit_one(fpath, gal, args.out, priors, e8_hook=args.e8)
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
