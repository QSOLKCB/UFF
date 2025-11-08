
import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from uff_model import v_circ_uff

# ---------- Utilities ----------

def load_sparc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["R_kpc", "V_obs_kms", "e_V_kms"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Fill optional baryonics if missing
    for col in ["V_gas_kms", "V_disk_kms", "V_bul_kms"]:
        if col not in df.columns:
            df[col] = 0.0
    return df

def model_v(R, theta, df):
    # If you want to include baryons explicitly, combine them here before/after UFF
    # For now, we fit UFF directly to V_obs. Extend as needed.
    return v_circ_uff(R, theta)

def neg_loglike(theta, R, V, eV, df):
    Vmod = model_v(R, theta, df)
    chi2 = np.sum(((V - Vmod) / eV)**2)
    # Gaussian errors
    return 0.5 * chi2 + np.sum(np.log(eV + 1e-12))

def prior_logprob(theta, priors: Dict[str, Tuple[float,float]]):
    # Uniform priors in [lo, hi]
    lp = 0.0
    names = list(priors.keys())
    for i, name in enumerate(names):
        lo, hi = priors[name]
        if not (lo <= theta[i] <= hi):
            return -np.inf
        # flat prior adds constant; we can ignore constants
    return lp

def posterior_logprob(theta, R, V, eV, df, priors):
    lp = prior_logprob(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp - neg_loglike(theta, R, V, eV, df)

def mh_sampler(theta0, logprob_fn, steps=20000, step_scale=0.05, burn=5000, thin=5, random_state=0):
    rng = np.random.default_rng(random_state)
    theta = np.array(theta0, dtype=float)
    ndim = theta.size
    # Proposal: Gaussian with diag covariance scaled by |theta| or 1
    scale = step_scale * (np.abs(theta) + 1.0)
    current_lp = logprob_fn(theta)
    chain = []
    accepts = 0
    for s in range(steps):
        prop = theta + rng.normal(0, scale, size=ndim)
        prop_lp = logprob_fn(prop)
        if np.log(rng.random()) < (prop_lp - current_lp):
            theta, current_lp = prop, prop_lp
            accepts += 1
        if s >= burn and (s - burn) % thin == 0:
            chain.append(np.concatenate([theta, [current_lp]]))
    chain = np.array(chain)
    acc_rate = accepts / steps
    return chain, acc_rate

def aic_bic(loglike_max, k, n):
    AIC = 2*k - 2*loglike_max
    BIC = k*np.log(n) - 2*loglike_max
    return AIC, BIC

def summarize_chain(chain, names):
    # chain columns: [theta..., lp]
    thetas = chain[:, :len(names)]
    stats = {}
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

def fit_one(csv_path, galname, outdir, priors):
    df = load_sparc_csv(csv_path)
    R = df["R_kpc"].values
    V = df["V_obs_kms"].values
    eV = df["e_V_kms"].values

    param_names = list(priors.keys())
    theta0 = np.array([(priors[n][0] + priors[n][1]) / 2 for n in param_names])

    def lp(theta):
        return posterior_logprob(theta, R, V, eV, df, priors)

    chain, acc = mh_sampler(theta0, lp, steps=30000, step_scale=0.10, burn=8000, thin=10, random_state=42)
    stats = summarize_chain(chain, param_names)

    # MAP approx: take best lp
    idx = np.argmax(chain[:, -1])
    theta_map = chain[idx, :len(param_names)]
    logpost_max = chain[idx, -1]
    # For Gaussian errors, loglike = -neg_loglike
    loglike_max = logpost_max  # flat priors up to constants

    # AIC/BIC
    AIC, BIC = aic_bic(loglike_max, k=len(param_names), n=len(R))

    # Plot fit
    Rgrid = np.linspace(max(1e-3, np.min(R)), np.max(R), 200)
    Vmap = v_circ_uff(Rgrid, theta_map)

    plt.figure()
    plt.errorbar(R, V, yerr=eV, fmt='o', label='Observed')
    plt.plot(Rgrid, Vmap, label='UFF (MAP)')
    plt.xlabel("R [kpc]")
    plt.ylabel("V_circ [km/s]")
    plt.title(f"{galname} — UFF fit")
    plt.legend()
    fig_path = os.path.join(outdir, f"{galname}_fit.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Save summary
    summary = {
        "galaxy": galname,
        "accept_rate": acc,
        "params": stats,
        "theta_map": {n: float(v) for n, v in zip(param_names, theta_map)},
        "loglike_max": float(loglike_max),
        "AIC": float(AIC),
        "BIC": float(BIC),
        "n_points": int(len(R)),
        "figure": fig_path,
    }
    with open(os.path.join(outdir, f"{galname}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {galname}: acc={acc:.2f}, AIC={AIC:.2f}, BIC={BIC:.2f}")
    print(f" -> Plot: {fig_path}")
    print(f" -> Summary: {os.path.join(outdir, f'{galname}_summary.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to SPARC-like CSV for a single galaxy")
    ap.add_argument("--gal", type=str, help="Galaxy name (for labeling outputs)")
    ap.add_argument("--batch", type=str, help="Directory containing multiple CSVs")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    ap.add_argument("--V0", type=float, nargs=2, default=[50.0, 350.0], help="Prior range for V0 [km/s]")
    ap.add_argument("--Rc", type=float, nargs=2, default=[0.1, 30.0], help="Prior range for Rc [kpc]")
    ap.add_argument("--beta", type=float, nargs=2, default=[-1.0, 2.0], help="Prior range for beta")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    priors = {
        "V0": tuple(args.V0),
        "Rc": tuple(args.Rc),
        "beta": tuple(args.beta),
    }

    if args.csv and args.gal:
        fit_one(args.csv, args.gal, args.out, priors)
    elif args.batch:
        for fname in os.listdir(args.batch):
            if not fname.lower().endswith(".csv"):
                continue
            gal = os.path.splitext(fname)[0]
            fit_one(os.path.join(args.batch, fname), gal, args.out, priors)
    else:
        print("Provide either --csv + --gal, or --batch DIR")

if __name__ == "__main__":
    main()
