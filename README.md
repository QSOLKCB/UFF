# 🌌 QSOL UFF — UFF v3.0.0 “Spectral Gravity Upgrade” Rotation Curve Analysis Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17669627.svg)](https://doi.org/10.5281/zenodo.17669627)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Language](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

**Author:** Trent Slade (QSOL IMC)  
**Status:** Active Research Release · v3.0.0 · November 2025

---

## 🧠 Overview

**QSOL UFF** is a lightweight, fully transparent **rotation-curve analysis engine** built around the **Unified Field Framework (UFF)**.

Version **3.0.0 — “Spectral Gravity Upgrade”** turns the original SPARC-style fitter into a full analysis suite featuring: :contentReference[oaicite:0]{index=0}  

- **Full-covariance adaptive MCMC** (multivariate Σ with Robbins–Monro adaptation)  
- **NumPy-only corner plots** (no heavy plotting libs)  
- **Posterior predictive bands** over baryonic rotation curves  
- **Spectral sonification** — galaxies “sing” their rotation curves to WAV  
- **E₈-projected parameter walk** for chain geometry visualization  
- **UFF vs MOND vs NFW comparison mode** for side-by-side theory tests  

Still no black boxes: pure NumPy, explicit priors, deterministic seeds, and files you can actually read.

---

## 🗂️ Repository Structure

Typical layout for the `QSOLKCB/UFF` repo at tag `v3.0.0`:

```text
QSOL_UFF/
├── analyze_sparc.py          # Main CLI for fitting, diagnostics, & comparisons
├── uff_model.py              # Unified Field circular-velocity law(s)
├── e8_visualization.py       # E₈ parameter walk & chain geometry tools
├── DEMO_GALAXY.csv           # Example SPARC-style dataset
├── UFF_SPARC_Template.ipynb  # Notebook walkthrough (optional)
├── requirements.txt          # Minimal Python dependencies
├── venv_setup.sh             # Auto setup script (Linux/macOS)
├── venv_setup.ps1            # Auto setup script (Windows)
├── MERGE_INSTRUCTIONS.md     # Git flow & Zenodo tagging notes
├── merge_pr1.sh              # Example scripted merge (legacy from v1.x)
└── README.md                 # You’re reading it
(Exact filenames may evolve; check the tagged release on GitHub for truth.)

⚙️ Environment Setup
Option A — Linux / macOS (Recommended)
bash
Copy code
git clone https://github.com/QSOLKCB/UFF.git
cd UFF
git checkout v3.0.0
./venv_setup.sh
source .venv/bin/activate
Option B — Windows (PowerShell)
powershell
Copy code
git clone https://github.com/QSOLKCB/UFF.git
cd UFF
git checkout v3.0.0
.\venv_setup.ps1
.\.venv\Scripts\Activate.ps1
Prompt should show:

text
Copy code
(.venv) UFF>
To deactivate:

bash
Copy code
deactivate
🧩 Quick Start Example
Fit the included demo galaxy with full diagnostics:

bash
Copy code
python analyze_sparc.py \
  --csv DEMO_GALAXY.csv \
  --gal DEMO_GALAXY \
  --out outputs \
  --corner \
  --postpred \
  --compare \
  --sonify \
  --walk-e8
Generated Outputs (in /outputs)
File	Description
DEMO_GALAXY_fit.png	Observed vs model rotation curve + posterior predictive bands
DEMO_GALAXY_corner.png	NumPy-only corner plot of posterior
DEMO_GALAXY_compare.png	UFF vs MOND vs NFW rotation curves on same axes
DEMO_GALAXY_e8_walk.png	E₈-projected MCMC trajectory vs root lattice slice
DEMO_GALAXY_summary.json	MAP parameters, posterior stats, AIC/BIC, diagnostic file paths, etc.
DEMO_GALAXY_posterior.txt	Chain statistics and text summary
DEMO_GALAXY_sonify.wav	Spectral sonification of the MAP rotation curve

📊 Model Comparison
UFF v3 ships with a comparison mode overlaying: 
Zenodo

UFF (Unified Field Framework law)

MOND (simple μ-interpolator)

NFW halo (analytic approximation for V(r))

Standard information criteria are computed from the MAP likelihood:

### Information Criteria

The standard information criteria are:

\[
\text{AIC} = 2k - 2 \ln L_{\max}, \qquad
\text{BIC} = k \ln n - 2 \ln L_{\max}
\]

Where:  
- \(k\) = number of free parameters  
- \(n\) = number of data points  
- \(L_{\max}\) = maximum likelihood (evaluated at the MAP estimate)

​
 ,BIC=klnn−2lnL 
max
​ 
k = number of free parameters

n = number of data points

Lower AIC/BIC → more parsimonious model for the same dataset.

🧮 Method Notes
Core philosophy: minimal dependencies, explicit assumptions, reproducible chains.

Sampler: pure NumPy Metropolis–Hastings with adaptive full covariance

Σ updated every N steps via damped Robbins–Monro

Automatic fallback to diagonal proposals for pathological posteriors 
Zenodo

Priors: explicitly defined in analyze_sparc.py

Start broad → tighten once convergence and residuals look sane

Posterior predictive bands:

Dense R-grid, full baryonic interpolation

16–84% credible interval shaded directly on the rotation curve 
Zenodo

Deterministic seeds:

Fixed RNG seeds for exact reproducibility of chains and plots

NumPy-only corner plots:

1D marginals on the diagonal, 2D scatter below, auto labels, no extra libs

Extend uff_model.py to test:

lensing-equivalent mass profiles

Tully–Fisher scaling

cluster-scale fits or alternative UFF parameterizations

🔊 Spectral Sonification
UFF v3 includes spectral sonification flags:

Radius → time

Velocity → pitch

Optional normalization + fades for WAV export

Result: each galaxy yields a short audio “glyph” encoding its rotation curve.

Use:

bash
Copy code
python analyze_sparc.py --csv DEMO_GALAXY.csv --gal DEMO_GALAXY --out outputs --sonify
Check DEMO_GALAXY_sonify.wav and feed it into your DAW, sampler, or QSOL-IMC sound engine.

🧊 E₈-Projected Parameter Walk
The E₈ walk tools let you view the chain as a trajectory through a random 3-slice of the E₈ root lattice: 
Zenodo

Chain points projected into an 8D–>3D slice

Overlaid on 240 E₈ roots

Integer vs half-integer classes separated

Deterministic projection for reproducible figures

Useful for diagnosing multi-modal posteriors and correlated parameters in a way that looks like fan-art for algebraic geometers.

🔍 Outputs and Diagnostics Overview
Posterior text summary (MAP, mean, σ)

AIC/BIC and log-evidence-adjacent metrics

Residual plots + posterior predictive envelopes

NumPy corner plots

E₈ chain visualizations

UFF / MOND / NFW comparison panels

Optional sonification WAVs

Everything is intended to be scriptable and batchable for full-SPARC sweeps.

🤝 Contributing
Pull requests welcome via GitHub issues or discussions.

Guidelines:

Follow PEP8

Document public functions

Add a tiny test dataset (or SPARC subset) for each new model variant

Keep dependencies minimal; if you add a heavy one, make it optional and justified

🚧 v3.x+ Roadmap
Likely next steps:

GPU / numba-assisted samplers for large SPARC batches

Batch SPARC runner with per-galaxy JSON summaries and dashboard-ready outputs

More sonification modes (e.g., baryons vs total curve in stereo)

Optional interactive notebook gallery for teaching and demos

🚪 License
Apache License 2.0

You are free to use, modify, and redistribute the code, including in commercial contexts, provided you:

Preserve copyright and license notices

Respect the Apache 2.0 terms for contributions and patents

See LICENSE or https://www.apache.org/licenses/LICENSE-2.0 for full details. 
Zenodo

📚 Citation
If you use UFF v3.0.0 — “Spectral Gravity Upgrade” in a publication, please cite the software record:

Slade, T. (2025). QSOLKCB/UFF: UFF v3.0.0 — “Spectral Gravity Upgrade”.
Zenodo. https://doi.org/10.5281/zenodo.17669627

For the broader theoretical and sonic context, also see:

Slade, T. (2024). Spectral Algebraics: Audible Geometry via E8-Inspired Signal Synthesis and 3D Visualization. Zenodo.
(Concept DOI for earlier UFF work: 10.5281/zenodo.17510648)

🔗 Related Projects
QSOLKCB / UFF — Unified Field Framework core rotation-curve engine 
Zenodo

QSOLKCB / QEC — Quantum Error Correction Framework

QSOLKCB / QAI-UFT — Unified Field Theory core modelling

Spectral Algebraics (Zenodo) — E₈-inspired audio/visual geometry

Truth Compiled · QSOL IMC 2025 · “Spectral Gravity Upgrade” Edition
