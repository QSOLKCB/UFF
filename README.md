# 🌌 QSOL UFF — Unified Field Framework Rotation Curve Analysis Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17510648.svg)](https://doi.org/10.5281/zenodo.17510648)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Language](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

**Author:** Trent Slade (QSOL IMC)
**Status:** Active Research Release · v1.0 · November 2025

---

## 🧠 Overview

**QSOL UFF** implements a lightweight, fully transparent rotation-curve analysis pipeline inspired by the *SPARC* galaxy database but using your **Unified Field Framework (UFF)** model.
It combines:

* pure-NumPy **Metropolis–Hastings** Bayesian fitting,
* explicit prior definitions,
* analytic AIC/BIC model comparison,
* publication-quality plots and posterior summaries.

No heavy external dependencies. Everything runs locally, reproducibly, and fast.

---

## 🗂️ Repository Structure

```
QSOL_UFF/
├── analyze_sparc.py          # Main CLI for fitting and diagnostics
├── uff_model.py              # Define your Unified-Field circular-velocity law here
├── DEMO_GALAXY.csv           # Example SPARC-style dataset
├── UFF_SPARC_Template.ipynb  # Notebook walkthrough for visual learners
├── requirements.txt          # Minimal Python dependencies
├── venv_setup.sh             # Auto setup script (Linux/macOS)
├── venv_setup.ps1            # Auto setup script (Windows)
├── .copilot-instructions.md  # GitHub Copilot integration guide
├── MERGE_INSTRUCTIONS.md     # Instructions for merging PR #1
├── merge_pr1.sh              # Automated merge script for v1.1.0
└── README.md                 # You’re reading it
```

---

## ⚙️ Environment Setup

### Option A — Linux / macOS (Recommended)

```bash
git clone https://github.com/QSOLKCB/QSOL_UFF.git
cd QSOL_UFF
./venv_setup.sh
source .venv/bin/activate
```

### Option B — Windows (PowerShell)

```powershell
git clone https://github.com/QSOLKCB/QSOL_UFF.git
cd QSOL_UFF
.\venv_setup.ps1
.\.venv\Scripts\Activate.ps1
```

After activation you’ll see:

```
(.venv) PS C:\QSOL_UFF>
```

To deactivate:

```bash
deactivate
```

---

## 🧩 Quick Start Example

Fit the included demo galaxy:

```bash
python analyze_sparc.py --csv DEMO_GALAXY.csv --gal DEMO_GALAXY --out outputs
```

### Generated Outputs (in `/outputs`)

| File                        | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `DEMO_GALAXY_fit.png`       | Observed vs model rotation curve with posterior predictive bands |
| `DEMO_GALAXY_summary.json`  | MAP parameters, AIC/BIC, log-evidence                            |
| `DEMO_GALAXY_posterior.txt` | Chain statistics and corner-like summary                         |

---

## 📊 Model Comparison

Model evidence is estimated from the **maximum a posteriori (MAP)** likelihood.

[
\text{AIC} = 2k - 2\ln L_\text{max}, \qquad
\text{BIC} = k\ln n - 2\ln L_\text{max}
]

* *k* = number of free parameters
* *n* = data points
  Lower AIC/BIC → more parsimonious model.

---

## 🧮 Method Notes

* Sampler: pure NumPy Metropolis–Hastings.
  → No PyMC, no Stan, zero black box.
* Priors must be explicit (`analyze_sparc.py`).
  → Start broad, then tighten as posterior stabilizes.
* Extend `uff_model.py` to test lensing mass, Tully–Fisher relations, cluster-scale fits, etc.

---

## 🧪 Example Model Stub

Inside `uff_model.py`:

```python
def v_circ_uff(R_kpc, theta):
    """
    Unified Field Framework circular velocity law.
    Replace this stub with your analytic expression.
    Parameters
    ----------
    R_kpc : array-like
        Galactocentric radius [kpc]
    theta : iterable
        Model parameters (e.g. V0, Rc, beta)
    Returns
    -------
    np.ndarray
        Circular velocity [km/s]
    """
    V0, Rc, beta = theta
    R = np.asarray(R_kpc, dtype=float)
    return V0 * (1 - np.exp(-(R / Rc) ** beta))
```

---

## 🔍 Outputs and Diagnostics

1. Posterior text summary (MAP, mean, σ)
2. AIC/BIC, log-evidence (harmonic-mean approx.)
3. Residual plots + posterior predictive bands
4. Corner-style visualization (coming soon)

---

## 🧮 Contributing

Pull requests welcome via GitHub Issues or Discussions.
Follow PEP8, document public functions, and include a 1-line test dataset for every new model variant.

---

## 🔀 For Repository Maintainers: Merging PR #1

To merge the Copilot integration and environment setup branch (v1.1.0):

### Quick Merge
```bash
./merge_pr1.sh
```

### Manual Merge
See [MERGE_INSTRUCTIONS.md](MERGE_INSTRUCTIONS.md) for detailed step-by-step instructions, including:
- Standard git merge commands
- Tagging for Zenodo synchronization
- Rollback procedures if needed
- Zenodo webhook verification

---

## 🚪 License

Creative Commons Attribution 4.0 International (CC-BY 4.0).
Feel free to reuse and extend with attribution.

---

## 📚 Citation

If you use QSOL UFF in a publication, please cite:

> **Slade, T. (2025).** *Spectral Algebraics: Audible Geometry via E8-Inspired Signal Synthesis and 3D Visualization.*
> Zenodo. [https://doi.org/10.5281/zenodo.17557660](https://doi.org/10.5281/zenodo.17557660)

All versions are covered by the concept DOI [10.5281/zenodo.17510648](https://doi.org/10.5281/zenodo.17510648).

---

## 🔗 Related Projects

* [QSOL KCB / QEC](https://github.com/QSOLKCB/QEC) — Quantum Error Correction Framework
* [QSOL KCB / QAI-UFT](https://github.com/QSOLKCB/QAI-UFT) — Unified Field Theory Core
* [Spectral Algebraics Paper on Zenodo](https://doi.org/10.5281/zenodo.17557660)

---

## 🧩 v2 Roadmap

* 🔄 Adaptive proposal covariance in MCMC
* 🧒 3D E₈ visualization hooks via Matplotlib projections
* 📈 Batch SPARC benchmark runner
* 🚐 UFF extensions for baryonic feedback & dark-field interactions

---

## 👨‍🔬 Acknowledgments

Thanks to QSOL IMC and open-science collaborators for providing the framework to test non-standard cosmologies through reproducible code and sound spectra.

---

*Truth Compiled · QSOL IMC 2025 · Arch Linux Certified*
