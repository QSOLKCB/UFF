# QSOL UFF

[![CI](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml/badge.svg)](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17669627.svg)](https://doi.org/10.5281/zenodo.17669627)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**UFF v4.0.0 — Galaxy Dynamics and Compact-Object Research Laboratory**

QSOL UFF is a transparent Python toolkit for fitting galaxy rotation curves,
comparing physical and phenomenological model families, and reporting the
scale separation between galaxy dynamics, Kerr supermassive black holes
(SMBHs), and loop-quantum-gravity-inspired (LQG) compact-object research.

Version 4 replaces the old placeholder comparators with dimensionally explicit
models, validated input handling, deterministic multi-start fitting, testable
limiting cases, and machine-readable provenance.

> **Scientific boundary:** UFF is a research framework, not evidence that a
> unified field theory, MOND, dark matter, or an LQG black-hole model is
> correct. The repository-specific UFF curve is explicitly labelled
> empirical. LQG is not used to explain galaxy rotation curves.

## What v4 fixes

| Area | Before v4 | v4 |
|---|---|---|
| SPARC baryons | Velocity scale factors were squared; signed gas was lost | Mass-to-light ratios scale `V²`; `Vgas × abs(Vgas)` is preserved |
| NFW | Shape-only approximation using `Vmax` and `Rs` | Physical `M200`, `c200`, `r200`, and configurable `H0` |
| MOND | Gas, disk, and bulge speeds were added before squaring | Correct baryonic acceleration with simple, standard, and RAR relations |
| SMBH | No central compact object | Optional fixed/fitted point mass plus separate Kerr horizon/photon-orbit/ISCO report |
| LQG | Not scoped | Area-gap scale diagnostic and opt-in bookkeeping ansatz, isolated from galaxy likelihoods |
| Galaxy systematics | Unchecked CSV rows | Strict validation plus optional SPARC distance and inclination nuisance fits |
| Model selection | One UFF fit with visual overlays | Same-data likelihoods, χ², RMSE, AIC/AICc/BIC, ΔBIC, and relative weights |
| Reproducibility | Generated files and legacy scripts mixed into source | SHA-256 input receipt, deterministic seeds, JSON schema, CI, and tests |

## Included models

| CLI name | Family | Free structural parameters | Status |
|---|---|---|---|
| `baryons` | Newtonian baseline | Stellar M/L, optional nuisance parameters | Established weak-field calculation |
| `nfw` | ΛCDM halo baseline | `log10(M200/M☉)`, `c200` | Standard collisionless-halo profile |
| `burkert` | Cored halo baseline | `log10(ρ0)`, core radius | Empirical dark-matter profile |
| `mond-rar` | MOND/RAR | Optional `a0` | Empirical exponential acceleration relation |
| `mond-simple` | MOND | Optional `a0` | Algebraic simple interpolating function |
| `mond-standard` | MOND | Optional `a0` | Algebraic standard interpolating function |
| `mond-efe` | MOND sensitivity test | Fixed external field and orientation | Approximate algebraic proxy, not a field-equation solver |
| `uff-empirical` | QSOL UFF | Asymptotic speed, core radius, bounded shape term | Repository-specific research law |

Every galaxy model can also include a central SMBH. See
[Model definitions](docs/MODELS.md) for equations, units, and limitations.
Run `python -m uff models` to list the canonical CLI names.

## Install

```bash
git clone https://github.com/QSOLKCB/UFF.git
cd UFF
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

The compatibility script still works, but `python -m uff` or the installed
`uff` command is preferred.

## Quick start

Run the default four-way comparison:

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --gal DEMO_GALAXY \
  --out outputs \
  --sonify
```

Equivalent legacy entry point:

```bash
python analyze_sparc.py --csv DEMO_GALAXY.csv --gal DEMO_GALAXY --out outputs
```

Compare a wider candidate set and fit the MOND acceleration scale:

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --models baryons,nfw,burkert,mond-rar,mond-simple,mond-standard,uff-empirical \
  --fit-a0 \
  --restarts 24 \
  --out outputs
```

Use the external-field sensitivity proxy only when an external field has been
specified:

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --models mond-rar,mond-efe \
  --external-field-a0 0.03 \
  --external-field-angle-deg 60
```

Fit SPARC-style distance and inclination nuisance parameters:

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --models nfw,burkert,mond-rar \
  --fit-distance \
  --fit-inclination
```

The input needs `INC_deg` metadata, or use `--inclination-deg`. Nuisance
parameters increase model complexity and should be given informed priors in
publication-grade analyses; v4's bounded ranges are transparent exploratory
defaults.

## Outputs

For a galaxy named `DEMO_GALAXY`, the CLI writes:

| File | Purpose |
|---|---|
| `DEMO_GALAXY_summary.json` | Configuration, input hash, fitted values, full residual arrays, diagnostics, and warnings |
| `DEMO_GALAXY_comparison.csv` | Flat model-ranking table |
| `DEMO_GALAXY_models.png` | Rotation curves and standardized residuals |
| `DEMO_GALAXY_<model>_phase_glyph.wav` | Optional deterministic stereo sonification |
| `DEMO_GALAXY_e8_reference.png` | Optional legacy E₈ visualization, explicitly outside the fit |
| `DEMO_GALAXY_<model>_posterior.npz` | Optional retained multi-chain posterior samples |
| `DEMO_GALAXY_<model>_corner.png` | Optional posterior parameter plot |
| `DEMO_GALAXY_<model>_postpred.png` | Optional 16–84% posterior curve band |

Model weights are relative information-criterion weights for the models in the
candidate set. They are not posterior probabilities that a physical theory is
true.

### Optional posterior sampling

The deterministic optimizer is the default. To retain posterior draws for one
candidate, enable the bounded full-covariance Metropolis sampler:

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --models nfw,burkert,mond-rar \
  --mcmc-steps 12000 \
  --mcmc-burn 4000 \
  --mcmc-chains 4 \
  --corner \
  --postpred
```

The sampler adapts its proposal only during burn-in and freezes the transition
kernel for retained draws. It writes a compressed NPZ, R-hat, approximate
effective sample sizes, parameter quantiles, a corner plot, and posterior curve
bands. Treat R-hat above 1.05 or low ESS warnings as non-convergence, not as a
cosmetic diagnostic.

## SMBH and LQG scale report

Galaxy fitting uses an SMBH only as a weak-field central point mass. Strong
field quantities belong to a separate command:

```bash
python -m uff compact-object \
  --mass-msun 4300000 \
  --spin 0.5 \
  --velocity-dispersion-kms 100 \
  --out outputs/sgr-a-scale-report.json
```

The report contains the Kerr gravitational radius, outer horizon, equatorial
photon orbit, ISCO, sphere of influence, LQG area-gap convention, and
`Δ/r²` scale ratio. It does not select or validate a particular effective LQG
metric. See [Scientific status, July 2026](docs/SCIENCE_STATUS_2026.md).

## Input data

Canonical CSV columns are:

```text
R_kpc,V_obs_kms,e_V_kms,V_gas_kms,V_disk_kms,V_bul_kms
```

Short SPARC aliases (`Rad`, `Vobs`, `errV`, `Vgas`, `Vdisk`, `Vbul`) are also
accepted. Full details are in [Data format](docs/DATA_FORMAT.md). The official
[SPARC database](https://astroweb.case.edu/SPARC/) remains the authoritative
source for the original 175-galaxy data release.

## QSOL project bridges

The v4 diagnostic layer makes narrow, labelled connections to three related
QSOL repositories:

- [QAI-UFT](https://github.com/QSOLKCB/QAI-UFT): π/2 phase fingerprint and phase-glyph convention.
- [QNTOY](https://github.com/QSOLKCB/QNTOY): normalized entropy telemetry for ambiguity among model weights.
- [TFT](https://github.com/QSOLKCB/TFT): covariance eigenspectrum and norm invariants.

These transforms do not change a model prediction or likelihood. No field
equation was copied from those projects. See
[Related-project interoperability](docs/RELATED_PROJECTS.md).

## Repository layout

```text
uff/
  compact.py       # SMBH/Kerr and LQG scale diagnostics
  constants.py     # explicit physical constants and units
  data.py          # validated canonical/SPARC CSV loading
  diagnostics.py   # plots, entropy, invariants, phase fingerprints, WAV
  fitting.py       # deterministic bounded fitting and information criteria
  models.py        # baryons, halos, MOND/RAR, UFF empirical law
  sampling.py      # opt-in burn-in-adapted posterior sampler
  cli.py           # fit, batch, and compact-object commands
tests/             # analytic limits, synthetic recovery, and CLI tests
docs/              # equations, data contract, science status, project bridges
analyze_sparc.py   # compatibility launcher
uff_model.py       # compatibility function
```

## Research status and limitations

- Rotation curves alone do not settle the dark-matter-versus-modified-gravity
  question.
- The algebraic MOND relations are not full AQUAL/QUMOND solvers for flattened
  disks. `mond-efe` is a sensitivity proxy with an explicit warning.
- AIC/BIC rankings depend on data quality, priors/bounds, nuisance treatment,
  and the candidate set.
- A central SMBH is only constrainable if the observations resolve its sphere
  of influence.
- Current LQG black-hole phenomenology contains multiple effective metrics;
  there is no accepted LQG galaxy rotation law implemented here.
- The UFF empirical profile is falsifiable as a curve family but is not yet
  derived from a covariant action, lensing law, or cosmological solution.

The detailed evidence boundary and current literature snapshot are maintained
in [docs/SCIENCE_STATUS_2026.md](docs/SCIENCE_STATUS_2026.md).

## Citation

Please cite the software record:

> Slade, T. (2026). *QSOL UFF v4.0.0: Galaxy Dynamics and Compact-Object
> Research Laboratory*. Zenodo. <https://doi.org/10.5281/zenodo.17669627>

Machine-readable metadata are in [CITATION.cff](CITATION.cff). Cite the
original scientific papers for every model used in an analysis; references are
listed in [docs/MODELS.md](docs/MODELS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).

Maintainer: **Trent Slade / QSOL-IMC**
GitHub: [QSOLKCB](https://github.com/QSOLKCB)
