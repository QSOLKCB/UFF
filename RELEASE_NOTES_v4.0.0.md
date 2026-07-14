# UFF v4.0.0 — Galaxy Dynamics and Compact-Object Research Laboratory

**Release date:** 14 July 2026

**Release type:** Major scientific and architectural correction

Version 4 turns UFF from a single-script placeholder fitter into a tested Python
package with explicit physics boundaries.

## Highlights

- Correct SPARC baryonic mass modeling, including signed gas and linear M/L
  scaling in `V²`.
- Physical NFW (`M200`, `c200`) and Burkert halo implementations.
- MOND simple, standard, empirical RAR, and explicitly approximate EFE proxy.
- Central SMBH contribution across galaxy models.
- Separate Kerr horizon, photon-orbit, ISCO, sphere-of-influence, and LQG
  area-gap scale report.
- Optional distance and inclination nuisance fits using SPARC scaling rules.
- Deterministic bounded multi-start optimization and normalized Gaussian
  likelihoods.
- Opt-in multi-chain posterior sampling with burn-in-only full-covariance
  adaptation, R-hat, ESS estimates, corner plots, and predictive curve bands.
- χ², reduced χ², RMSE, AIC, AICc, BIC, ΔBIC, and relative criterion weights.
- SHA-256 input receipts and a versioned JSON summary schema.
- QAI-UFT phase fingerprints, QNTOY-style model-weight entropy, and TFT-style
  covariance invariants, all isolated from the physical fit.
- Python package/CLI, 23-test initial suite, and GitHub Actions matrix for
  Python 3.10–3.13.

## Breaking changes

- `v_circ_uff` now evaluates the bounded v4 empirical law. Use a v3 tag for
  exact reproduction of the historical placeholder curve.
- The old additive power-law “dark field” is removed from the default model;
  it lacked a defined dimensional normalization and was degenerate with the
  placeholder UFF curve.
- The former approximate `nfw_curve(Vmax, Rs)` and incorrect MOND comparator
  are removed.
- Generated output artifacts and one-off v1 release-maintenance scripts are no
  longer stored on the source branch.

## Scientific interpretation

This release makes no claim that UFF, MOND, dark matter, or an effective LQG
metric has been validated. The empirical UFF law remains a curve family. LQG
is restricted to compact-object scale diagnostics and is not used in the
galaxy likelihood.

See `docs/SCIENCE_STATUS_2026.md` and `docs/MODELS.md` before publishing results.
