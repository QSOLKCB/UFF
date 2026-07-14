# Changelog

Notable changes to QSOL UFF are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-07-14

### Added

- Installable `uff` Python package and `python -m uff` CLI.
- Validated canonical/SPARC CSV loader with aliases and SHA-256 input receipts.
- Physical NFW (`M200`, `c200`) and Burkert halo models.
- MOND simple, standard, empirical RAR, and labelled EFE sensitivity models.
- Central SMBH terms plus separate Kerr and LQG scale-report command.
- Optional distance and inclination nuisance parameters.
- Deterministic multi-start optimization and normalized likelihood statistics.
- Opt-in multi-chain full-covariance Metropolis sampling with adaptation limited
  to burn-in, retained-draw R-hat, ESS estimates, corner plots, and curve bands.
- AIC, AICc, BIC, ΔBIC, relative criterion weights, and residual diagnostics.
- QAI-UFT phase fingerprints, QNTOY-style model ambiguity entropy, and
  TFT-style covariance invariants outside the physical likelihood.
- Automated tests, synthetic NFW recovery, CLI smoke tests, and CI across
  Python 3.10–3.13.
- Model equations, data contract, July 2026 science-status boundary, citation
  metadata, contributing guide, and security policy.

### Changed

- Replaced the historical UFF placeholder with a bounded cored empirical law.
- Rebuilt the demo workflow around explicit same-data model comparison.
- Corrected baryonic mass-to-light scaling and SPARC's signed gas convention.
- Replaced stored generated figures with reproducible commands.
- Retained `analyze_sparc.py` and `uff_model.py` as compatibility entry points.

### Removed

- Incorrect MOND velocity addition and shape-only NFW approximation.
- Dimensionally ambiguous additive power-law dark-field term.
- One-off v1 merge/tag scripts, generated caches, and stale output artifacts.

### Scientific boundary

- LQG is not used in galaxy fits. UFF remains an empirical research model, not
  a claimed completed fundamental theory.

## [3.0.0] - 2025-11-22

### Added

- Full-covariance adaptive Metropolis-Hastings option.
- Corner plots, posterior-predictive figures, sonification, E₈ walk, and
  preliminary UFF/MOND/NFW overlays.

### Known limitations corrected in 4.0.0

- MOND combined baryonic component velocities incorrectly.
- NFW was a visual shape approximation rather than an `M200,c200` halo.
- Adaptive sampling continued after burn-in and lacked convergence tests.
- Documentation overstated the maturity of placeholder equations.

## [1.1.0] - 2025-11-08

### Added

- Environment setup and initial release-maintenance documentation.
- GitHub Copilot integration guidance.

## [1.0.0] - 2025-11-08

### Added

- Initial rotation-curve fitter, demo CSV, notebook, README, and Apache-2.0
  license.

## Release links

- v4.0.0 — release tag pending review of this upgrade
- [v3.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v3.0.0)
- [v1.1.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.0.0)

Concept DOI: <https://doi.org/10.5281/zenodo.17669627>
