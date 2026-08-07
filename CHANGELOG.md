# Changelog

Notable changes to QSOL UFF are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.1.0] - 2026-08-07

### Added

- QEC-inspired `uff.qec_gate` fail-closed evidence boundary with canonical JSON,
  exact artifact allowlists, physical bundle closure, child/hash recomputation,
  embedded-contract validation, deterministic roots, optional external anchors,
  and replay-required admission.
- SPECTRAL-inspired `uff.spectral_witness` pre-observation commit/reveal workflow
  for contract, catalogue, and Sheridan support-grid identities.
- Statistical-mechanics interpretation guardrail separating computational replay
  from future null-ensemble calibration.
- SONIFICATION-inspired `uff.audit_events` receiver-neutral read-only forensic
  telemetry outside the evidence bundle.
- Formal v5.1.0 defense-in-depth technical report source plus rendered archival PDF.
- Zenodo v5.1.0 upload guidance, metadata snapshot, manifest, and checksums.

### Changed

- Bumped package, README, citation, and Zenodo metadata to v5.1.0.
- Expanded the assurance model to distinguish input commitment, integrity,
  replay, future ensemble calibration, and external scientific judgement.
- Bound SPECTRAL reveal to the canonical contract digest actually verified from
  the replayed recipe.
- Made integrity-only mode incapable of replay or admission even when replay
  inputs are supplied.
- Made `qec_gate.json` receipt verification exact by reconstructing the complete
  deterministic receipt payload.
- Made malformed-manifest failures observable through deterministic telemetry.

### Validation

- Python 3.10-3.13 CI passes for the merged defense layer.
- Added regression coverage for contract substitution, malformed-manifest
  telemetry, integrity-only replay suppression, and exact receipt validation.

### Scientific boundary

- `REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH`.
- A local witness establishes identity, not historical chronology.
- Telemetry is an observation aid and has no authority over evidence admission
  or scientific verdicts.
- `uff.sheridan-crucible.v2` and `ENSEMBLE_CALIBRATED` remain future work.

Full details: [RELEASE_NOTES_v5.1.0.md](RELEASE_NOTES_v5.1.0.md).

## [5.0.0] - 2026-08-07

### Added

- UFF-SLFA v1.0.0 with frozen `uff.sky-lattice-claim.v1` contracts,
  anti-circularity validation, independent-catalogue/holdout requirements,
  RA-shift, SO(3), and stratified-label nulls, finite empirical p-values,
  Holm correction, complete node tables, SHA-256 bundles, and numerical replay.
- Sheridan Crucible v1.1.0 with explicit survey-support quadrature,
  masks/completeness, spherical von Mises-Fisher weighted KDE, adaptive
  bandwidths, edge correction, survey-matched SO(3) rotations, nuisance-model
  comparison, predictive checks, synthetic injection calibration, and replay.
- Content-addressed public-claim ledger and machine-readable
  `uff.public-claim-profile.v1` provenance records.
- Independent methodological assessment rendition, source manifest, response
  matrix, and `uff.independent-assessment-response.v1` action ledger.
- Formalisation roadmap for the planned breaking
  `uff.sheridan-crucible.v2` contract family.
- Review-driven regression coverage for geometry, artifact completeness,
  replay failure handling, finite statistics, bounded survey quadrature,
  optimiser evaluation, source provenance, and exact action priorities.
- Consolidated v5 README and release notes.
- Published Zenodo software archive with version DOI
  [`10.5281/zenodo.21830630`](https://doi.org/10.5281/zenodo.21830630).

### Changed

- Reframed the repository as a reproducible astrophysics and falsification
  laboratory while preserving the complete v4 galaxy and compact-object core.
- Bumped package and citation metadata to v5.0.0.
- Separated galaxy fitting, historical claim reproduction, diagnostic tests,
  survey-corrected enrichment, and prospective confirmation.
- Replaced the obsolete archive reference with the published v5 Zenodo DOI in
  the README, package URLs, citation metadata, release notes, and canonical
  `.zenodo.json` deposit metadata.

### Scientific boundary

- Frozen contracts, hashes and replay establish specification and
  computational consistency; they do not prove that a sampling frame or null
  model is scientifically adequate.
- A supported catalogue association does not identify its physical cause.
- `uff.sheridan-crucible.v2` is a roadmap and is not implemented in v5.0.0.

Full details: [RELEASE_NOTES_v5.0.0.md](RELEASE_NOTES_v5.0.0.md).

## [4.0.0] - 2026-07-14

### Added

- Installable `uff` Python package and `python -m uff` CLI.
- Validated canonical and SPARC-style CSV loader with aliases and SHA-256 input receipts.
- Signed gas handling and dimensionally explicit baryonic scaling.
- Physical NFW and empirical Burkert halo models.
- MOND/RAR variants and an explicitly approximate external-field sensitivity proxy.
- The repository-specific empirical UFF rotation-curve family.
- Optional weak-field central SMBH fitting.
- Deterministic bounded multi-start optimisation.
- Chi-squared, RMSE, AIC, AICc, BIC and relative model weights.
- Optional bounded Metropolis posterior sampling, R-hat and ESS diagnostics.
- Rotation-curve plots, residuals, deterministic sonification and SHA-256 input receipts.
- Separate Kerr/SMBH and LQG scale reporting outside the galaxy likelihood.

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
- Corner plots, posterior-predictive figures, sonification, E8 walk, and
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

- v5.1.0 - GitHub tag and Zenodo version DOI pending publication
- [v5.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v5.0.0) -
  [Zenodo archive](https://doi.org/10.5281/zenodo.21830630)
- v4.0.0 - historical release state retained in repository history
- [v3.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v3.0.0)
- [v1.1.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.0.0)
