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
- Formal v5.1.0 defense-in-depth technical report in Markdown and PDF.
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
- Sheridan Crucible v1.1.0 with survey-support quadrature, masks/completeness,
  spherical von Mises-Fisher KDE, adaptive bandwidths, edge correction,
  survey-matched SO(3) rotations, nuisance-model comparison, predictive checks,
  synthetic injection calibration, and replay.
- Content-addressed public-claim ledger and independent methodological assessment.
- Formalisation roadmap for the planned `uff.sheridan-crucible.v2` family.
- Published Zenodo software archive DOI `10.5281/zenodo.21830630`.

### Changed

- Reframed UFF as a reproducible astrophysics and falsification laboratory while
  preserving the complete v4 galaxy and compact-object core.
- Separated galaxy fitting, historical reproduction, diagnostic tests,
  survey-corrected enrichment, and prospective confirmation.

Full details: [RELEASE_NOTES_v5.0.0.md](RELEASE_NOTES_v5.0.0.md).

## [4.0.0] - 2026-07-14

- Installable `uff` Python package and CLI.
- Validated canonical/SPARC ingestion and SHA-256 input receipts.
- Physical NFW and Burkert halos, MOND/RAR variants, UFF empirical law, and
  optional central SMBH term.
- Deterministic optimization, information criteria, optional posterior sampling,
  diagnostics, plots, sonification, tests, and CI.
- Separate compact-object Kerr/LQG scale reporting outside the galaxy likelihood.

## [3.0.0] - 2025-11-22

- Full-covariance adaptive Metropolis-Hastings option.
- Corner plots, posterior-predictive figures, sonification, E8 walk, and
  preliminary UFF/MOND/NFW overlays.

## [1.1.0] - 2025-11-08

- Environment setup and release-maintenance documentation.
- GitHub Copilot integration guidance.

## [1.0.0] - 2025-11-08

- Initial rotation-curve fitter, demo CSV, notebook, README, and Apache-2.0 license.

## Release links

- v5.1.0 - GitHub tag and Zenodo version DOI pending publication
- [v5.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v5.0.0) -
  [Zenodo archive](https://doi.org/10.5281/zenodo.21830630)
- v4.0.0 - historical release state retained in repository history
- [v3.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v3.0.0)
- [v1.1.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.0.0)
