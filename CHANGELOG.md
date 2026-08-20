# Changelog

Notable changes to QSOL UFF are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.3.0] - 2026-08-20

### Added

- Reusable nonclaim-calibration vocabulary separating existence, robustness,
  genericity, prevalence, empirical support, and physical truth.
- Additional claim-scope dimensions for causal interpretation, universality,
  predictive direction, and static equilibrium versus general dynamical
  stability.
- Worked machine-readable `uff.nonclaim-reference.v1` record for Jampolski &
  Rezzolla, *Formation of gravastars*, arXiv:2509.15302v2.
- Explicit provenance for whether a nonclaim boundary is source-explicit, an UFF
  interpretation, or a combination of both.
- Regression tests requiring the nonclaim calibration surface and common
  forbidden epistemic promotions to remain machine-readable.
- Zenodo v5.3.0 metadata snapshot and exact-release publication guide.

### Changed

- Bumped package, citation, README, release, and Zenodo metadata to v5.3.0.
- Extended `formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md` to reference the v5.3
  calibration layer without extending the Lean theorem surface.
- Added the governing nonclaim boundary:
  `EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT != PHYSICAL_TRUTH`.
- Preserved the existing assurance boundary
  `REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH` unchanged.
- Bound the published v5.3.0 Zenodo version DOI
  `10.5281/zenodo.22026554` into release-facing metadata while retaining the
  immutable v5.2.0 DOI `10.5281/zenodo.21911644` as historical provenance.
- Preserved tag `v5.3.0` on exact archived source commit
  `1e310b257ec51c92cccf12271900cec5aa972c50`; DOI synchronization is a
  post-publication metadata update on `main`, not a retagged release.

### Validation

- Nonclaim calibration JSON is required to expose all eight calibration
  dimensions and explicit source-basis / UFF-nonclaim fields.
- Tests reject loss of the principal forbidden promotions, including
  `can form -> likely to form`, `fine-tuned success -> robust success`, and
  `theoretical possibility -> empirical occurrence`.
- Release metadata tests require the v5.3.0 DOI, exact release commit, and
  no-retagging publication rule to remain explicit.
- Existing Python and Lean assurance behavior is unchanged by the calibration
  layer.

### Scientific boundary

- The Jampolski & Rezzolla paper is a calibration reference, not evidence that
  astrophysical gravastars are generic, common, observed, or physically
  preferred.
- A successful theoretical construction may establish model possibility while
  leaving robustness, genericity, prevalence, and observation open.
- Backward target-conditioned construction is not silently promoted to forward
  predictive genericity.
- Static equilibrium in a specified construction is not silently promoted to a
  general proof of dynamical stability.

Full details: [RELEASE_NOTES_v5.3.0.md](RELEASE_NOTES_v5.3.0.md).

## [5.2.0] - 2026-08-12

### Added

- Lean 4 formal assurance library under `formal/lean/`, pinned to an exact
  toolchain and focused on software/epistemic invariants rather than physical
  truth claims.
- Machine-checked assurance separation showing that replay verification does not
  itself satisfy ensemble calibration and ensemble calibration does not itself
  satisfy scientific defensibility.
- Monotone replay-promotion theorem proving replay verification cannot downgrade
  an already higher, separately earned assurance state.
- Formal identity models for frozen recipes, claim boundaries, and runtime
  replay contracts.
- Formal fail-closed transaction model in which cancellation produces no
  archival bundle and cannot be exported.
- Formal one-way observation model in which sonification preserves the numerical
  result and telemetry has zero evidence-admission authority.
- Typed manifest-core/envelope construction that makes no-self-hash sequencing
  explicit in the formal interface.
- Explicit machine-visible nonclaims for SHA-256 collision resistance, catalogue
  correctness, null-model adequacy, and physical truth.
- Exact theorem declaration manifest, `#print axioms` audit surface, proof-hole
  rejection, and project-defined axiom/constant rejection including supported
  attribute/modifier-prefixed declarations.
- Dedicated `UFF Lean 4 formal verification` GitHub Actions workflow, including
  post-merge pushes to `main`.
- Runtime-correspondence and NEXUS v1.0.0 architectural-lineage documentation.
- Zenodo v5.2.0 metadata and exact-commit publication checklist.

### Changed

- Bumped package, citation, release, and Zenodo metadata to v5.2.0.
- Bound citation/package metadata to assigned Zenodo version DOI
  `10.5281/zenodo.21911644` while keeping exact commit/tag binding separate.
- Preserved the v5.1.0 scientific schemas and runtime behavior while adding an
  independent formal specification layer.
- Made the distinction between formal proof scope and empirical scientific
  validation part of the release contract.

### Validation

- Lean CI pins the exact Lean release artifact and verifies the v5.1.0 base
  ancestry before building the formal library.
- Formal CI rejects `sorry`, `admit`, and project-defined `axiom`/`constant`
  declarations including supported modified forms, requires exact
  theorem-manifest synchronization including supported modified theorem/lemma
  declarations, and audits advertised theorem dependencies.
- Existing Python 3.10-3.13 CI remains authoritative for runtime regressions.

### Scientific boundary

- `REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH` remains governing.
- Lean proves selected properties of the formal specification only.
- The formal layer does not prove source-catalogue correctness, null-model
  adequacy, SHA-256 security, causal interpretation, or physical ontology.
- `uff.sheridan-crucible.v2` and runtime `ENSEMBLE_CALIBRATED` remain future
  work.

Full details: [RELEASE_NOTES_v5.2.0.md](RELEASE_NOTES_v5.2.0.md).

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
- Validated canonical/SPARC CSV loader with aliases and SHA-256 input receipts.
- Physical NFW (`M200`, `c200`) and empirical Burkert halo models.
- MOND/RAR variants and an explicitly approximate external-field sensitivity proxy.
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

- [v5.3.0](https://github.com/QSOLKCB/UFF/releases/tag/v5.3.0) — [Zenodo version DOI 10.5281/zenodo.22026554](https://doi.org/10.5281/zenodo.22026554); tag bound to exact archived source commit `1e310b257ec51c92cccf12271900cec5aa972c50`
- v5.2.0 — [Zenodo version DOI 10.5281/zenodo.21911644](https://doi.org/10.5281/zenodo.21911644); retained as immutable historical provenance
- [v5.1.0](https://github.com/QSOLKCB/UFF/releases/tag/v5.1.0) - Zenodo version DOI pending metadata patch
- [v5.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v5.0.0) —
  [Zenodo archive](https://doi.org/10.5281/zenodo.21830630)
- v4.0.0 — historical release state retained in repository history
- [v3.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v3.0.0)
- [v1.1.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.1.0)
- [v1.0.0](https://github.com/QSOLKCB/UFF/releases/tag/v1.0.0)
