# QSOL UFF v5.0.0 — Release notes

**Release date:** 7 August 2026  
**Theme:** Reproducible astrophysics, frozen falsification contracts, and survey-aware adversarial evaluation

UFF v5.0.0 is a major repository release. It retains the validated galaxy-dynamics and compact-object laboratory from v4 while adding a complete second research track for formalising, testing and replaying disputed catalogue-level celestial-node claims.

The release combines:

- the v4 galaxy and compact-object analysis core;
- UFF-SLFA v1.0.0;
- the Sheridan Crucible v1.1.0 survey-aware engine;
- content-addressed public-claim provenance;
- an independent methodological assessment and formalisation roadmap; and
- review-driven correctness, memory and replay hardening.

## Highlights

### One repository, four separated layers

UFF now exposes four intentionally distinct layers:

1. **Galaxy and compact-object laboratory** — explicit astrophysical model fitting and scale diagnostics.
2. **UFF-SLFA** — preregistered catalogue-level tests of frozen celestial-node claims.
3. **Sheridan Crucible** — survey-aware density reconstruction, competing models and injection calibration.
4. **Provenance and assessment governance** — immutable claim versions, source manifests, blockers and external review responses.

The default `uff` command remains galaxy-focused. SLFA and Sheridan use separate interfaces and verdict namespaces so model fitting, historical reproduction, diagnostic tests and prospective confirmation cannot be confused.

## Retained from v4.0.0

The complete galaxy-dynamics and compact-object research laboratory remains available:

- validated canonical and SPARC-style CSV ingestion;
- signed gas handling and dimensionally explicit baryonic scaling;
- physical NFW and empirical Burkert halo models;
- MOND/RAR variants and an explicitly approximate external-field sensitivity proxy;
- the repository-specific empirical UFF rotation-curve family;
- optional weak-field central SMBH fitting;
- deterministic bounded multi-start optimisation;
- chi-squared, RMSE, AIC, AICc, BIC and relative model weights;
- optional bounded Metropolis posterior sampling, R-hat and ESS diagnostics;
- rotation-curve plots, residuals, deterministic sonification and SHA-256 input receipts; and
- separate Kerr/SMBH and LQG scale reporting outside the galaxy likelihood.

No v4 physical model is reinterpreted as evidence for the celestial-node claims introduced in v5.

## Added: UFF-SLFA v1.0.0

UFF-SLFA turns a celestial-node assertion into a frozen, replayable catalogue test.

### Contracts and anti-circularity

- Added machine-readable `uff.sky-lattice-claim.v1` contracts.
- Froze node IDs, ICRS coordinates, cap radius, anomaly predicate, holdout, weights, strata, null model, seed, alpha, minimum effect and required node support.
- Rejected node-targeted catalogue selection as confirmatory evidence.
- Required either an independent catalogue or an untouched declared holdout for confirmatory execution.
- Preserved failed and untestable nodes in the denominator and output tables.

### Statistics and null models

- Added weighted inside-versus-outside anomaly-rate contrasts.
- Added shared RA-shift nulls.
- Added Haar-uniform proper SO(3) rotation nulls for the complete rigid architecture.
- Added stratified anomaly-label permutation.
- Added plus-one empirical p-values so permutation p-values cannot be reported as zero.
- Added Holm step-down family-wise correction for node-level tests.
- Added Haldane-Anscombe odds ratios as diagnostics without replacing the frozen primary statistic.

### Geometry verification

- Added TFT-derived checks for `R^T R = I` and `det(R) = +1`.
- Added Gram-matrix and pairwise angular-separation invariance checks for every geometric null transform.

### Evidence bundles

- Added canonical recipe, observations and complete node artifacts.
- Added SHA-256 manifests.
- Separated artifact integrity from numerical replay.
- Added deterministic verification against the exact frozen catalogue.

## Added: Sheridan Crucible v1.1.0

Sheridan wraps an ordinary SLFA claim in `uff.sheridan-crucible.v1` and makes the survey geometry part of the experiment.

### Survey representation

- Added explicit spherical support quadrature with solid-angle weights.
- Added fractional coverage and mask handling.
- Added completeness filtering and normalised inverse-completeness analysis weights.
- Added deterministic equal-area Fibonacci support-grid generation.

### Spherical density reconstruction

- Added normalised von Mises-Fisher weighted KDE on the sphere.
- Added leave-one-out global bandwidth selection.
- Added adaptive local bandwidths.
- Added mask-aware kernel-mass and field-edge renormalisation.
- Added a frozen `maximum_exact_sources` guard rather than silently changing algorithm or exhausting memory.

### Survey-aware geometric nulls

- Added survey-availability scoring at each node.
- Added acceptance-matched Haar-uniform SO(3) rotations that retain node testability and comparable survey support.
- Retained all rigid-geometry invariance checks from SLFA.

### Competing models and calibration

- Added nuisance-only versus nuisance-plus-node weighted logistic comparison.
- Added transparent pseudo-BIC reporting, convergence status and coefficient bound-hit detection.
- Added Laplace-approximate predictive checks.
- Added nuisance-baseline anomaly-label injection using real in-footprint catalogue rows.
- Added recovery-rate and false-positive-rate reporting.
- Added component-wise frozen decisions so a result cannot redefine which tests count after execution.

### Sheridan evidence bundles

- Added `recipe.json`, `density.json`, `nodes.csv`, `models.json`, `injection.json`, `decision.json` and `manifest.json`.
- Added complete integrity checking and numerical replay against the frozen catalogue and support grid.

## Added: public-claim provenance ledger

UFF v5 records public Logvinovich claim variants without silently reconciling them.

- Added a human-readable public claim ledger.
- Added machine-readable `uff.public-claim-profile.v1` provenance records.
- Recorded incompatible node tables, radii, thresholds, temporal cuts, query hotspots, reported counts and claimed p-values.
- Marked profiles with unresolved primary fields as `not-ready-for-confirmatory-run`.
- Recorded absent Gaia query text and thresholds as blockers rather than inferred values.
- Added content-addressed source acquisition metadata, byte counts, timestamps and SHA-256 identities.
- Added claim-level `source_refs` and a source manifest.
- Added tests that preserve the exact incomplete rotated E-node set and prohibit accidental execution.

The ledger distinguishes a historical claim audit from a valid blind confirmatory experiment.

## Added: independent methodological assessment

The repository now includes a repository-native rendition and response package for the supplied *Independent Research Assessment of the Logvinovich Claim and Sheridan Audit*.

- Added the assessment rendition keyed to the authoritative PDF SHA-256.
- Added acquisition and visual-verification metadata.
- Added a formal response matrix.
- Added a machine-readable `uff.independent-assessment-response.v1` action ledger.
- Added exact action-priority regression tests.
- Accepted the governing judgement that the crucible syntax is largely formalised while statistical calibration and one stationary claimant specification remain incomplete.

The assessment is methodological critique. It is not represented as an independent execution of UFF or validation of the underlying claim.

## Formalisation roadmap

The current ten-field abstraction:

```text
C = (N, r, A, D, S, H, N_null, alpha, delta, k)
```

is proposed for a future breaking schema as:

```text
C* = (P, N, r, A, D, S, H, T, E, M, N_null, alpha, delta, k, Q, R)
```

where the added fields freeze:

- `P` — immutable provenance and historical claim version;
- `T` — exact statistic and node aggregation;
- `E` — scientific estimand;
- `M` — the complete multiplicity family;
- `Q` — quality, deduplication, cross-match and missing-data rules; and
- `R` — reproducible environment and RNG specification.

The planned breaking identifier is `uff.sheridan-crucible.v2`. It is a roadmap and is **not implemented in v5.0.0**.

The roadmap also separates:

- `S_survey` — footprint, masks, exposure, completeness, depth and cadence; and
- `S_astro` — ordinary source density, crowding, extinction and expected catalogue-failure structure.

## Review-driven hardening

Copilot and subsequent review passes produced substantive corrections across SLFA, Sheridan and the governance layer.

### SLFA corrections

- Normalised right ascension at the celestial poles before duplicate-node detection.
- Rejected confirmatory holdout columns without a frozen holdout value.
- Rejected missing stratum values rather than allowing non-exchangeable labels to remain frozen accidentally.
- Corrected null-generation completion logic at the final allowed attempt.
- Required the exact v1 artifact set before integrity verification can pass.
- Validated manifest paths before membership checks.
- Replayed sparse untestable nodes with NaN-aware numerical comparison.
- Converted invalid replay contracts and audit failures into explicit failed verification reports.
- Rejected non-finite observed statistics and non-finite node geometry.
- Added deterministic RA-shift and stratified-label regression coverage.
- Rejected non-empty output directories before expensive analysis begins.

### Sheridan corrections

- Passed the joint objective and gradient to SciPy without duplicate coefficient evaluation.
- Streamed support-grid and kernel-centre blocks to bound peak memory during survey quadrature.
- Added shape validation for support vectors, area weights and coverage arrays.
- Added regression tests for bounded streaming, malformed quadrature and the optimiser contract.

### Provenance and assessment corrections

- Replaced mutable bare source URLs with structured, content-addressed acquisition records.
- Added claim-level source references.
- Locked the exact ten-ID complement of the incomplete rotated E-node list.
- Preserved the established `uff.sheridan-crucible` schema family for the planned v2 contract.
- Added the missing P0 estimand/effect decision-rule action.
- Added the missing P2 cross-catalogue-dependence action.
- Locked the exact contract delta and all action-to-priority mappings in regression tests.

## Documentation

The README has been completely rewritten for the current repository architecture.

New and updated documentation includes:

- the SLFA protocol;
- the Sheridan Siege Engine protocol;
- standalone SLFA and Sheridan release notes;
- the public claim ledger and source manifest;
- the independent assessment rendition, source manifest and formal response;
- example contracts and machine-readable governance ledgers; and
- canonical COSMOS-Web methodological BibTeX metadata.

## Metadata and citation

- Bumped the package version from `4.0.0` to `5.0.0`.
- Updated the package description and research keywords.
- Updated `CITATION.cff` for the full v5 scope.
- Added documentation and changelog project URLs.
- Published the complete v5.0.0 software-and-documentation package on Zenodo.
- Added the canonical version DOI to the README, package URLs, citation metadata,
  changelog, and root `.zenodo.json` deposit metadata.

Please cite the archived software release as:

> Slade, T. (2026). *QSOL UFF v5.0.0: Reproducible Astrophysics and
> Falsification Laboratory* (Version 5.0.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21830630

## Compatibility

- Existing galaxy CLI commands remain available through `python -m uff` and the installed `uff` command.
- `analyze_sparc.py` remains a compatibility launcher.
- SLFA remains available through `python sky_lattice_audit.py`.
- Sheridan remains available through `python -m uff.sheridan`.
- Existing executable schemas remain `uff.sky-lattice-claim.v1` and `uff.sheridan-crucible.v1`.
- No existing schema is silently reinterpreted as the planned v2 contract.

## Scientific boundary

UFF v5 can preserve a claim, freeze an experiment, model an explicit survey support function, calibrate detection power, compare declared alternatives and replay a result. It does not establish that:

- a catalogue diagnostic is a physical object;
- a broadband colour is a spectral line;
- multiple catalogues are statistically independent;
- a selected null model is uniquely correct;
- a positive spatial association proves its proposed cause; or
- a failed frozen claim excludes every future modified theory.

A reproducible workflow can reproduce a biased query perfectly. This release makes that distinction explicit and machine-auditable.
