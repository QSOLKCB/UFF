# QSOL UFF v5.1.0 - Release notes

**Release date:** 7 August 2026  
**Theme:** Defense-in-depth trust, pre-observation witnessing, ensemble guardrails, and receiver-neutral forensic telemetry

UFF v5.1.0 is an additive assurance release. It keeps the v5.0.0 galaxy/compact-object, UFF-SLFA, Sheridan Crucible, provenance, and assessment systems intact while adding a deliberately small defense layer around evidence admission and observation.

No existing executable scientific schema is redefined in this release. The new mechanisms strengthen how evidence is committed, checked, replayed, sealed, observed, and interpreted.

## Highlights

### QEC-inspired computational bouncer

Added `uff.qec_gate` with:

- strict canonical UTF-8 JSON;
- duplicate-key, BOM, NaN, and Infinity rejection;
- exact profile-specific artifact allowlists;
- path-traversal, symbolic-link, and unlisted-file rejection;
- child byte-count and SHA-256 recomputation;
- embedded-contract digest recomputation;
- manifest/decision cross-checks;
- deterministic bundle roots with self-hash exclusion;
- optional external root anchoring;
- exact deterministic `qec_gate.json` receipt validation; and
- explicit `INTEGRITY_ONLY`, `REPLAY_VERIFIED`, and `REJECTED` assurance states.

`admitted=true` requires fresh numerical replay through the existing SLFA or Sheridan domain verifier. Integrity-only inspection cannot replay or admit, even when replay inputs are supplied.

### SPECTRAL-inspired pre-observation witness

Added `uff.spectral_witness` to commit identity-bearing inputs before observation:

- raw and canonical contract SHA-256;
- catalogue SHA-256 and byte count;
- Sheridan support-grid SHA-256 and byte count;
- domain-separated deterministic commitment;
- optional externally anchored precommit digest; and
- reveal only through a QEC `REPLAY_VERIFIED` bundle.

The witness is now cryptographically bound to the canonical contract actually verified inside the replayed recipe. A witness for contract A cannot admit a replay-valid bundle built from contract B.

A local witness establishes identity, not chronology. Historical preregistration still requires an independent timestamped or signed anchor.

### Statistical-mechanics ensemble guardrail

Added `docs/STATISTICAL_MECHANICS_GUARDRAIL.md` and formalized the separation:

```text
REPLAY_VERIFIED != ENSEMBLE_CALIBRATED
ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH
```

Quantum statistical mechanics is used only as a methodological analogy for separating exact microscopic/computational evolution from ensemble-level inference. No quantum many-body model is imported into UFF catalogue analysis.

A future `ENSEMBLE_CALIBRATED` state must be separately earned through prespecified type-I-error, power, negative-control, survey-systematic, convergence, seed-block, and multiplicity calibration.

### SONIFICATION-inspired forensic telemetry

Added `uff.audit_events`:

- deterministic receiver-neutral event documents;
- integrity, replay, admission, and optional external-root states;
- already-recorded UFF decisions and selected metrics;
- fixed canonical event fields; and
- explicit separation from tempo, hertz, MIDI, timbre, loudness, waveform, and rendered audio.

Telemetry is read-only, has no scientific or admission authority, and may not be written inside a closed evidence bundle.

Malformed manifests now remain observable as deterministic trust-boundary failure events rather than causing telemetry to fail before representing the rejection.

## Review-driven correctness fixes

The defense layer received a dedicated Copilot review pass. Four substantive issues were fixed and regression-locked:

1. SPECTRAL now verifies its committed contract digest against the contract actually verified by QEC from the replayed recipe.
2. Audit telemetry loads scientific manifest fields only after integrity passes, preserving rejected-manifest telemetry.
3. `require_replay=False` now disables replay absolutely, regardless of supplied catalogue/support paths.
4. Existing `qec_gate.json` receipts must equal the complete deterministic receipt payload exactly; missing, changed, or extra fields are rejected.

## Validation

The full CI matrix passes on Python 3.10, 3.11, 3.12, and 3.13.

Regression coverage includes:

- integrity-only versus replay assurance;
- noncanonical JSON tampering;
- hidden/unlisted payload smuggling;
- external root mismatch;
- exact sealed-receipt validation;
- pre-observation input substitution;
- contract-A/witness versus contract-B/replay mismatch;
- deterministic witness commitments;
- deterministic receiver-neutral telemetry;
- malformed-manifest telemetry; and
- telemetry/evidence-bundle isolation.

## Documentation and archival package

Added a versioned formal technical-report source at `papers/UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md`. A rendered archival PDF, `UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.pdf`, is included in the Zenodo upload bundle.

The report documents the threat model, architecture, authority matrix, assurance ladder, review-driven hardening, preservation requirements, and scientific limitations of the defense layer.

A Zenodo v5.1.0 upload package is also provided under `zenodo/v5.1.0/` with metadata guidance, a file manifest, and SHA-256 checksums for the supporting archival documents.

## Compatibility

- Existing `python -m uff` galaxy and compact-object commands remain unchanged.
- UFF-SLFA remains available through `python sky_lattice_audit.py`.
- Sheridan remains available through `python -m uff.sheridan`.
- Existing executable schemas remain `uff.sky-lattice-claim.v1` and `uff.sheridan-crucible.v1`.
- The planned `uff.sheridan-crucible.v2` remains a roadmap and is not implemented by v5.1.0.

## Scientific boundary

This release strengthens computational assurance, not physical certainty.

It does not establish that:

- source catalogues are correct or unbiased;
- analysts were historically blind merely because a local commitment exists;
- the selected null ensemble is scientifically adequate;
- catalogues are statistically independent;
- an association is causal; or
- a proposed physical ontology is true.

The governing assurance rule is:

```text
INPUTS_COMMITTED
      -> INTEGRITY_VERIFIED
      -> REPLAY_VERIFIED
      -> ENSEMBLE_CALIBRATED      (future; separately earned)
      -> SCIENTIFICALLY_DEFENSIBLE (external scientific judgement)
```

No lower rung implies a higher rung.

## Zenodo versioning

This release supersedes the archived v5.0.0 software state:

> Slade, T. (2026). *QSOL UFF v5.0.0: Reproducible Astrophysics and Falsification Laboratory* (Version 5.0.0) [Computer software]. Zenodo. DOI: 10.5281/zenodo.21830630.

Create the v5.1.0 deposit using Zenodo's **New version** workflow so the v5.0.0 DOI remains immutable and citable. The v5.1.0 version DOI will be assigned when the new version is published.
