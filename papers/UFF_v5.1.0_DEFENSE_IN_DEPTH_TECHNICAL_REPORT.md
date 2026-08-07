# QSOL UFF v5.1.0
## Defense-in-Depth Trust Boundaries for Reproducible Astrophysical Falsification Workflows

**Technical Report**  
**Author:** Trent Slade (QSOL-IMC)  
**ORCID:** 0009-0002-4515-9237  
**Date:** 7 August 2026  
**Software:** QSOLKCB/UFF  
**License:** Apache-2.0  
**Previous archived software version:** QSOL UFF v5.0.0, DOI 10.5281/zenodo.21830630  
**v5.1.0 DOI:** Assigned when the new Zenodo version is published.

## Abstract

QSOL UFF v5.1.0 adds a deliberately small defense-in-depth assurance layer above the existing UFF Sky-Lattice Falsification Audit (SLFA) and Sheridan Crucible evidence systems. The release does not replace the existing scientific engines and does not transplant the full architectures of the QSOLKCB/QEC, QSOLKCB/SPECTRAL, or QSOLKCB/SONIFICATION projects. Instead, it adopts one narrow engineering mechanism from each: a fail-closed computational boundary gate, a pre-observation input-identity witness, and a receiver-neutral forensic telemetry stream. A fourth component, motivated by statistical mechanics, formalizes the distinction between exact computational replay and validation of a statistical ensemble.

The resulting assurance ladder separates input commitment, structural integrity, numerical replay, future ensemble calibration, and scientific interpretation. The central rule is:

`REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH`

This report specifies the threat model, authority boundaries, deterministic hashing and replay semantics, commit/reveal workflow, telemetry contract, review-driven hardening, validation strategy, limitations, and preservation requirements for the v5.1.0 release.

## 1. Scope and scientific boundary

UFF is a reproducible astrophysics and falsification laboratory. Its existing v5 line contains two intentionally separated research tracks: explicit galaxy/compact-object model analysis, and frozen catalogue-level testing of disputed celestial-node claims through SLFA and the survey-aware Sheridan Crucible.

The v5.1.0 release adds assurance mechanisms around the latter evidence workflow. These mechanisms can establish computational properties such as byte identity, canonical representation, replay consistency, and pre-observation input identity. They cannot establish that a catalogue is unbiased, that a null model represents nature, that an analyst was historically blind, that two catalogues are independent, or that an observed association has a particular physical cause.

Accordingly, no hash state, replay state, witness state, telemetry state, or sonified pattern is represented as scientific truth.

## 2. Existing UFF evidence model

SLFA freezes a machine-readable `uff.sky-lattice-claim.v1` contract and evaluates declared anomaly-rate statistics under specified null transformations. Sheridan wraps a frozen SLFA claim in `uff.sheridan-crucible.v1` and adds survey support, masks, completeness, spherical density reconstruction, nuisance-model comparison, survey-matched rotations, predictive checks, and synthetic injection calibration.

Both systems produce deterministic evidence bundles with canonical JSON artifacts, manifests, SHA-256 identities, and numerical replay support. Prior to v5.1.0, those domain-specific verifiers already separated artifact integrity from numerical replay. The new defense layer adds a stricter domain-neutral admission boundary and explicit pre-observation and telemetry contracts.

## 3. Defense-in-depth architecture

The v5.1.0 architecture is:

```text
BEFORE OBSERVATION
    SPECTRAL-inspired witness
 contract + catalogue + support identity commit
              |
              v
        UFF computation
              |
              v
       QEC boundary gate
 strict structure + hashes + cross-links + replay
       ADMIT / REJECT
              |
              v
 statistical-mechanics guardrail
 replay != ensemble calibration
              |
              v
 SONIFICATION-inspired telemetry
 receiver-neutral events; optional external receivers
```

Each layer has one job and one authority boundary. No layer is allowed to infer a stronger scientific claim merely because a lower-level computational check succeeds.

## 4. Threat model

The defense layer is designed against practical evidence-pipeline failure modes rather than against an abstract omnipotent attacker. The covered failure classes include:

1. permissive JSON parsing that admits duplicate keys, non-finite values, BOMs, or noncanonical encodings;
2. path traversal, symbolic-link substitution, and unlisted-file smuggling inside evidence directories;
3. stale or rewritten manifest hashes that do not reflect the actual child bytes;
4. divergence between manifest decisions, decision artifacts, embedded contracts, and recipe identities;
5. integrity-only verification being mistaken for successful numerical replay;
6. stored receipts being trusted without reconstruction;
7. pre-observation commitments being checked only against supplied files rather than against the contract actually replayed by the evidence bundle;
8. telemetry crashing precisely when a malformed manifest should be represented as a rejected trust-boundary event; and
9. computational replay being rhetorically promoted into statistical or physical validation.

The release does not claim resistance to compromised operating systems, malicious interpreters, stolen signing keys, fabricated source catalogues, or collusion among data custodians. Those require separate operational controls.

## 5. QEC-inspired computational boundary gate

### 5.1 Purpose

`uff.qec_gate` is the computational bouncer. It borrows a narrow set of mature proof-system ideas from QSOLKCB/QEC: canonical JSON, recompute-not-trust validation, child-before-aggregate hashing, self-hash exclusion, exact allowlists, explicit assurance states, and deterministic externally anchorable roots.

### 5.2 Assurance states

The gate exposes three states:

- `INTEGRITY_ONLY`: structural and cryptographic checks passed, but no fresh numerical replay was performed. The bundle is not admitted.
- `REPLAY_VERIFIED`: strict checks passed and the appropriate UFF domain verifier reproduced the stored numerical result from frozen inputs. The bundle is admitted.
- `REJECTED`: a structural, semantic, hash, replay, receipt, or external-anchor requirement failed.

A crucial v5.1.0 invariant is that `require_replay=False` disables replay even when catalogue or support-grid arguments are supplied. Diagnostic integrity mode therefore cannot accidentally upgrade itself to admission.

### 5.3 Strict bundle validation

Before replay, the gate verifies canonical UTF-8 JSON, rejects duplicate object keys and non-finite constants, enforces exact manifest entry fields, checks profile-specific artifact allowlists, rejects unsafe paths and symbolic links, closes the physical bundle against unlisted payloads, recomputes byte counts and SHA-256 values, and parses JSON children canonically even if manifest hashes were rewritten.

The gate also recomputes the canonical SHA-256 of the embedded contract in `recipe.json`, validates the frozen catalogue/support hashes, and cross-checks the manifest result against the actual decision artifact.

### 5.4 Deterministic root and receipt

After structural validation, the gate computes a deterministic root over the canonical manifest digest and sorted child path/size/digest tuples. The receipt `qec_gate.json` is excluded from its own root.

In v5.1.0, seal verification is exact rather than partial. Sealing and verification share the same deterministic receipt constructor, and an existing receipt must equal the expected payload field-for-field. Missing fields, changed booleans, altered boundary text, or arbitrary extra keys fail closed.

A locally computed root does not establish authorship. Authenticity requires an independent anchor such as a signed Git tag/release, DOI record, preregistration, or separately signed message.

## 6. SPECTRAL-inspired pre-observation witness

### 6.1 Identity commitment

`uff.spectral_witness` adds a commit/reveal workflow for identity-bearing inputs before result inspection. The witness commits to:

- raw contract file SHA-256;
- canonical contract SHA-256;
- catalogue SHA-256 and byte count; and
- Sheridan support-grid SHA-256 and byte count when applicable.

Filenames, paths, file modification times, wall-clock time, and UI state are excluded from identity. The commitment uses domain separation and excludes its own digest from the preimage.

### 6.2 Reveal semantics

Reveal succeeds only if current input identities match the frozen commitment and the QEC boundary independently returns `REPLAY_VERIFIED`.

A key review-driven correction in v5.1.0 binds the committed contract to the contract actually verified inside the replayed bundle. The QEC gate exposes the canonical digest it recomputes from the embedded recipe contract, and SPECTRAL reveal requires equality with the witness-committed canonical digest. This prevents a witness for contract A from admitting a replay-valid bundle produced from contract B merely because both use the same profile and catalogue.

### 6.3 Chronology boundary

A local commitment proves identity, not chronology. A historical preregistration claim requires the commitment digest to be placed in an independent timestamped or signed location before observation. The reveal interface can then require the externally anchored commitment digest.

## 7. Statistical-mechanics interpretation guardrail

The statistical-mechanics component is intentionally non-executable in v5.1.0. It contributes an epistemic separation rather than a quantum physical model of catalogue data.

The methodological lesson is that exact microscopic dynamics does not, by itself, establish the validity of a chosen statistical ensemble. UFF applies the analogous rule:

**Exact computational replay does not, by itself, establish that a chosen null/resampling ensemble is the correct statistical description of the data-generating process.**

The assurance ladder therefore distinguishes:

1. input commitment;
2. structural integrity;
3. numerical replay;
4. ensemble calibration; and
5. scientific interpretation.

A future `ENSEMBLE_CALIBRATED` state must be separately earned through prespecified type-I-error, power, negative-control, survey-mask/systematic, convergence, seed-block, and multiplicity calibration. Deterministic replay cannot automatically promote a result to that state.

No Hilbert-space, Hamiltonian, density-matrix, microcanonical, canonical, thermalization, or eigenstate-thermalization model is imported into UFF catalogue analysis by this guardrail.

## 8. SONIFICATION-inspired receiver-neutral telemetry

`uff.audit_events` turns live QEC gate outcomes and already-recorded UFF outputs into a deterministic event document suitable for external human or machine receivers.

Canonical event fields include event order, channel, code, state, polarity, authority, fixed-point values, and integer values. Receiver choices such as tempo, hertz, MIDI notes, timbre, loudness, stereo placement, waveform, or rendered audio remain noncanonical.

Telemetry has zero authority over evidence-bundle admission or scientific verdicts and is forbidden inside the closed evidence-bundle directory.

A review-driven v5.1.0 correction ensures manifest-level gate failures remain observable. The manifest is loaded for scientific fields only after gate integrity passes; malformed or missing manifests can therefore produce deterministic `INTEGRITY=FAIL`, `REPLAY=ABSENT`, and `ADMISSION=REJECT` events rather than causing the telemetry layer to fail before representing the rejection.

## 9. Authority matrix

| Layer | Computational admission | Scientific result | Physical truth |
|---|---|---|---|
| UFF scientific engine | Produces evidence | Executes frozen method | No authority |
| QEC boundary gate | May block | Cannot change | No authority |
| SPECTRAL witness | May block reveal | Cannot change | No authority |
| Statistical-mechanics guardrail | Interpretation constraint only | Cannot change | No authority |
| SONIFICATION telemetry | No authority | Cannot change | No authority |

The design deliberately prevents authority escalation between layers.

## 10. Assurance ladder

```text
INPUTS_COMMITTED
      |
      v
INTEGRITY_VERIFIED
      |
      v
REPLAY_VERIFIED
      |
      v
ENSEMBLE_CALIBRATED      (future; separately earned)
      |
      v
SCIENTIFICALLY_DEFENSIBLE (external scientific judgement)
```

No lower rung implies a higher rung.

## 11. Review-driven hardening and regression coverage

The v5.1.0 defense layer received an additional Copilot review pass that identified four substantive boundary defects. All four were corrected and given dedicated regression tests:

1. witness commitment is now bound to the actual replayed recipe contract;
2. malformed-manifest telemetry now emits rejection events rather than raising prematurely;
3. integrity-only mode cannot replay even when replay inputs are supplied; and
4. gate receipts require exact deterministic schema/value equality.

The resulting test matrix passes on Python 3.10, 3.11, 3.12, and 3.13. Existing tests also cover noncanonical child JSON, hidden-file smuggling, external root mismatch, pre-observation input substitution, deterministic commitments, deterministic event streams, and telemetry/evidence-bundle isolation.

## 12. Reproducibility and preservation requirements

For an archival software release, preservation should distinguish the software source from supporting scientific documentation.

Recommended Zenodo v5.1.0 record contents are:

- the GitHub release source archive for tag `v5.1.0`;
- this formal technical report in PDF;
- the report's Markdown source;
- `RELEASE_NOTES_v5.1.0.md`;
- `CITATION.cff`;
- `.zenodo.json` or the equivalent exported metadata snapshot;
- the Apache-2.0 license;
- a SHA-256 checksum manifest for the uploaded supporting files; and
- an upload/readme manifest explaining the relationship between the software archive, report, and prior v5.0.0 DOI.

The new Zenodo version should be created through Zenodo's versioning workflow so the prior v5.0.0 record remains immutable and citable while the record series points to the newer software state.

## 13. Limitations and future work

The defense layer is intentionally narrower than full experimental validation. Remaining work includes publication-grade null-ensemble calibration, protected holdout execution, independent-custodian workflows, cross-catalogue dependence modelling, and the planned breaking `uff.sheridan-crucible.v2` contract family.

Future work may implement a machine-verifiable `ENSEMBLE_CALIBRATED` assurance state, but only if calibration requirements are made explicit and regression-testable. That state must never be inferred from replay alone.

## 14. Conclusion

UFF v5.1.0 strengthens the evidence pipeline without enlarging the scientific claim. QEC guards computational admission, SPECTRAL binds pre-observation identities to the replayed experiment, statistical mechanics prevents exact replay from masquerading as ensemble validation, and SONIFICATION exposes a read-only diagnostic event bus.

The governing principle is:

> QEC guards the door. SPECTRAL proves what was handed to the door. Statistical mechanics stops us mistaking one perfectly replayed run for a validated ensemble. SONIFICATION tells us what the door is doing. None of them gets to impersonate nature.

## References

1. Slade, T. (2026). QSOL UFF v5.0.0: Reproducible Astrophysics and Falsification Laboratory. Zenodo. DOI: 10.5281/zenodo.21830630.
2. QSOLKCB/UFF, `docs/QEC_BOUNDARY_GATE.md`, v5.1.0 release state.
3. QSOLKCB/UFF, `docs/UFF_DEFENSE_IN_DEPTH.md`, v5.1.0 release state.
4. QSOLKCB/UFF, `docs/STATISTICAL_MECHANICS_GUARDRAIL.md`, v5.1.0 release state.
5. Deutsch, J. M. (1991). Quantum statistical mechanics in a closed system. Physical Review A, 43, 2046-2049. DOI: 10.1103/PhysRevA.43.2046.
6. Basti, G., & Cenatiempo, S. An invitation to Quantum Statistical Mechanics. Lecture notes, updated 12 December 2023.
7. Schwartz, M. D. Lecture 10: Quantum Statistical Mechanics. Statistical Mechanics, Harvard University, Spring 2021.
