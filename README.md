# QSOL UFF

[![CI](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml/badge.svg)](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v5.1.0-4c1.svg)](RELEASE_NOTES_v5.1.0.md)
[![Zenodo v5.0.0 archive](https://img.shields.io/badge/Zenodo-v5.0.0%20archive-1682D4.svg)](https://doi.org/10.5281/zenodo.21830630)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**QSOL UFF v5.1.0 - Reproducible Astrophysics, Falsification, and Defense-in-Depth Assurance Laboratory**

QSOL UFF is a transparent Python research laboratory for two kinds of work that
should never be confused:

1. fitting and comparing explicit astrophysical models; and
2. testing extraordinary catalogue-level spatial claims under frozen,
   replayable, survey-aware, and now fail-closed assurance rules.

UFF v5.1.0 retains the galaxy-dynamics and compact-object tools, UFF-SLFA,
Sheridan Crucible, claim provenance, and independent assessment programme from
v5.0.0. It adds a deliberately small defense-in-depth layer around the evidence
workflow: a QEC-inspired computational gate, a SPECTRAL-inspired pre-observation
witness, a statistical-mechanics ensemble guardrail, and SONIFICATION-inspired
receiver-neutral audit telemetry.

> **Scientific boundary:** UFF can formalise a claim, expose circular selection,
> model survey geometry, freeze input identities, verify bundle integrity,
> reproduce a deterministic result, and expose trust-boundary telemetry. It
> cannot turn catalogue diagnostics into physical objects, prove analyst
> blindness from a local hash, guarantee that a chosen null ensemble represents
> nature, or promote replay into physical truth.

The v5.1.0 assurance rule is explicit:

```text
REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH
```

## What UFF contains

| Layer | Purpose | Primary interface | Status |
|---|---|---|---|
| Galaxy and compact-object laboratory | Fit rotation curves, compare baryonic/halo/MOND/UFF model families, and report separate Kerr/LQG scales | `python -m uff` | Stable v4 core retained |
| UFF-SLFA | Test a frozen anomaly-rate claim inside fixed celestial node caps | `python sky_lattice_audit.py` | Preregistration-ready reference implementation |
| Sheridan Crucible | Add masks, completeness, spherical density reconstruction, nuisance models, survey-matched rotations, and injection calibration | `python -m uff.sheridan` | Exact survey-aware reference implementation |
| Provenance and assessment | Preserve incompatible public claim versions, source hashes, blockers, and methodological review | JSON ledgers and Markdown records | Governance / audit layer |
| Defense-in-depth assurance | Freeze identities, fail closed on bundle/replay defects, separate replay from ensemble calibration, and export read-only telemetry | `python -m uff.spectral_witness`, `python -m uff.qec_gate`, `python -m uff.audit_events` | New in v5.1.0 |

The default `uff` CLI remains focused on galaxy and compact-object analysis.
Sky-audit and assurance interfaces remain separate so model fitting, evidence
admission, and scientific interpretation cannot silently borrow authority from
one another.

## Why v5.1 exists

UFF v5.0.0 made disputed catalogue-level celestial-node claims testable under
frozen and survey-aware contracts. v5.1.0 addresses the next question:

> Even when the scientific method is frozen, how do we make the evidence
> boundary itself fail closed and make its assurance level impossible to
> misread?

The resulting stack is:

```text
BEFORE OBSERVATION
    SPECTRAL witness
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
 SONIFICATION audit telemetry
 receiver-neutral events; external receivers optional
```

A claim that is incomplete remains `CONTRACT_NOT_EXECUTABLE`. A bundle that is
intact but not replayed remains `INTEGRITY_ONLY` and is **not admitted**. A
successful replay remains computational assurance, not proof that the null
ensemble or physical interpretation is correct.

## Installation

```bash
git clone https://github.com/QSOLKCB/UFF.git
cd UFF
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## 1. Galaxy dynamics and compact objects

### Fit and compare rotation-curve models

```bash
python -m uff fit \
  --csv DEMO_GALAXY.csv \
  --gal DEMO_GALAXY \
  --models baryons,nfw,burkert,mond-rar,uff-empirical \
  --restarts 24 \
  --out outputs \
  --sonify
```

Canonical CSV columns are:

```text
R_kpc,V_obs_kms,e_V_kms,V_gas_kms,V_disk_kms,V_bul_kms
```

Short SPARC aliases are also accepted. See [Data format](docs/DATA_FORMAT.md) and
[Model definitions](docs/MODELS.md).

Included model families are Newtonian baryons, NFW and Burkert halos, MOND/RAR
variants including an explicitly approximate EFE sensitivity proxy, a
repository-specific empirical UFF curve family, and an optional weak-field
central SMBH term.

The fit pipeline reports likelihood diagnostics, chi-squared, RMSE, AIC/AICc,
BIC, relative information-criterion weights, bound hits, full residual arrays,
and SHA-256 input receipts. Optional posterior sampling, plots, and
deterministic sonification remain available.

### Compact-object scale report

```bash
python -m uff compact-object \
  --mass-msun 4300000 \
  --spin 0.5 \
  --velocity-dispersion-kms 100 \
  --out outputs/sgr-a-scale-report.json
```

This reports Kerr characteristic radii, the sphere of influence, the selected
LQG area-gap convention, and scale ratios. LQG is not used in the galaxy
likelihood. See [Scientific status](docs/SCIENCE_STATUS_2026.md).

## 2. UFF-SLFA: frozen sky-lattice claims

UFF-SLFA asks a narrow question:

> Does an independently selected catalogue show a preregistered excess of a
> declared anomaly inside spherical caps around frozen celestial nodes, under a
> null model that preserves the relevant selection structure?

A `uff.sky-lattice-claim.v1` contract freezes node IDs and ICRS coordinates,
one cap radius, the anomaly predicate, catalogue/holdout/weight/stratum
declarations, the null model and deterministic seed, decision thresholds, and
anti-circularity declarations.

Run and verify an audit:

```bash
python sky_lattice_audit.py run \
  --catalogue frozen_catalogue.csv \
  --contract examples/sky_lattice_contract.example.json \
  --out runs/frozen-claim

python sky_lattice_audit.py verify \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv
```

SLFA supports shared RA shifts, Haar-uniform proper SO(3) rotations, and
stratified label permutations. It uses finite plus-one empirical p-values, Holm
family-wise correction, and TFT-derived invariance checks for geometric null
transforms.

Read the full [Sky-Lattice Falsification Protocol](docs/SKY_LATTICE_FALSIFICATION_PROTOCOL.md).

## 3. Sheridan Crucible: survey-aware falsification

Sheridan wraps an ordinary SLFA claim in `uff.sheridan-crucible.v1` and makes
telescope/catalogue geometry part of the frozen experiment.

It adds explicit survey-support quadrature, masks and fractional coverage,
completeness filtering and inverse-completeness weights, normalized spherical
von Mises-Fisher KDE, leave-one-out/adaptive bandwidths, mask-aware edge
renormalization, survey-availability-matched SO(3) rotations, nuisance-model
comparison, predictive checks, synthetic anomaly-label injection, bounded exact
source execution, and replayable SHA-256 evidence bundles.

Generate a deterministic full-sky support grid:

```bash
python -m uff.sheridan support-grid \
  --points 4096 \
  --out full_sky_support.csv
```

Run and replay a frozen Sheridan contract:

```bash
python -m uff.sheridan run \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv \
  --contract examples/sheridan_contract.example.json \
  --out runs/sheridan-example

python -m uff.sheridan verify \
  runs/sheridan-example/manifest.json \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv
```

Read the full [Sheridan Siege Engine protocol](docs/SHERIDAN_SIEGE_ENGINE.md).

## 4. Defense in depth

Read [UFF Defense in Depth](docs/UFF_DEFENSE_IN_DEPTH.md) for the complete
authority model and the versioned [v5.1.0 technical report source](papers/UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md). The rendered PDF is included in the Zenodo upload bundle.

### QEC boundary gate - computational admission

The QEC-inspired gate performs strict canonical JSON and artifact validation,
recomputes child hashes and contract cross-links, constructs an externally
anchorable deterministic root, and then requires fresh domain replay before
admission.

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv
```

Integrity inspection is deliberately weaker and never admits:

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --integrity-only
```

Assurance states:

| State | Meaning | Admitted? |
|---|---|---:|
| `INTEGRITY_ONLY` | Strict structure and hashes passed; no fresh replay | No |
| `REPLAY_VERIFIED` | Strict checks passed and frozen result replayed | Yes |
| `REJECTED` | Structural, semantic, hash, receipt, anchor, or replay failure | No |

See [QEC Boundary Gate](docs/QEC_BOUNDARY_GATE.md).

### SPECTRAL witness - pre-observation identity

Commit identity-bearing inputs before running or inspecting the audit:

```bash
python -m uff.spectral_witness commit precommit.json \
  --contract frozen_contract.json \
  --catalogue frozen_catalogue.csv
```

For Sheridan add `--support frozen_support.csv`.

Reveal through a replay-verified evidence bundle:

```bash
python -m uff.spectral_witness reveal \
  precommit.json runs/frozen-claim/manifest.json \
  --contract frozen_contract.json \
  --catalogue frozen_catalogue.csv \
  --expected-commit <externally-anchored-digest>
```

v5.1.0 binds the witness's canonical contract digest to the contract actually
verified from the replayed recipe. A witness for contract A cannot admit a
replay-valid bundle produced from contract B.

A local commitment proves identity, not chronology. Historical preregistration
still requires an independent timestamped or signed anchor.

### Statistical-mechanics guardrail - replay is not calibration

The interpretation guardrail formalizes:

```text
INPUTS_COMMITTED
      -> INTEGRITY_VERIFIED
      -> REPLAY_VERIFIED
      -> ENSEMBLE_CALIBRATED       (future; separately earned)
      -> SCIENTIFICALLY_DEFENSIBLE (external scientific judgement)
```

No lower rung implies a higher rung. A future `ENSEMBLE_CALIBRATED` state must
be earned with explicit type-I-error, power, negative-control, survey-systematic,
convergence, seed-block, and multiplicity calibration. No quantum many-body
model is imported into ordinary catalogue resampling by this analogy.

See [Statistical Mechanics Guardrail](docs/STATISTICAL_MECHANICS_GUARDRAIL.md).

### SONIFICATION audit telemetry - read-only receiver bus

Generate deterministic receiver-neutral telemetry outside the evidence bundle:

```bash
python -m uff.audit_events \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv \
  --out telemetry/frozen-claim-events.json
```

Canonical event fields describe trust-boundary and already-recorded scientific
states. Tempo, hertz, MIDI, timbre, loudness, waveform, and rendered audio are
noncanonical receiver choices. Telemetry has **zero authority** over bundle
admission or scientific verdicts.

## Evidence bundles and verdict boundaries

SLFA, Sheridan, and the v5.1.0 gate separate five questions that are often
blurred together:

1. **Was the claim fully specified?**
2. **Are the artifacts intact?**
3. **Does numerical replay reproduce the stored result?**
4. **Is the statistical ensemble calibrated for the inferential claim?**
5. **Is the scientific model and sampling design defensible?**

A bundle may be computationally perfect and scientifically biased. Hashes prove
byte identity; deterministic replay proves computational consistency; neither
proves that the sampling frame, anomaly predicate, null distribution, or causal
interpretation is appropriate.

Sheridan bundles contain:

```text
recipe.json
density.json
nodes.csv
models.json
injection.json
decision.json
manifest.json
```

The QEC gate may additionally write `qec_gate.json` after successful replay;
the receipt is self-hash-excluded and validated against the full deterministic
receipt payload.

Failed and untestable nodes remain visible. Null outcomes are not deleted. A
positive association remains an association, not automatic evidence for its
proposed cause.

## Claim provenance and independent assessment

The repository includes a content-addressed public-claim ledger for the
Logvinovich celestial-node claims. It preserves incompatible coordinate sets,
radii, query predicates, reported counts, and unresolved fields without
choosing a preferred version on the claimant's behalf.

Key records include:

- [Public Claim Ledger](docs/PUBLIC_CLAIM_LEDGER_2026-08-07.md)
- [Public Claim Source Manifest](docs/PUBLIC_CLAIM_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable public claim profile](examples/public_claim_profile_2026-08-07.json)
- [Independent assessment response](docs/INDEPENDENT_ASSESSMENT_RESPONSE_2026-08-07.md)
- [Independent assessment source manifest](docs/INDEPENDENT_ASSESSMENT_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable assessment action ledger](examples/independent_assessment_actions_2026-08-07.json)

The governing assessment remains intentionally uncomfortable:

> The crucible's syntax is largely formalised. Its statistical calibration
> still needs validation, and the claimant has not supplied one stationary
> claim to place inside it.

The proposed `uff.sheridan-crucible.v2` expansion remains a roadmap, not an
implemented contract in v5.1.0.

## Machine-readable schemas and protocols

| Schema / protocol | Role | Executable? |
|---|---|---:|
| `uff.rotation-curve-summary.v4` | Galaxy fit and comparison result | Output schema |
| `uff.sky-lattice-claim.v1` | Frozen catalogue-level celestial-node claim | Yes, when complete |
| `uff.sheridan-crucible.v1` | Survey-aware wrapper around a frozen SLFA claim | Yes, when complete |
| `uff.qec-bundle-root.v1` | Deterministic evidence-root payload | Yes, verifier-internal |
| `uff.qec-boundary-gate.v1` | Replay-verified gate receipt | Yes, verifier-generated |
| `uff.spectral-witness.v1` | Pre-observation input-identity commitment | Yes |
| `uff.audit-event-stream.v1` | Receiver-neutral read-only audit telemetry | Yes, non-authoritative |
| `uff.public-claim-profile.v1` | Provenance record containing unresolved public claim versions | No by design |
| `uff.independent-assessment-response.v1` | Machine-readable implementation roadmap | No; governance record |
| `uff.sheridan-crucible.v2` | Proposed publication-grade contract expansion | Planned, not implemented |
| `ENSEMBLE_CALIBRATED` | Future assurance state | Planned, not implemented |

## Repository layout

```text
uff/
  cli.py                    # galaxy and compact-object CLI
  models.py                 # baryons, halos, MOND/RAR and UFF empirical law
  fitting.py                # deterministic model fitting and comparison
  sampling.py               # optional posterior sampler
  compact.py                # Kerr/SMBH and LQG scale diagnostics
  sky_contract.py           # SLFA contract validation
  sky_geometry.py           # spherical geometry and SO(3) invariants
  sky_statistics.py         # audit statistics and null models
  sky_artifacts.py          # SLFA bundles, integrity and replay
  sky_audit.py              # SLFA public API and CLI
  sheridan_contract.py      # survey-aware contract validation
  sheridan_density.py       # vMF KDE, masks and edge correction
  sheridan_models.py        # nuisance comparison and injection recovery
  sheridan_artifacts.py     # Sheridan bundles and replay
  sheridan.py               # Sheridan public API and CLI
  qec_gate.py               # strict defense-in-depth admission boundary
  spectral_witness.py       # pre-observation input identity commit/reveal
  audit_events.py           # receiver-neutral read-only forensic telemetry
examples/                   # frozen example contracts and governance ledgers
tests/                      # model, geometry, replay, provenance and assurance regressions
docs/                       # protocols, scientific boundaries and manifests
papers/                     # methods papers, formal reports and references
zenodo/                     # release-specific archival upload guidance
```

## Research status and limitations

- Rotation curves alone do not settle dark matter versus modified gravity.
- Algebraic MOND relations are not full AQUAL/QUMOND solvers for flattened disks.
- Information-criterion rankings depend on the candidate set and data contract.
- The UFF empirical profile is not derived from a covariant field theory.
- Catalogue diagnostics such as excess noise, uncertainty, or missing values are
  not physical objects without an independently validated object-level model.
- Broadband colour differences are not spectral resonances without a
  bandpass-aware spectral model.
- Cross-catalogue agreement is not automatic statistical independence when
  catalogues share objects, source-density structure, or systematics.
- Preregistration prevents later rule changes; it does not repair a biased
  sampling frame or make previously inspected data blind.
- `REPLAY_VERIFIED` is computational assurance, not ensemble calibration.
- A local SPECTRAL witness establishes identity, not historical chronology.
- SONIFICATION telemetry is an observation aid, not additional evidence.
- The current exact Sheridan KDE is intentionally bounded by a frozen source
  limit rather than silently changing algorithm or exhausting memory.

## Release notes and technical report

- [QSOL UFF v5.1.0](RELEASE_NOTES_v5.1.0.md)
- [v5.1.0 Defense-in-Depth Technical Report source](papers/UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md) - the rendered PDF is included in the Zenodo bundle
- [Zenodo v5.1.0 upload guide](zenodo/v5.1.0/ZENODO_UPLOAD_README.md)
- [QSOL UFF v5.0.0](RELEASE_NOTES_v5.0.0.md)
- [UFF Sheridan Crucible v1.1.0](RELEASE_NOTES_SHERIDAN_v1.1.0.md)
- [UFF-SLFA v1.0.0](RELEASE_NOTES_SLFA_v1.0.0.md)

## Citation and Zenodo versioning

The currently published archive is v5.0.0:

> Slade, T. (2026). *QSOL UFF v5.0.0: Reproducible Astrophysics and
> Falsification Laboratory* (Version 5.0.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21830630

v5.1.0 is prepared as a **new Zenodo version** of that record. Its version DOI
will be assigned when the v5.1.0 deposit is published. Until then, the DOI above
must be treated as the immutable v5.0.0 archive, not as the v5.1.0 DOI.

Machine-readable release metadata are in [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json). The supporting Zenodo package includes a
post-publication checklist identifying every place where the newly assigned
v5.1.0 DOI should be patched after publication.

Analyses must also cite the primary scientific sources for every physical model,
catalogue, and statistical method used.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Maintainer: **Trent Slade / QSOL-IMC**  
GitHub: [QSOLKCB](https://github.com/QSOLKCB)
