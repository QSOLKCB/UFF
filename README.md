# QSOL UFF

[![CI](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml/badge.svg)](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml)
[![Lean 4](https://github.com/QSOLKCB/UFF/actions/workflows/lean-formal.yml/badge.svg)](https://github.com/QSOLKCB/UFF/actions/workflows/lean-formal.yml)
[![Release](https://img.shields.io/badge/release-v5.2.0-4c1.svg)](RELEASE_NOTES_v5.2.0.md)
[![Zenodo v5.0.0 archive](https://img.shields.io/badge/Zenodo-v5.0.0%20archive-1682D4.svg)](https://doi.org/10.5281/zenodo.21830630)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**QSOL UFF v5.2.0 — Reproducible Astrophysics, Falsification, Defense-in-Depth Assurance, and Lean 4 Formal Claim Boundaries**

QSOL UFF is a transparent research laboratory for two kinds of work that should never be confused:

1. fitting and comparing explicit astrophysical models; and
2. testing extraordinary catalogue-level spatial claims under frozen, replayable, survey-aware, fail-closed evidence rules.

v5.2.0 retains the galaxy-dynamics and compact-object laboratory, UFF-SLFA, Sheridan Crucible, claim provenance, independent assessment, SPECTRAL-style input witnessing, QEC-style replay admission, ensemble guardrails, and receiver-neutral audit telemetry from v5.1.0. It adds a deliberately small **Lean 4 formal assurance layer** for selected software and epistemic invariants.

> **Scientific boundary:** UFF can formalise a claim, expose circular selection, model survey geometry, freeze input identities, verify bundle integrity, reproduce a deterministic result, machine-check selected assurance invariants, and expose trust-boundary telemetry. It cannot turn catalogue diagnostics into physical objects, prove analyst blindness from a local hash, guarantee that a chosen null ensemble represents nature, prove SHA-256 collision resistance, or promote formal/replay consistency into physical truth.

The governing rule remains:

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
| Defense-in-depth assurance | Freeze identities, fail closed on bundle/replay defects, separate replay from ensemble calibration, and export read-only telemetry | `python -m uff.spectral_witness`, `python -m uff.qec_gate`, `python -m uff.audit_events` | v5.1.0 |
| Lean 4 formal assurance | Machine-check selected assurance, identity, transaction, observation, manifest, and replay invariants | `cd formal/lean && lake build && bash audit.sh` | New in v5.2.0 |

The default `uff` CLI remains focused on galaxy and compact-object analysis. Sky-audit, defense, telemetry, and formal-verification interfaces remain separate so model fitting, evidence admission, theorem checking, and scientific interpretation cannot silently borrow authority from one another.

## Why v5.2 exists

v5.0.0 made disputed catalogue-level claims testable under frozen and survey-aware contracts. v5.1.0 made the evidence boundary itself fail closed and separated replay from calibration. v5.2.0 addresses the next question:

> Which of those assurance and provenance boundaries can be made machine-checkable without pretending that a theorem prover proves nature?

The resulting trust stack is:

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
              |
              v
      Lean 4 assurance model
 machine-check selected invariants and nonclaims
              |
              v
       external science
 calibration, peer review, replication, judgement
```

A claim that is incomplete remains `CONTRACT_NOT_EXECUTABLE`. A bundle that is intact but not replayed remains `INTEGRITY_ONLY` and is **not admitted**. A successful replay remains computational assurance. A Lean theorem remains a statement about the formal specification. Neither is automatically an empirical or physical verdict.

## Installation

```bash
git clone https://github.com/QSOLKCB/UFF.git
cd UFF
python3 -m venv .venv
source .venv/bin/activate
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

Short SPARC aliases are also accepted. See [Data format](docs/DATA_FORMAT.md) and [Model definitions](docs/MODELS.md).

Included model families are Newtonian baryons, NFW and Burkert halos, MOND/RAR variants including an explicitly approximate EFE sensitivity proxy, a repository-specific empirical UFF curve family, and an optional weak-field central SMBH term.

The fit pipeline reports likelihood diagnostics, chi-squared, RMSE, AIC/AICc, BIC, relative information-criterion weights, bound hits, full residual arrays, and SHA-256 input receipts. Optional posterior sampling, plots, and deterministic sonification remain available.

### Compact-object scale report

```bash
python -m uff compact-object \
  --mass-msun 4300000 \
  --spin 0.5 \
  --velocity-dispersion-kms 100 \
  --out outputs/sgr-a-scale-report.json
```

This reports Kerr characteristic radii, the sphere of influence, the selected LQG area-gap convention, and scale ratios. LQG is not used in the galaxy likelihood. See [Scientific status](docs/SCIENCE_STATUS_2026.md).

## 2. UFF-SLFA: frozen sky-lattice claims

UFF-SLFA asks a narrow question:

> Does an independently selected catalogue show a preregistered excess of a declared anomaly inside spherical caps around frozen celestial nodes, under a null model that preserves the relevant selection structure?

A `uff.sky-lattice-claim.v1` contract freezes node IDs and ICRS coordinates, one cap radius, the anomaly predicate, catalogue/holdout/weight/stratum declarations, the null model and deterministic seed, decision thresholds, and anti-circularity declarations.

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

SLFA supports shared RA shifts, Haar-uniform proper SO(3) rotations, and stratified label permutations. It uses finite plus-one empirical p-values, Holm family-wise correction, and TFT-derived invariance checks for geometric null transforms.

Read the full [Sky-Lattice Falsification Protocol](docs/SKY_LATTICE_FALSIFICATION_PROTOCOL.md).

## 3. Sheridan Crucible: survey-aware falsification

Sheridan wraps an ordinary SLFA claim in `uff.sheridan-crucible.v1` and makes telescope/catalogue geometry part of the frozen experiment.

It adds explicit survey-support quadrature, masks and fractional coverage, completeness filtering and inverse-completeness weights, normalized spherical von Mises-Fisher KDE, leave-one-out/adaptive bandwidths, mask-aware edge renormalization, survey-availability-matched SO(3) rotations, nuisance-model comparison, predictive checks, synthetic anomaly-label injection, bounded exact source execution, and replayable SHA-256 evidence bundles.

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

Read [UFF Defense in Depth](docs/UFF_DEFENSE_IN_DEPTH.md) and the [v5.1.0 technical report source](papers/UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md).

### QEC boundary gate — computational admission

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv
```

Integrity-only inspection is deliberately weaker and never admits:

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --integrity-only
```

| State | Meaning | Admitted? |
|---|---|---:|
| `INTEGRITY_ONLY` | Strict structure and hashes passed; no fresh replay | No |
| `REPLAY_VERIFIED` | Strict checks passed and frozen result replayed | Yes |
| `REJECTED` | Structural, semantic, hash, receipt, anchor, or replay failure | No |

See [QEC Boundary Gate](docs/QEC_BOUNDARY_GATE.md).

### SPECTRAL witness — pre-observation identity

```bash
python -m uff.spectral_witness commit precommit.json \
  --contract frozen_contract.json \
  --catalogue frozen_catalogue.csv
```

For Sheridan add `--support frozen_support.csv`. Reveal only through a replay-verified bundle. A local commitment establishes identity, not chronology; historical preregistration still requires an independent timestamped or signed anchor.

### Statistical-mechanics guardrail — replay is not calibration

```text
INPUTS_COMMITTED
      -> INTEGRITY_VERIFIED
      -> REPLAY_VERIFIED
      -> ENSEMBLE_CALIBRATED       (future; separately earned)
      -> SCIENTIFICALLY_DEFENSIBLE (external scientific judgement)
```

No lower rung implies a higher rung. A future `ENSEMBLE_CALIBRATED` state must be earned with explicit type-I-error, power, negative-control, survey-systematic, convergence, seed-block, and multiplicity calibration.

See [Statistical Mechanics Guardrail](docs/STATISTICAL_MECHANICS_GUARDRAIL.md).

### SONIFICATION audit telemetry — read-only receiver bus

```bash
python -m uff.audit_events \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv \
  --out telemetry/frozen-claim-events.json
```

Telemetry has **zero authority** over bundle admission or scientific verdicts. Tempo, hertz, MIDI, timbre, loudness, waveform, and rendered audio are noncanonical receiver choices.

## 5. Lean 4 formal assurance

The v5.2.0 formal layer lives in [`formal/lean/`](formal/lean/) and is intentionally compact. It models selected UFF invariants instead of attempting to reproduce the numerical astrophysics in a theorem prover.

Build and audit it with:

```bash
cd formal/lean
lake build
bash audit.sh
```

The advertised theorem surface includes machine-checked statements that:

- replay promotion preserves the modeled external scientific judgement;
- a replay-verified state does not itself satisfy ensemble calibration;
- an ensemble-calibrated state does not itself satisfy scientific defensibility;
- changing a claim boundary or frozen recipe changes formal experiment identity;
- a cancelled transaction produces no bundle and is not exportable;
- sonification preserves the underlying numerical result;
- telemetry leaves evidence state unchanged;
- manifest sealing computes the envelope digest from a pre-existing core;
- same deterministic engine + frozen recipe + runtime contract gives the same formal replay result; and
- a changed runtime contract changes replay identity.

The formal layer also makes its **nonclaims** machine-visible: it does not claim to prove SHA-256 collision resistance, catalogue correctness, null-model adequacy, or physical truth.

CI rejects `sorry`/`admit`, rejects project-defined `axiom`/`constant` declarations, requires [`AUDIT_MANIFEST.tsv`](formal/lean/AUDIT_MANIFEST.tsv) to match the theorem declarations exactly, and audits each advertised theorem with `#print axioms`.

Read:

- [v5.2.0 Formal Assurance Report](docs/UFF_FORMAL_ASSURANCE_V5.2.md)
- [Formal README](formal/lean/README.md)
- [Assumptions and nonclaims](formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md)
- [Runtime correspondence](formal/lean/RUNTIME_CORRESPONDENCE.md)
- [Advertised theorem surface](formal/lean/THEOREMS.md)

## Evidence bundles and verdict boundaries

SLFA, Sheridan, the v5.1.0 gate, and the v5.2.0 formal layer separate questions that are often blurred together:

1. **Was the claim fully specified?**
2. **Are the artifacts intact?**
3. **Does numerical replay reproduce the stored result?**
4. **Do selected formal assurance definitions satisfy their stated invariants?**
5. **Is the statistical ensemble calibrated for the inferential claim?**
6. **Is the scientific model and sampling design defensible?**

A bundle may be computationally perfect and scientifically biased. A formal model may be internally proved and fail to correspond to a buggy implementation. Hashes prove byte identity; deterministic replay proves computational consistency; Lean proves theorems about its definitions; none of those facts alone proves that the sampling frame, anomaly predicate, null distribution, or causal interpretation is scientifically appropriate.

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

The QEC gate may additionally write `qec_gate.json` after successful replay. Failed and untestable nodes remain visible. Null outcomes are not deleted. A positive association remains an association, not automatic evidence for its proposed cause.

## Claim provenance and independent assessment

The repository includes a content-addressed public-claim ledger for the Logvinovich celestial-node claims. It preserves incompatible coordinate sets, radii, query predicates, reported counts, and unresolved fields without choosing a preferred version on the claimant's behalf.

Key records include:

- [Public Claim Ledger](docs/PUBLIC_CLAIM_LEDGER_2026-08-07.md)
- [Public Claim Source Manifest](docs/PUBLIC_CLAIM_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable public claim profile](examples/public_claim_profile_2026-08-07.json)
- [Independent assessment response](docs/INDEPENDENT_ASSESSMENT_RESPONSE_2026-08-07.md)
- [Independent assessment source manifest](docs/INDEPENDENT_ASSESSMENT_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable assessment action ledger](examples/independent_assessment_actions_2026-08-07.json)

The proposed `uff.sheridan-crucible.v2` expansion remains a roadmap, not an implemented contract in v5.2.0.

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
| UFF Lean 4 v5.2 model | Formal assurance specification | Yes, theorem-checked; non-runtime |
| `uff.public-claim-profile.v1` | Provenance record containing unresolved public claim versions | No by design |
| `uff.independent-assessment-response.v1` | Machine-readable implementation roadmap | No; governance record |
| `uff.sheridan-crucible.v2` | Proposed publication-grade contract expansion | Planned, not implemented |
| `ENSEMBLE_CALIBRATED` | Future runtime assurance state | Planned, not implemented |

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
formal/lean/                # v5.2.0 Lean 4 assurance specification and audit
examples/                   # frozen example contracts and governance ledgers
tests/                      # model, geometry, replay, provenance and assurance regressions
docs/                       # protocols, formal/scientific boundaries and manifests
papers/                     # methods papers, formal reports and references
zenodo/                     # release-specific archival upload guidance
```

## Research status and limitations

- Rotation curves alone do not settle dark matter versus modified gravity.
- Algebraic MOND relations are not full AQUAL/QUMOND solvers for flattened disks.
- Information-criterion rankings depend on the candidate set and data contract.
- The UFF empirical profile is not derived from a covariant field theory.
- Catalogue diagnostics are not physical objects without an independently validated object-level model.
- Cross-catalogue agreement is not automatic statistical independence when catalogues share objects, source-density structure, or systematics.
- Preregistration prevents later rule changes; it does not repair a biased sampling frame or make previously inspected data blind.
- `REPLAY_VERIFIED` is computational assurance, not ensemble calibration.
- A local SPECTRAL witness establishes identity, not historical chronology.
- SONIFICATION telemetry is an observation aid, not additional evidence.
- Lean theorems prove properties of the formal model, not automatic semantic equivalence with Python or truth about nature.

## Release notes and archival package

- [QSOL UFF v5.2.0](RELEASE_NOTES_v5.2.0.md)
- [v5.2.0 Formal Assurance Report](docs/UFF_FORMAL_ASSURANCE_V5.2.md)
- [Zenodo v5.2.0 upload guide](zenodo/v5.2.0/ZENODO_UPLOAD_README.md)
- [QSOL UFF v5.1.0](RELEASE_NOTES_v5.1.0.md)
- [v5.1.0 Defense-in-Depth Technical Report](papers/UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md)
- [QSOL UFF v5.0.0](RELEASE_NOTES_v5.0.0.md)
- [UFF Sheridan Crucible v1.1.0](RELEASE_NOTES_SHERIDAN_v1.1.0.md)
- [UFF-SLFA v1.0.0](RELEASE_NOTES_SLFA_v1.0.0.md)

## Citation and Zenodo versioning

The immutable published v5.0.0 archive remains:

> Slade, T. (2026). *QSOL UFF v5.0.0: Reproducible Astrophysics and Falsification Laboratory* (Version 5.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21830630

v5.2.0 is prepared for Zenodo's **New version** workflow. Its version DOI must be assigned by Zenodo only after the exact merged v5.2.0 release tree is frozen and tagged. Do not relabel an earlier version DOI as v5.2.0.

Machine-readable release metadata are in [CITATION.cff](CITATION.cff), [.zenodo.json](.zenodo.json), and [`zenodo/v5.2.0/metadata.json`](zenodo/v5.2.0/metadata.json).

Analyses must also cite the primary scientific sources for every physical model, catalogue, and statistical method used.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Maintainer: **Trent Slade / QSOL-IMC**  
GitHub: [QSOLKCB](https://github.com/QSOLKCB)
