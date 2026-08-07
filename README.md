# QSOL UFF

[![CI](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml/badge.svg)](https://github.com/QSOLKCB/UFF/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v5.0.0-4c1.svg)](RELEASE_NOTES_v5.0.0.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**QSOL UFF v5.0.0 — Reproducible Astrophysics and Falsification Laboratory**

QSOL UFF is a transparent Python research laboratory for two kinds of work that
should never be confused:

1. fitting and comparing explicit astrophysical models; and
2. testing extraordinary catalogue-level spatial claims under frozen,
   replayable and survey-aware rules.

The repository retains the validated galaxy-dynamics and compact-object tools
from v4, then adds the UFF Sky-Lattice Falsification Audit (SLFA), the Sheridan
Crucible survey engine, content-addressed public-claim provenance, and an
independent methodological assessment programme.

> **Scientific boundary:** UFF can formalise a claim, expose circular selection,
> model survey geometry, calibrate a detection procedure and replay an exact
> decision. It cannot turn catalogue diagnostics into physical objects, prove a
> vacuum ontology, make an inspected dataset blind again, or guarantee that a
> chosen null model represents nature.

## What UFF contains

| Layer | Purpose | Primary interface | Status |
|---|---|---|---|
| Galaxy and compact-object laboratory | Fit rotation curves, compare baryonic/halo/MOND/UFF model families, and report separate Kerr/LQG scales | `python -m uff` | Stable v4 core retained in v5 |
| UFF-SLFA | Test a frozen anomaly-rate claim inside fixed celestial node caps | `python sky_lattice_audit.py` | Preregistration-ready reference implementation |
| Sheridan Crucible | Add masks, completeness, spherical density reconstruction, nuisance models, survey-matched rotations and injection calibration | `python -m uff.sheridan` | Exact survey-aware reference implementation |
| Provenance and assessment layer | Preserve incompatible public claim versions, source hashes, blockers and external methodological review | JSON ledgers and Markdown records | Audit governance; deliberately non-executable where fields remain unresolved |

The default `uff` CLI remains focused on galaxy and compact-object analysis.
The sky-audit interfaces are separate so their contracts, evidence bundles and
scientific verdicts cannot be accidentally mixed with model-fitting outputs.

## Why v5 exists

UFF began as a galaxy rotation-curve and compact-object laboratory. During v5,
the project gained a second, rigorously bounded role: converting disputed
celestial-node claims into tests that cannot silently change coordinates,
radii, anomaly definitions, null models or success criteria after results are
seen.

The resulting stack is:

```text
public claim record
        ↓
frozen UFF-SLFA contract
        ↓
survey-aware Sheridan contract
        ↓
integrity + numerical replay bundle
        ↓
separate empirical, diagnostic and scientific interpretation boundaries
```

A claim that is incomplete remains `CONTRACT_NOT_EXECUTABLE`. Failure to specify
an experiment is not treated as empirical falsification.

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

Short SPARC aliases are also accepted. See
[Data format](docs/DATA_FORMAT.md) and [Model definitions](docs/MODELS.md).

Included model families are:

- Newtonian baryons;
- NFW and Burkert dark-matter baselines;
- MOND/RAR variants, including an explicitly approximate EFE sensitivity proxy;
- a repository-specific empirical UFF curve family; and
- an optional weak-field central SMBH term.

The fit pipeline reports likelihood diagnostics, chi-squared, RMSE, AIC/AICc,
BIC, relative information-criterion weights, bound hits, full residual arrays
and SHA-256 input receipts. Optional posterior sampling, plots and deterministic
sonification remain available.

### Compact-object scale report

```bash
python -m uff compact-object \
  --mass-msun 4300000 \
  --spin 0.5 \
  --velocity-dispersion-kms 100 \
  --out outputs/sgr-a-scale-report.json
```

This command reports Kerr characteristic radii, the sphere of influence, the
selected LQG area-gap convention and scale ratios. LQG is not used in the galaxy
likelihood. See [Scientific status](docs/SCIENCE_STATUS_2026.md).

## 2. UFF-SLFA: frozen sky-lattice claims

UFF-SLFA asks a narrow question:

> Does an independently selected catalogue show a preregistered excess of a
> declared anomaly inside spherical caps around frozen celestial nodes, under a
> null model that preserves the relevant selection structure?

A `uff.sky-lattice-claim.v1` contract freezes:

- node IDs and ICRS coordinates;
- one cap radius;
- the anomaly predicate;
- catalogue, holdout, weight and stratum declarations;
- the null model, permutation count and deterministic seed;
- alpha, minimum effect and required supported-node count; and
- explicit anti-circularity statements.

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

SLFA supports shared RA shifts, Haar-uniform proper SO(3) rotations and
stratified label permutations. It uses finite plus-one empirical p-values,
Holm family-wise correction and TFT-derived invariance checks for every
geometric null transform.

Read the full [Sky-Lattice Falsification Protocol](docs/SKY_LATTICE_FALSIFICATION_PROTOCOL.md).

## 3. Sheridan Crucible: the survey-aware layer

Sheridan wraps an ordinary SLFA claim in `uff.sheridan-crucible.v1` and makes
the telescope and catalogue geometry part of the frozen experiment.

It adds:

- explicit survey-support quadrature with masks and fractional coverage;
- completeness filtering and inverse-completeness weights;
- normalized spherical von Mises–Fisher weighted KDE;
- leave-one-out bandwidth selection and adaptive local bandwidths;
- mask-aware edge renormalization;
- survey-availability-matched SO(3) rotations;
- nuisance-only versus nuisance-plus-node logistic comparison;
- predictive checks and synthetic anomaly-label injection;
- bounded exact-source execution; and
- SHA-256 evidence bundles with complete numerical replay.

Generate a deterministic full-sky support grid:

```bash
python -m uff.sheridan support-grid \
  --points 4096 \
  --out full_sky_support.csv
```

Run a frozen Sheridan contract:

```bash
python -m uff.sheridan run \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv \
  --contract examples/sheridan_contract.example.json \
  --out runs/sheridan-example
```

Verify integrity and optionally replay the calculation:

```bash
python -m uff.sheridan verify \
  runs/sheridan-example/manifest.json \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv
```

Read the full [Sheridan Siege Engine protocol](docs/SHERIDAN_SIEGE_ENGINE.md).

## Evidence bundles and verdict boundaries

SLFA and Sheridan separate four questions that are often blurred together:

1. **Was the claim fully specified?**
2. **Are the artifacts intact?**
3. **Does numerical replay reproduce the stored result?**
4. **Is the scientific model and sampling design defensible?**

A bundle may be computationally perfect and scientifically biased. Hashes prove
byte identity; deterministic replay proves computational consistency; neither
proves that the sampling frame, anomaly predicate or null distribution is
appropriate.

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

Failed and untestable nodes remain visible. Null outcomes are not deleted. A
positive association remains an association, not automatic evidence for its
proposed cause.

## Claim provenance and independent assessment

The current repository includes a content-addressed public-claim ledger for the
Logvinovich celestial-node claims. It preserves incompatible coordinate sets,
radii, query predicates, reported counts and unresolved fields without choosing
a preferred version on the claimant's behalf.

Key records include:

- [Public Claim Ledger](docs/PUBLIC_CLAIM_LEDGER_2026-08-07.md)
- [Public Claim Source Manifest](docs/PUBLIC_CLAIM_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable public claim profile](examples/public_claim_profile_2026-08-07.json)
- [Independent assessment response](docs/INDEPENDENT_ASSESSMENT_RESPONSE_2026-08-07.md)
- [Independent assessment source manifest](docs/INDEPENDENT_ASSESSMENT_SOURCE_MANIFEST_2026-08-07.md)
- [Machine-readable assessment action ledger](examples/independent_assessment_actions_2026-08-07.json)

The governing assessment is intentionally uncomfortable:

> The crucible's syntax is largely formalised. Its statistical calibration
> still needs validation, and the claimant has not supplied one stationary
> claim to place inside it.

The assessment proposes a future breaking contract family,
`uff.sheridan-crucible.v2`, with explicit provenance, statistic, estimand,
multiplicity, quality and reproducible-environment fields. **That v2 schema is a
roadmap, not an implemented contract in this release.**

## Machine-readable schemas

| Schema | Role | Executable? |
|---|---|---:|
| `uff.rotation-curve-summary.v4` | Galaxy fit and comparison result | Output schema |
| `uff.sky-lattice-claim.v1` | Frozen catalogue-level celestial-node claim | Yes, when complete |
| `uff.sheridan-crucible.v1` | Survey-aware wrapper around a frozen SLFA claim | Yes, when complete |
| `uff.public-claim-profile.v1` | Provenance record containing unresolved public claim versions | No by design |
| `uff.independent-assessment-response.v1` | Machine-readable implementation roadmap | No; governance record |
| `uff.sheridan-crucible.v2` | Proposed publication-grade contract expansion | Planned, not implemented |

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
examples/                   # frozen example contracts and governance ledgers
tests/                      # model, geometry, replay and provenance regressions
docs/                       # protocols, scientific boundaries and manifests
papers/                     # methods papers, assessment rendition and references
```

## Research status and limitations

- Rotation curves alone do not settle dark matter versus modified gravity.
- Algebraic MOND relations are not full AQUAL/QUMOND solvers for flattened disks.
- Information-criterion rankings depend on the candidate set and data contract.
- The UFF empirical profile is not derived from a covariant field theory.
- Catalogue diagnostics such as excess noise, uncertainty or missing values are
  not physical objects without an independently validated object-level model.
- Broadband colour differences are not spectral resonances without a
  bandpass-aware spectral model.
- Cross-catalogue agreement is not automatic statistical independence when the
  catalogues share objects, source-density structure or systematics.
- Preregistration prevents later rule changes; it does not repair a biased
  sampling frame or make previously inspected data blind.
- The current exact Sheridan KDE is intentionally bounded by a frozen source
  limit rather than silently changing algorithm or exhausting memory.

## Release notes

- [QSOL UFF v5.0.0](RELEASE_NOTES_v5.0.0.md)
- [UFF Sheridan Crucible v1.1.0](RELEASE_NOTES_SHERIDAN_v1.1.0.md)
- [UFF-SLFA v1.0.0](RELEASE_NOTES_SLFA_v1.0.0.md)

## Citation

The previous README DOI has been removed because it does not resolve to a valid
record for the current repository state. A new archive record and DOI will be
added after the v5.0.0 release package is deposited.

Until then, cite the repository release as:

> Slade, T. (2026). *QSOL UFF v5.0.0: Reproducible Astrophysics and
> Falsification Laboratory*. QSOL-IMC. https://github.com/QSOLKCB/UFF

Machine-readable metadata are in [CITATION.cff](CITATION.cff). Analyses must
also cite the primary scientific sources for every physical model, catalogue
and statistical method used.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Maintainer: **Trent Slade / QSOL-IMC**  
GitHub: [QSOLKCB](https://github.com/QSOLKCB)
