# UFF Defense in Depth

UFF uses three small, independent ideas from QSOLKCB/QEC, QSOLKCB/SPECTRAL,
and QSOLKCB/SONIFICATION. None of the parent architectures is transplanted.
Each borrowed mechanism has one job and one authority boundary.

```text
             BEFORE OBSERVATION
                     │
          SPECTRAL-inspired witness
       contract + catalogue + support
                identity commit
                     │
                     ▼
              UFF computation
                     │
                     ▼
             QEC boundary gate
       strict structure + hashes +
       cross-links + numerical replay
                     │
              ADMIT / REJECT
                     │
                     ▼
       SONIFICATION-inspired telemetry
          receiver-neutral events
          audio/visualization optional
```

## 1. QEC: the bouncer

Module: `uff.qec_gate`

QEC contributes the hard computational trust boundary:

- strict canonical JSON;
- duplicate-key and non-finite-value rejection;
- exact schema and artifact allowlists;
- path, symlink and unlisted-file rejection;
- child hash and byte-count recomputation;
- embedded-contract hash recomputation;
- decision/manifest cross-checks;
- deterministic bundle-root construction;
- optional external root anchoring; and
- most importantly, no admission without fresh numerical replay.

The key distinction is explicit:

- `INTEGRITY_ONLY` — intact, but **not admitted**;
- `REPLAY_VERIFIED` — intact and reproduced, admitted;
- `REJECTED` — failed a trust-boundary check.

This layer has authority over computational bundle admission only.

## 2. SPECTRAL: the witness

Module: `uff.spectral_witness`

SPECTRAL contributes a commit/reveal pattern for identity-bearing inputs.
Before observing a result, the witness commits to:

- the raw contract file SHA-256;
- the canonical contract SHA-256;
- the frozen catalogue SHA-256 and byte count; and
- for Sheridan, the frozen support-grid SHA-256 and byte count.

Filenames, paths, modification times, wall-clock time and UI state are excluded
from identity. The commitment is domain separated and excludes its own digest
from the digest preimage.

Create the witness **before** running or inspecting the audit:

```bash
python -m uff.spectral_witness commit precommit.json \
  --contract frozen_contract.json \
  --catalogue frozen_catalogue.csv
```

For Sheridan add:

```bash
--support frozen_support.csv
```

The command prints the commitment SHA-256. To make a historical precommitment
claim, place that digest in an independent timestamped/signed location before
observation: for example a preregistration, signed Git release, DOI record, or
other external trust anchor.

Reveal only after a UFF bundle exists:

```bash
python -m uff.spectral_witness reveal \
  precommit.json runs/frozen-claim/manifest.json \
  --contract frozen_contract.json \
  --catalogue frozen_catalogue.csv \
  --expected-commit <externally-anchored-digest>
```

A reveal is admitted only when the frozen identities still match and the QEC
gate independently returns `REPLAY_VERIFIED`.

A local commitment alone does **not** prove chronology or analyst blindness.
It only proves identity. The historical claim comes from the independent anchor.

## 3. SONIFICATION: the diagnostic receiver bus

Module: `uff.audit_events`

SONIFICATION contributes a receiver-neutral event protocol, not a truth oracle.
The module performs a live QEC boundary verification and emits deterministic
telemetry describing:

- integrity state;
- replay state;
- admission state;
- external bundle-root match when supplied;
- the already-recorded UFF scientific decision; and
- selected already-recorded diagnostic metrics.

Generate telemetry outside the evidence bundle:

```bash
python -m uff.audit_events \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv \
  --out telemetry/frozen-claim-events.json
```

For Sheridan add `--support frozen_support.csv`.

The canonical layer contains event order, channel, code, state, polarity,
authority and fixed-point/integer observations. It deliberately does **not**
define tempo, hertz, MIDI note, timbre, loudness, stereo position, waveform or
rendered audio.

Those are receiver choices. A future browser or audio receiver can make failure,
drift and replay state audible without changing one bit of UFF evidence or one
scientific verdict.

Telemetry is forbidden inside the closed evidence-bundle directory so an
observation tool cannot accidentally become part of the evidence identity.

## Authority matrix

| Layer | Can block computational admission? | Can change scientific result? | Can claim physical truth? |
|---|---:|---:|---:|
| UFF scientific engine | Produces the recorded result | Yes, by executing the frozen method | No |
| QEC boundary gate | **Yes** | No | No |
| SPECTRAL witness | **Yes, when reveal identity fails** | No | No |
| SONIFICATION telemetry | **No** | **No** | No |

## Gaps these layers close

Together the three small additions close several practical gaps without
rebuilding UFF around another architecture:

- integrity-only success cannot masquerade as replay verification;
- permissive JSON and hidden bundle payloads fail closed;
- manifest, decision and embedded-contract identities are cross-checked;
- a bundle can be tied to an independent external root;
- frozen pre-observation inputs can be committed before result inspection;
- post-commit input substitution is detectable; and
- trust-boundary failures and scientific diagnostics can be exported as a
  deterministic receiver-neutral event stream for human inspection.

## Gaps they intentionally do not close

No software receipt can establish all of experimental validity. These layers do
not prove:

- that source data are correct or unbiased;
- that an analyst never saw the data before a commitment;
- that a claimant's historical specification was stationary;
- that the null model is scientifically adequate;
- that multiplicity or selection effects are fully calibrated;
- that two catalogues are independent;
- that an association is causal; or
- that a proposed physical ontology is true.

Those remain the job of preregistration, independent data, calibration,
protected holdouts, provenance, external review and scientific argument.

The design principle is therefore simple:

> **QEC guards the door. SPECTRAL proves what was handed to the door. SONIFICATION tells the operator what the door is doing. None of them gets to impersonate nature.**
