# UFF QEC Boundary Gate

## Purpose

UFF already has domain-specific integrity and replay machinery in UFF-SLFA and
the Sheridan Crucible. The QEC Boundary Gate adds one small, domain-neutral
trust boundary above those systems.

It deliberately borrows only a narrow set of mature ideas from `QSOLKCB/QEC`:

- canonical JSON as a byte-level contract;
- recompute-not-trust validation;
- child-before-aggregate hashing;
- self-hash exclusion;
- fail-closed schema and artifact allowlists;
- explicit separation of integrity from replay assurance; and
- a deterministic root that can be anchored outside the bundle.

It does **not** import the QEC decoder architecture, runtime, governance stack,
telemetry, switching work, quantum semantics, or historical release machinery.

## The rule that matters most

The gate has three practical outcomes:

| Outcome | Meaning |
|---|---|
| `INTEGRITY_ONLY` | The bundle is structurally intact, but no fresh numerical replay was performed. It is **not admitted**. |
| `REPLAY_VERIFIED` | Strict bundle checks passed and the appropriate UFF domain verifier reproduced the stored result from the frozen inputs. The bundle is admitted. |
| `REJECTED` | A structural, hash, semantic, replay, receipt, or external-anchor check failed. |

This closes an ambiguity that is acceptable in a low-level integrity API but is
not acceptable at a security boundary: an intact bundle must never be mistaken
for a replay-verified bundle.

## What the gate checks

Before calling the SLFA or Sheridan numerical replay code, the boundary performs
strict checks that are intentionally more defensive than ordinary JSON and
manifest parsing:

1. `manifest.json` and every JSON child must be canonical UTF-8 JSON.
2. Duplicate object keys are rejected.
3. `NaN`, `Infinity`, `-Infinity`, and UTF-8 BOMs are rejected.
4. The manifest schema selects one exact, known gate profile.
5. Artifact entries contain exactly `path`, `media_type`, `bytes`, and `sha256`.
6. Artifact paths are an exact allowlist for that profile.
7. Path traversal, backslashes, absolute paths, and symbolic links are rejected.
8. The physical bundle directory may not contain unlisted payload files.
9. Byte counts and SHA-256 values are recomputed from child artifacts.
10. JSON child bytes must remain canonical even if an attacker rewrites the
    manifest hashes to match altered formatting or duplicate-key content.
11. Recipe schema and algorithm IDs must match the selected profile.
12. The embedded contract's canonical SHA-256 is recomputed and compared with
    `contract_canonical_sha256`.
13. Frozen catalogue hashes must be present and well formed; Sheridan also
    requires a frozen support-grid hash.
14. The manifest result must match the actual decision artifact.
15. SLFA's claim boundary must agree between `observations.json` and the manifest.
16. The existing UFF domain verifier must then reproduce the numerical output
    from the supplied frozen inputs before the gate returns `admitted=true`.

## Deterministic bundle root

After all strict structural checks pass, the gate computes a root over:

- the canonical manifest SHA-256; and
- the sorted child artifact paths, byte counts, and SHA-256 values.

The root payload uses schema `uff.qec-bundle-root.v1`.

`qec_gate.json` is explicitly excluded from its own root. This is the same
self-hash-exclusion principle used in proof-receipt systems: the receipt commits
to the evidence, not recursively to itself.

A self-contained hash does **not** prove who created a bundle. For authenticity,
copy the root to an independent trust anchor such as a signed Git release,
Zenodo record, preregistration, or separately signed message, then supply that
root with `--expected-root` during verification.

## Usage

### SLFA: strict replay gate

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv
```

### Sheridan: strict replay gate

```bash
python -m uff.qec_gate \
  runs/sheridan-example/manifest.json \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv
```

### Inspect integrity without admitting the bundle

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --integrity-only
```

A successful integrity-only inspection still reports `admitted=false` and
`assurance="INTEGRITY_ONLY"`.

### Seal a replay-verified bundle

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv \
  --seal
```

This writes canonical `qec_gate.json` only after fresh numerical replay passes.
The receipt records the deterministic root and the fact that authenticity still
requires an independent anchor.

### Verify against an external root

```bash
python -m uff.qec_gate \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv \
  --expected-root 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

## What this closes

The gate closes computational trust-boundary gaps including:

- green-light ambiguity between integrity and replay;
- permissive JSON parsing at the security boundary;
- duplicate-key and non-finite JSON ambiguity;
- unlisted-file and symbolic-link smuggling;
- manifest/decision divergence;
- embedded-contract hash drift;
- trusting a stored receipt without recomputation; and
- lack of a deterministic root suitable for an external trust anchor.

## What this cannot close

The QEC Boundary Gate is intentionally not a scientific oracle.

It cannot establish:

- that a catalogue is correct or unbiased;
- that a claimant supplied a stationary historical specification;
- that a null model represents nature;
- that the multiplicity family is scientifically adequate;
- that a protected holdout was genuinely unseen by every analyst;
- that a source is independent merely because its bytes differ;
- that a statistical association is causal; or
- that any proposed physical ontology is true.

Those remain experimental-design, provenance, calibration, and scientific
interpretation questions. The gate's job is narrower: **nothing crosses the UFF
computational trust boundary unless its structure is strict, its evidence is
intact, and its deterministic result has actually replayed.**
