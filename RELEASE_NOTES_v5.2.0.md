# QSOL UFF v5.2.0 — Release notes

**Release date:** 12 August 2026  
**Theme:** Machine-checked assurance and formal claim boundaries

UFF v5.2.0 is an additive formal-assurance release. It retains the v5.1.0 scientific implementation and defense-in-depth evidence workflow while adding a Lean 4 specification layer for selected invariants around assurance state, experiment identity, replay, observation, transaction closure, and manifest construction.

No astrophysical likelihood, catalogue schema, SLFA contract, or Sheridan contract is redefined by this release.

## Highlights

### Lean 4 formal assurance library

Added `formal/lean/` with a pinned Lean toolchain and machine-checked theorems covering:

- replay does not manufacture ensemble calibration;
- ensemble calibration does not manufacture scientific defensibility;
- replay promotion preserves external scientific judgement;
- changed claim boundaries and frozen recipes change formal experiment identity;
- cancelled transactions cannot export an archival bundle;
- sonification preserves the underlying numerical result;
- receiver-neutral telemetry has zero evidence-admission authority;
- manifest-core sealing is modeled without circular self-hash input;
- deterministic replay is pure re-evaluation of frozen recipe + runtime contract; and
- changed runtime contracts change replay identity.

### Explicit formal nonclaims

The formal layer contains machine-visible scope markers stating that UFF Lean does not prove:

- SHA-256 collision resistance;
- catalogue correctness or unbiasedness;
- null-model adequacy;
- statistical independence or causal interpretation; or
- physical truth.

This prevents the theorem prover from being used as rhetorical authority outside its actual proof scope.

### Audited theorem surface

Added a strict formal audit:

- `sorry` and `admit` are forbidden;
- project-defined `axiom` and `constant` declarations are forbidden in the UFF theorem modules;
- `AUDIT_MANIFEST.tsv` must exactly match advertised theorem/lemma declarations;
- every advertised theorem is covered by `#print axioms`; and
- unexpected axiom dependencies fail CI.

### NEXUS architectural lineage

The formal specification documents its design lineage to the archived QSOL-NEXUS v1.0.0 deterministic workbench, particularly frozen transactions, claim-boundary identity, one-way sonification, replay semantics, and no-self-hash construction. The NEXUS archive is a provenance source, not a proof authority.

## Scientific boundary

The v5.1.0 governing rule remains intact:

```text
REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH
```

v5.2.0 strengthens the first part of that boundary by making selected assurance relationships machine-checkable. It does not convert computational consistency into empirical calibration or physical truth.

## Compatibility

- Existing Python APIs and CLIs remain unchanged apart from the package version.
- Existing executable schemas remain `uff.sky-lattice-claim.v1` and `uff.sheridan-crucible.v1`.
- `uff.sheridan-crucible.v2` remains a roadmap.
- `ENSEMBLE_CALIBRATED` remains a future runtime assurance state that must be separately earned.

## Zenodo versioning

Publish v5.2.0 through Zenodo's **New version** workflow. Do not overwrite the immutable v5.0.0 or v5.1.0 version records. The v5.2.0 version DOI is assigned by Zenodo when the new version is published.

The release package under `zenodo/v5.2.0/` provides metadata and an operator checklist for binding the deposit to the exact merged release commit.
