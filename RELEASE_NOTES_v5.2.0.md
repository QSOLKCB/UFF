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
- replay promotion preserves external scientific judgement and never lowers a higher, separately earned assurance state;
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
- project-defined `axiom` and `constant` declarations are forbidden in the UFF theorem modules, including supported attribute/modifier-prefixed forms such as `private axiom`;
- `AUDIT_MANIFEST.tsv` must exactly match advertised theorem/lemma declarations, including supported modified declarations such as `private theorem`;
- every advertised theorem is covered by `#print axioms`; and
- unexpected axiom dependencies fail CI.

The formal workflow runs on the review branch, pull requests into `main`, and pushes to `main`, so the exact merged formal tree is re-verified after merge.

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

The assigned v5.2.0 Zenodo version DOI is **10.5281/zenodo.21911644**.

DOI assignment and exact software-tree binding are deliberately separate. PR #15 remains the release candidate until merged; after merge, the `v5.2.0` GitHub tag must resolve to the exact merged commit used for the archival payload. The DOI must not be treated as proof that an arbitrary branch head is the archived software identity.

The release package under `zenodo/v5.2.0/` provides metadata and an operator checklist for completing that exact-commit binding.
