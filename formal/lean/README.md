# UFF v5.2.0 Lean 4 formal assurance layer

This directory contains a deliberately small Lean 4 model of selected UFF assurance and provenance invariants. It formalizes computational and epistemic boundaries; it does **not** attempt to prove that an astrophysical theory is physically true.

## Machine-checked scope

The library models and proves properties of:

- the ordered UFF assurance ladder, including replay promotion that never lowers an already higher assurance state;
- separation of replay from ensemble calibration and external scientific judgement;
- claim-boundary and frozen-recipe identity;
- cancellation as a non-exportable transaction state;
- one-way sonification and zero-authority telemetry;
- manifest-core sealing without circular self-hash input;
- deterministic replay as pure re-evaluation of frozen inputs; and
- runtime-fingerprint identity for replay-sensitive work.

## Explicit nonclaims

The formal layer does not prove SHA-256 collision resistance, catalogue correctness, survey unbiasedness, null-model adequacy, statistical independence, causal interpretation, or physical truth. Those boundaries are represented explicitly in `UFF/Assumptions.lean` and `ASSUMPTIONS_AND_NONCLAIMS.md`.

## Build

```bash
cd formal/lean
lake build
bash audit.sh
```

CI additionally rejects `sorry`/`admit`, rejects project-defined `axiom`/`constant` declarations even when prefixed by supported Lean attributes/modifiers, checks `AUDIT_MANIFEST.tsv` against all theorem/lemma declarations including modified declarations, and audits each advertised theorem with `#print axioms`.

The toolchain is pinned by `lean-toolchain`.
