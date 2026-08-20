# Assumptions and nonclaims

UFF v5.2.0 uses Lean 4 to verify a compact model of software-level and epistemic invariants. The formal model is intentionally narrower than the scientific system.

## Not proved by Lean

- SHA-256 collision resistance or preimage resistance.
- Correctness, completeness, independence, or lack of bias in source catalogues.
- Adequacy of a chosen null ensemble for nature.
- Historical analyst blindness from a local commitment.
- Statistical calibration beyond the explicitly modeled assurance label.
- Causality, ontology, or physical truth.
- Bit-for-bit equivalence of the Python implementation to the Lean model.

## What the proofs mean

The theorems show that the **formal definitions in this directory** satisfy the stated invariants. Runtime correspondence is documented separately and must be maintained by review and regression testing. A theorem about the formal model is not a proof that every Python execution or external dataset satisfies its preconditions.

## v5.3 nonclaim calibration layer

UFF v5.3.0 adds an epistemic calibration reference outside the Lean theorem surface. The governing separation is:

```text
EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT != PHYSICAL_TRUTH
```

The worked reference is Jampolski & Rezzolla, **“Formation of gravastars,”** arXiv:2509.15302v2. It is used because the source makes several useful boundaries visible: successful formation is obtained under fine-tuned conditions; the construction is found by backward integration from the desired final state; and more realistic assumptions plus the relative likelihood of gravastar versus black-hole formation are left open.

This reference does **not** extend the Lean proofs and is not evidence that gravastars are generic, common, observed, or physically preferred. It calibrates how UFF records absent warrant. See [`../../docs/NONCLAIM_CALIBRATION.md`](../../docs/NONCLAIM_CALIBRATION.md) and [`../../examples/nonclaim_reference_gravastar_2026.json`](../../examples/nonclaim_reference_gravastar_2026.json).
