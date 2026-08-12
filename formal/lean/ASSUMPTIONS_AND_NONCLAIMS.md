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
