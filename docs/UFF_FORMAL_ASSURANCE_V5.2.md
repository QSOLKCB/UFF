# UFF v5.2.0 — Machine-Checked Assurance and Formal Claim Boundaries

UFF v5.2.0 adds a Lean 4 specification layer over selected evidence-assurance invariants introduced operationally in v5.0.0 and v5.1.0.

The objective is deliberately narrow: make important **software and epistemic boundaries** machine-checkable without pretending that a theorem prover establishes astrophysical truth.

## Formalized invariants

The v5.2.0 Lean layer covers:

1. **Assurance ordering.** `REPLAY_VERIFIED` does not itself satisfy `ENSEMBLE_CALIBRATED`; `ENSEMBLE_CALIBRATED` does not itself satisfy `SCIENTIFICALLY_DEFENSIBLE`; replay promotion raises a state to at least replay verification and never discards a higher, separately earned assurance level.
2. **External scientific judgement separation.** Promoting a computational state to at least replay-verified leaves the modeled external scientific judgement unchanged.
3. **Identity-bearing claim boundaries.** Changing a claim boundary or frozen recipe changes experiment identity in the formal model.
4. **Fail-closed cancellation.** A cancelled transaction has no archival bundle and is not exportable.
5. **One-way observation.** Sonification retains the underlying numerical result; telemetry leaves evidence state unchanged.
6. **No-self-hash manifest construction.** The digest function consumes the manifest core before an envelope carries that digest.
7. **Deterministic replay model.** A deterministic engine is a pure function of frozen recipe and runtime contract; same formal inputs yield the same replay result.
8. **Runtime-sensitive identity.** A changed recorded runtime contract changes replay identity.

## Nonclaims

Lean does not establish SHA-256 collision resistance, catalogue correctness, survey unbiasedness, null-ensemble adequacy, statistical independence, causal interpretation, or physical truth. It also does not prove full semantic equivalence between the Python runtime and the Lean model.

## Provenance

The formal layer draws on UFF v5.1.0's assurance model and the archived QSOL-NEXUS v1.0.0 deterministic workbench architecture. NEXUS supplied reusable architectural ideas—frozen recipes, identity-bearing claim boundaries, one-way observation mappings, replay validation, and no-self-hash construction—while UFF v5.2.0 formalizes only the subset relevant to UFF's evidence pipeline.

The base UFF release commit pinned by CI is:

`136b63089f5b70cd2a7356b6f647d482f3b3273c`

The NEXUS v1.0.0 archive is referenced as a design ancestor, not copied as an authority source.

## Verification discipline

CI pins Lean 4, rejects proof holes, rejects project-defined axiom/constant declarations including supported attribute/modifier-prefixed declarations, checks the theorem manifest against source declarations including modified theorems/lemmas, builds the complete Lean library, and runs `#print axioms` over every advertised theorem.

The workflow runs for the review branch, pull requests into `main`, and pushes to `main`, so the exact merged formal tree is re-verified after merge rather than relying only on a pre-merge checkout.

The resulting verification report is computational evidence about the formal specification. It remains distinct from empirical scientific validation.
