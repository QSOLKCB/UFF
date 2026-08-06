# UFF Sheridan Crucible v1.1.0 — Release notes

The Sheridan Crucible extends UFF-SLFA with an exact survey-aware reference engine for adversarial evaluation of frozen celestial-node claims.

## Added

- Nested `uff.sheridan-crucible.v1` contracts around an ordinary SLFA claim.
- Explicit survey-support quadrature with solid-angle weights and fractional coverage/masks.
- Completeness filtering and inverse-completeness analysis weights.
- Normalized spherical von Mises–Fisher weighted KDE.
- Leave-one-out global bandwidth selection and adaptive local bandwidths.
- Mask-aware kernel-mass and field-edge renormalization.
- Survey-availability-matched Haar-uniform SO(3) lattice rotations.
- Holm-corrected node density tests retaining failed and untestable nodes.
- Nuisance-only versus nuisance-plus-node weighted logistic model comparison.
- Transparent pseudo-BIC, bound-hit reporting, and Laplace predictive checks.
- Predictive nuisance-baseline anomaly injection with recovery and false-positive rates.
- Component-wise frozen final decisions; no result can silently redefine which tests count.
- Deterministic full-sky Fibonacci support-grid generator.
- SHA-256 evidence bundles and complete numerical replay.
- Exact-source safety limit preventing silent quadratic resource exhaustion.
- Protocol documentation, example contract, example catalogue, Zenodo metadata, canonical BibTeX metadata, and regression tests.

## Scientific boundary

Sheridan tests survey-corrected catalogue associations and declared competing models. It does not identify the physical cause of a surviving pattern, convert catalogue diagnostics into objects, simulate telescope images, or prove that no future modified claim can be proposed.
