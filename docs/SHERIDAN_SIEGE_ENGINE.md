# UFF Sheridan Crucible v1.1.0 — Survey-Aware Siege Engine

**Status:** exact, preregistration-ready reference implementation  
**Scope:** frozen celestial-node claims evaluated against explicit survey geometry, catalogue selection, nuisance models, and calibrated detection power

## Why this layer exists

UFF-SLFA v1.0 asks whether a declared catalogue anomaly is more common inside frozen node caps than outside them. The Sheridan Crucible adds the observational machinery needed when the telescope and catalogue have their own geometry.

The engine is deliberately symmetric. It does not presume that the proposed lattice is real, and it does not presume that every anomaly is instrumental. It freezes the test, models the survey, and makes the competing explanations face the same data and decision rule.

## Five locked phases

1. **Protocol lock.** The nested `uff.sky-lattice-claim.v1` contract freezes nodes, cap radius, anomaly rule, holdout, and claim boundary.
2. **Catalogue and survey forensics.** Completeness, quadrature area, coverage, masks, and nuisance columns are validated before analysis.
3. **Spherical density reconstruction.** A von Mises–Fisher weighted KDE uses leave-one-out bandwidth selection, adaptive smoothing, mask-aware quadrature, and edge renormalization.
4. **Competing model stress test.** A nuisance-only weighted logistic model is compared with the identical model plus one frozen node-membership term.
5. **Calibration and multiplicity.** Survey-matched SO(3) rotations, Holm correction, Laplace predictive checks, and synthetic anomaly-label injection quantify significance, detection power, and false positives.

## Survey-support quadrature

The support CSV is an explicit numerical representation of the usable survey field. Each row contains:

```text
ra_deg,dec_deg,area_weight_sr,coverage
```

- `area_weight_sr` is the solid angle represented by the support point.
- `coverage` is a frozen value in `[0, 1]`; zero represents a mask and fractional values can represent partial exposure or usable-area probability.
- The sum of area weights cannot exceed `4π` steradians.

A deterministic full-sky grid can be generated with:

```bash
python -m uff.sheridan support-grid \
  --points 4096 \
  --out full_sky_support.csv
```

For a real survey, replace the generated `coverage` column with values derived from the authoritative exposure/mask product. Do not infer the footprint from the claimed node detections.

## Spherical density model

For source direction `x_g`, evaluation direction `x`, and adaptive angular bandwidth `b_g`, the engine uses the normalized von Mises–Fisher kernel on `S²`:

```text
κ_g = 1 / b_g²
K(x; x_g, b_g) = κ_g exp(κ_g x·x_g) / (4π sinh κ_g)
```

The global bandwidth is selected from a frozen candidate list by weighted leave-one-out log likelihood. A pilot field then produces Abramson-style adaptive bandwidths:

```text
b_g = b_global [G / f_pilot(x_g)]^α
```

with frozen lower and upper factors.

### Mask and edge correction

The usable kernel mass for source `g` is evaluated by quadrature:

```text
η_g = Σ_j area_j coverage_j K(s_j; x_g, b_g)
```

The corrected density is:

```text
f(x) = [Σ_g w_g K(x; x_g, b_g) / η_g] / Σ_g w_g
```

This prevents masked areas and truncated kernels from masquerading as physical voids or overdensities. The algorithm is conceptually informed by the survey-geometry, vMF-wKDE, adaptive-bandwidth, mask, and edge-correction methodology described by Hatamnia et al. (2026), *The Astrophysical Journal* 1002:192, DOI `10.3847/1538-4357/ae5bac`. The UFF implementation is independently written under Apache-2.0.

## Survey-matched geometric null

A naive full-sky rotation can compare well-observed nodes with unobserved controls. Sheridan therefore computes a smoothed survey-availability value at every node. A proper SO(3) rotation is accepted only when:

- the same node identities remain testable; and
- the preregistered normalized availability distance is within tolerance.

Every accepted transform is checked for:

```text
RᵀR = I
 det(R) = +1
```

and for preserved Gram matrix and pairwise node angles.

## Competing anomaly models

The nuisance-only model contains:

- an intercept;
- frozen standardized numeric covariates; and
- optional survey/quality stratum indicators.

The node model adds exactly one binary term: membership in the union of frozen node caps. Both models use the same completeness-adjusted catalogue weights. The output reports log likelihood and **pseudo-BIC**. It is called pseudo-BIC because non-integer selection weights are normalized to the catalogue row count; it is not presented as an exact marginal likelihood.

A node term is preferred only when:

- both fits converge;
- the node coefficient is positive and does not hit its bound; and
- `BIC_null - BIC_node` reaches the frozen threshold.

Laplace-approximate predictive simulations test whether each fitted model reproduces the observed weighted inside-versus-outside anomaly contrast.

## Synthetic injection calibration

The engine fits the nuisance-only model, draws predictive anomaly baselines that preserve declared covariate and stratum effects, and injects anomaly labels into actual in-footprint rows inside each node. It then measures:

- recovery rate after injection; and
- false-positive rate before injection.

This is a catalogue-level power calibration. It does **not** simulate detector images, source extraction, or telescope hardware. A publication claiming image-level recovery must supply a separate image-injection pipeline.

## Exact reference limit

The current adaptive KDE implementation is intentionally exact and has a frozen `maximum_exact_sources` contract field, defaulting to 3000. It refuses larger inputs rather than silently exhausting memory or changing algorithms after seeing the data.

For larger catalogues, preregister one of the following:

- a representative weighted sample selected independently of the nodes;
- a physically justified stratum-by-stratum analysis; or
- a future validated sparse/accelerated backend with equivalence tests against this reference implementation.

Raising the limit is allowed only by changing the frozen contract before analysis and accepting the resulting runtime and memory requirements.

## Bundle and replay

Each run writes:

- `recipe.json` — nested contract, input hashes, row counts, survey area, and method boundaries;
- `density.json` — bandwidth, edge-mass, survey-matched null, and global density result;
- `nodes.csv` — complete node-level density and multiplicity results;
- `models.json` — nuisance and node fits plus predictive checks;
- `injection.json` — recovery and false-positive calibration;
- `decision.json` — logical conjunction of only the components frozen as required;
- `manifest.json` — byte sizes, SHA-256 hashes, runtime versions, and final result.

Verification separates artifact integrity from numerical replay:

```bash
python -m uff.sheridan verify \
  runs/example/manifest.json \
  --catalogue frozen_catalogue.csv \
  --support frozen_support.csv
```

## Interpretation boundary

- A positive density result establishes a survey-corrected spatial association, not its cause.
- A preferred node term establishes conditional predictive improvement within the declared candidate set, not a vacuum ontology.
- Passing injection calibration establishes that this catalogue-level procedure could detect the injected effect under the frozen assumptions.
- A failed frozen result is not a proof that no modified theory can ever be written.
- Database timeouts, missing values, uncertainties, and classification diagnostics remain metadata until an object-level physical model is independently validated.

## Reference metadata

The canonical BibTeX record for the COSMOS-Web methodological source is stored in
[`papers/sheridan_references.bib`](../papers/sheridan_references.bib). The entry is
keyed by DOI and preserves the capitalization of COSMOS-Web and JWST.
