# UFF-SLFA v1.0.0: Sky-Lattice Falsification Audit

**Status:** preregistration-ready method and reference implementation  
**Scope:** catalogue-level claims of anomalies concentrated around fixed celestial nodes

## Research question

UFF-SLFA asks one deliberately narrow question:

> Does a complete, independently selected catalogue exhibit a preregistered excess of a declared anomaly inside fixed spherical caps around frozen celestial nodes, relative to a null model that preserves the relevant survey selection structure?

The protocol does not decide whether space is a crystal, a condensate, a standing wave, or any other ontology. A statistically supported catalogue association would be a reason for further investigation, not automatic confirmation of a field theory.

## Non-negotiable declarations

A contract must declare that catalogue selection was independent of the proposed node masks. A node-targeted query cannot validate node clustering. Confirmatory runs must also use either:

- an independent catalogue not used to discover or tune the nodes; or
- a declared holdout split that was not inspected while fixing coordinates, thresholds, and decision rules.

Failed nodes remain in the denominator. Null results remain in the bundle. Thresholds are frozen before the confirmatory run.

## Frozen claim contract

The schema `uff.sky-lattice-claim.v1` records:

- ICRS node coordinates and identifiers;
- one spherical-cap radius;
- the anomaly column, comparison operator, and threshold;
- catalogue and discovery-catalogue SHA-256 values where applicable;
- holdout, weight, and stratum columns;
- null model, permutation count, and deterministic seed;
- alpha, minimum rate contrast, and required supported-node count; and
- explicit anti-circularity declarations.

Canonical JSON hashing makes any later change to the contract visible.

## Test statistic

For anomaly indicator `Y_i`, positive survey weight `w_i`, and union-cap membership `M_i`, define

\[
\hat p_{in}=\frac{\sum_i w_iY_iM_i}{\sum_i w_iM_i},
\qquad
\hat p_{out}=\frac{\sum_i w_iY_i(1-M_i)}{\sum_i w_i(1-M_i)}.
\]

The preregistered global statistic is

\[
T=\hat p_{in}-\hat p_{out}.
\]

The same contrast is calculated separately for every node. A Haldane-Anscombe corrected odds ratio is reported as a diagnostic; it does not silently replace the frozen statistic.

## Null models

### RA shift

All nodes receive the same random rotation about the ICRS z-axis. Node declinations, cap sizes, and mutual geometry remain fixed. This is useful when declination-dependent exposure must be retained and right ascension is the exchangeable direction.

### Uniform SO(3) rotation

The complete node configuration receives a Haar-uniform proper three-dimensional rotation. This preserves every pairwise angular separation. It is suitable for effectively full-sky samples or samples with an explicitly modelled selection function.

### Stratified label permutation

Anomaly labels are shuffled within preregistered strata while coordinates remain fixed. Strata can encode survey, depth, crowding, Galactic latitude band, exposure, or quality regime.

## TFT-derived invariance bridge

For each geometric null transform, UFF-SLFA verifies

\[
R^TR=I,\qquad \det R=+1,
\]

and checks that the node Gram matrix and all pairwise angular separations remain invariant. This prevents the null generator from accidentally deforming the claimed lattice. The invariant check validates the transformation used by the audit; it does not validate the physical claim.

## Empirical significance and multiplicity

For `B` null replicates,

\[
p=\frac{1+\sum_{b=1}^{B}\mathbf{1}[T_b\ge T_{obs}]}{B+1}.
\]

This finite-sample correction prevents reported permutation p-values of zero. Node-wise p-values are adjusted with Holm's step-down family-wise procedure.

A node is supported only when:

1. its Holm-adjusted p-value is no greater than the frozen alpha; and
2. its rate contrast reaches the frozen minimum effect.

The global decision `EMPIRICAL_CRITERIA_MET` requires both the global criterion and the preregistered number of supported nodes. Otherwise the result is `EMPIRICAL_CRITERIA_NOT_MET`.

## Artifact and replay contract

Each run writes:

- `recipe.json` - frozen contract, input hashes, row counts, and replay boundary;
- `observations.json` - global result, node count result, invariant residuals, and claim boundary;
- `nodes.csv` - all nodes, including failures, raw p-values, Holm p-values, and effects;
- `manifest.json` - artifact byte sizes, SHA-256 hashes, runtime metadata, and result.

Verification separates two questions:

1. **Integrity:** do the stored files still match their manifest hashes?
2. **Numerical replay:** does the exact frozen catalogue reproduce the recorded decision and numerical observations?

A bundle can be internally consistent and still be based on a poor scientific model. Hashes and replay establish computational consistency, not truth.

## Prohibited interpretations

- A timeout, missing value, high uncertainty, excess-noise field, or poor fit is not a "pipeline crash" without an operational failure log.
- A catalogue diagnostic is not a new physical object without an independently validated object-level model.
- A broadband colour difference is not a spectral resonance without a bandpass-aware spectral model.
- A successful association test does not prove a causal ontology.
- A failed frozen claim does not prove that no future modified claim can be written; it falsifies the exact tested claim.

## Usage

```bash
python sky_lattice_audit.py run \
  --catalogue frozen_catalogue.csv \
  --contract frozen_claim.json \
  --out runs/frozen-claim

python sky_lattice_audit.py verify \
  runs/frozen-claim/manifest.json \
  --catalogue frozen_catalogue.csv
```

## Scientific posture

The protocol is intended to be equally inconvenient for believers and skeptics. A claimed pattern that survives a frozen, footprint-aware holdout deserves attention. A pattern that disappears under those controls must not be advertised as confirmed by the tested catalogue.
