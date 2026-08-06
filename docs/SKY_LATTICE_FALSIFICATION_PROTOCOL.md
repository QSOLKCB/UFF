# UFF-SLFA: Sky-Lattice Falsification Audit

**Protocol version:** 1.0.0  
**Status:** preregistration-ready research method  
**Scope:** empirical celestial-node claims and survey-systematics controls

## Purpose

UFF-SLFA converts a claimed celestial lattice into a frozen, executable claim
contract. It asks a narrow question:

> Does a complete, independently defined catalogue exhibit a preregistered
> excess of anomalies inside fixed spherical caps around the claimed nodes,
> relative to a null model that preserves the relevant survey selection
> structure?

It does **not** decide whether space is a crystal, a quantum condensate, a
standing wave, or any other ontology. Those are causal interpretations requiring
additional predictions.

## Why this protocol exists

A query that first selects high-error records inside proposed node windows and
then reports that those records occur inside the windows is circular. Likewise,
a full-sky binomial null is invalid for a sample drawn only from targeted boxes.
UFF-SLFA prevents these errors by requiring:

1. a complete catalogue sample independent of the node masks;
2. a frozen node list, cap radius, anomaly rule, null model, and decision rule;
3. a holdout set or genuinely new catalogue where feasible;
4. rotations or label permutations that preserve declared selection effects;
5. family-wise correction for node-by-node claims;
6. hashes for the contract, catalogue, receipt, and output table.

## Contract

The JSON contract uses schema `uff.sky-lattice-claim.v1` and freezes:

- ICRS node coordinates;
- spherical-cap radius;
- anomaly column, comparison operator, and threshold;
- holdout, weight, and stratum columns;
- null model, permutation count, and random seed;
- global alpha, minimum effect size, and required supported-node count;
- optionally, the expected SHA-256 of the catalogue.

Once deposited or committed, changing any field produces a different canonical
contract fingerprint.

## Test statistic

For anomaly indicator \(Y_i\in\{0,1\}\), positive weight \(w_i\), and union
node-cap membership \(M_i\), define

\[
\hat p_{\mathrm{in}}=
\frac{\sum_i w_iY_iM_i}{\sum_i w_iM_i},\qquad
\hat p_{\mathrm{out}}=
\frac{\sum_i w_iY_i(1-M_i)}{\sum_i w_i(1-M_i)}.
\]

The preregistered global statistic is

\[
T=\hat p_{\mathrm{in}}-\hat p_{\mathrm{out}}.
\]

The same statistic is calculated separately for every node. Odds ratios are
reported as diagnostics with the Haldane-Anscombe correction, but they are not
used to silently replace the frozen decision statistic.

## Null models

### RA-shift

A common random right-ascension shift is applied to every node. This preserves
node declinations, mutual geometry, cap radius, and the catalogue footprint. It
is appropriate when declination-dependent exposure is important and RA is the
exchangeable direction.

### SO(3) rotation

A uniform random three-dimensional rotation is applied to the complete node
configuration. This preserves all angular separations and is appropriate only
for effectively full-sky, isotropically selected samples or when the selection
function is explicitly represented by weights.

### Stratified-label permutation

Anomaly labels are shuffled within preregistered strata while coordinates and
node masks remain fixed. Strata may encode survey, depth, crowding, Galactic
latitude band, exposure, or quality regime. This tests whether the anomaly label
is unusually associated with the nodes after preserving those factors.

## Empirical p-value

For \(B\) null replicates \(T_b\), the one-sided empirical p-value is

\[
p=\frac{1+\sum_{b=1}^{B}\mathbf 1[T_b\ge T_{\mathrm{obs}}]}{B+1}.
\]

Node-wise p-values are corrected using Holm's step-down family-wise procedure.
A node counts as supported only when its Holm-adjusted p-value is at most the
frozen alpha and its rate contrast reaches the frozen minimum effect.

## Decision

`EMPIRICAL_CRITERIA_MET` requires both:

1. the global test satisfies alpha and minimum effect; and
2. at least the preregistered number of nodes survive Holm correction.

Otherwise the result is `EMPIRICAL_CRITERIA_NOT_MET`.

Neither label means that an entire metaphysical or mathematical framework is
proved or permanently unpatchable. It means the exact empirical claim encoded
in that exact contract did or did not pass.

## Required anti-goalpost rules

- The discovery data cannot also serve as the confirmatory holdout.
- Failed nodes remain in the denominator.
- Null results and excluded rows are retained.
- Thresholds cannot be tuned after inspecting the holdout.
- Catalogue diagnostics are not described as physical objects without an
  independently validated object-level model.
- A timeout, null field, large uncertainty, or poor catalogue fit is not called a
  "pipeline crash" unless an actual operational failure log establishes that.
- Photometric colour differences are not re-labelled as spectral resonances
  without a bandpass-aware spectral model.

## Usage

```bash
python sky_lattice_audit.py \
  --catalog frozen_catalog.csv \
  --contract frozen_claim.json \
  --out outputs/claim-id
```

The command writes:

- a JSON receipt containing hashes, configuration, global result, and boundary;
- a CSV table containing every node, effect, raw p-value, Holm p-value, and pass;
- a JSON artifact manifest containing output SHA-256 hashes.

## Interpretation boundary

UFF-SLFA is intentionally hostile to both credulity and reflexive dismissal. A
claimed pattern that survives a fair, frozen, footprint-aware holdout deserves
further study. A pattern that disappears under those controls must not be
advertised as confirmed by the tested catalogue.
