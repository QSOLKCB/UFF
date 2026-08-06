# UFF-SLFA v1.0.0
## A Preregistered Sky-Lattice Falsification Audit

**Trent Slade, QSOL-IMC**  
**7 August 2026**

### Abstract

Claims that astronomical catalogue anomalies cluster around a rigid celestial lattice are straightforward to state but unusually easy to test circularly. A query can select high-error records inside proposed node windows and then report that the selected records occur at the nodes. UFF-SLFA formalizes a neutral counter-framework that prevents this failure mode. A machine-readable contract freezes node coordinates, cap radius, anomaly rule, data split, null model, multiplicity correction, effect threshold, and decision rule before confirmatory analysis. The implementation compares anomaly rates inside and outside the frozen caps under footprint-aware right-ascension shifts, proper SO(3) rotations, or stratified label permutations. Every geometric null transform is checked for proper-rotation and lattice-invariance residuals, following the reproducibility discipline developed in QSOL TFT. Runs produce hashed recipe, observation, node-table, and manifest artifacts and support independent numerical replay. The result applies only to the exact catalogue-level association claim. It neither proves a physical ontology when positive nor claims that no later modified theory can be written when negative.

### 1. Problem

A celestial-node claim has three separable layers:

1. a mathematical configuration of fixed sky coordinates;
2. an empirical claim that a declared anomaly occurs unusually often near those coordinates; and
3. a physical interpretation of any association.

Only the second layer is tested here. Mixing the layers allows a valid geometry to be used as rhetorical protection for an invalid empirical test, or an interesting catalogue association to be promoted directly into a field theory.

The central anti-circularity rule is simple: catalogue selection must be independent of the proposed node masks. A node-targeted sample cannot validate concentration at the nodes. Confirmatory evaluation also requires an independent catalogue or an untouched holdout split.

### 2. Frozen contract

The contract records ICRS coordinates, spherical-cap radius, anomaly threshold, catalogue hashes, holdout and weighting columns, null model, random seed, permutation count, family-wise alpha, minimum effect, and required supported-node count. The canonical contract hash changes if any field changes.

For anomaly indicator `Y_i`, positive weight `w_i`, and membership `M_i` in the union of node caps, define

`p_in = sum(w_i Y_i M_i) / sum(w_i M_i)`

`p_out = sum(w_i Y_i (1-M_i)) / sum(w_i (1-M_i))`

and use `T = p_in - p_out` as the preregistered global statistic. The identical contrast is calculated for each node.

### 3. Selection-aware nulls

UFF-SLFA supports three null families.

**RA shift.** A common random rotation about the ICRS z-axis preserves declinations, cap radii, and mutual node geometry while moving the configuration through right ascension.

**SO(3) rotation.** A Haar-uniform proper rotation preserves the full internal geometry of the lattice. It is appropriate for effectively full-sky data or explicitly modelled selection functions.

**Stratified label permutation.** Anomaly labels are shuffled only within declared survey or quality strata, preserving coordinates and the stratified anomaly burden.

For `B` null replicates the one-sided empirical p-value is

`p = (1 + count(T_b >= T_obs)) / (B + 1)`.

Node-wise p-values are corrected using Holm's step-down family-wise procedure. A node survives only if its adjusted p-value and effect size both meet the frozen criteria.

### 4. TFT invariance bridge

A null rotation must move the lattice without changing it. For every geometric replicate the implementation checks

`R^T R = I` and `det(R) = +1`,

then verifies invariance of the node Gram matrix and all pairwise angular separations. The run records the maximum residual observed across all null transformations. This is a transformation audit, not evidence that the proposed celestial structure is physically real.

### 5. Decision and evidence bundle

A positive decision requires the global test to meet alpha and minimum effect and at least the preregistered number of nodes to survive Holm correction. Every node remains visible, including failures and untestable sparse caps.

The bundle separates requested computation from observed result:

- `recipe.json` stores the frozen contract and input hashes;
- `observations.json` stores the decision and numerical diagnostics;
- `nodes.csv` stores complete node-wise results;
- `manifest.json` closes the bundle with byte sizes and SHA-256 hashes.

Verification first checks integrity and then, when supplied with the exact frozen catalogue, reruns the analysis and compares the recorded observations. Integrity and replay establish computational consistency, not physical truth.

### 6. Interpretation boundary

UFF-SLFA deliberately refuses two symmetrical overclaims. A positive catalogue association does not prove a quantum vacuum crystal, standing wave, topological defect, or other ontology. A negative result does not establish that an author could never modify a theory. It rejects the exact empirical claim encoded in the frozen contract.

Operational database outcomes also require disciplined language. Missing values, high uncertainty, excess-noise fields, morphology-fit diagnostics, and query timeouts are not physical objects or supercomputer crashes by default. Broadband colour differences are not spectral resonances without a bandpass-aware spectral model.

### 7. Conclusion

The framework replaces personality contests and self-adjudicated bounties with an executable claim contract. The decisive question is no longer whether a critic can prove an entire research programme permanently unpatchable. It is whether a specific, frozen prediction survives an independent, selection-aware, replayable test.

### Software and related work

Reference implementation: QSOLKCB/UFF, module `uff.sky_audit`. The rotation-invariance and artifact-boundary design is conceptually bridged from QSOLKCB/TFT: *Reproducible Tensor Invariance and Quadrature Sonification*. UFF-SLFA reimplements the required mechanisms under the UFF Apache-2.0 codebase rather than copying TFT's CC BY 4.0 implementation.

### References

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.

Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Bioinformatics*, 26(2), 250-251.
