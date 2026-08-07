# Independent Research Assessment of the Logvinovich Claim and Sheridan Audit

**Repository rendition of the independent assessment supplied on 7 August 2026**  
**Author metadata in source PDF:** ChatGPT Deep Research  
**Authoritative source identity:** `sha256:4af1ba265770b88b41d70d08b93eb73c2b1ff3992b0b041431deea40f1a4ea07`

## Executive assessment

The central diagnosis is substantially correct: the evaluation framework is much closer to formalisation than the underlying Logvinovich empirical claim. The existing ten-part contract captures many essential ingredients of a falsifiable spatial-catalogue claim, and preventing unresolved fields from being executed is methodologically sound.

The framework can represent and replay a specified test, subject to independent implementation verification. It cannot yet run one uniquely attributable “Logvinovich test” because the public record does not identify one stable node set, radius, anomaly definition, sampling frame or null procedure.

The project is not merely one claimant clarification away from full scientific readiness. A clarification packet may make a retrospective contract syntactically executable, but confirmation additionally requires:

1. an independently defined sampling frame;
2. a calibrated survey-selection and astrophysical-background model;
3. an untouched dataset or genuinely protected holdout;
4. a prespecified statistic, estimand and multiplicity family;
5. implementation validation under synthetic nulls and known injected alternatives.

The strongest summary is:

> The crucible’s syntax is largely formalised. Its statistical calibration still needs validation, and the claimant has not supplied one stationary claim to place inside it.

## Main methodological findings

### Formalisation achieved

The current framework usefully separates geometry, radius, anomaly predicate, catalogue, survey support, holdout, null generator, significance threshold, effect threshold and required node support. It also correctly preserves incompatible historical claim versions rather than allowing an auditor to choose the most convenient one after observing outcomes.

Injection recovery and deterministic replay are valuable controls, but neither proves that real data contain the injected structure or that the chosen null model is scientifically valid. A reproducible workflow can reproduce biased sampling or a misspecified null exactly.

### Geometry

The ideal twelve-node frame, incomplete rotated E-node list and thirty-six-node hierarchy are materially different hypotheses. Missing E-7 and E-12 are fatal to executing that rotated twelve-node version as a claimant-frozen hypothesis. The audit must not infer those coordinates from symmetry unless clearly labelled as an exploratory auditor reconstruction.

A rigid architecture should generally be rotated as one body under Haar-uniform `SO(3)` draws. Independently randomising nodes destroys the claimed geometry and tests a different hypothesis.

### Radius and multiplicity

Testing `1.5°`, `3°` and `5°` is not inherently improper. Selecting the strongest radius after inspection without calibrating the family is improper. The multiplicity family includes not only radii, but node lists, catalogues, thresholds, baseline cuts, match radii and endpoints.

A defensible design freezes one primary radius and treats others as sensitivity analyses, or repeats the full maximisation inside every null replicate using a max-statistic.

### Anomaly definitions

Large proper motion and large proper-motion uncertainty are different populations with different interpretations. Predicates must freeze units, null handling, release/schema version, deduplication, blending rules and quality diagnostics.

`W3-W4 > 1.5` is executable but not sufficient as an astrophysical anomaly definition without signal-to-noise, non-detection, saturation, contamination, extension and source-class rules.

A Gaia `astrometric_excess_noise` threshold must be accompanied by magnitude, colour, crowding, scanning and other quality controls. RUWE and related diagnostics are robustness checks, not interchangeable post-hoc replacements.

### Sampling and estimands

A target-restricted query produces an in-cap count. It does not estimate either `P(anomaly | inside)` or `P(inside | anomaly)` without an eligible denominator or full anomaly sampling frame.

A suitable primary estimand may be a rate ratio, odds ratio or excess density. A full-sky download is not always required: probability-sampled controls, matched caps, case-control designs or inverse-probability weighting can be valid when inclusion probabilities are declared.

### Survey support and astrophysical background

Survey footprint and completeness are only one layer. Ordinary sky source density, extinction, crowding and expected measurement-failure structure are another. The contract should distinguish:

- `S_survey`: where and how well the survey could observe;
- `S_astro`: the expected astrophysical population and background failure intensity.

A uniform sky rotation is not automatically a valid null for an anisotropic astronomical catalogue. The null must preserve or condition on the principal nuisance structure.

### Monte Carlo significance

For `M` null simulations and `B` values at least as extreme as observed, the empirical p-value is:

```text
p_hat = (B + 1) / (M + 1)
```

With 100,000 ordinary draws and zero exceedances, the directly resolved value is approximately `9.9999e-6`, not zero and not `1e-300`. A substantially smaller claim requires a separately declared exact calculation, importance sampler, parametric tail model or analytic extrapolation.

The generator, version, initialisation, sampling algorithm and complete statistic must be frozen. Conclusions should also be stable across independent seed streams.

### Holdout integrity

Existing public catalogue work is discovery analysis because filters and queries changed while results were visible. A later contract cannot make those observations blind retroactively.

A holdout declaration must document both data independence and analyst blindness. The strongest design uses an external custodian who receives the frozen contract, executes it against a protected partition and returns only preregistered outputs, with contract, container and holdout-manifest hashes published before execution.

## Contract delta

The assessment recommends extending:

```text
C = (N, r, A, D, S, H, N_null, alpha, delta, k)
```

to:

```text
C* = (P, N, r, A, D, S, H, T, E, M, N_null, alpha, delta, k, Q, R)
```

where:

| Field | Required content |
|---|---|
| `P` | provenance and historical claim version |
| `T` | exact test statistic and node-aggregation function |
| `E` | estimand: rate ratio, odds ratio, excess density, etc. |
| `M` | complete multiplicity family and correction rule |
| `Q` | quality, deduplication, cross-match and missing-data rules |
| `R` | reproducible environment, software hash and RNG specification |

Any unresolved primary field should return a terminal validation error such as:

```text
CONTRACT_NOT_EXECUTABLE:
  unresolved_primary_fields = [
    canonical_node_table,
    primary_endpoint,
    gaia_predicate,
    null_sampling_measure
  ]
```

Failure to specify a claim is not empirical falsification.

## Recommended prospective design

A clean confirmatory programme should freeze one primary rigid node table, one radius, one catalogue-specific anomaly predicate, one estimand and one global endpoint. Secondary radii, catalogues and predicates should be labelled before data access.

A model-based endpoint could estimate a node effect after prespecified adjustment for Galactic position, ecliptic latitude, depth, cadence, magnitude and crowding. Spatial dependence should be handled through block-level inference, matched regional randomisation or another calibrated method.

The complementary design-based test should rotate the complete architecture against the fixed catalogue while preserving survey support and matching major sky covariates. Every null replicate must repeat the entire analysis, including radius selection, node aggregation and catalogue combination.

Confirmation should require statistical and practical significance, a globally calibrated node pattern, successful injection recovery and negative controls that do not show comparable enrichment.

## Retrospective reproduction boundary

A historical `(270°, 45°)`, `1.5°` NSC query is a reasonable reproduction target. Its outputs must distinguish:

```text
HISTORICAL_QUERY_REPRODUCED
HISTORICAL_QUERY_NOT_REPRODUCED
DIAGNOSTIC_FALSIFIER_TRIGGERED
DIAGNOSTIC_FALSIFIER_NOT_TRIGGERED
```

Reproducing a row count or isotropic position-angle distribution is not equivalent to `CRUCIBLE_CRITERIA_MET`. Isotropy rejects only one narrow coherent-drift alternative; random noise, crowding, confusion and heterogeneous calibration effects may also lack one coherent direction.

## Final judgement

The audit framework is sufficiently developed to preserve and evaluate a genuinely frozen specification, but the current public material supports only versioned retrospective reconstructions. Claimant clarification can remove semantic ambiguity; it cannot replace an independent sampling frame, calibrated null model or untouched confirmation set.

## Principal sources cited by the assessment

- Budavári, T. & Szalay, A. S. (2008), “Probabilistic Cross-Identification of Astronomical Sources,” *The Astrophysical Journal*, 679, 301–309.
- Cutri, R. M. et al. (2013), *Explanatory Supplement to the AllWISE Data Release Products*.
- Fabricius, C. et al. (2021), “Gaia Early Data Release 3: Catalogue validation,” *Astronomy & Astrophysics*, 649, A5.
- Holm, S. (1979), “A Simple Sequentially Rejective Multiple Test Procedure,” *Scandinavian Journal of Statistics*, 6, 65–70.
- Horvitz, D. G. & Thompson, D. J. (1952), “A Generalization of Sampling Without Replacement from a Finite Universe,” *Journal of the American Statistical Association*, 47, 663–685.
- Lindegren, L. et al. (2021), “Gaia Early Data Release 3: The astrometric solution,” *Astronomy & Astrophysics*, 649, A2.
- Nidever, D. L. et al. (2021), “The Second Data Release of the NOIRLab Source Catalog,” *The Astronomical Journal*, 161, 192.
- Nosek, B. A. et al. (2018), “The Preregistration Revolution,” *PNAS*, 115, 2600–2606.
- Phipson, B. & Smyth, G. K. (2010), “Permutation P-values Should Never Be Zero,” *Statistical Applications in Genetics and Molecular Biology*, 9, Article 39.
- Rubin, D. B. (2007), “The Design versus the Analysis of Observational Studies for Causal Effects,” *Statistics in Medicine*, 26, 20–36.
- Westfall, P. H. & Young, S. S. (1993), *Resampling-Based Multiple Testing*, Wiley.

## Source boundary

This Markdown file is a repository-native rendition of the supplied 11-page PDF. The PDF identified by the SHA-256 above remains the authoritative source artifact. The rendition preserves the assessment’s core conclusions and implementation recommendations but is not a byte-for-byte textual export.
