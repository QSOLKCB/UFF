# Independent Assessment Response — 7 August 2026

**Status:** accepted methodological review; implementation follow-up required  
**Assessment:** *Independent Research Assessment of the Logvinovich Claim and Sheridan Audit*  
**Purpose:** convert the external review into an auditable formalisation roadmap without misrepresenting the assessment as independent verification of UFF code or of the underlying empirical claim

## Wake-up call

The review's central judgement is accepted:

> The crucible's syntax is largely formalised. Its statistical calibration still needs validation, and the claimant has not supplied one stationary claim to place inside it.

That distinction is now the governing project boundary. A syntactically executable retrospective contract is not the same thing as a valid confirmatory experiment. Claimant clarification can resolve semantic ambiguity; it cannot create an independent sampling frame, a scientifically adequate null model, or an untouched holdout.

## What the assessment validates

The review agrees that the existing architecture has made real progress:

- the claim-expression language is strong and near complete;
- unresolved primary fields are prevented from becoming executable claims;
- provenance and historical-version separation are methodologically sound;
- injection recovery, deterministic replay and evidence bundles are useful controls;
- proof-assistant work is currently lower priority than simulation calibration, property testing, code review and frozen environments.

The review does **not** independently certify the implementation, the source ledger, the reported GitHub state, or any Logvinovich numerical claim. Those remain separate verification tasks.

## Accepted formalisation delta

The current ten-field abstraction

```text
C = (N, r, A, D, S, H, N_null, alpha, delta, k)
```

should be extended for publication-grade scientific execution to:

```text
C* = (P, N, r, A, D, S, H, T, E, M, N_null, alpha, delta, k, Q, R)
```

where:

| Field | Required content |
|---|---|
| `P` | provenance and immutable historical claim version |
| `T` | exact primary statistic and node-aggregation function |
| `E` | estimand, such as rate ratio, odds ratio or excess density |
| `M` | full multiplicity family and correction/max-statistic rule |
| `Q` | quality, deduplication, cross-match and missing-data rules |
| `R` | execution environment, software/container hash and RNG specification |

The existing `S` field must also be decomposed conceptually into:

- `S_survey`: footprint, masks, exposure, completeness, depth and cadence;
- `S_astro`: ordinary astrophysical source density, crowding, extinction and expected catalogue-failure structure.

A footprint mask answers where an instrument could observe. It does not by itself model what the ordinary sky should look like.

## Response matrix

| Review finding | Response | State |
|---|---|---|
| Syntax is ahead of scientific readiness | Adopt as the top-level readiness statement | Accepted |
| One claimant packet is insufficient for confirmation | Separate retrospective execution from prospective confirmation | Accepted |
| Freeze statistic and estimand explicitly | Add `T` and `E` to the next contract schema | Planned - P0 |
| Freeze the complete analysis family | Add `M`, including radii, node sets, predicates, catalogues and endpoint family | Planned - P0 |
| Freeze data-quality and cross-match rules | Add `Q` with units, null handling, flags, deduplication and release identifiers | Planned - P0 |
| Freeze environment and RNG | Add `R` with software hash, dependency lock, generator/version and seed-stream policy | Planned - P0 |
| Reject unresolved primary fields terminally | Preserve `CONTRACT_NOT_EXECUTABLE`; never convert missing specification into a negative scientific verdict | Partially implemented; hardening planned |
| Distinguish survey support from astrophysical background | Extend Sheridan nuisance contracts and simulation fixtures | Planned - P1 |
| Validate type-I error under anisotropic skies | Add a calibration harness using irregular masks, spatially varying source density and catalogue-specific failures | Planned - P1 |
| Correct the whole search path, not only nominal radii | Add max-statistic/randomisation support for declared correlated families | Planned - P1 |
| Monte Carlo p-values must use plus-one correction | Retain `(B+1)/(M+1)` and prohibit zero or unsupported extrapolated p-values | Already implemented in core; add reporting guard |
| Historical query reproduction is not confirmation | Add separate reproduction and diagnostic verdict namespaces | Planned - P1 |
| Isotropic position angles reject only one narrow systematic | Encode as a diagnostic, never as positive proof of topology | Accepted; ledger boundary retained |
| Holdout requires data independence and analyst blindness | Add protected-holdout manifest and external-custodian execution design | Planned - P2 |
| Cross-catalogue observations are not automatically independent | Require declared shared-source and shared-systematic dependency structure | Planned - P2 |

## Required verdict separation

Retrospective reproduction must produce verdicts distinct from the prospective crucible:

```text
HISTORICAL_QUERY_REPRODUCED
HISTORICAL_QUERY_NOT_REPRODUCED
DIAGNOSTIC_FALSIFIER_TRIGGERED
DIAGNOSTIC_FALSIFIER_NOT_TRIGGERED
CONTRACT_NOT_EXECUTABLE
```

None of those is equivalent to:

```text
CRUCIBLE_CRITERIA_MET
```

A historical row count or isotropic angle distribution can be reproduced while the causal interpretation remains unsupported.

## Monte Carlo reporting boundary

For `M` null simulations and `B` statistics at least as extreme as observed, the empirical p-value must be reported as:

```text
p_hat = (B + 1) / (M + 1)
```

With 100,000 ordinary null draws and zero exceedances, the finest directly resolved empirical value is approximately `9.9999e-6`, not zero and not `1e-300`. More extreme claims require a separately declared and validated exact calculation, importance sampler, parametric tail model or analytic extrapolation.

## Implementation sequence

### P0 — contract hardening

1. Introduce `uff.sheridan-crucible.v2` with `P`, `T`, `E`, `M`, `Q` and `R`.
2. Add terminal validation with an explicit `unresolved_primary_fields` list.
3. Bind `delta` to the scale of `E` and bind `k` to a globally calibrated decision rule.
4. Freeze catalogue release, table/schema version, units, query text and query hash.

### P1 — statistical calibration

1. Add synthetic anisotropic-sky generators with survey masks and spatially varying failure rates.
2. Measure false-positive control for the complete decision pipeline, not isolated functions.
3. Add max-statistic calibration for radius and endpoint families.
4. Add negative-control geometries, predicates and matched fields.
5. Add historical reproduction verdicts and CLI output separation.

### P2 — prospective confirmation operations

1. Define a protected-holdout manifest containing both data-independence and analyst-blindness declarations.
2. Hash the frozen contract, container and holdout manifest before execution.
3. Support external-custodian execution that returns only preregistered outputs.
4. Record cross-catalogue object overlap and systematic-dependency assumptions.

## Scientific boundary

The assessment strengthens the audit programme; it does not validate the extraordinary claim. The immediate deliverable is a more precise formalisation target and a sharper separation among:

1. historical claim preservation;
2. retrospective query reproduction;
3. diagnostic testing;
4. survey-corrected enrichment analysis;
5. prospective confirmatory execution.

The wake-up call is therefore constructive: the machinery is real, but it must now demonstrate calibrated error control under realistic skies before anyone mistakes a reproducible workflow for a validated scientific inference.
