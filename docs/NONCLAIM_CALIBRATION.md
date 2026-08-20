# Nonclaim calibration

UFF uses **nonclaims** to prevent a result from acquiring more epistemic authority than its source actually earns.

A nonclaim is not a criticism of a source. It is an explicit boundary around what a result does **not** establish. UFF treats those boundaries as first-class provenance because many scientific overstatements arise from silent promotion between logically different questions.

## Core rule

```text
EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT != PHYSICAL_TRUTH
```

A result may establish one rung while leaving every later rung open.

## Calibration dimensions

| Dimension | Question |
|---|---|
| `existence` | Does at least one admissible construction, trajectory, solution, or dataset realization exist? |
| `robustness` | Does the result persist under perturbations of assumptions, parameters, numerics, or data? |
| `genericity` | Does it arise across a broad, non-fine-tuned region of admissible initial conditions or models? |
| `prevalence` | How often should it occur under a justified measure, ensemble, population, or generative process? |
| `empirical_support` | Is there observational or experimental evidence that the modeled phenomenon occurs in nature? |
| `causal_interpretation` | Does the evidence identify the proposed cause rather than merely an association or compatible mechanism? |
| `universality` | Is a model-bound limit or relation established beyond the model class from which it was derived? |
| `predictive_direction` | Was the result obtained by forward prediction from independently specified initial conditions, or by target-conditioned/backward construction? |

## Forbidden promotions

UFF does not silently promote:

- `can occur` into `is likely to occur`;
- `one construction exists` into `the construction is generic`;
- `fine-tuned success` into `robust success`;
- `static equilibrium` into `dynamical stability`;
- `model-bound limit` into `universal physical law`;
- `backward target-conditioned reconstruction` into `forward predictive genericity`;
- `simulation or analytic possibility` into `empirical occurrence`;
- `association` into `causation`; or
- `formal consistency` into `physical truth`.

## Reference calibration: Jampolski & Rezzolla (2026)

The worked reference record in [`examples/nonclaim_reference_gravastar_2026.json`](../examples/nonclaim_reference_gravastar_2026.json) uses:

> Daniel Jampolski and Luciano Rezzolla, **“Formation of gravastars,”** arXiv:2509.15302v2, 11 June 2026.

Source: <https://arxiv.org/abs/2509.15302>

The paper is useful for UFF because the authors themselves keep several claim boundaries visible. They demonstrate gravastar formation in a specified general-relativistic construction **under fine-tuned conditions**, use backward integration to identify successful conditions, and explicitly leave more realistic assumptions and the likelihood of gravastar versus black-hole formation to future study.

UFF therefore records the paper as a **nonclaim calibration reference**, not as evidence that astrophysical gravastars are common, generic, observationally established, or physically preferred.

The reference is particularly useful for distinguishing:

```text
POSSIBLE_WITHIN_MODEL
        !=
ROBUST_UNDER_PERTURBATION
        !=
GENERIC_FROM_INITIAL_DATA
        !=
LIKELY_IN_A_POPULATION
        !=
OBSERVED_IN_NATURE
```

## Source fidelity

Nonclaim records must separate three things:

1. what the source explicitly demonstrates;
2. what the source explicitly leaves open or qualifies; and
3. UFF's own conservative interpretation of what must not be inferred.

UFF must not attribute a stronger disclaimer to an author than the source supports. When a boundary is an UFF inference rather than an explicit author statement, the machine-readable record marks it as such.

## Role in UFF

This calibration layer is epistemic infrastructure only. It does not change galaxy likelihoods, compact-object calculations, SLFA contracts, Sheridan contracts, replay semantics, evidence admission, or the Lean theorem surface.

Its job is narrower and important: make **absence of warrant** machine-readable before prose turns possibility into certainty by grammatical osmosis.
