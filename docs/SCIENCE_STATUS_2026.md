# Scientific status snapshot — 14 July 2026

This document records why UFF v4 separates galaxy dynamics, SMBH strong-field
physics, and LQG-inspired compact-object phenomenology. It is a scope statement,
not a literature review or a claim that one model family has won.

## Galaxy data

The original SPARC release contains 175 nearby disk galaxies with 3.6 μm
photometry, H I/Hα rotation curves, and baryonic mass models. It remains the
default reference dataset for this repository. The ongoing BIG-SPARC program
aims at a much larger, more homogeneous sample; its 2024 project paper describes
about 4,000 planned galaxies, so UFF treats BIG-SPARC compatibility as a data
interface direction rather than claiming the future catalog is complete.

- [Official SPARC database](https://astroweb.case.edu/SPARC/)
- [Lelli, McGaugh & Schombert (2016)](https://arxiv.org/abs/1606.09251)
- [Haubner et al. (2024), BIG-SPARC](https://arxiv.org/abs/2411.13329)

## Dark halos and MOND/RAR

NFW and Burkert remain useful cusped and cored baselines, but fitting a curve is
not the same as testing the full ΛCDM formation history. Likewise, the algebraic
RAR reproduces a strong empirical correlation but is not by itself a complete
relativistic MOND theory.

Recent analyses emphasize that rankings depend on data quality, nuisance
parameters, priors, and the chosen comparison. Desmond's joint SPARC inference
found a characteristic acceleration near `1.19e-10 m/s²`, very small inferred
intrinsic scatter under the adopted error model, and weak evidence for an
external-field effect. A 2024 analysis by Khelashvili, Rudakovskyi and
Hossenfelder reported a preference for cored dark-matter fits over its MOND/RAR
candidate set. Another 2024 study found tension between MOND transition
functions inferred from the galaxy RAR and Solar-System constraints. These are
active statistical and theoretical disputes, not settled facts.

- [McGaugh, Lelli & Schombert (2016), RAR](https://arxiv.org/abs/1609.05917)
- [Desmond (2023), underlying RAR](https://arxiv.org/abs/2303.11314)
- [Khelashvili, Rudakovskyi & Hossenfelder (2024)](https://arxiv.org/abs/2401.10202)
- [Desmond, Hees & Famaey (2024)](https://arxiv.org/abs/2401.04796)
- [Beck et al. (2026), LITTLE THINGS/SPARC comparisons](https://arxiv.org/abs/2605.27217) — recent preprint

UFF's response is methodological neutrality: use correct equations, expose
assumptions, fit the same data, report parameter-bound hits, and avoid turning
relative BIC weights into claims of physical truth.

## Supermassive black holes

At kiloparsec radii an SMBH is accurately represented by a central point mass.
Near the horizon that approximation is not enough. UFF therefore provides Kerr
characteristic scales separately and never plots a Newtonian `sqrt(GM/r)` curve
through the horizon.

The Event Horizon Telescope observations of M87* and Sgr A* opened direct
horizon-scale tests. Current images are consistent with Kerr/GR within their
astrophysical and observational uncertainties; they do not establish that every
alternative effective metric is excluded.

- [EHT Collaboration (2019), M87* first results](https://arxiv.org/abs/1906.11238)
- [EHT Collaboration (2022), Sgr A* first results](https://arxiv.org/abs/2204.01840)

The SMBH mass parameter in a galaxy fit should be interpreted only when the
innermost observations resolve its approximate sphere of influence.

## Loop quantum gravity and black holes

LQG is a quantum-gravity research program, not a single agreed effective black-
hole metric. Effective models differ in quantization choices, covariance,
interior dynamics, and exterior corrections. Galaxy rotation curves occur at
scales where a Planck-area correction such as `Δ/r²` is fantastically small;
UFF therefore does not advertise LQG as a solution to galactic missing mass.

Research since the general 2017 review has investigated potentially observable
compact-object signatures. Examples include non-zero tidal Love numbers for
specific loop-quantized black-hole models and a May 2026 preprint constraining a
rotating holonomy-corrected model with EHT shadow observables. These results are
model-specific and should not be generalized to “LQG predicts” without the
qualifier.

- [Perez (2017), Black Holes in Loop Quantum Gravity](https://arxiv.org/abs/1703.09149)
- [Motaharfar & Singh (2025), Love numbers](https://arxiv.org/abs/2501.09151)
- [Motaharfar & Singh (2025), covariant loop quantum black-hole Love numbers](https://arxiv.org/abs/2505.14784)
- [Ali & Ghosh (2026), rotating holonomy-corrected EHT test](https://arxiv.org/abs/2605.28871) — recent preprint

UFF v4 implements only:

1. an explicit LQG area-gap convention;
2. scale ratios at a user-selected radius; and
3. an opt-in, clearly labelled bookkeeping ansatz for API experiments.

Adding a named effective metric requires its full equation, classical limit,
domain of validity, source citation, and independent tests.

## UFF status

The v4 UFF curve is a bounded empirical family designed for falsifiable
rotation-curve comparisons. It is not yet a unified field theory in the usual
physics sense because the repository does not supply a covariant action, field
equations, stress-energy prescription, lensing sector, cosmological background,
or independent parameter prediction.

That boundary is deliberate. A future theoretical UFF model can be added under
a new name when those structures exist; it should not silently replace the
empirical law.

## What a result can support

A UFF run can support statements such as:

- “Within this candidate set and error model, model A has lower BIC than model B.”
- “The fitted NFW parameters are X and Y under the declared H0 and M/L treatment.”
- “The MOND/RAR residuals show this radial pattern for the chosen a0.”
- “The data do or do not resolve the assumed SMBH sphere of influence.”

It cannot by itself support statements such as:

- “Dark matter/MOND/UFF has been proven.”
- “LQG explains flat galaxy rotation curves.”
- “A good empirical curve fit is a completed fundamental field theory.”
- “Relative BIC weights are posterior probabilities of physical truth.”
