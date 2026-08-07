# Statistical Mechanics Guardrail for UFF

## Purpose

UFF is deterministic where its computational contracts require determinism, but
deterministic replay is not the same claim as statistical representativeness.

This document records a methodological guardrail motivated by quantum
statistical mechanics, especially the distinction between microscopic state,
macroscopic observable, and statistical ensemble, together with J. M. Deutsch's
1991 analysis of statistical mechanics in a closed quantum system.

The guardrail is **not** a claim that UFF's astronomical catalogues, permutation
nulls, or software receipts are quantum many-body systems. No Hilbert-space,
density-matrix, Hamiltonian, thermalization, or eigenstate-thermalization model
is imported into UFF astrophysics by this document.

The useful lesson is narrower:

> **Exact microscopic evolution does not, by itself, establish that a chosen
> statistical ensemble is the correct macroscopic description.**

For UFF:

> **Exact computational replay does not, by itself, establish that a chosen
> null ensemble is the correct statistical description of the data-generating
> process.**

## Source motivation

Introductory quantum statistical mechanics separates microscopic many-particle
states from equilibrium ensembles and macroscopic observables. Density matrices
and reduced density matrices make that distinction explicit: a reduced
observable description can be sufficient for a declared task without retaining
or determining the full microscopic state.

The microcanonical and canonical ensembles likewise correspond to different
macroscopic constraints. Selecting an ensemble is therefore a modelling choice
with physical assumptions; it is not a consequence of merely having exact
microscopic equations.

Deutsch (1991), *Quantum statistical mechanics in a closed system*, gives an
especially useful warning for UFF's philosophy. A closed system with exact
unitary dynamics need not automatically yield time averages agreeing with the
microcanonical distribution. Deutsch discusses violations for uncoupled degrees
of freedom and shows recovery of statistical-mechanical behaviour after a small
generic perturbation in the model studied there.

UFF adopts only the epistemic lesson: **exact evolution and ensemble validity
are separate propositions.**

## The UFF separation rule

UFF therefore keeps four assertions distinct:

1. **Integrity** — are the recorded artifacts byte-identical to the manifest?
2. **Replay** — does the frozen computation reproduce the recorded result?
3. **Ensemble calibration** — does the declared null/resampling family behave
   as required for the inferential claim under relevant synthetic and negative
   controls?
4. **Scientific interpretation** — is the selected model, sampling frame,
   estimand and causal/physical interpretation defensible?

QEC can establish 1 and 2 at the computational boundary.

SPECTRAL-style pre-observation witnessing can establish that specified input
identities did not change between commitment and reveal, provided chronology is
independently anchored when a historical precommitment claim is made.

SONIFICATION-style telemetry can expose recorded diagnostics to a human or
machine receiver.

None of those three mechanisms establishes 3 or 4.

## No `thermalization` shortcut

UFF must not use terms such as `thermalized`, `ETH-compliant`, `microcanonical`,
`canonical`, `density matrix`, or `quantum equilibrium` as decorative labels for
ordinary catalogue resampling.

A UFF permutation distribution is a statistical null distribution defined by a
specific data and transformation contract. It is not a quantum ensemble unless
a separate, physically justified quantum model is explicitly defined and
validated.

Likewise, a deterministic seed sweep is not an analogue of quantum time
evolution in any evidentiary sense. Analogies may be used pedagogically, but
never as proof.

## Practical consequence for Sheridan / SLFA

A bundle may be `REPLAY_VERIFIED` and still fail to deserve a publication-grade
scientific verdict because its null ensemble is poorly calibrated.

Future UFF contracts should therefore reserve a distinct assurance state such as
`ENSEMBLE_CALIBRATED` for cases in which the relevant statistical family has
passed a declared calibration battery. Until such a battery is implemented and
passed, replay verification must remain computational assurance only.

### Minimum calibration requirements

A future ensemble-calibration layer should require, at minimum:

- the null/resampling family frozen before analysis;
- the complete multiplicity family frozen before analysis;
- deterministic seed schedules or independently committed random sources;
- synthetic null cases with measured type-I error;
- synthetic positive controls with measured power/recovery;
- irregular-mask, anisotropy, incompleteness and catalogue-failure stress cases
  relevant to the declared survey;
- stability checks across prespecified seed blocks / simulation batches;
- convergence diagnostics for the reported tail probability or test statistic;
- frozen negative-control geometries and predicates;
- explicit failure when the requested p-value resolution exceeds what the
  permutation count can support;
- separation between exploratory, historical-reproduction, diagnostic and
  prospective-confirmatory verdicts; and
- no promotion from `REPLAY_VERIFIED` to an ensemble-calibrated scientific
  assurance state merely because the same deterministic result reproduces.

These are ordinary statistical and experimental-design requirements. Quantum
statistical mechanics motivates the separation of levels; it does not supply a
shortcut around calibration.

## Coarse observables and reduced descriptions

Quantum statistical mechanics also supplies a useful discipline for UFF's
reporting boundary: a task may legitimately depend on a reduced set of
observables without claiming those observables uniquely determine the entire
underlying system.

Applied to UFF:

- catalogue-level excess noise is an observable, not an object ontology;
- node-cap counts are observables, not physical nodes;
- KDE overdensity is an observable/statistic, not a field measurement;
- model-comparison scores are observables of a fitted model family, not proof of
  the uniquely correct physical theory; and
- a sonified or visualized diagnostic is a receiver representation of an
  observable, not additional evidence.

This is consistent with UFF's existing scientific boundary: reduced summaries
can answer a declared statistical question while remaining radically
insufficient for stronger physical claims.

## Assurance ladder

The intended UFF assurance ladder is therefore:

```text
INPUTS_COMMITTED
      ↓
INTEGRITY_VERIFIED
      ↓
REPLAY_VERIFIED
      ↓
ENSEMBLE_CALIBRATED      (future; must be earned separately)
      ↓
SCIENTIFICALLY_DEFENSIBLE (external scientific judgement; never a hash state)
```

No lower rung implies a higher rung.

In particular:

```text
REPLAY_VERIFIED ≠ ENSEMBLE_CALIBRATED
ENSEMBLE_CALIBRATED ≠ PHYSICAL_TRUTH
```

## Relationship to the UFF defense layers

The complete small-layer architecture is now conceptually:

```text
SPECTRAL witness
  freeze identity-bearing inputs
           ↓
UFF scientific computation
           ↓
QEC boundary gate
  strict integrity + fresh replay
           ↓
statistical-mechanics guardrail
  do not confuse one exact computation
  with validation of its statistical ensemble
           ↓
SONIFICATION telemetry
  receiver-neutral observation of the result
```

The statistical-mechanics guardrail is deliberately non-executable in the
current UFF v5 contract family. It constrains interpretation and specifies what
a future calibration implementation must earn.

## References

- J. M. Deutsch, "Quantum statistical mechanics in a closed system," *Physical
  Review A* **43**, 2046–2049 (1991). DOI: 10.1103/PhysRevA.43.2046.
- G. Basti and S. Cenatiempo, *An invitation to Quantum Statistical Mechanics*,
  lecture notes, last updated 12 December 2023.
- M. D. Schwartz, *Lecture 10: Quantum Statistical Mechanics*, Statistical
  Mechanics, Harvard University, Spring 2021.
- "Quantum statistical mechanics," Wikipedia, used only as a general orientation
  source; primary/lecture sources above govern the methodological statements.

## Final boundary

The value of quantum statistical mechanics here is not that it makes UFF
"more quantum." It does the opposite: it gives us a precise reason **not** to
confuse exact microscopic/computational behaviour with justified ensemble-level
inference.

That distinction belongs directly beside UFF's reproducibility and
falsification machinery.
