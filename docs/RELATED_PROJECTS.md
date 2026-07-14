# Related QSOL project interoperability

UFF v4 was audited against three related QSOL repositories on 14 July 2026.
They do not contain a drop-in galaxy gravity law, so their useful ideas are
connected through diagnostics rather than inserted into the likelihood.

## QAI-UFT

Repository: <https://github.com/QSOLKCB/QAI-UFT>

Useful element: the Tensor Phase Cube's deterministic π/2 phase convention.

UFF use:

- standardized residuals are normalized and encoded as unit complex phases;
- the fingerprint is written to the summary JSON; and
- the optional stereo WAV uses sine/cosine quadrature.

Boundary: phase encoding is a provenance and sonification transform. It is not
a gravitational degree of freedom or evidence of self-duality in the galaxy.

## QNTOY

Repository: <https://github.com/QSOLKCB/QNTOY>

Useful element: normalized Shannon entropy telemetry on a three-state field.

UFF use:

- relative BIC weights are normalized onto a simplex;
- Shannon entropy is divided by `log(number_of_models)`; and
- the value summarizes ambiguity within the declared candidate set.

Boundary: this is ordinary information entropy, not von Neumann entropy, a
quantum state, or evidence that the candidate models form a physical ensemble.

## TFT

Repository: <https://github.com/QSOLKCB/TFT>

Useful element: basis-invariant tensor diagnostics under orthogonal congruence.

UFF use:

- local fit covariance matrices are symmetrized;
- rank, trace, Frobenius norm, and sorted eigenvalues are reported; and
- tests confirm eigenspectrum and Frobenius norm are unchanged by an orthogonal
  change of parameter basis.

Boundary: parameter units still matter. These invariants compare one declared
parameterization with its rotations; they do not make different models or units
automatically commensurate.

## Provenance and licensing

The UFF implementations use standard mathematical definitions and were written
independently for this repository. No source file was copied from the related
repositories. The links above credit the conceptual project lineage while
keeping UFF's Apache-2.0 code boundary clear.
