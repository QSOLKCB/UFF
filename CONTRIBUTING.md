# Contributing

Contributions are welcome when they preserve the repository's central rule:
every physical claim must be narrower than or equal to what the code and source
literature establish.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Adding a model

A new model must include:

1. a unique, descriptive CLI name;
2. an explicit status (`baseline`, `phenomenological`, or `research model`);
3. equations, units, parameters, and domain of validity in `docs/MODELS.md`;
4. a primary-source citation;
5. tests for dimensional behavior and at least two analytic limiting cases;
6. a test showing finite, non-negative predictions over its declared domain;
7. an explicit classical/standard limit if it modifies an existing theory; and
8. no suggestion that curve-fitting alone validates a fundamental theory.

For a named LQG effective metric, also provide the full metric, quantization
convention, parameter mapping, horizon/classical limits, and a clear statement
that results are model-specific.

## Data changes

- Preserve the signed SPARC gas convention.
- Never silently fill required observations or uncertainties.
- Record transformations in the JSON configuration and input receipt.
- Do not commit third-party datasets unless their license permits redistribution.

## Statistical changes

- Compare candidates on identical observations and likelihoods.
- Count every fitted nuisance parameter in AIC/AICc/BIC.
- Label approximate covariance errors as local approximations.
- Do not call information-criterion weights posterior theory probabilities.
- Add synthetic-recovery tests before introducing a new optimizer or sampler.

## Pull requests

Keep commits focused. Include:

- a concise scientific and software rationale;
- tests run and their results;
- any changed model assumptions;
- before/after output for user-visible changes; and
- updated documentation and changelog entries.
