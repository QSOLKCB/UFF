# Zenodo upload guide — QSOL UFF v5.3.0

The v5.3.0 record metadata is prepared, but a new Zenodo version DOI has **not** been assigned in this repository.

Do not reuse the immutable v5.2.0 DOI `10.5281/zenodo.21911644` as the v5.3.0 identifier.

## Release binding steps

1. Merge the v5.3.0 release changes only after Python CI and the existing Lean formal workflow are green on the final review head.
2. Record the exact merged commit SHA.
3. Confirm post-merge CI is green for that exact commit.
4. Create the GitHub `v5.3.0` tag from that exact merged commit.
5. Verify the tag resolves to the intended commit.
6. Create a **new version** of the existing UFF Zenodo record and use `metadata.json` in this directory as the metadata source.
7. Allow Zenodo to assign the new version DOI. Do not predeclare or guess it.
8. Bind the assigned DOI back into release-facing citation/package metadata in a follow-up metadata commit if required by the publication workflow.
9. Upload the exact GitHub-generated source archive for tag `v5.3.0`.
10. Confirm the Zenodo description preserves both UFF boundaries:

```text
REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH
EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT != PHYSICAL_TRUTH
```

11. Confirm the Jampolski & Rezzolla paper is represented as a **reference used for nonclaim calibration**, not as empirical evidence for gravastars.

## Suggested archival payload

- GitHub-generated source archive for tag `v5.3.0`
- `RELEASE_NOTES_v5.3.0.md`
- `docs/NONCLAIM_CALIBRATION.md`
- `examples/nonclaim_reference_gravastar_2026.json`
- `formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md`
- `zenodo/v5.3.0/metadata.json`
- CI results from the exact release commit

## Exact release identity

The Zenodo record, GitHub tag, release notes, and archived source must all refer to the same exact merged commit. A DOI assignment alone is not proof of that binding.

Do not publish placeholder checksums for a commit that is not yet frozen. Generate archival checksums only after the final release tree exists.
