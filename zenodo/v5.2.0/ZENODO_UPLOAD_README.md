# Zenodo upload guide — QSOL UFF v5.2.0

Use Zenodo's **New version** action on the existing UFF software record. Do not edit or replace an earlier immutable version.

## Before publishing

1. Merge the v5.2.0 release PR only after Python CI and `UFF Lean 4 formal verification` are green.
2. Record the exact merged commit SHA.
3. Create the GitHub `v5.2.0` tag from that exact merged commit.
4. Verify the tag resolves to the intended commit.
5. Use `metadata.json` in this directory as the deposit metadata source.
6. Upload the GitHub release/source archive and, if desired, the formal verification report captured from CI.
7. Confirm the Zenodo description preserves the scientific boundary and does not imply that Lean proves physical truth.
8. Publish the new version and record the newly assigned v5.2.0 version DOI.
9. Patch `CITATION.cff`, README, package URLs, and release links with the assigned DOI in a post-publication metadata-only follow-up if needed.

## Exact release identity

The formal CI is rooted in UFF v5.1.0 base commit:

`136b63089f5b70cd2a7356b6f647d482f3b3273c`

The final Zenodo v5.2.0 deposit must bind to the exact merged v5.2.0 commit, not merely to a branch name.

## Suggested archival payload

- GitHub-generated source archive for tag `v5.2.0`
- `RELEASE_NOTES_v5.2.0.md`
- `docs/UFF_FORMAL_ASSURANCE_V5.2.md`
- `formal/lean/README.md`
- `formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md`
- `formal/lean/RUNTIME_CORRESPONDENCE.md`
- `formal/lean/THEOREMS.md`
- CI formal verification report from the exact release candidate commit

Do not publish placeholder checksums for a commit that has not yet been merged. Generate any archival checksums only after the exact release tree is frozen.
