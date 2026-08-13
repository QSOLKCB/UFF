# Zenodo upload guide — QSOL UFF v5.2.0

The v5.2.0 Zenodo version DOI has been assigned:

`10.5281/zenodo.21911644`

DOI assignment is an identifier event; it is not by itself proof of the exact Git commit archived by the release. Keep the release identity gate below intact.

## Remaining release binding steps

1. Merge the v5.2.0 release PR only after Python CI and `UFF Lean 4 formal verification` are green on the final PR head.
2. Record the exact merged commit SHA.
3. Confirm the post-merge `main` run of `UFF Lean 4 formal verification` is green for that merged commit.
4. Create the GitHub `v5.2.0` tag from that exact merged commit.
5. Verify the tag resolves to the intended commit.
6. Use `metadata.json` in this directory as the deposit metadata source and ensure the Zenodo record uses DOI `10.5281/zenodo.21911644`.
7. Upload or finalize the GitHub release/source archive and, if desired, the formal verification report captured from the exact release commit.
8. Confirm the Zenodo description preserves the scientific boundary and does not imply that Lean proves physical truth.
9. Record the exact archived/tagged commit alongside the DOI in the final release notes or publication handoff.

## Exact release identity

The formal CI is rooted in UFF v5.1.0 base commit:

`136b63089f5b70cd2a7356b6f647d482f3b3273c`

The final Zenodo v5.2.0 deposit must bind to the exact merged v5.2.0 commit, not merely to a branch name or to the existence of the DOI.

## Suggested archival payload

- GitHub-generated source archive for tag `v5.2.0`
- `RELEASE_NOTES_v5.2.0.md`
- `docs/UFF_FORMAL_ASSURANCE_V5.2.md`
- `formal/lean/README.md`
- `formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md`
- `formal/lean/RUNTIME_CORRESPONDENCE.md`
- `formal/lean/THEOREMS.md`
- CI formal verification report from the exact release commit

Do not publish placeholder checksums for a commit that has not yet been merged. Generate any archival checksums only after the exact release tree is frozen.
