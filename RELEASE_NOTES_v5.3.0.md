# QSOL UFF v5.3.0 — Release notes

**Release date:** 20 August 2026  
**Theme:** Nonclaim calibration and evidence-scope discipline

UFF v5.3.0 is an additive epistemic-assurance release. It retains the v5.2.0 scientific implementation, defense-in-depth evidence workflow, and Lean 4 formal assurance layer while adding a reusable nonclaim calibration vocabulary and a worked scientific reference showing how to keep model possibility separate from robustness, genericity, prevalence, empirical support, and physical truth.

No astrophysical likelihood, catalogue schema, SLFA contract, Sheridan contract, QEC gate, replay rule, or Lean theorem is redefined by this release.

## Highlights

### Nonclaim calibration vocabulary

Added [`docs/NONCLAIM_CALIBRATION.md`](docs/NONCLAIM_CALIBRATION.md) with the governing separation:

```text
EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT != PHYSICAL_TRUTH
```

The calibration layer also distinguishes causal interpretation, universality, predictive direction, and static equilibrium from general dynamical stability.

### Worked gravastar reference

Added [`examples/nonclaim_reference_gravastar_2026.json`](examples/nonclaim_reference_gravastar_2026.json), a machine-readable calibration record for:

Daniel Jampolski and Luciano Rezzolla, **“Formation of gravastars,”** arXiv:2509.15302v2, 11 June 2026.

The paper is used because it provides a clean real-world example of disciplined claim scope. It demonstrates gravastar formation in a specified general-relativistic construction under fine-tuned conditions, identifies successful conditions through backward integration from the desired final configuration, and explicitly leaves more realistic assumptions and the relative likelihood of gravastar versus black-hole formation to future work.

UFF therefore records several forbidden promotions, including:

- `can form -> likely to form`;
- `constructed trajectory -> generic outcome`;
- `fine-tuned success -> robust success`;
- `static equilibrium -> general dynamical stability`;
- `model-bound threshold -> universal physical law`;
- `backward target-conditioned construction -> forward predictive genericity`; and
- `theoretical possibility -> empirical occurrence`.

The source is a calibration reference, **not** evidence that astrophysical gravastars are generic, common, observed, or physically preferred.

### Source-fidelity rule

The nonclaim record separates:

1. what the source explicitly demonstrates;
2. what the source explicitly leaves open or qualifies; and
3. UFF's conservative interpretation of what must not be inferred.

Machine-readable entries mark whether a boundary is source-explicit, an UFF interpretation, or a combination of both. This avoids putting stronger disclaimers into an author's mouth than the paper itself supports.

### Formal-layer connection without authority leakage

`formal/lean/ASSUMPTIONS_AND_NONCLAIMS.md` now points to the v5.3 calibration layer while explicitly stating that the new reference does not extend the Lean theorem surface.

The existing rule remains:

```text
REPLAY_VERIFIED != ENSEMBLE_CALIBRATED != PHYSICAL_TRUTH
```

v5.3.0 adds a complementary rule for scientific prose and interpretation rather than another theorem-proving claim.

## Validation

Added regression tests that require:

- the nonclaim reference to remain valid JSON;
- the complete eight-dimension calibration surface to remain present;
- source-basis and UFF-nonclaim fields to remain explicit;
- boundary provenance to stay machine-readable; and
- the most common forbidden epistemic promotions to remain encoded.

## Compatibility

- Existing Python APIs and CLIs are unchanged apart from release metadata.
- Existing executable schemas remain `uff.sky-lattice-claim.v1` and `uff.sheridan-crucible.v1`.
- The Lean theorem surface is unchanged.
- `uff.nonclaim-reference.v1` is a documentation/provenance record, not an executable evidence-admission contract.
- `uff.sheridan-crucible.v2` and runtime `ENSEMBLE_CALIBRATED` remain future work.

## Archival record

The v5.3.0 Zenodo metadata is prepared under `zenodo/v5.3.0/` and mirrored in `.zenodo.json`. The arXiv reference is recorded as a related identifier with relation `references`.

A new Zenodo version DOI must be assigned by Zenodo before final DOI binding. The previous v5.2.0 DOI, `10.5281/zenodo.21911644`, remains immutable historical provenance and must not be relabeled as the v5.3.0 DOI.

The final GitHub `v5.3.0` tag and archival payload must bind to the exact merged release commit before publication.
