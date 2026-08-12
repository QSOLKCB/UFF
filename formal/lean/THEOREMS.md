# Advertised theorem surface

The canonical machine-readable declaration list is `AUDIT_MANIFEST.tsv`. CI requires the manifest to match the actual theorem/lemma declarations exactly.

## Assurance boundaries

- `replay_preserves_external_scientific_judgement`
- `replay_verified_is_not_ensemble_calibrated`
- `ensemble_calibrated_is_not_scientifically_defensible`
- `ensemble_calibrated_includes_replay`

## Identity and replay

- `changed_claim_boundary_changes_identity`
- `changed_frozen_recipe_changes_identity`
- `deterministic_replay_same_inputs`
- `changed_runtime_contract_changes_replay_identity`

## Transaction and observation

- `cancellation_produces_no_bundle`
- `cancelled_transaction_not_exportable`
- `sonification_preserves_numerical_result`
- `telemetry_has_zero_admission_authority`

## Manifest construction

- `sealing_does_not_modify_manifest_core`
- `manifest_digest_is_computed_from_core_only`

## Explicit nonclaims

- `no_hash_collision_resistance_claim`
- `no_catalogue_correctness_claim`
- `no_null_model_adequacy_claim`
- `no_physical_truth_claim`
