from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "examples" / "public_claim_profile_2026-08-07.json"


def _collect_source_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        declared = value.get("source_refs", [])
        assert isinstance(declared, list)
        refs.update(str(item) for item in declared)
        for child in value.values():
            refs.update(_collect_source_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_collect_source_refs(child))
    return refs


def test_public_claim_profile_remains_non_executable_and_internally_consistent() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))

    assert payload["schema"] == "uff.public-claim-profile.v1"
    assert payload["status"] == "not-ready-for-confirmatory-run"

    versions = payload["claim_versions"]
    ideal = versions["ideal_oh_12"]
    ideal_nodes = ideal["nodes"]
    assert ideal["node_count"] == len(ideal_nodes) == 12
    assert len({(node["ra_deg"], node["dec_deg"]) for node in ideal_nodes}) == 12

    rotated = versions["rotated_e_nodes_partial"]
    all_e_ids = {f"E-{index}" for index in range(1, 13)}
    captured_ids = [node["id"] for node in rotated["nodes"]]
    missing_ids = set(rotated["missing_ids"])
    assert rotated["declared_node_count"] == len(all_e_ids) == 12
    assert rotated["captured_node_count"] == len(captured_ids) == 10
    assert len(captured_ids) == len(set(captured_ids))
    assert missing_ids == {"E-7", "E-12"}
    assert set(captured_ids) == all_e_ids - missing_ids
    assert missing_ids == all_e_ids - set(captured_ids)

    gaia = payload["catalogue_predicates"]["gaia_dr3"]
    assert gaia["query_present_in_snapshot"] is False
    assert gaia["frozen_threshold_present"] is False

    assert payload["blocking_ambiguities"]
    assert payload["required_before_confirmation"]


def test_public_claim_profile_has_content_addressed_provenance() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    source_records = payload["source_records"]
    snapshot = source_records["snapshot_counter_intel_2026_08_07"]

    assert snapshot["acquired_at"] == "2026-08-07T09:08:07+09:30"
    assert snapshot["byte_count"] == 203874
    assert snapshot["sha256"] == "edcb8d46241728dca0e8bafa92265ed57ba65ed9a8d7cf1c665169f9d6859cc3"
    assert snapshot["immutable_identifier"] == f"sha256:{snapshot['sha256']}"
    assert len(snapshot["sha256"]) == 64

    source_ids = set(source_records)
    for source_id, source in source_records.items():
        snapshot_ref = source.get("snapshot_ref")
        if snapshot_ref is not None:
            assert snapshot_ref in source_ids, source_id

    used_refs = _collect_source_refs({
        "claim_versions": payload["claim_versions"],
        "catalogue_predicates": payload["catalogue_predicates"],
        "reported_assertions": payload["reported_assertions"],
        "declared_discriminants": payload["declared_discriminants"],
    })
    assert used_refs
    assert used_refs <= source_ids
    assert "snapshot_counter_intel_2026_08_07" in used_refs

    manifest = ROOT / payload["source_manifest"]
    assert manifest.is_file()
