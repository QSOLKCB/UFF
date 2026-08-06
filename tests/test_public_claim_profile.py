from __future__ import annotations

import json
from pathlib import Path


PROFILE = Path(__file__).parents[1] / "examples" / "public_claim_profile_2026-08-07.json"


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
    assert rotated["declared_node_count"] == 12
    assert rotated["captured_node_count"] == len(rotated["nodes"]) == 10
    assert set(rotated["missing_ids"]) == {"E-7", "E-12"}

    gaia = payload["catalogue_predicates"]["gaia_dr3"]
    assert gaia["query_present_in_snapshot"] is False
    assert gaia["frozen_threshold_present"] is False

    assert payload["blocking_ambiguities"]
    assert payload["required_before_confirmation"]
