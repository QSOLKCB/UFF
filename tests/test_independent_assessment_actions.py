from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
ACTIONS = ROOT / "examples" / "independent_assessment_actions_2026-08-07.json"
RENDITION = ROOT / "papers" / "Independent_Research_Assessment_Logvinovich_Claim_and_Sheridan_Audit.md"
MANIFEST = ROOT / "docs" / "INDEPENDENT_ASSESSMENT_SOURCE_MANIFEST_2026-08-07.md"


def test_independent_assessment_manifest_and_action_priorities() -> None:
    payload = json.loads(ACTIONS.read_text(encoding="utf-8"))

    assert payload["schema"] == "uff.independent-assessment-response.v1"
    assert payload["status"] == "accepted-with-implementation-follow-up"

    source = payload["source"]
    assert source["sha256"] == "4af1ba265770b88b41d70d08b93eb73c2b1ff3992b0b041431deea40f1a4ea07"
    assert source["byte_count"] == 135639
    assert source["pages"] == 11
    assert source["repository_rendition"] == str(RENDITION.relative_to(ROOT))
    assert RENDITION.is_file()
    assert MANIFEST.is_file()

    current = set(payload["contract_delta"]["current"])
    proposed = set(payload["contract_delta"]["proposed"])
    assert proposed - current == {"P", "T", "E", "M", "Q", "R"}

    actions = payload["actions"]
    ids = [action["id"] for action in actions]
    assert len(ids) == len(set(ids))
    assert {action["priority"] for action in actions} == {"P0", "P1", "P2"}
    assert all(action["status"] == "planned" for action in actions)

    boundaries = payload["boundaries"]
    assert any("does not validate the Logvinovich claim" in item for item in boundaries)
    assert any("not an independent execution of UFF" in item for item in boundaries)
