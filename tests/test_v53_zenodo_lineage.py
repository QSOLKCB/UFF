from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
ZENODO = ROOT / ".zenodo.json"
ZENODO_SNAPSHOT = ROOT / "zenodo" / "v5.3.0" / "metadata.json"
PREVIOUS_DOI = "10.5281/zenodo.21911644"


def test_v5_3_zenodo_lineage_is_machine_readable() -> None:
    metadata = json.loads(ZENODO.read_text(encoding="utf-8"))
    snapshot = json.loads(ZENODO_SNAPSHOT.read_text(encoding="utf-8"))

    assert metadata == snapshot
    assert any(
        item["identifier"] == PREVIOUS_DOI
        and item["relation"] == "isNewVersionOf"
        and item["scheme"] == "doi"
        for item in metadata["related_identifiers"]
    )
    assert any(
        item["identifier"] == "https://arxiv.org/abs/2509.15302"
        and item["relation"] == "references"
        for item in metadata["related_identifiers"]
    )
