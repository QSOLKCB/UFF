from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECORD = ROOT / "examples" / "nonclaim_reference_gravastar_2026.json"
DOC = ROOT / "docs" / "NONCLAIM_CALIBRATION.md"

EXPECTED_DIMENSIONS = {
    "existence",
    "robustness",
    "genericity",
    "prevalence",
    "empirical_support",
    "predictive_direction",
    "stability",
    "universality",
}


def test_nonclaim_reference_is_machine_readable_and_complete() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["schema"] == "uff.nonclaim-reference.v1"
    assert record["record_role"] == "nonclaim-calibration"
    assert record["record_id"] == "arxiv:2509.15302v2"
    assert record["source"]["title"] == "Formation of gravastars"
    assert record["source"]["identifier"] == "arXiv:2509.15302v2"
    assert record["source"]["version_date"] == "2026-06-11"

    dimensions = {item["dimension"] for item in record["dimensions"]}
    assert dimensions == EXPECTED_DIMENSIONS

    assert all(item["source_status"] for item in record["dimensions"])
    assert all(item["source_basis"] for item in record["dimensions"])
    assert all(item["uff_nonclaim"] for item in record["dimensions"])
    assert all(
        item["boundary_origin"]
        in {
            "source-explicit",
            "uff-interpretation",
            "source-explicit-plus-uff-interpretation",
        }
        for item in record["dimensions"]
    )


def test_nonclaim_reference_forbids_common_epistemic_promotions() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    promotions = set(record["forbidden_promotions"])

    assert "can form -> likely to form" in promotions
    assert "constructed trajectory -> generic outcome" in promotions
    assert "fine-tuned success -> robust success" in promotions
    assert "static equilibrium -> general dynamical stability" in promotions
    assert (
        "backward target-conditioned construction -> forward predictive genericity"
        in promotions
    )
    assert "theoretical possibility -> empirical occurrence" in promotions
    assert "association -> causation" in promotions


def test_nonclaim_doc_preserves_governing_boundary_and_dimension_contract() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert (
        "EXISTENCE != ROBUSTNESS != GENERICITY != PREVALENCE != EMPIRICAL_SUPPORT "
        "!= PHYSICAL_TRUTH"
        in text
    )
    assert "canonical `uff.nonclaim-reference.v1` vocabulary has eight dimensions" in text
    assert "| `stability` |" in text
    assert "Causal interpretation is enforced as a **forbidden promotion**" in text
    assert "arXiv:2509.15302v2" in text
    assert "nonclaim calibration reference" in text
    assert "does not change galaxy likelihoods" in text
