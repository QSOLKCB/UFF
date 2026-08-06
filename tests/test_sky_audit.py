from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uff.sky_audit import audit_files, holm_adjust, load_contract, radec_to_unit


def _write_contract(path: Path, permutations: int = 399) -> None:
    payload = {
        "schema": "uff.sky-lattice-claim.v1",
        "claim_id": "synthetic-two-node",
        "title": "Synthetic two-node recovery test",
        "node_radius_deg": 8.0,
        "nodes": [
            {"id": "A", "ra_deg": 0.0, "dec_deg": 0.0},
            {"id": "B", "ra_deg": 180.0, "dec_deg": 0.0},
        ],
        "anomaly_rule": {
            "column": "score",
            "operator": ">=",
            "threshold": 1.0,
        },
        "analysis": {
            "null_model": "so3",
            "permutations": permutations,
            "seed": 1234,
        },
        "decision_rule": {
            "alpha": 0.05,
            "minimum_rate_contrast": 0.20,
            "minimum_supported_nodes": 1,
        },
        "data": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_radec_vectors_are_unit_length():
    vectors = radec_to_unit(
        np.array([0.0, 90.0, 180.0]), np.array([0.0, 45.0, -45.0])
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_holm_adjustment_is_monotone_in_sorted_order():
    p = np.array([0.03, 0.001, 0.02, 0.2])
    adjusted = holm_adjust(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)
    assert np.all((adjusted >= p) & (adjusted <= 1.0))


def test_contract_rejects_duplicate_nodes(tmp_path):
    path = tmp_path / "contract.json"
    _write_contract(path)
    payload = json.loads(path.read_text())
    payload["nodes"][1]["id"] = "A"
    path.write_text(json.dumps(payload))
    try:
        load_contract(path)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate node ids should fail")


def test_audit_recovers_preregistered_signal(tmp_path):
    contract_path = tmp_path / "contract.json"
    catalog_path = tmp_path / "catalog.csv"
    output = tmp_path / "out"
    _write_contract(contract_path)

    rng = np.random.default_rng(8)
    background = pd.DataFrame(
        {
            "ra_deg": rng.uniform(0.0, 360.0, 3000),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, 3000))),
            "score": rng.binomial(1, 0.08, 3000),
        }
    )
    signal_a = pd.DataFrame(
        {
            "ra_deg": rng.normal(0.0, 1.5, 80) % 360.0,
            "dec_deg": rng.normal(0.0, 1.5, 80),
            "score": np.ones(80),
        }
    )
    signal_b = pd.DataFrame(
        {
            "ra_deg": rng.normal(180.0, 1.5, 80) % 360.0,
            "dec_deg": rng.normal(0.0, 1.5, 80),
            "score": np.ones(80),
        }
    )
    pd.concat([background, signal_a, signal_b], ignore_index=True).to_csv(
        catalog_path, index=False
    )

    outputs = audit_files(catalog_path, contract_path, output)
    receipt = json.loads(outputs["receipt"].read_text())
    nodes = pd.read_csv(outputs["nodes"])

    assert receipt["decision"] == "EMPIRICAL_CRITERIA_MET"
    assert receipt["global_test"]["empirical_p"] <= 0.05
    assert receipt["global_test"]["rate_contrast"] > 0.2
    assert nodes["survives_holm"].sum() >= 1
    assert outputs["manifest"].exists()


def test_holdout_filter_is_enforced(tmp_path):
    contract_path = tmp_path / "contract.json"
    catalog_path = tmp_path / "catalog.csv"
    output = tmp_path / "out"
    _write_contract(contract_path, permutations=199)
    payload = json.loads(contract_path.read_text())
    payload["data"] = {"holdout_column": "split", "holdout_value": "test"}
    contract_path.write_text(json.dumps(payload))

    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "ra_deg": rng.uniform(0, 360, 600),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1, 1, 600))),
            "score": rng.binomial(1, 0.15, 600),
            "split": np.where(np.arange(600) % 2 == 0, "train", "test"),
        }
    )
    frame.to_csv(catalog_path, index=False)
    outputs = audit_files(catalog_path, contract_path, output)
    receipt = json.loads(outputs["receipt"].read_text())
    assert receipt["inputs"]["catalog_rows_after_holdout"] == 300
