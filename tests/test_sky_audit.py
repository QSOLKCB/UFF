from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uff.sky_audit import (
    AuditError,
    holm_adjust,
    load_contract,
    verify_bundle,
    write_bundle,
)
from uff.sky_geometry import (
    pairwise_angles,
    radec_to_unit,
    random_so3,
    validate_lattice_invariance,
)


def _contract_payload(permutations: int = 399) -> dict[str, object]:
    return {
        "schema": "uff.sky-lattice-claim.v1",
        "claim_id": "synthetic-two-node",
        "title": "Synthetic two-node recovery test",
        "node_radius_deg": 8.0,
        "nodes": [
            {"id": "A", "ra_deg": 0.0, "dec_deg": 0.0},
            {"id": "B", "ra_deg": 180.0, "dec_deg": 0.0},
        ],
        "anomaly_rule": {"column": "score", "operator": ">=", "threshold": 1.0},
        "analysis": {"null_model": "so3", "permutations": permutations, "seed": 1234},
        "decision_rule": {
            "alpha": 0.05,
            "minimum_rate_contrast": 0.20,
            "minimum_supported_nodes": 1,
        },
        "data": {},
        "declarations": {
            "selection_independent_of_nodes": True,
            "confirmatory": True,
            "independent_catalogue": True,
        },
    }


def _write_contract(path: Path, permutations: int = 399) -> None:
    path.write_text(json.dumps(_contract_payload(permutations)), encoding="utf-8")


def _write_signal_catalogue(path: Path) -> None:
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
    pd.concat([background, signal_a, signal_b], ignore_index=True).to_csv(path, index=False)


def test_so3_preserves_lattice_geometry():
    nodes = radec_to_unit(np.array([0.0, 45.0, 180.0]), np.array([0.0, 30.0, -10.0]))
    rotation = random_so3(np.random.default_rng(22))
    transformed = nodes @ rotation.T
    residuals = validate_lattice_invariance(nodes, transformed, rotation)
    assert residuals.orthogonality_frobenius < 1e-12
    assert residuals.determinant_abs_error < 1e-12
    assert residuals.gram_max_abs < 1e-12
    assert residuals.pairwise_angle_max_abs_rad < 1e-12
    assert np.allclose(pairwise_angles(nodes), pairwise_angles(transformed))


def test_holm_adjustment_is_monotone_and_conservative():
    p_values = np.array([0.03, 0.001, 0.02, 0.2])
    adjusted = holm_adjust(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1.0)


def test_contract_rejects_duplicate_nodes(tmp_path):
    path = tmp_path / "contract.json"
    payload = _contract_payload()
    payload["nodes"][1]["id"] = "A"  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuditError, match="unique"):
        load_contract(path)


def test_contract_rejects_node_targeted_selection(tmp_path):
    path = tmp_path / "contract.json"
    payload = _contract_payload()
    payload["declarations"]["selection_independent_of_nodes"] = False  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuditError, match="node-targeted"):
        load_contract(path)


def test_bundle_recovers_signal_and_replays(tmp_path):
    contract = tmp_path / "contract.json"
    catalogue = tmp_path / "catalogue.csv"
    output = tmp_path / "bundle"
    _write_contract(contract)
    _write_signal_catalogue(catalogue)
    manifest = write_bundle(output, catalogue_path=catalogue, contract_path=contract)
    observations = json.loads((output / "observations.json").read_text())
    nodes = pd.read_csv(output / "nodes.csv")
    assert observations["decision"] == "EMPIRICAL_CRITERIA_MET"
    assert observations["global_test"]["empirical_p"] <= 0.05
    assert observations["global_test"]["rate_contrast"] > 0.2
    assert nodes["survives_holm"].sum() >= 1
    report = verify_bundle(manifest, catalogue)
    assert report.passed
    assert report.integrity_passed
    assert report.replay_passed is True


def test_holdout_filter_is_enforced(tmp_path):
    contract = tmp_path / "contract.json"
    catalogue = tmp_path / "catalogue.csv"
    output = tmp_path / "bundle"
    payload = _contract_payload(permutations=199)
    payload["declarations"]["independent_catalogue"] = False  # type: ignore[index]
    payload["data"] = {"holdout_column": "split", "holdout_value": "test"}
    contract.write_text(json.dumps(payload), encoding="utf-8")
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "ra_deg": rng.uniform(0.0, 360.0, 600),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, 600))),
            "score": rng.binomial(1, 0.15, 600),
            "split": np.where(np.arange(600) % 2 == 0, "train", "test"),
        }
    )
    frame.to_csv(catalogue, index=False)
    write_bundle(output, catalogue_path=catalogue, contract_path=contract)
    recipe = json.loads((output / "recipe.json").read_text())
    assert recipe["inputs"]["catalogue_rows_before_holdout"] == 600
    assert recipe["inputs"]["catalogue_rows_after_holdout"] == 300


def test_tamper_detection_fails_integrity(tmp_path):
    contract = tmp_path / "contract.json"
    catalogue = tmp_path / "catalogue.csv"
    output = tmp_path / "bundle"
    _write_contract(contract, permutations=199)
    _write_signal_catalogue(catalogue)
    manifest = write_bundle(output, catalogue_path=catalogue, contract_path=contract)
    observations = output / "observations.json"
    observations.write_text(observations.read_text() + " ", encoding="utf-8")
    report = verify_bundle(manifest, catalogue)
    assert not report.passed
    assert not report.integrity_passed
    assert any("SHA-256 mismatch" in error or "byte-size mismatch" in error for error in report.errors)
