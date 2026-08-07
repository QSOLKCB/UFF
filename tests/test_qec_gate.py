from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uff.qec_gate import GATE_FILENAME, seal_boundary, verify_boundary
from uff.sky_artifacts import write_bundle
from uff.sky_contract import canonical_json_bytes, sha256_bytes


def _write_contract(path: Path) -> None:
    payload = {
        "schema": "uff.sky-lattice-claim.v1",
        "claim_id": "qec-gate-test",
        "title": "QEC boundary gate regression",
        "node_radius_deg": 15.0,
        "nodes": [
            {"id": "A", "ra_deg": 0.0, "dec_deg": 0.0},
            {"id": "B", "ra_deg": 180.0, "dec_deg": 0.0},
        ],
        "anomaly_rule": {"column": "score", "operator": ">=", "threshold": 1.0},
        "analysis": {
            "null_model": "stratified-label",
            "permutations": 99,
            "seed": 20260807,
        },
        "decision_rule": {
            "alpha": 0.05,
            "minimum_rate_contrast": 0.0,
            "minimum_supported_nodes": 0,
        },
        "data": {"stratum_column": "survey"},
        "declarations": {
            "selection_independent_of_nodes": True,
            "confirmatory": True,
            "independent_catalogue": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_catalogue(path: Path) -> None:
    rng = np.random.default_rng(90210)
    size = 600
    frame = pd.DataFrame(
        {
            "ra_deg": rng.uniform(0.0, 360.0, size),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size))),
            "score": rng.binomial(1, 0.15, size),
            "survey": np.where(np.arange(size) % 2 == 0, "north", "south"),
        }
    )
    frame.to_csv(path, index=False)


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    contract = tmp_path / "contract.json"
    catalogue = tmp_path / "catalogue.csv"
    output = tmp_path / "bundle"
    _write_contract(contract)
    _write_catalogue(catalogue)
    manifest = write_bundle(
        output,
        catalogue_path=catalogue,
        contract_path=contract,
    )
    return manifest, catalogue


def test_integrity_alone_never_admits_bundle(tmp_path):
    manifest, _catalogue = _bundle(tmp_path)
    report = verify_boundary(manifest, require_replay=False)
    assert report.integrity_passed
    assert report.replay_passed is None
    assert report.assurance == "INTEGRITY_ONLY"
    assert not report.admitted


def test_gate_requires_replay_for_admission(tmp_path):
    manifest, catalogue = _bundle(tmp_path)
    missing = verify_boundary(manifest)
    assert missing.integrity_passed
    assert missing.assurance == "INTEGRITY_ONLY"
    assert not missing.admitted
    assert any("requires the frozen catalogue" in error for error in missing.errors)

    replayed = verify_boundary(manifest, catalogue_path=catalogue)
    assert replayed.integrity_passed
    assert replayed.replay_passed is True
    assert replayed.assurance == "REPLAY_VERIFIED"
    assert replayed.admitted
    assert replayed.root_sha256 is not None


def test_gate_rejects_noncanonical_child_even_if_manifest_hash_is_rewritten(tmp_path):
    manifest_path, catalogue = _bundle(tmp_path)
    observations_path = manifest_path.parent / "observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    noncanonical = json.dumps(observations, indent=2).encode("utf-8")
    observations_path.write_bytes(noncanonical)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        if entry["path"] == "observations.json":
            entry["bytes"] = len(noncanonical)
            entry["sha256"] = sha256_bytes(noncanonical)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    report = verify_boundary(manifest_path, catalogue_path=catalogue)
    assert not report.admitted
    assert not report.integrity_passed
    assert any("not canonical JSON" in error for error in report.errors)


def test_gate_rejects_smuggled_unlisted_files(tmp_path):
    manifest, catalogue = _bundle(tmp_path)
    (manifest.parent / "unlisted_payload.txt").write_text("not part of the evidence contract\n")
    report = verify_boundary(manifest, catalogue_path=catalogue)
    assert not report.admitted
    assert not report.integrity_passed
    assert any("unlisted files" in error for error in report.errors)


def test_seal_is_recomputed_and_can_be_externally_anchored(tmp_path):
    manifest, catalogue = _bundle(tmp_path)
    before = verify_boundary(manifest, catalogue_path=catalogue)
    assert before.admitted
    assert before.root_sha256 is not None

    receipt = seal_boundary(manifest, catalogue_path=catalogue)
    assert receipt.name == GATE_FILENAME
    sealed = verify_boundary(
        manifest,
        catalogue_path=catalogue,
        expected_root=before.root_sha256,
    )
    assert sealed.admitted
    assert any("sealed gate receipt" in check for check in sealed.checks)
    assert any("external trust anchor" in check for check in sealed.checks)

    wrong = verify_boundary(
        manifest,
        catalogue_path=catalogue,
        expected_root="0" * 64,
    )
    assert not wrong.admitted
    assert not wrong.integrity_passed
    assert any("externally supplied trust anchor" in error for error in wrong.errors)
