from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uff.audit_events import TelemetryError, build_event_stream, write_event_stream
from uff.spectral_witness import create_witness, reveal_witness
from uff.sky_artifacts import write_bundle


def _write_contract(path: Path) -> None:
    payload = {
        "schema": "uff.sky-lattice-claim.v1",
        "claim_id": "defense-layer-test",
        "title": "Defense layer regression",
        "node_radius_deg": 15.0,
        "nodes": [
            {"id": "A", "ra_deg": 0.0, "dec_deg": 0.0},
            {"id": "B", "ra_deg": 180.0, "dec_deg": 0.0},
        ],
        "anomaly_rule": {"column": "score", "operator": ">=", "threshold": 1.0},
        "analysis": {
            "null_model": "stratified-label",
            "permutations": 99,
            "seed": 8080,
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
    rng = np.random.default_rng(1212)
    size = 500
    pd.DataFrame(
        {
            "ra_deg": rng.uniform(0.0, 360.0, size),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size))),
            "score": rng.binomial(1, 0.12, size),
            "survey": np.where(np.arange(size) % 2 == 0, "north", "south"),
        }
    ).to_csv(path, index=False)


def _prepared(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    contract = tmp_path / "contract.json"
    catalogue = tmp_path / "catalogue.csv"
    witness = tmp_path / "precommit.json"
    bundle = tmp_path / "bundle"
    _write_contract(contract)
    _write_catalogue(catalogue)
    create_witness(
        witness,
        contract_path=contract,
        catalogue_path=catalogue,
    )
    manifest = write_bundle(
        bundle,
        catalogue_path=catalogue,
        contract_path=contract,
    )
    return contract, catalogue, witness, manifest


def test_spectral_witness_is_deterministic_and_reveals_through_qec(tmp_path):
    contract, catalogue, witness, manifest = _prepared(tmp_path)
    first_bytes = witness.read_bytes()
    second = tmp_path / "precommit-2.json"
    create_witness(second, contract_path=contract, catalogue_path=catalogue)
    assert second.read_bytes() == first_bytes

    commit = json.loads(witness.read_text())["commit_sha256"]
    report = reveal_witness(
        witness,
        manifest,
        contract_path=contract,
        catalogue_path=catalogue,
        expected_commit_sha256=commit,
    )
    assert report.admitted
    assert report.witness_verified
    assert report.qec_admitted
    assert report.external_anchor_verified is True
    assert any("QEC-verified bundle contract" in check for check in report.checks)


def test_spectral_witness_detects_post_commit_input_change(tmp_path):
    contract, catalogue, witness, manifest = _prepared(tmp_path)
    frame = pd.read_csv(catalogue)
    frame.loc[0, "score"] = 1 - int(frame.loc[0, "score"])
    frame.to_csv(catalogue, index=False)

    report = reveal_witness(
        witness,
        manifest,
        contract_path=contract,
        catalogue_path=catalogue,
    )
    assert not report.admitted
    assert any("frozen witness identity" in error for error in report.errors)


def test_spectral_witness_binds_contract_to_replayed_bundle(tmp_path):
    contract, catalogue, witness, _manifest = _prepared(tmp_path)
    different_contract = tmp_path / "different-contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["claim_id"] = "different-replayed-claim"
    payload["title"] = "Different replayed contract"
    payload["analysis"]["seed"] = 9090
    different_contract.write_text(json.dumps(payload), encoding="utf-8")

    different_bundle = tmp_path / "different-bundle"
    different_manifest = write_bundle(
        different_bundle,
        catalogue_path=catalogue,
        contract_path=different_contract,
    )
    report = reveal_witness(
        witness,
        different_manifest,
        contract_path=contract,
        catalogue_path=catalogue,
    )
    assert report.qec_admitted
    assert not report.admitted
    assert not report.witness_verified
    assert any("contract embedded in the replayed bundle" in error for error in report.errors)


def test_receiver_neutral_audit_events_are_deterministic(tmp_path):
    _contract, catalogue, _witness, manifest = _prepared(tmp_path)
    first = build_event_stream(manifest, catalogue_path=catalogue)
    second = build_event_stream(manifest, catalogue_path=catalogue)
    assert first == second
    assert first["source"]["gate_assurance"] == "REPLAY_VERIFIED"
    assert first["events"][0]["code"] == "INTEGRITY"
    assert first["events"][1]["code"] == "REPLAY"
    assert first["events"][2]["code"] == "ADMISSION"
    assert first["receiver_contract"]["noncanonical"]
    assert first["event_stream_sha256"]


def test_rejected_manifest_still_emits_boundary_telemetry(tmp_path):
    manifest = tmp_path / "bad-bundle" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"{not valid json")

    stream = build_event_stream(manifest)
    assert stream["source"]["gate_assurance"] == "REJECTED"
    assert [event["code"] for event in stream["events"][:3]] == [
        "INTEGRITY",
        "REPLAY",
        "ADMISSION",
    ]
    assert [event["state"] for event in stream["events"][:3]] == [
        "FAIL",
        "ABSENT",
        "REJECT",
    ]


def test_audit_events_cannot_be_written_into_closed_bundle(tmp_path):
    _contract, catalogue, _witness, manifest = _prepared(tmp_path)
    with pytest.raises(TelemetryError, match="outside the closed evidence bundle"):
        write_event_stream(
            manifest.parent / "telemetry.json",
            manifest,
            catalogue_path=catalogue,
        )

    output = tmp_path / "telemetry" / "audit-events.json"
    written = write_event_stream(output, manifest, catalogue_path=catalogue)
    assert written == output
    assert json.loads(output.read_text())["schema"] == "uff.audit-event-stream.v1"
