from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uff.sheridan import (
    fibonacci_support,
    load_sheridan_contract_payload,
    verify_sheridan_bundle,
    vmf_kernel,
    write_sheridan_bundle,
)
from uff.sheridan_contract import load_support
from uff.sheridan_density import support_kernel_mass
from uff.sky_contract import AuditError
from uff.sky_geometry import radec_to_unit


def contract_payload(*, permutations: int = 99, injection_runs: int = 4) -> dict[str, object]:
    return {
        "schema": "uff.sheridan-crucible.v1",
        "title": "Synthetic Sheridan recovery",
        "claim": {
            "schema": "uff.sky-lattice-claim.v1",
            "claim_id": "synthetic-siege",
            "title": "Synthetic two-node claim",
            "node_radius_deg": 15.0,
            "nodes": [
                {"id": "A", "ra_deg": 0.0, "dec_deg": 0.0},
                {"id": "B", "ra_deg": 180.0, "dec_deg": 0.0},
            ],
            "anomaly_rule": {"column": "score", "operator": ">=", "threshold": 1.0},
            "analysis": {"null_model": "so3", "permutations": 99, "seed": 44},
            "decision_rule": {
                "alpha": 0.05,
                "minimum_rate_contrast": 0.1,
                "minimum_supported_nodes": 1,
            },
            "data": {},
            "declarations": {
                "selection_independent_of_nodes": True,
                "confirmatory": True,
                "independent_catalogue": True,
            },
        },
        "survey": {
            "area_weight_column": "area_weight_sr",
            "coverage_column": "coverage",
            "completeness_column": "completeness",
            "minimum_completeness": 0.5,
        },
        "density": {
            "bandwidth_candidates_deg": [10.0, 15.0, 22.0],
            "adaptive_alpha": 0.5,
            "minimum_bandwidth_factor": 0.6,
            "maximum_bandwidth_factor": 1.8,
            "permutations": permutations,
            "seed": 101,
            "availability_tolerance": 0.08,
            "minimum_availability": 0.5,
            "maximum_rotation_attempt_multiplier": 20,
        },
        "models": {
            "stratum_column": "survey",
            "covariate_columns": ["magnitude"],
            "predictive_draws": 99,
            "delta_pseudo_bic_threshold": 4.0,
        },
        "injection": {
            "enabled": True,
            "runs": injection_runs,
            "injected_per_node": 8,
            "minimum_recovery_rate": 0.5,
            "maximum_false_positive_rate": 0.5,
            "delta_pseudo_bic_threshold": 2.0,
        },
        "decision_rule": {
            "alpha": 0.05,
            "minimum_mean_overdensity": 0.5,
            "minimum_supported_nodes": 1,
            "required_components": ["density", "model"],
        },
    }


def write_catalogue(path: Path) -> None:
    rng = np.random.default_rng(2026)
    n_background = 500
    background = pd.DataFrame(
        {
            "ra_deg": rng.uniform(0.0, 360.0, n_background),
            "dec_deg": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_background))),
            "score": rng.binomial(1, 0.06, n_background),
            "completeness": rng.uniform(0.65, 1.0, n_background),
            "magnitude": rng.normal(20.0, 1.2, n_background),
            "survey": np.where(np.arange(n_background) % 2, "north", "south"),
        }
    )
    clusters = []
    for centre in (0.0, 180.0):
        count = 90
        clusters.append(
            pd.DataFrame(
                {
                    "ra_deg": rng.normal(centre, 4.0, count) % 360.0,
                    "dec_deg": rng.normal(0.0, 4.0, count),
                    "score": rng.binomial(1, 0.75, count),
                    "completeness": rng.uniform(0.7, 1.0, count),
                    "magnitude": rng.normal(19.5, 1.0, count),
                    "survey": np.where(np.arange(count) % 2, "north", "south"),
                }
            )
        )
    pd.concat([background, *clusters], ignore_index=True).to_csv(path, index=False)


def test_vmf_kernel_integrates_on_full_sky():
    support = fibonacci_support(4096)
    vectors = radec_to_unit(support.ra_deg.to_numpy(), support.dec_deg.to_numpy())
    centre = radec_to_unit(np.array([33.0]), np.array([-12.0]))
    mass = support_kernel_mass(
        centre,
        np.deg2rad(20.0),
        vectors,
        support.area_weight_sr.to_numpy(),
        support.coverage.to_numpy(),
    )
    assert mass[0] == pytest.approx(1.0, rel=0.015, abs=0.015)
    kernel = vmf_kernel(centre, centre, np.deg2rad(20.0))
    assert kernel.shape == (1, 1)
    assert kernel[0, 0] > 0.0


def test_contract_rejects_double_completeness_weighting():
    payload = contract_payload()
    payload["claim"]["data"] = {"weight_column": "completeness"}  # type: ignore[index]
    with pytest.raises(AuditError, match="same correction twice"):
        load_sheridan_contract_payload(payload)


def test_support_rejects_zero_usable_area(tmp_path):
    path = tmp_path / "support.csv"
    frame = fibonacci_support(64)
    frame["coverage"] = 0.0
    frame.to_csv(path, index=False)
    config = load_sheridan_contract_payload(contract_payload()).survey
    with pytest.raises(AuditError, match="zero usable area"):
        load_support(path, config)


def test_bundle_recovers_signal_and_replays(tmp_path):
    catalogue = tmp_path / "catalogue.csv"
    support = tmp_path / "support.csv"
    contract = tmp_path / "contract.json"
    output = tmp_path / "bundle"
    write_catalogue(catalogue)
    fibonacci_support(1536).to_csv(support, index=False)
    contract.write_text(json.dumps(contract_payload()), encoding="utf-8")

    manifest = write_sheridan_bundle(
        output,
        catalogue_path=catalogue,
        support_path=support,
        contract_path=contract,
    )
    density = json.loads((output / "density.json").read_text())
    models = json.loads((output / "models.json").read_text())
    decision = json.loads((output / "decision.json").read_text())
    nodes = pd.read_csv(output / "nodes.csv")

    assert density["decision"] == "SURVEY_AWARE_DENSITY_CRITERIA_MET"
    assert models["decision"] == "NODE_TERM_PREFERRED"
    assert decision["decision"] == "CRUCIBLE_CRITERIA_MET"
    assert nodes["supported"].sum() >= 1

    report = verify_sheridan_bundle(
        manifest,
        catalogue_path=catalogue,
        support_path=support,
    )
    assert report.passed, report.errors
    assert report.replay_passed is True


def test_tamper_detection(tmp_path):
    catalogue = tmp_path / "catalogue.csv"
    support = tmp_path / "support.csv"
    contract = tmp_path / "contract.json"
    output = tmp_path / "bundle"
    write_catalogue(catalogue)
    fibonacci_support(1024).to_csv(support, index=False)
    contract.write_text(json.dumps(contract_payload(injection_runs=2)), encoding="utf-8")
    manifest = write_sheridan_bundle(
        output,
        catalogue_path=catalogue,
        support_path=support,
        contract_path=contract,
    )
    density = output / "density.json"
    density.write_text(density.read_text() + " ", encoding="utf-8")
    report = verify_sheridan_bundle(manifest)
    assert not report.passed
    assert not report.integrity_passed
    assert any("mismatch" in error for error in report.errors)


def test_mask_quadrature_reduces_kernel_mass():
    support = fibonacci_support(4096)
    vectors = radec_to_unit(support.ra_deg.to_numpy(), support.dec_deg.to_numpy())
    coverage = support.coverage.to_numpy().copy()
    coverage[support.dec_deg.to_numpy() > 35.0] = 0.0
    centres = radec_to_unit(np.array([0.0, 0.0]), np.array([70.0, -70.0]))
    mass = support_kernel_mass(
        centres,
        np.deg2rad(15.0),
        vectors,
        support.area_weight_sr.to_numpy(),
        coverage,
    )
    assert mass[0] < 0.2
    assert mass[1] > 0.95


def test_exact_source_limit_is_enforced(tmp_path):
    from uff.sheridan_contract import prepare_catalogue
    from uff.sheridan_density import fit_density
    from uff.sky_contract import load_catalogue

    catalogue_path = tmp_path / "catalogue.csv"
    support_path = tmp_path / "support.csv"
    write_catalogue(catalogue_path)
    fibonacci_support(256).to_csv(support_path, index=False)
    payload = contract_payload()
    payload["density"]["maximum_exact_sources"] = 64  # type: ignore[index]
    contract = load_sheridan_contract_payload(payload)
    catalogue, _ = load_catalogue(catalogue_path, contract.claim)
    catalogue, _ = prepare_catalogue(catalogue, contract.survey, contract.models)
    support = load_support(support_path, contract.survey)
    with pytest.raises(AuditError, match="exact adaptive KDE is capped"):
        fit_density(catalogue, support, contract.density)
