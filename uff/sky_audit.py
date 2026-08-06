"""Deterministic, preregistered audit for claimed celestial-node lattices.

The audit tests a frozen catalogue-level claim. It does not infer a causal
ontology from a spatial association.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SCHEMA = "uff.sky-lattice-audit.v1"
CONTRACT_SCHEMA = "uff.sky-lattice-claim.v1"
_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


@dataclass(frozen=True)
class Node:
    node_id: str
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class AuditContract:
    claim_id: str
    title: str
    nodes: tuple[Node, ...]
    radius_deg: float
    anomaly_column: str
    anomaly_operator: str
    anomaly_threshold: float
    permutations: int
    seed: int
    null_model: str
    alpha: float
    minimum_effect: float
    minimum_supported_nodes: int
    holdout_column: str | None = None
    holdout_value: str | None = None
    weight_column: str | None = None
    stratum_column: str | None = None
    expected_catalog_sha256: str | None = None


@dataclass(frozen=True)
class RegionSummary:
    anomaly_inside: float
    normal_inside: float
    anomaly_outside: float
    normal_outside: float

    @property
    def inside_total(self) -> float:
        return self.anomaly_inside + self.normal_inside

    @property
    def outside_total(self) -> float:
        return self.anomaly_outside + self.normal_outside

    @property
    def inside_rate(self) -> float:
        return self.anomaly_inside / self.inside_total if self.inside_total else math.nan

    @property
    def outside_rate(self) -> float:
        return self.anomaly_outside / self.outside_total if self.outside_total else math.nan

    @property
    def rate_contrast(self) -> float:
        return self.inside_rate - self.outside_rate

    @property
    def odds_ratio(self) -> float:
        ai, ni = self.anomaly_inside + 0.5, self.normal_inside + 0.5
        ao, no = self.anomaly_outside + 0.5, self.normal_outside + 0.5
        return ai * no / (ni * ao)

    def to_dict(self) -> dict[str, float]:
        return {
            "anomaly_inside": self.anomaly_inside,
            "normal_inside": self.normal_inside,
            "anomaly_outside": self.anomaly_outside,
            "normal_outside": self.normal_outside,
            "inside_total": self.inside_total,
            "outside_total": self.outside_total,
            "inside_rate": self.inside_rate,
            "outside_rate": self.outside_rate,
            "rate_contrast": self.rate_contrast,
            "odds_ratio": self.odds_ratio,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_contract(path: Path) -> tuple[AuditContract, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA!r}")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("contract must contain at least one node")
    nodes: list[Node] = []
    seen: set[str] = set()
    for item in raw_nodes:
        node_id = str(item.get("id", "")).strip()
        if not node_id or node_id in seen:
            raise ValueError("node ids must be non-empty and unique")
        seen.add(node_id)
        ra = _number(item.get("ra_deg"), f"node {node_id} ra_deg")
        dec = _number(item.get("dec_deg"), f"node {node_id} dec_deg")
        if not 0 <= ra < 360 or not -90 <= dec <= 90:
            raise ValueError(f"node {node_id} coordinates are outside ICRS bounds")
        nodes.append(Node(node_id, ra, dec))

    anomaly = payload.get("anomaly_rule", {})
    analysis = payload.get("analysis", {})
    decision = payload.get("decision_rule", {})
    data = payload.get("data", {})
    operator = str(anomaly.get("operator", ">="))
    if operator not in _OPERATORS:
        raise ValueError("unsupported anomaly operator")
    radius = _number(payload.get("node_radius_deg"), "node_radius_deg")
    if not 0 < radius <= 45:
        raise ValueError("node_radius_deg must be in (0, 45]")
    permutations = int(analysis.get("permutations", 10_000))
    if permutations < 99:
        raise ValueError("analysis.permutations must be at least 99")
    null_model = str(analysis.get("null_model", "ra-shift"))
    if null_model not in {"ra-shift", "so3", "stratified-label"}:
        raise ValueError("invalid null model")
    alpha = _number(decision.get("alpha", 0.05), "decision_rule.alpha")
    if not 0 < alpha < 1:
        raise ValueError("decision_rule.alpha must be in (0, 1)")
    required_nodes = int(decision.get("minimum_supported_nodes", 1))
    if not 0 <= required_nodes <= len(nodes):
        raise ValueError("minimum_supported_nodes is outside the node count")

    contract = AuditContract(
        claim_id=str(payload.get("claim_id", "")).strip(),
        title=str(payload.get("title", "")).strip(),
        nodes=tuple(nodes),
        radius_deg=radius,
        anomaly_column=str(anomaly.get("column", "anomaly_score")),
        anomaly_operator=operator,
        anomaly_threshold=_number(anomaly.get("threshold"), "anomaly threshold"),
        permutations=permutations,
        seed=int(analysis.get("seed", 20260807)),
        null_model=null_model,
        alpha=alpha,
        minimum_effect=_number(
            decision.get("minimum_rate_contrast", 0.0),
            "decision_rule.minimum_rate_contrast",
        ),
        minimum_supported_nodes=required_nodes,
        holdout_column=str(data["holdout_column"]) if data.get("holdout_column") else None,
        holdout_value=str(data["holdout_value"]) if data.get("holdout_value") is not None else None,
        weight_column=str(data["weight_column"]) if data.get("weight_column") else None,
        stratum_column=str(data["stratum_column"]) if data.get("stratum_column") else None,
        expected_catalog_sha256=(
            str(data["catalog_sha256"]).lower() if data.get("catalog_sha256") else None
        ),
    )
    if not contract.claim_id or not contract.title:
        raise ValueError("claim_id and title are required")
    if contract.null_model == "stratified-label" and not contract.stratum_column:
        raise ValueError("stratified-label requires data.stratum_column")
    return contract, payload


def load_catalog(path: Path, contract: AuditContract) -> pd.DataFrame:
    if contract.expected_catalog_sha256:
        actual = sha256_file(path)
        if actual != contract.expected_catalog_sha256:
            raise ValueError(
                f"catalog SHA-256 mismatch: expected {contract.expected_catalog_sha256}, got {actual}"
            )
    frame = pd.read_csv(path).copy()
    required = {"ra_deg", "dec_deg", contract.anomaly_column}
    required.update(
        column
        for column in (
            contract.weight_column,
            contract.holdout_column,
            contract.stratum_column,
        )
        if column
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"catalog is missing columns: {', '.join(missing)}")
    for column in ("ra_deg", "dec_deg", contract.anomaly_column):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["ra_deg", "dec_deg", contract.anomaly_column]]).all().all():
        raise ValueError("catalog coordinates and anomaly values must be finite")
    if not frame["ra_deg"].between(0, 360, inclusive="left").all():
        raise ValueError("catalog ra_deg must be in [0, 360)")
    if not frame["dec_deg"].between(-90, 90).all():
        raise ValueError("catalog dec_deg must be in [-90, 90]")
    if contract.holdout_column:
        frame = frame[
            frame[contract.holdout_column].astype(str) == contract.holdout_value
        ].copy()
        if frame.empty:
            raise ValueError("holdout filter produced an empty catalogue")
    if contract.weight_column:
        frame["_weight"] = pd.to_numeric(frame[contract.weight_column], errors="raise")
        if not np.isfinite(frame["_weight"]).all() or (frame["_weight"] <= 0).any():
            raise ValueError("weights must be finite and strictly positive")
    else:
        frame["_weight"] = 1.0
    frame["_anomaly"] = _OPERATORS[contract.anomaly_operator](
        frame[contract.anomaly_column].to_numpy(float), contract.anomaly_threshold
    )
    if frame["_anomaly"].nunique() < 2:
        raise ValueError("catalog must contain anomalous and non-anomalous records")
    return frame.reset_index(drop=True)


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.column_stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )


def unit_to_radec(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, float)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return (
        np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360,
        np.rad2deg(np.arcsin(np.clip(vectors[:, 2], -1, 1))),
    )


def node_membership(
    catalog_vectors: np.ndarray, node_vectors: np.ndarray, radius_deg: float
) -> np.ndarray:
    return catalog_vectors @ node_vectors.T >= math.cos(math.radians(radius_deg))


def summarize_region(
    mask: np.ndarray, anomaly: np.ndarray, weights: np.ndarray
) -> RegionSummary:
    mask, anomaly, weights = np.asarray(mask, bool), np.asarray(anomaly, bool), np.asarray(weights, float)
    return RegionSummary(
        float(weights[mask & anomaly].sum()),
        float(weights[mask & ~anomaly].sum()),
        float(weights[~mask & anomaly].sum()),
        float(weights[~mask & ~anomaly].sum()),
    )


def _rotation(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    y = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotate_nodes(
    nodes: np.ndarray, null_model: str, rng: np.random.Generator
) -> np.ndarray:
    if null_model == "so3":
        return nodes @ _rotation(rng).T
    angle = rng.uniform(0, 2 * math.pi)
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    return nodes @ matrix.T


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), float)
    order, adjusted, running = np.argsort(values), np.empty_like(values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _p_value(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _shuffle_labels(
    anomaly: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = anomaly.copy()
    for value in pd.unique(strata):
        indexes = np.flatnonzero(strata == value)
        result[indexes] = rng.permutation(result[indexes])
    return result


def run_audit(
    catalog: pd.DataFrame,
    contract: AuditContract,
    *,
    contract_payload: dict[str, Any],
    catalog_path: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    catalog_vectors = radec_to_unit(
        catalog["ra_deg"].to_numpy(float), catalog["dec_deg"].to_numpy(float)
    )
    node_vectors = radec_to_unit(
        np.array([node.ra_deg for node in contract.nodes]),
        np.array([node.dec_deg for node in contract.nodes]),
    )
    membership = node_membership(catalog_vectors, node_vectors, contract.radius_deg)
    anomaly = catalog["_anomaly"].to_numpy(bool)
    weights = catalog["_weight"].to_numpy(float)
    global_summary = summarize_region(membership.any(axis=1), anomaly, weights)
    if not global_summary.inside_total or not global_summary.outside_total:
        raise ValueError("node caps and complement must both contain records")
    node_summaries = [
        summarize_region(membership[:, index], anomaly, weights)
        for index in range(len(contract.nodes))
    ]
    testable = np.array(
        [bool(summary.inside_total and summary.outside_total) for summary in node_summaries]
    )
    observed_nodes = np.array(
        [summary.rate_contrast if ok else 0.0 for summary, ok in zip(node_summaries, testable)]
    )
    rng = np.random.default_rng(contract.seed)
    global_null: list[float] = []
    node_null: list[list[float]] = [[] for _ in contract.nodes]

    if contract.null_model in {"ra-shift", "so3"}:
        for _ in range(contract.permutations * 100):
            if len(global_null) == contract.permutations and all(
                not testable[i] or len(node_null[i]) == contract.permutations
                for i in range(len(contract.nodes))
            ):
                break
            rotated = node_membership(
                catalog_vectors,
                _rotate_nodes(node_vectors, contract.null_model, rng),
                contract.radius_deg,
            )
            summary = summarize_region(rotated.any(axis=1), anomaly, weights)
            if len(global_null) < contract.permutations and summary.inside_total and summary.outside_total:
                global_null.append(summary.rate_contrast)
            for index in range(len(contract.nodes)):
                if not testable[index] or len(node_null[index]) == contract.permutations:
                    continue
                summary = summarize_region(rotated[:, index], anomaly, weights)
                if summary.inside_total and summary.outside_total:
                    node_null[index].append(summary.rate_contrast)
        else:
            raise ValueError("unable to generate enough non-empty rotated null regions")
    else:
        strata = catalog[contract.stratum_column].to_numpy()
        for _ in range(contract.permutations):
            shuffled = _shuffle_labels(anomaly, strata, rng)
            global_null.append(
                summarize_region(membership.any(axis=1), shuffled, weights).rate_contrast
            )
            for index in range(len(contract.nodes)):
                if testable[index]:
                    node_null[index].append(
                        summarize_region(membership[:, index], shuffled, weights).rate_contrast
                    )

    global_null_array = np.asarray(global_null)
    global_p = _p_value(global_summary.rate_contrast, global_null_array)
    raw_p = np.ones(len(contract.nodes))
    for index in range(len(contract.nodes)):
        if testable[index]:
            raw_p[index] = _p_value(observed_nodes[index], np.asarray(node_null[index]))
    holm_p = holm_adjust(raw_p)

    rows: list[dict[str, Any]] = []
    for index, (node, summary) in enumerate(zip(contract.nodes, node_summaries)):
        rows.append(
            {
                "node_id": node.node_id,
                "ra_deg": node.ra_deg,
                "dec_deg": node.dec_deg,
                **summary.to_dict(),
                "testable": bool(testable[index]),
                "empirical_p": float(raw_p[index]),
                "holm_p": float(holm_p[index]),
                "survives_holm": bool(
                    testable[index]
                    and holm_p[index] <= contract.alpha
                    and summary.rate_contrast >= contract.minimum_effect
                ),
            }
        )
    node_table = pd.DataFrame(rows)
    supported_nodes = int(node_table["survives_holm"].sum())
    global_pass = bool(
        global_p <= contract.alpha
        and global_summary.rate_contrast >= contract.minimum_effect
    )
    passed = global_pass and supported_nodes >= contract.minimum_supported_nodes
    receipt = {
        "schema": SCHEMA,
        "claim_id": contract.claim_id,
        "title": contract.title,
        "decision": "EMPIRICAL_CRITERIA_MET" if passed else "EMPIRICAL_CRITERIA_NOT_MET",
        "interpretation_boundary": (
            "This decision applies only to the frozen catalogue-level claim; "
            "it does not prove or disprove a causal ontology."
        ),
        "inputs": {
            "contract_path": str(contract_path),
            "contract_file_sha256": sha256_file(contract_path),
            "contract_canonical_sha256": contract_fingerprint(contract_payload),
            "catalog_path": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "catalog_rows_after_holdout": len(catalog),
        },
        "configuration": {
            "coordinate_frame": "ICRS",
            "node_count": len(contract.nodes),
            "node_radius_deg": contract.radius_deg,
            "anomaly_rule": {
                "column": contract.anomaly_column,
                "operator": contract.anomaly_operator,
                "threshold": contract.anomaly_threshold,
            },
            "null_model": contract.null_model,
            "permutations": contract.permutations,
            "seed": contract.seed,
            "alpha": contract.alpha,
            "minimum_rate_contrast": contract.minimum_effect,
            "minimum_supported_nodes": contract.minimum_supported_nodes,
            "holdout_column": contract.holdout_column,
            "holdout_value": contract.holdout_value,
            "weight_column": contract.weight_column,
            "stratum_column": contract.stratum_column,
        },
        "global_test": {
            **global_summary.to_dict(),
            "empirical_p": global_p,
            "null_mean": float(global_null_array.mean()),
            "null_std": float(global_null_array.std(ddof=1)),
            "passes": global_pass,
        },
        "node_test": {
            "supported_nodes": supported_nodes,
            "required_supported_nodes": contract.minimum_supported_nodes,
            "holm_familywise_alpha": contract.alpha,
        },
    }
    return receipt, node_table


def audit_files(
    catalog_path: Path, contract_path: Path, output_dir: Path
) -> dict[str, Path]:
    contract, payload = load_contract(contract_path)
    catalog = load_catalog(catalog_path, contract)
    receipt, node_table = run_audit(
        catalog,
        contract,
        contract_payload=payload,
        catalog_path=catalog_path,
        contract_path=contract_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f"{contract.claim_id}_receipt.json"
    nodes_path = output_dir / f"{contract.claim_id}_nodes.csv"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    node_table.to_csv(nodes_path, index=False)
    manifest_path = output_dir / f"{contract.claim_id}_manifest.json"
    manifest = {
        "schema": "uff.sky-lattice-artifact-manifest.v1",
        "claim_id": contract.claim_id,
        "artifacts": [
            {"path": receipt_path.name, "sha256": sha256_file(receipt_path)},
            {"path": nodes_path.name, "sha256": sha256_file(nodes_path)},
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"receipt": receipt_path, "nodes": nodes_path, "manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uff-sky-audit",
        description="Run a preregistered celestial-node and survey-systematics audit.",
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = audit_files(args.catalog, args.contract, args.out)
    for label, path in outputs.items():
        print(f"[OK] {label}: {path}")
    return 0


__all__ = [
    "AuditContract",
    "CONTRACT_SCHEMA",
    "Node",
    "RegionSummary",
    "SCHEMA",
    "audit_files",
    "contract_fingerprint",
    "holm_adjust",
    "load_catalog",
    "load_contract",
    "main",
    "node_membership",
    "radec_to_unit",
    "run_audit",
    "sha256_file",
    "summarize_region",
    "unit_to_radec",
]
