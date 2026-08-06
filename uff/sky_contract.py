"""Frozen claim and catalogue validation for UFF-SLFA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTRACT_SCHEMA = "uff.sky-lattice-claim.v1"
_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


class AuditError(RuntimeError):
    """Raised when a frozen claim, catalogue, bundle, or replay is invalid."""


@dataclass(frozen=True, slots=True)
class Node:
    node_id: str
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True, slots=True)
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
    confirmatory: bool
    independent_catalogue: bool
    holdout_column: str | None = None
    holdout_value: str | None = None
    weight_column: str | None = None
    stratum_column: str | None = None
    expected_catalog_sha256: str | None = None
    discovery_catalog_sha256: str | None = None


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AuditError(f"value cannot be encoded as finite canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite")
    return result


def load_contract_payload(payload: dict[str, Any]) -> AuditContract:
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise AuditError(f"contract schema must be {CONTRACT_SCHEMA!r}")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise AuditError("contract must contain at least one node")
    nodes: list[Node] = []
    identifiers: set[str] = set()
    coordinates: set[tuple[float, float]] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise AuditError("each node must be an object")
        node_id = str(raw.get("id", "")).strip()
        ra = _number(raw.get("ra_deg"), f"node {node_id or '?'} ra_deg")
        dec = _number(raw.get("dec_deg"), f"node {node_id or '?'} dec_deg")
        if not node_id or node_id in identifiers:
            raise AuditError("node ids must be non-empty and unique")
        if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
            raise AuditError(f"node {node_id} coordinates are outside ICRS bounds")
        # Right ascension is undefined at either celestial pole. Normalize it
        # before duplicate detection so, for example, (0, +90) and
        # (180, +90) cannot become duplicate statistical tests of one cap.
        normalized_ra = 0.0 if abs(dec) == 90.0 else round(ra, 12)
        coordinate_key = (normalized_ra, round(dec, 12))
        if coordinate_key in coordinates:
            raise AuditError("duplicate node coordinates are not allowed")
        identifiers.add(node_id)
        coordinates.add(coordinate_key)
        nodes.append(Node(node_id, ra, dec))

    anomaly = payload.get("anomaly_rule", {})
    analysis = payload.get("analysis", {})
    decision = payload.get("decision_rule", {})
    data = payload.get("data", {})
    declarations = payload.get("declarations", {})
    if not all(isinstance(item, dict) for item in (anomaly, analysis, decision, data, declarations)):
        raise AuditError("contract sections must be JSON objects")
    if declarations.get("selection_independent_of_nodes") is not True:
        raise AuditError(
            "declarations.selection_independent_of_nodes must be true; "
            "node-targeted selection cannot validate node clustering"
        )
    operator = str(anomaly.get("operator", ">="))
    if operator not in _OPERATORS:
        raise AuditError(f"unsupported anomaly operator: {operator}")
    radius = _number(payload.get("node_radius_deg"), "node_radius_deg")
    if not 0.0 < radius <= 45.0:
        raise AuditError("node_radius_deg must be in (0, 45]")
    permutations = int(analysis.get("permutations", 10_000))
    if permutations < 99:
        raise AuditError("analysis.permutations must be at least 99")
    null_model = str(analysis.get("null_model", "ra-shift"))
    if null_model not in {"ra-shift", "so3", "stratified-label"}:
        raise AuditError("analysis.null_model must be ra-shift, so3, or stratified-label")
    alpha = _number(decision.get("alpha", 0.05), "decision_rule.alpha")
    if not 0.0 < alpha < 1.0:
        raise AuditError("decision_rule.alpha must be in (0, 1)")
    required = int(decision.get("minimum_supported_nodes", 1))
    if not 0 <= required <= len(nodes):
        raise AuditError("minimum_supported_nodes is outside the node count")
    confirmatory = declarations.get("confirmatory", True) is True
    independent = declarations.get("independent_catalogue", False) is True
    holdout_column = str(data["holdout_column"]) if data.get("holdout_column") else None
    holdout_value_present = "holdout_value" in data and data["holdout_value"] is not None
    if holdout_column and not holdout_value_present:
        raise AuditError("data.holdout_column requires a non-null data.holdout_value")
    if holdout_value_present and not holdout_column:
        raise AuditError("data.holdout_value requires data.holdout_column")
    holdout_value = str(data["holdout_value"]) if holdout_value_present else None
    if confirmatory and not (independent or holdout_column):
        raise AuditError(
            "a confirmatory contract requires declarations.independent_catalogue=true "
            "or a declared holdout column/value"
        )
    stratum_column = str(data["stratum_column"]) if data.get("stratum_column") else None
    if null_model == "stratified-label" and not stratum_column:
        raise AuditError("stratified-label requires data.stratum_column")
    claim_id = str(payload.get("claim_id", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not claim_id or not title:
        raise AuditError("claim_id and title are required")
    return AuditContract(
        claim_id=claim_id,
        title=title,
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
        minimum_supported_nodes=required,
        confirmatory=confirmatory,
        independent_catalogue=independent,
        holdout_column=holdout_column,
        holdout_value=holdout_value,
        weight_column=str(data["weight_column"]) if data.get("weight_column") else None,
        stratum_column=stratum_column,
        expected_catalog_sha256=(str(data["catalog_sha256"]).lower() if data.get("catalog_sha256") else None),
        discovery_catalog_sha256=(
            str(data["discovery_catalog_sha256"]).lower()
            if data.get("discovery_catalog_sha256")
            else None
        ),
    )


def load_contract(path: Path) -> tuple[AuditContract, dict[str, Any]]:
    payload = load_json(path)
    return load_contract_payload(payload), payload


def load_catalogue(path: Path, contract: AuditContract) -> tuple[pd.DataFrame, int]:
    catalogue_sha = sha256_file(path)
    if contract.expected_catalog_sha256 and catalogue_sha != contract.expected_catalog_sha256:
        raise AuditError(
            f"catalogue SHA-256 mismatch: expected {contract.expected_catalog_sha256}, got {catalogue_sha}"
        )
    if (
        contract.confirmatory
        and contract.discovery_catalog_sha256 == catalogue_sha
        and not contract.holdout_column
    ):
        raise AuditError("confirmatory catalogue matches the declared discovery catalogue")
    try:
        frame = pd.read_csv(path).copy()
    except Exception as exc:
        raise AuditError(f"cannot read catalogue CSV {path}: {exc}") from exc
    original_rows = len(frame)
    required = {"ra_deg", "dec_deg", contract.anomaly_column}
    required.update(
        column
        for column in (contract.weight_column, contract.holdout_column, contract.stratum_column)
        if column
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditError(f"catalogue is missing columns: {', '.join(missing)}")
    for column in ("ra_deg", "dec_deg", contract.anomaly_column):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame[["ra_deg", "dec_deg", contract.anomaly_column]].to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise AuditError("catalogue coordinates and anomaly values must be finite")
    if not frame["ra_deg"].between(0.0, 360.0, inclusive="left").all():
        raise AuditError("catalogue ra_deg must be in [0, 360)")
    if not frame["dec_deg"].between(-90.0, 90.0).all():
        raise AuditError("catalogue dec_deg must be in [-90, 90]")
    if contract.holdout_column:
        frame = frame[
            frame[contract.holdout_column].astype(str) == contract.holdout_value
        ].copy()
        if frame.empty:
            raise AuditError("holdout filter produced an empty catalogue")
    if contract.stratum_column and frame[contract.stratum_column].isna().any():
        raise AuditError("stratum values must be non-missing")
    if contract.weight_column:
        frame["_weight"] = pd.to_numeric(frame[contract.weight_column], errors="raise")
        weights = frame["_weight"].to_numpy(float)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise AuditError("weights must be finite and strictly positive")
    else:
        frame["_weight"] = 1.0
    frame["_anomaly"] = _OPERATORS[contract.anomaly_operator](
        frame[contract.anomaly_column].to_numpy(float), contract.anomaly_threshold
    )
    if frame["_anomaly"].nunique() < 2:
        raise AuditError("catalogue must contain anomalous and non-anomalous rows")
    return frame.reset_index(drop=True), original_rows
