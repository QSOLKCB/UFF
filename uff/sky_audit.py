"""Preregistered, deterministic audit for claimed celestial-node lattices.

UFF-SLFA tests a frozen catalogue-level spatial claim. It does not infer a
causal ontology from a spatial association and it does not treat catalogue
quality diagnostics as physical objects by default.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

CONTRACT_SCHEMA = "uff.sky-lattice-claim.v1"
OBSERVATIONS_SCHEMA = "uff.sky-lattice-observations.v1"
MANIFEST_SCHEMA = "uff.sky-lattice-manifest.v1"
ALGORITHM_ID = "uff-slfa-v1"
ROTATION_TOLERANCE = 1.0e-10
GRAM_TOLERANCE = 1.0e-10

_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


class AuditError(RuntimeError):
    """Raised when an audit or verification contract cannot be satisfied."""


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
    holdout_column: str | None = None
    holdout_value: str | None = None
    weight_column: str | None = None
    stratum_column: str | None = None
    expected_catalog_sha256: str | None = None


@dataclass(frozen=True, slots=True)
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
        # Haldane-Anscombe correction makes the diagnostic finite.
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


@dataclass(frozen=True, slots=True)
class VerificationReport:
    passed: bool
    integrity_passed: bool
    replay_passed: bool | None
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AuditError(f"value cannot be encoded as finite canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite")
    return result


def load_contract(path: Path) -> tuple[AuditContract, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read contract: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CONTRACT_SCHEMA:
        raise AuditError(f"contract schema must be {CONTRACT_SCHEMA!r}")

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise AuditError("contract must contain at least one node")
    nodes: list[Node] = []
    seen: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            raise AuditError("every node must be an object")
        node_id = str(item.get("id", "")).strip()
        if not node_id or node_id in seen:
            raise AuditError("node ids must be non-empty and unique")
        seen.add(node_id)
        ra = _number(item.get("ra_deg"), f"node {node_id} ra_deg")
        dec = _number(item.get("dec_deg"), f"node {node_id} dec_deg")
        if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
            raise AuditError(f"node {node_id} coordinates are outside ICRS bounds")
        nodes.append(Node(node_id, ra, dec))

    anomaly = payload.get("anomaly_rule", {})
    analysis = payload.get("analysis", {})
    decision = payload.get("decision_rule", {})
    data = payload.get("data", {})
    if not all(isinstance(value, dict) for value in (anomaly, analysis, decision, data)):
        raise AuditError("anomaly_rule, analysis, decision_rule, and data must be objects")

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
        raise AuditError("null_model must be ra-shift, so3, or stratified-label")
    alpha = _number(decision.get("alpha", 0.05), "decision_rule.alpha")
    if not 0.0 < alpha < 1.0:
        raise AuditError("decision_rule.alpha must be in (0, 1)")
    minimum_supported_nodes = int(decision.get("minimum_supported_nodes", 1))
    if not 0 <= minimum_supported_nodes <= len(nodes):
        raise AuditError("minimum_supported_nodes is outside the node count")

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
        minimum_supported_nodes=minimum_supported_nodes,
        holdout_column=str(data["holdout_column"]) if data.get("holdout_column")-®éÜj×o&¬²+-j{]¹ëh{²È¯zW§‚Ø