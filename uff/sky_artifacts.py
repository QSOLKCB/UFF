"""Evidence bundles and replay verification for UFF-SLFA."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from .sky_contract import (
    AuditError,
    canonical_json_bytes,
    load_catalogue,
    load_contract,
    load_contract_payload,
    load_json,
    sha256_bytes,
    sha256_file,
)
from .sky_statistics import ALGORITHM_ID, run_audit

RECIPE_SCHEMA = "uff.sky-lattice-recipe.v1"
MANIFEST_SCHEMA = "uff.sky-lattice-manifest.v1"
SOFTWARE_VERSION = "1.0.0"
REPLAY_TOLERANCE = 1.0e-12
REQUIRED_ARTIFACTS = frozenset({"recipe.json", "observations.json", "nodes.csv"})


@dataclass(frozen=True, slots=True)
class VerificationReport:
    passed: bool
    integrity_passed: bool
    replay_passed: bool | None
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _node_csv_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _entry(path: str, data: bytes, media_type: str) -> dict[str, Any]:
    return {
        "path": path,
        "media_type": media_type,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def write_bundle(
    output_dir: Path,
    *,
    catalogue_path: Path,
    contract_path: Path,
    force: bool = False,
) -> Path:
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise AuditError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise AuditError(f"output directory is not empty: {output}")
    contract, payload = load_contract(contract_path)
    catalogue, original_rows = load_catalogue(catalogue_path, contract)
    observations, nodes = run_audit(catalogue, contract)
    output.mkdir(parents=True, exist_ok=True)
    recipe = {
        "schema": RECIPE_SCHEMA,
        "algorithm": ALGORITHM_ID,
        "software": {"name": "QSOL UFF-SLFA", "version": SOFTWARE_VERSION},
        "contract": payload,
        "inputs": {
            "catalogue_path_label": catalogue_path.name,
            "catalogue_sha256": sha256_file(catalogue_path),
            "catalogue_rows_before_holdout": original_rows,
            "catalogue_rows_after_holdout": len(catalogue),
            "contract_path_label": contract_path.name,
            "contract_file_sha256": sha256_file(contract_path),
            "contract_canonical_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        "replay_contract": {
            "integrity": "Manifest byte-size and SHA-256 checks protect the stored bundle.",
            "numerical": "The exact frozen catalogue reruns the deterministic algorithm.",
            "boundary": "Integrity and replay establish consistency, not physical truth.",
        },
    }
    payloads = [
        ("recipe.json", canonical_json_bytes(recipe), "application/json"),
        ("observations.json", canonical_json_bytes(observations), "application/json"),
        ("nodes.csv", _node_csv_bytes(nodes), "text/csv"),
    ]
    entries = []
    for relative, data, media_type in payloads:
        _atomic_write(output / relative, data)
        entries.append(_entry(relative, data, media_type))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "algorithm": ALGORITHM_ID,
        "runtime": {
            "qsol_uff_slfa": SOFTWARE_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "result": observations["decision"],
        "claim_boundary": observations["claim_boundary"],
        "artifacts": sorted(entries, key=lambda item: item["path"]),
    }
    manifest_path = output / "manifest.json"
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    return manifest_path


def _safe_path(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise AuditError("artifact path must be a non-empty POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise AuditError(f"unsafe artifact path: {relative!r}")
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise AuditError(f"artifact escapes bundle directory: {relative!r}")
    return resolved


def _compare(recorded: Any, replayed: Any, location: str, errors: list[str]) -> None:
    if isinstance(replayed, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(replayed):
            errors.append(f"{location} has missing or unexpected fields")
            return
        for key in sorted(replayed):
            _compare(recorded[key], replayed[key], f"{location}.{key}", errors)
    elif isinstance(replayed, list):
        if not isinstance(recorded, list) or len(recorded) != len(replayed):
            errors.append(f"{location} has the wrong list shape")
            return
        for index, (left, right) in enumerate(zip(recorded, replayed, strict=True)):
            _compare(left, right, f"{location}[{index}]", errors)
    elif isinstance(replayed, bool) or replayed is None or isinstance(replayed, str):
        if recorded != replayed:
            errors.append(f"{location} does not replay exactly")
    elif isinstance(replayed, int):
        if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded != replayed:
            errors.append(f"{location} does not replay exactly")
    elif isinstance(replayed, float):
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or not np.isclose(
            float(recorded), replayed, rtol=REPLAY_TOLERANCE, atol=REPLAY_TOLERANCE
        ):
            errors.append(f"{location} does not numerically replay")
    elif recorded != replayed:
        errors.append(f"{location} does not replay")


def verify_bundle(manifest_path: Path, catalogue_path: Path | None = None) -> VerificationReport:
    checks: list[str] = []
    errors: list[str] = []
    manifest = load_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("algorithm") != ALGORITHM_ID:
        return VerificationReport(False, False, None, (), ("incompatible manifest",))
    base = manifest_path.parent
    resolved: dict[str, Path] = {}
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        return VerificationReport(False, False, None, (), ("manifest artifacts must be a list",))
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest artifact entry is not an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            errors.append("artifact path must be a non-empty POSIX relative path")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative!r}")
            continue
        seen.add(relative)
        try:
            path = _safe_path(base, relative)
        except AuditError as exc:
            errors.append(str(exc))
            continue
        resolved[str(relative)] = path
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"byte-size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"SHA-256 mismatch: {relative}")
    missing_artifacts = sorted(REQUIRED_ARTIFACTS - seen)
    unexpected_artifacts = sorted(seen - REQUIRED_ARTIFACTS)
    if missing_artifacts:
        errors.append(f"manifest is missing required artifacts: {', '.join(missing_artifacts)}")
    if unexpected_artifacts:
        errors.append(f"manifest contains unexpected artifacts: {', '.join(unexpected_artifacts)}")
    integrity = not errors
    if integrity:
        checks.append("artifact byte sizes and SHA-256 hashes match the manifest")
    replay: bool | None = None
    if catalogue_path is not None and integrity:
        try:
            recipe = load_json(resolved["recipe.json"])
            recorded = load_json(resolved["observations.json"])
            recorded_nodes = pd.read_csv(resolved["nodes.csv"])
        except (KeyError, AuditError, OSError, ValueError) as exc:
            errors.append(f"cannot load replay artifacts: {exc}")
            replay = False
        else:
            try:
                if recipe.get("schema") != RECIPE_SCHEMA or recipe.get("algorithm") != ALGORITHM_ID:
                    raise AuditError("incompatible recipe")
                if sha256_file(catalogue_path) != recipe.get("inputs", {}).get("catalogue_sha256"):
                    raise AuditError("supplied replay catalogue does not match recipe SHA-256")
                contract = load_contract_payload(recipe["contract"])
                catalogue, _ = load_catalogue(catalogue_path, contract)
                replayed, replayed_nodes = run_audit(catalogue, contract)
                replay_errors: list[str] = []
                _compare(recorded, replayed, "observations", replay_errors)
                if list(recorded_nodes.columns) != list(replayed_nodes.columns) or len(recorded_nodes) != len(replayed_nodes):
                    replay_errors.append("nodes.csv shape does not replay")
                else:
                    for column in replayed_nodes.columns:
                        left, right = recorded_nodes[column], replayed_nodes[column]
                        if pd.api.types.is_numeric_dtype(right):
                            if not np.allclose(
                                left.to_numpy(float), right.to_numpy(float),
                                rtol=REPLAY_TOLERANCE, atol=REPLAY_TOLERANCE,
                                equal_nan=True,
                            ):
                                replay_errors.append(f"nodes.csv column {column} does not replay")
                        elif left.astype(str).tolist() != right.astype(str).tolist():
                            replay_errors.append(f"nodes.csv column {column} does not replay")
                if replay_errors:
                    errors.extend(replay_errors)
                    replay = False
                else:
                    replay = True
                    checks.append("numerical replay matches observations and node table")
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"numerical replay failed: {exc}")
                replay = False
    return VerificationReport(
        passed=integrity and replay is not False,
        integrity_passed=integrity,
        replay_passed=replay,
        checks=tuple(checks),
        errors=tuple(errors),
    )
