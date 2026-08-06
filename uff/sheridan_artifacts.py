"""Deterministic Sheridan Crucible evidence bundles and replay verification."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy

from .sheridan_contract import (
    SHERIDAN_CONTRACT_SCHEMA,
    load_sheridan_contract,
    load_sheridan_contract_payload,
    load_support,
    prepare_catalogue,
)
from .sheridan_density import fit_density, score_nodes
from .sheridan_models import run_injection_recovery, run_model_comparison
from .sky_contract import (
    AuditError,
    canonical_json_bytes,
    load_catalogue,
    load_json,
    sha256_bytes,
    sha256_file,
)

SHERIDAN_ALGORITHM_ID = "uff-sheridan-v1"
SHERIDAN_SOFTWARE_VERSION = "1.1.0"
SHERIDAN_RECIPE_SCHEMA = "uff.sheridan-recipe.v1"
SHERIDAN_MANIFEST_SCHEMA = "uff.sheridan-manifest.v1"
REPLAY_TOLERANCE = 1.0e-11
REQUIRED_ARTIFACTS = frozenset(
    {
        "recipe.json",
        "density.json",
        "nodes.csv",
        "models.json",
        "injection.json",
        "decision.json",
    }
)


@dataclass(frozen=True, slots=True)
class SheridanVerificationReport:
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


def _entry(path: str, data: bytes, media_type: str) -> dict[str, Any]:
    return {
        "path": path,
        "media_type": media_type,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def _nodes_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")


def _component_passes(
    density: dict[str, Any],
    models: dict[str, Any],
    injection: dict[str, Any],
) -> dict[str, bool]:
    return {
        "density": density["decision"] == "SURVEY_AWARE_DENSITY_CRITERIA_MET",
        "model": models["decision"] == "NODE_TERM_PREFERRED",
        "injection": injection["decision"] == "INJECTION_CALIBRATION_PASSED",
    }


def run_sheridan_analysis(
    *,
    catalogue_path: Path,
    support_path: Path,
    contract_path: Path | None = None,
    contract_payload: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (contract_path is None) == (contract_payload is None):
        raise AuditError("provide exactly one of contract_path or contract_payload")
    if contract_path is not None:
        contract, payload = load_sheridan_contract(contract_path)
    else:
        assert contract_payload is not None
        payload = contract_payload
        contract = load_sheridan_contract_payload(payload)

    catalogue, original_rows = load_catalogue(catalogue_path, contract.claim)
    catalogue, completeness_removed = prepare_catalogue(
        catalogue,
        contract.survey,
        contract.models,
    )
    support = load_support(support_path, contract.survey)
    density_fit = fit_density(catalogue, support, contract.density)
    density, nodes = score_nodes(
        density_fit,
        contract.claim,
        contract.density,
        contract.decision,
    )
    models = run_model_comparison(
        catalogue,
        contract.claim,
        contract.models,
        seed=contract.density.seed + 1,
    )
    injection = run_injection_recovery(
        catalogue,
        contract.claim,
        contract.models,
        contract.injection,
        seed=contract.density.seed + 2,
    )
    component_pass = _component_passes(density, models, injection)
    required = contract.decision.required_components
    passed = all(component_pass[name] for name in required)
    decision = {
        "schema": "uff.sheridan-decision.v1",
        "algorithm": SHERIDAN_ALGORITHM_ID,
        "decision": "CRUCIBLE_CRITERIA_MET" if passed else "CRUCIBLE_CRITERIA_NOT_MET",
        "required_components": list(required),
        "component_passes": component_pass,
        "interpretation": (
            "The final decision is the logical conjunction of only the components "
            "frozen as required before analysis. It is not a proof of physical ontology."
        ),
    }
    recipe = {
        "schema": SHERIDAN_RECIPE_SCHEMA,
        "algorithm": SHERIDAN_ALGORITHM_ID,
        "software": {
            "name": "QSOL UFF Sheridan Crucible",
            "version": SHERIDAN_SOFTWARE_VERSION,
        },
        "contract": payload,
        "inputs": {
            "catalogue_path_label": catalogue_path.name,
            "catalogue_sha256": sha256_file(catalogue_path),
            "support_path_label": support_path.name,
            "support_sha256": sha256_file(support_path),
            "contract_canonical_sha256": sha256_bytes(canonical_json_bytes(payload)),
            "catalogue_rows_before_holdout": original_rows,
            "catalogue_rows_after_holdout_and_completeness": len(catalogue),
            "catalogue_rows_removed_by_completeness": completeness_removed,
            "support_points": len(support),
            "support_total_area_sr": float(support["_area_weight_sr"].sum()),
            "support_usable_area_sr": float(
                np.dot(support["_area_weight_sr"], support["_coverage"])
            ),
        },
        "method_boundary": {
            "density": (
                "vMF weighted KDE with leave-one-out bandwidth selection, adaptive "
                "smoothing, coverage quadrature, and edge renormalization"
            ),
            "models": (
                "weighted logistic nuisance model versus the same model plus a frozen "
                "node-membership term; reported BIC is explicitly pseudo-BIC"
            ),
            "injection": (
                "stratum-preserving anomaly-label injection into real in-footprint rows; "
                "not an image-level telescope simulator"
            ),
        },
    }
    return recipe, density, nodes, models, injection, decision


def write_sheridan_bundle(
    output_dir: Path,
    *,
    catalogue_path: Path,
    support_path: Path,
    contract_path: Path,
    force: bool = False,
) -> Path:
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise AuditError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise AuditError(f"output directory is not empty: {output}")
    recipe, density, nodes, models, injection, decision = run_sheridan_analysis(
        catalogue_path=catalogue_path,
        support_path=support_path,
        contract_path=contract_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        ("recipe.json", canonical_json_bytes(recipe), "application/json"),
        ("density.json", canonical_json_bytes(density), "application/json"),
        ("nodes.csv", _nodes_bytes(nodes), "text/csv"),
        ("models.json", canonical_json_bytes(models), "application/json"),
        ("injection.json", canonical_json_bytes(injection), "application/json"),
        ("decision.json", canonical_json_bytes(decision), "application/json"),
    ]
    entries = []
    for relative, data, media_type in payloads:
        _atomic_write(output / relative, data)
        entries.append(_entry(relative, data, media_type))
    manifest = {
        "schema": SHERIDAN_MANIFEST_SCHEMA,
        "algorithm": SHERIDAN_ALGORITHM_ID,
        "runtime": {
            "qsol_uff_sheridan": SHERIDAN_SOFTWARE_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "result": decision["decision"],
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
        if (
            not isinstance(recorded, (int, float))
            or isinstance(recorded, bool)
            or not np.isclose(
                float(recorded),
                replayed,
                rtol=REPLAY_TOLERANCE,
                atol=REPLAY_TOLERANCE,
                equal_nan=True,
            )
        ):
            errors.append(f"{location} does not numerically replay")
    elif recorded != replayed:
        errors.append(f"{location} does not replay")


def verify_sheridan_bundle(
    manifest_path: Path,
    *,
    catalogue_path: Path | None = None,
    support_path: Path | None = None,
) -> SheridanVerificationReport:
    checks: list[str] = []
    errors: list[str] = []
    try:
        manifest = load_json(manifest_path)
    except AuditError as exc:
        return SheridanVerificationReport(False, False, None, (), (str(exc),))
    if (
        manifest.get("schema") != SHERIDAN_MANIFEST_SCHEMA
        or manifest.get("algorithm") != SHERIDAN_ALGORITHM_ID
    ):
        return SheridanVerificationReport(False, False, None, (), ("incompatible manifest",))
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        return SheridanVerificationReport(
            False,
            False,
            None,
            (),
            ("manifest artifacts must be a list",),
        )
    base = manifest_path.parent
    resolved: dict[str, Path] = {}
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
        resolved[relative] = path
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"byte-size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"SHA-256 mismatch: {relative}")
    missing = sorted(REQUIRED_ARTIFACTS - seen)
    unexpected = sorted(seen - REQUIRED_ARTIFACTS)
    if missing:
        errors.append(f"manifest is missing required artifacts: {', '.join(missing)}")
    if unexpected:
        errors.append(f"manifest contains unexpected artifacts: {', '.join(unexpected)}")
    integrity = not errors
    if integrity:
        checks.append("artifact byte sizes and SHA-256 hashes match the manifest")

    replay: bool | None = None
    replay_requested = catalogue_path is not None or support_path is not None
    if replay_requested and (catalogue_path is None or support_path is None):
        errors.append("numerical replay requires both catalogue_path and support_path")
        replay = False
    elif integrity and catalogue_path is not None and support_path is not None:
        try:
            recipe = load_json(resolved["recipe.json"])
            if (
                recipe.get("schema") != SHERIDAN_RECIPE_SCHEMA
                or recipe.get("algorithm") != SHERIDAN_ALGORITHM_ID
            ):
                raise AuditError("incompatible recipe")
            inputs = recipe.get("inputs", {})
            if sha256_file(catalogue_path) != inputs.get("catalogue_sha256"):
                raise AuditError("supplied catalogue does not match recipe SHA-256")
            if sha256_file(support_path) != inputs.get("support_sha256"):
                raise AuditError("supplied support grid does not match recipe SHA-256")
            replayed = run_sheridan_analysis(
                catalogue_path=catalogue_path,
                support_path=support_path,
                contract_payload=recipe["contract"],
            )
            recorded = (
                recipe,
                load_json(resolved["density.json"]),
                pd.read_csv(resolved["nodes.csv"]),
                load_json(resolved["models.json"]),
                load_json(resolved["injection.json"]),
                load_json(resolved["decision.json"]),
            )
            replay_errors: list[str] = []
            for name, left, right in zip(
                ("recipe", "density", "nodes", "models", "injection", "decision"),
                recorded,
                replayed,
                strict=True,
            ):
                if name == "nodes":
                    if list(left.columns) != list(right.columns) or len(left) != len(right):
                        replay_errors.append("nodes.csv shape does not replay")
                        continue
                    for column in right.columns:
                        if pd.api.types.is_numeric_dtype(right[column]):
                            if not np.allclose(
                                left[column].to_numpy(float),
                                right[column].to_numpy(float),
                                rtol=REPLAY_TOLERANCE,
                                atol=REPLAY_TOLERANCE,
                                equal_nan=True,
                            ):
                                replay_errors.append(f"nodes.csv column {column} does not replay")
                        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
                            replay_errors.append(f"nodes.csv column {column} does not replay")
                else:
                    _compare(left, right, name, replay_errors)
            if replay_errors:
                errors.extend(replay_errors)
                replay = False
            else:
                replay = True
                checks.append("numerical replay matches all Sheridan artifacts")
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"numerical replay failed: {exc}")
            replay = False

    return SheridanVerificationReport(
        passed=integrity and replay is not False,
        integrity_passed=integrity,
        replay_passed=replay,
        checks=tuple(checks),
        errors=tuple(errors),
    )
