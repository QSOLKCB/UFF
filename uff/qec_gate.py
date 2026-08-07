"""QEC-inspired fail-closed trust boundary for UFF evidence bundles.

This module intentionally imports only the small subset of QEC ideas that UFF
needs at its evidence boundary: strict canonical JSON, child-before-root
validation, recompute-not-trust replay, explicit assurance states, closed bundle
contents, and an optional externally anchorable root hash.

It is not the QEC architecture and it does not promote computational integrity
into scientific truth.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


GATE_SCHEMA = "uff.qec-boundary-gate.v1"
ROOT_SCHEMA = "uff.qec-bundle-root.v1"
GATE_FILENAME = "qec_gate.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_BOUNDARY = (
    "This receipt establishes strict bundle integrity and successful deterministic replay. "
    "It does not establish source-data correctness, statistical adequacy, causal explanation, "
    "or physical ontology."
)


class GateError(RuntimeError):
    """Raised when a bundle cannot satisfy the QEC boundary contract."""


@dataclass(frozen=True, slots=True)
class _Profile:
    name: str
    manifest_schema: str
    algorithm: str
    recipe_schema: str
    artifacts: dict[str, str]
    decision_artifact: str
    decision_schema: str
    require_claim_boundary: bool = False


_PROFILES = {
    "uff.sky-lattice-manifest.v1": _Profile(
        name="uff-slfa-qec-gate-v1",
        manifest_schema="uff.sky-lattice-manifest.v1",
        algorithm="uff-slfa-v1",
        recipe_schema="uff.sky-lattice-recipe.v1",
        artifacts={
            "recipe.json": "application/json",
            "observations.json": "application/json",
            "nodes.csv": "text/csv",
        },
        decision_artifact="observations.json",
        decision_schema="uff.sky-lattice-observations.v1",
        require_claim_boundary=True,
    ),
    "uff.sheridan-manifest.v1": _Profile(
        name="uff-sheridan-qec-gate-v1",
        manifest_schema="uff.sheridan-manifest.v1",
        algorithm="uff-sheridan-v1",
        recipe_schema="uff.sheridan-recipe.v1",
        artifacts={
            "recipe.json": "application/json",
            "density.json": "application/json",
            "nodes.csv": "text/csv",
            "models.json": "application/json",
            "injection.json": "application/json",
            "decision.json": "application/json",
        },
        decision_artifact="decision.json",
        decision_schema="uff.sheridan-decision.v1",
    ),
}


@dataclass(frozen=True, slots=True)
class GateReport:
    """Unified gate verdict.

    ``admitted`` is deliberately stricter than the domain verifiers' historical
    ``passed`` field: admission requires fresh numerical replay, not merely an
    intact bundle.
    """

    admitted: bool
    assurance: str
    profile: str | None
    integrity_passed: bool
    replay_passed: bool | None
    root_sha256: str | None
    contract_canonical_sha256: str | None
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "uff.qec-boundary-gate-report.v1",
            "admitted": self.admitted,
            "assurance": self.assurance,
            "profile": self.profile,
            "integrity_passed": self.integrity_passed,
            "replay_passed": self.replay_passed,
            "root_sha256": self.root_sha256,
            "contract_canonical_sha256": self.contract_canonical_sha256,
            "checks": list(self.checks),
            "errors": list(self.errors),
        }


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
        raise GateError(f"value cannot be represented as finite canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GateError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise GateError(f"non-finite JSON constant is forbidden: {token}")


def _strict_json_bytes(
    data: bytes,
    *,
    label: str,
    require_canonical: bool = True,
) -> dict[str, Any]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise GateError(f"{label} must not contain a UTF-8 BOM")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except GateError:
        raise
    except json.JSONDecodeError as exc:
        raise GateError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} JSON root must be an object")
    if require_canonical and canonical_json_bytes(value) != data:
        raise GateError(f"{label} is valid JSON but not canonical JSON")
    return value


def _strict_json_file(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    return _strict_json_bytes(data, label=path.name, require_canonical=require_canonical)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _safe_child(base: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise GateError("artifact path must be a non-empty POSIX relative path")
    child = Path(relative)
    if child.is_absolute() or any(part in {"", ".", ".."} for part in child.parts):
        raise GateError(f"unsafe artifact path: {relative!r}")
    candidate = base / child
    if candidate.is_symlink():
        raise GateError(f"symbolic-link artifacts are forbidden: {relative}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise GateError(f"artifact escapes bundle directory: {relative!r}")
    return candidate


def _physical_files(base: Path) -> set[str]:
    files: set[str] = set()
    for path in base.rglob("*"):
        if path.is_symlink():
            raise GateError(f"symbolic links are forbidden inside evidence bundles: {path.name}")
        if path.is_file():
            files.add(path.relative_to(base).as_posix())
    return files


def _root_payload(
    profile: _Profile,
    manifest_bytes: bytes,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    children = [
        {
            "path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return {
        "schema": ROOT_SCHEMA,
        "profile": profile.name,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "children": children,
        "self_hash_exclusion": GATE_FILENAME,
    }


def _receipt_payload(
    *,
    profile: str,
    root_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Construct the one exact receipt shape accepted by the gate."""

    return {
        "schema": GATE_SCHEMA,
        "profile": profile,
        "assurance": "REPLAY_VERIFIED",
        "root_sha256": root_sha256,
        "manifest_sha256": manifest_sha256,
        "self_hash_exclusion": GATE_FILENAME,
        "recompute_not_trust": True,
        "external_anchor_required_for_authenticity": True,
        "boundary": _RECEIPT_BOUNDARY,
    }


def _inspect_bundle(manifest_path: Path, expected_root: str | None) -> tuple[
    _Profile | None,
    dict[str, Any] | None,
    str | None,
    str | None,
    list[str],
    list[str],
]:
    checks: list[str] = []
    errors: list[str] = []
    contract_sha256: str | None = None
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    if manifest_path.name != "manifest.json":
        errors.append("gate entry point must be a file named manifest.json")
        return None, None, None, None, checks, errors
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = _strict_json_bytes(
            manifest_bytes,
            label="manifest.json",
            require_canonical=True,
        )
    except (OSError, GateError) as exc:
        errors.append(str(exc))
        return None, None, None, None, checks, errors

    profile = _PROFILES.get(manifest.get("schema"))
    if profile is None:
        errors.append(f"unsupported UFF manifest schema: {manifest.get('schema')!r}")
        return None, manifest, None, None, checks, errors
    if manifest.get("algorithm") != profile.algorithm:
        errors.append("manifest algorithm does not match the selected gate profile")

    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, list):
        errors.append("manifest artifacts must be a list")
        return profile, manifest, None, None, checks, errors

    expected_paths = set(profile.artifacts)
    seen: set[str] = set()
    normalized_entries: list[dict[str, Any]] = []
    json_children: dict[str, dict[str, Any]] = {}

    for raw in raw_entries:
        if not isinstance(raw, dict):
            errors.append("manifest artifact entry is not an object")
            continue
        if set(raw) != {"path", "media_type", "bytes", "sha256"}:
            errors.append(
                "manifest artifact entries must contain exactly "
                "path, media_type, bytes and sha256"
            )
            continue
        relative = raw.get("path")
        if not isinstance(relative, str):
            errors.append("artifact path must be a string")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative!r}")
            continue
        seen.add(relative)
        if relative not in expected_paths:
            errors.append(f"artifact is not admitted by profile: {relative}")
            continue
        expected_media = profile.artifacts[relative]
        if raw.get("media_type") != expected_media:
            errors.append(f"media type mismatch for {relative}")
        byte_count = raw.get("bytes")
        if type(byte_count) is not int or byte_count < 0:
            errors.append(f"artifact byte count must be a non-negative integer: {relative}")
        digest = raw.get("sha256")
        if not _valid_sha256(digest):
            errors.append(f"artifact SHA-256 is malformed: {relative}")
        try:
            child = _safe_child(base, relative)
        except GateError as exc:
            errors.append(str(exc))
            continue
        if not child.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        try:
            data = child.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read artifact {relative}: {exc}")
            continue
        if type(byte_count) is int and len(data) != byte_count:
            errors.append(f"byte-size mismatch: {relative}")
        actual_digest = _sha256_bytes(data)
        if isinstance(digest, str) and actual_digest != digest:
            errors.append(f"SHA-256 mismatch: {relative}")
        if expected_media == "application/json":
            try:
                json_children[relative] = _strict_json_bytes(
                    data,
                    label=relative,
                    require_canonical=True,
                )
            except GateError as exc:
                errors.append(str(exc))
        normalized_entries.append(raw)

    missing = sorted(expected_paths - seen)
    if missing:
        errors.append(f"manifest is missing required artifacts: {', '.join(missing)}")
    if seen - expected_paths:
        errors.append("manifest contains artifacts outside the gate profile")

    try:
        physical = _physical_files(base)
    except GateError as exc:
        errors.append(str(exc))
        physical = set()
    allowed_physical = expected_paths | {"manifest.json", GATE_FILENAME}
    extra_physical = sorted(physical - allowed_physical)
    missing_physical = sorted((expected_paths | {"manifest.json"}) - physical)
    if extra_physical:
        errors.append(f"bundle contains unlisted files: {', '.join(extra_physical)}")
    if missing_physical:
        errors.append(f"bundle is physically incomplete: {', '.join(missing_physical)}")

    recipe = json_children.get("recipe.json")
    decision = json_children.get(profile.decision_artifact)
    if recipe is not None:
        if recipe.get("schema") != profile.recipe_schema or recipe.get("algorithm") != profile.algorithm:
            errors.append("recipe schema or algorithm is incompatible with the gate profile")
        contract = recipe.get("contract")
        inputs = recipe.get("inputs")
        if not isinstance(contract, dict) or not isinstance(inputs, dict):
            errors.append("recipe must contain object-valued contract and inputs")
        else:
            claimed_contract_hash = inputs.get("contract_canonical_sha256")
            contract_sha256 = _sha256_bytes(canonical_json_bytes(contract))
            if claimed_contract_hash != contract_sha256:
                errors.append(
                    "recipe contract_canonical_sha256 does not match the embedded contract"
                )
            if not _valid_sha256(inputs.get("catalogue_sha256")):
                errors.append("recipe catalogue_sha256 is missing or malformed")
            if profile.manifest_schema == "uff.sheridan-manifest.v1" and not _valid_sha256(
                inputs.get("support_sha256")
            ):
                errors.append("Sheridan recipe support_sha256 is missing or malformed")

    if decision is not None:
        if decision.get("schema") != profile.decision_schema or decision.get("algorithm") != profile.algorithm:
            errors.append("decision artifact schema or algorithm is incompatible with the gate profile")
        if manifest.get("result") != decision.get("decision"):
            errors.append("manifest result does not match the decision artifact")
        if profile.require_claim_boundary:
            boundary = decision.get("claim_boundary")
            if not isinstance(boundary, str) or not boundary.strip():
                errors.append("SLFA observations must contain a non-empty claim boundary")
            if manifest.get("claim_boundary") != boundary:
                errors.append("manifest claim_boundary does not match observations.json")

    root_sha256: str | None = None
    if not errors:
        root_sha256 = _sha256_bytes(
            canonical_json_bytes(_root_payload(profile, manifest_bytes, normalized_entries))
        )
        checks.append("strict canonical JSON, closed artifact set and child hashes verified")
        checks.append("recipe and decision cross-links recomputed instead of trusted")
        if expected_root is not None:
            if not _valid_sha256(expected_root):
                errors.append(
                    "expected_root must be a lowercase 64-character SHA-256 digest"
                )
            elif root_sha256 != expected_root:
                errors.append(
                    "bundle root does not match the externally supplied trust anchor"
                )
            else:
                checks.append("bundle root matches the external trust anchor")

    receipt_path = base / GATE_FILENAME
    if receipt_path.exists() and not errors and root_sha256 is not None:
        try:
            receipt = _strict_json_file(receipt_path, require_canonical=True)
        except GateError as exc:
            errors.append(str(exc))
        else:
            expected_receipt = _receipt_payload(
                profile=profile.name,
                root_sha256=root_sha256,
                manifest_sha256=_sha256_bytes(manifest_bytes),
            )
            if receipt != expected_receipt:
                errors.append(
                    "qec_gate.json does not exactly match the deterministic sealed receipt"
                )
            else:
                checks.append("sealed gate receipt exactly matches the recomputed receipt")

    return profile, manifest, root_sha256, contract_sha256, checks, errors


def verify_boundary(
    manifest_path: Path,
    *,
    catalogue_path: Path | None = None,
    support_path: Path | None = None,
    expected_root: str | None = None,
    require_replay: bool = True,
) -> GateReport:
    """Verify a UFF bundle through the strict QEC-inspired boundary.

    Integrity-only inspection is available for diagnostics, but never produces
    ``admitted=True``. Admission requires the appropriate domain verifier to
    recompute the numerical result from the frozen source inputs.
    """

    profile, _manifest, root, contract_sha256, checks, errors = _inspect_bundle(
        Path(manifest_path),
        expected_root,
    )
    structural_integrity = profile is not None and not errors
    replay: bool | None = None

    # Integrity-only is a hard mode boundary. Extra replay inputs are ignored so
    # diagnostics can never accidentally upgrade themselves to REPLAY_VERIFIED.
    if structural_integrity and profile is not None and require_replay:
        if profile.manifest_schema == "uff.sky-lattice-manifest.v1":
            if support_path is not None:
                errors.append("SLFA replay does not accept a support grid")
                replay = False
            elif catalogue_path is None:
                errors.append(
                    "SLFA gate admission requires the frozen catalogue for numerical replay"
                )
            else:
                from .sky_artifacts import verify_bundle

                domain = verify_bundle(Path(manifest_path), Path(catalogue_path))
                if not domain.integrity_passed:
                    errors.extend(f"SLFA verifier: {item}" for item in domain.errors)
                replay = domain.replay_passed is True
                if replay:
                    checks.append(
                        "SLFA numerical replay independently reproduced the stored result"
                    )
                else:
                    errors.extend(f"SLFA replay: {item}" for item in domain.errors)
        elif profile.manifest_schema == "uff.sheridan-manifest.v1":
            if catalogue_path is None or support_path is None:
                errors.append(
                    "Sheridan gate admission requires both frozen catalogue and support grid"
                )
            else:
                from .sheridan_artifacts import verify_sheridan_bundle

                domain = verify_sheridan_bundle(
                    Path(manifest_path),
                    catalogue_path=Path(catalogue_path),
                    support_path=Path(support_path),
                )
                if not domain.integrity_passed:
                    errors.extend(f"Sheridan verifier: {item}" for item in domain.errors)
                replay = domain.replay_passed is True
                if replay:
                    checks.append(
                        "Sheridan numerical replay independently reproduced the stored result"
                    )
                else:
                    errors.extend(f"Sheridan replay: {item}" for item in domain.errors)

    integrity = structural_integrity and not any(
        error.startswith("bundle root does not match") or error.startswith("expected_root")
        for error in errors
    )
    admitted = integrity and replay is True and not errors
    if admitted:
        assurance = "REPLAY_VERIFIED"
    elif integrity and replay is None:
        assurance = "INTEGRITY_ONLY"
    else:
        assurance = "REJECTED"

    return GateReport(
        admitted=admitted,
        assurance=assurance,
        profile=profile.name if profile is not None else None,
        integrity_passed=integrity,
        replay_passed=replay,
        root_sha256=root,
        contract_canonical_sha256=contract_sha256,
        checks=tuple(checks),
        errors=tuple(errors),
    )


def seal_boundary(
    manifest_path: Path,
    *,
    catalogue_path: Path,
    support_path: Path | None = None,
    expected_root: str | None = None,
) -> Path:
    """Write a replay-verified receipt beside a bundle.

    The receipt excludes itself from the bundle root. Its root becomes an
    authenticity claim only when copied to an independent trust anchor such as
    a signed release, DOI record, preregistration, or separately signed message.
    """

    report = verify_boundary(
        manifest_path,
        catalogue_path=catalogue_path,
        support_path=support_path,
        expected_root=expected_root,
        require_replay=True,
    )
    if not report.admitted or report.root_sha256 is None or report.profile is None:
        detail = "; ".join(report.errors) or "bundle was not admitted"
        raise GateError(f"refusing to seal bundle: {detail}")
    manifest_path = Path(manifest_path)
    receipt = _receipt_payload(
        profile=report.profile,
        root_sha256=report.root_sha256,
        manifest_sha256=_sha256_file(manifest_path),
    )
    output = manifest_path.parent / GATE_FILENAME
    output.write_bytes(canonical_json_bytes(receipt))
    return output


def _main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uff.qec_gate",
        description="Fail-closed QEC-inspired integrity and replay gate for UFF evidence bundles.",
    )
    parser.add_argument("manifest", type=Path, help="Path to a UFF manifest.json")
    parser.add_argument("--catalogue", type=Path, help="Frozen catalogue used for replay")
    parser.add_argument("--support", type=Path, help="Frozen Sheridan support grid")
    parser.add_argument(
        "--expected-root",
        help="Externally anchored lowercase SHA-256 bundle root",
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Inspect structure and hashes without replay; never returns an ADMIT verdict",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help=f"Write {GATE_FILENAME} after a successful replay-verified admission",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _main_parser()
    args = parser.parse_args(argv)
    if args.seal and args.integrity_only:
        parser.error("--seal cannot be combined with --integrity-only")
    if args.seal and args.catalogue is None:
        parser.error("--seal requires --catalogue")

    report = verify_boundary(
        args.manifest,
        catalogue_path=args.catalogue,
        support_path=args.support,
        expected_root=args.expected_root,
        require_replay=not args.integrity_only,
    )
    if args.seal:
        if not report.admitted:
            print(canonical_json_bytes(report.to_dict()).decode("utf-8"), end="")
            return 2
        seal_boundary(
            args.manifest,
            catalogue_path=args.catalogue,
            support_path=args.support,
            expected_root=args.expected_root,
        )
        report = verify_boundary(
            args.manifest,
            catalogue_path=args.catalogue,
            support_path=args.support,
            expected_root=args.expected_root,
            require_replay=True,
        )

    print(canonical_json_bytes(report.to_dict()).decode("utf-8"), end="")
    if args.integrity_only:
        return 0 if report.integrity_passed else 2
    return 0 if report.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
