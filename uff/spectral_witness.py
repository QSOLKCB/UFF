"""SPECTRAL-inspired pre-observation identity witness for UFF.

The witness freezes identity-bearing inputs before a result is admitted. It is
purposefully small: source hashes, canonical contract identity, domain-separated
commitment, and reveal through the QEC boundary gate.

A local commitment does not prove analyst blindness or chronology by itself.
For a historical precommitment claim, publish/sign the commitment digest outside
the analysis directory before observing the result and verify it with
``expected_commit_sha256`` during reveal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .qec_gate import canonical_json_bytes, verify_boundary


WITNESS_SCHEMA = "uff.spectral-witness.v1"
REPORT_SCHEMA = "uff.spectral-witness-report.v1"
_DOMAIN = b"UFF/SPECTRAL-WITNESS/v1\0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WitnessError(RuntimeError):
    """Raised when a SPECTRAL-style witness is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class WitnessReport:
    admitted: bool
    witness_verified: bool
    qec_admitted: bool
    external_anchor_verified: bool | None
    target_profile: str | None
    commit_sha256: str | None
    bundle_root_sha256: str | None
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "admitted": self.admitted,
            "witness_verified": self.witness_verified,
            "qec_admitted": self.qec_admitted,
            "external_anchor_verified": self.external_anchor_verified,
            "target_profile": self.target_profile,
            "commit_sha256": self.commit_sha256,
            "bundle_root_sha256": self.bundle_root_sha256,
            "checks": list(self.checks),
            "errors": list(self.errors),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WitnessError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise WitnessError(f"non-finite JSON constant is forbidden: {token}")


def _load_json(path: Path, *, require_canonical: bool) -> dict[str, Any]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise WitnessError(f"cannot read {path}: {exc}") from exc
    if data.startswith(b"\xef\xbb\xbf"):
        raise WitnessError(f"{Path(path).name} must not contain a UTF-8 BOM")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WitnessError(f"{Path(path).name} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except WitnessError:
        raise
    except json.JSONDecodeError as exc:
        raise WitnessError(f"{Path(path).name} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WitnessError(f"{Path(path).name} JSON root must be an object")
    if require_canonical and canonical_json_bytes(value) != data:
        raise WitnessError(f"{Path(path).name} is valid JSON but not canonical JSON")
    return value


def _target_profile(contract: dict[str, Any], support_path: Path | None) -> str:
    schema = contract.get("schema")
    if schema == "uff.sky-lattice-claim.v1":
        if support_path is not None:
            raise WitnessError("SLFA witness creation does not accept a support grid")
        return "uff-slfa-qec-gate-v1"
    if schema == "uff.sheridan-crucible.v1":
        if support_path is None:
            raise WitnessError("Sheridan witness creation requires a support grid")
        return "uff-sheridan-qec-gate-v1"
    raise WitnessError(f"unsupported UFF contract schema: {schema!r}")


def _identity(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    catalogue_path: Path,
    support_path: Path | None,
) -> dict[str, Any]:
    return {
        "contract_file_sha256": _sha256_file(contract_path),
        "contract_canonical_sha256": _sha256_bytes(canonical_json_bytes(contract)),
        "catalogue_sha256": _sha256_file(catalogue_path),
        "catalogue_bytes": catalogue_path.stat().st_size,
        "support_sha256": _sha256_file(support_path) if support_path is not None else None,
        "support_bytes": support_path.stat().st_size if support_path is not None else None,
    }


def _core(profile: str, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": WITNESS_SCHEMA,
        "protocol": "UFF_SPECTRAL_COMMIT_REVEAL_V1",
        "target_profile": profile,
        "identity": identity,
        "identity_excludes": [
            "filenames",
            "filesystem_paths",
            "file_modification_times",
            "wall_clock_time",
            "UI_state",
        ],
        "reveal_rule": (
            "The frozen identities must match the replay inputs and the QEC boundary gate "
            "must return REPLAY_VERIFIED before observation is admitted."
        ),
        "historical_precommit_requires_external_anchor": True,
        "boundary": (
            "This witness commits to input identity. It does not prove that an analyst had "
            "never seen the data, that the sampling frame is unbiased, that the null model "
            "is adequate, or that a scientific interpretation is true."
        ),
    }


def _commit_digest(core: dict[str, Any]) -> str:
    return _sha256_bytes(_DOMAIN + canonical_json_bytes(core))


def create_witness(
    output_path: Path,
    *,
    contract_path: Path,
    catalogue_path: Path,
    support_path: Path | None = None,
) -> Path:
    """Create a canonical identity commitment before running/observing an audit."""

    contract_path = Path(contract_path)
    catalogue_path = Path(catalogue_path)
    support_path = Path(support_path) if support_path is not None else None
    contract = _load_json(contract_path, require_canonical=False)
    profile = _target_profile(contract, support_path)
    identity = _identity(
        contract_path=contract_path,
        contract=contract,
        catalogue_path=catalogue_path,
        support_path=support_path,
    )
    core = _core(profile, identity)
    envelope = {
        **core,
        "commit_sha256": _commit_digest(core),
        "self_hash_exclusion": "commit_sha256",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(envelope))
    return output_path


def _verify_witness_file(path: Path) -> tuple[dict[str, Any], str]:
    envelope = _load_json(Path(path), require_canonical=True)
    if envelope.get("schema") != WITNESS_SCHEMA:
        raise WitnessError("witness schema is incompatible")
    if envelope.get("protocol") != "UFF_SPECTRAL_COMMIT_REVEAL_V1":
        raise WitnessError("witness protocol is incompatible")
    if envelope.get("self_hash_exclusion") != "commit_sha256":
        raise WitnessError("witness must declare commit_sha256 self-hash exclusion")
    claimed = envelope.get("commit_sha256")
    if not _valid_sha256(claimed):
        raise WitnessError("witness commit_sha256 is missing or malformed")
    core = dict(envelope)
    core.pop("commit_sha256", None)
    core.pop("self_hash_exclusion", None)
    if _commit_digest(core) != claimed:
        raise WitnessError("witness commitment does not recompute")
    return envelope, claimed


def reveal_witness(
    witness_path: Path,
    manifest_path: Path,
    *,
    contract_path: Path,
    catalogue_path: Path,
    support_path: Path | None = None,
    expected_commit_sha256: str | None = None,
    expected_bundle_root: str | None = None,
) -> WitnessReport:
    """Reveal a frozen witness only through a replay-verified QEC boundary."""

    checks: list[str] = []
    errors: list[str] = []
    anchor_verified: bool | None = None
    target_profile: str | None = None
    commit_sha256: str | None = None

    try:
        witness, commit_sha256 = _verify_witness_file(Path(witness_path))
        target_profile = str(witness.get("target_profile", "")) or None
        checks.append("witness commitment recomputed from canonical bytes")
        contract = _load_json(Path(contract_path), require_canonical=False)
        support = Path(support_path) if support_path is not None else None
        current_profile = _target_profile(contract, support)
        current_identity = _identity(
            contract_path=Path(contract_path),
            contract=contract,
            catalogue_path=Path(catalogue_path),
            support_path=support,
        )
        if target_profile != current_profile:
            errors.append("witness target profile does not match the supplied contract")
        if witness.get("identity") != current_identity:
            errors.append("frozen witness identity does not match the supplied inputs")
        else:
            checks.append("contract, catalogue and support identities match the frozen witness")
        if expected_commit_sha256 is not None:
            if not _valid_sha256(expected_commit_sha256):
                errors.append("expected_commit_sha256 must be a lowercase SHA-256 digest")
                anchor_verified = False
            elif expected_commit_sha256 != commit_sha256:
                errors.append("witness commitment does not match the external precommit anchor")
                anchor_verified = False
            else:
                checks.append("witness commitment matches the external precommit anchor")
                anchor_verified = True
    except (OSError, WitnessError) as exc:
        errors.append(str(exc))

    qec = verify_boundary(
        Path(manifest_path),
        catalogue_path=Path(catalogue_path),
        support_path=Path(support_path) if support_path is not None else None,
        expected_root=expected_bundle_root,
        require_replay=True,
    )
    if qec.admitted:
        checks.append("QEC boundary gate admitted the replayed bundle")
    else:
        errors.extend(f"QEC gate: {item}" for item in qec.errors)
    if target_profile is not None and qec.profile is not None and target_profile != qec.profile:
        errors.append("witness target profile does not match the QEC bundle profile")

    witness_verified = not errors and commit_sha256 is not None
    admitted = witness_verified and qec.admitted
    return WitnessReport(
        admitted=admitted,
        witness_verified=witness_verified,
        qec_admitted=qec.admitted,
        external_anchor_verified=anchor_verified,
        target_profile=target_profile,
        commit_sha256=commit_sha256,
        bundle_root_sha256=qec.root_sha256,
        checks=tuple(checks),
        errors=tuple(errors),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uff.spectral_witness",
        description="SPECTRAL-inspired commit/reveal witness for UFF frozen inputs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("commit", help="freeze input identities before observation")
    create.add_argument("output", type=Path)
    create.add_argument("--contract", type=Path, required=True)
    create.add_argument("--catalogue", type=Path, required=True)
    create.add_argument("--support", type=Path)

    reveal = sub.add_parser("reveal", help="verify the witness and replayed bundle")
    reveal.add_argument("witness", type=Path)
    reveal.add_argument("manifest", type=Path)
    reveal.add_argument("--contract", type=Path, required=True)
    reveal.add_argument("--catalogue", type=Path, required=True)
    reveal.add_argument("--support", type=Path)
    reveal.add_argument("--expected-commit")
    reveal.add_argument("--expected-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "commit":
        output = create_witness(
            args.output,
            contract_path=args.contract,
            catalogue_path=args.catalogue,
            support_path=args.support,
        )
        witness = _load_json(output, require_canonical=True)
        print(witness["commit_sha256"])
        return 0

    report = reveal_witness(
        args.witness,
        args.manifest,
        contract_path=args.contract,
        catalogue_path=args.catalogue,
        support_path=args.support,
        expected_commit_sha256=args.expected_commit,
        expected_bundle_root=args.expected_root,
    )
    print(canonical_json_bytes(report.to_dict()).decode("utf-8"), end="")
    return 0 if report.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
