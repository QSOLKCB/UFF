"""Public API and command-line interface for UFF-SLFA."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .sky_artifacts import VerificationReport, verify_bundle, write_bundle
from .sky_contract import (
    AuditContract,
    AuditError,
    CONTRACT_SCHEMA,
    Node,
    canonical_json_bytes,
    load_catalogue,
    load_contract,
    load_contract_payload,
    sha256_file,
)
from .sky_statistics import (
    ALGORITHM_ID,
    OBSERVATIONS_SCHEMA,
    RegionSummary,
    empirical_p,
    holm_adjust,
    run_audit,
    summarize_region,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uff-sky-audit",
        description="Run or verify a preregistered celestial-node audit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a frozen claim against a catalogue")
    run.add_argument("--catalogue", "--catalog", dest="catalogue", required=True, type=Path)
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--force", action="store_true")
    verify = subparsers.add_parser("verify", help="verify bundle integrity and optional replay")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--catalogue", "--catalog", dest="catalogue", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            manifest = write_bundle(
                args.out,
                catalogue_path=args.catalogue,
                contract_path=args.contract,
                force=args.force,
            )
            print(f"[OK] manifest: {manifest}")
            return 0
        report = verify_bundle(args.manifest, args.catalogue)
        for check in report.checks:
            print(f"[OK] {check}")
        for error in report.errors:
            print(f"[ERROR] {error}")
        return 0 if report.passed else 2
    except (AuditError, OSError, ValueError) as exc:
        print(f"uff-sky-audit: error: {exc}")
        return 2


__all__ = [
    "ALGORITHM_ID",
    "AuditContract",
    "AuditError",
    "CONTRACT_SCHEMA",
    "Node",
    "OBSERVATIONS_SCHEMA",
    "RegionSummary",
    "VerificationReport",
    "canonical_json_bytes",
    "empirical_p",
    "holm_adjust",
    "load_catalogue",
    "load_contract",
    "load_contract_payload",
    "main",
    "run_audit",
    "sha256_file",
    "summarize_region",
    "verify_bundle",
    "write_bundle",
]
