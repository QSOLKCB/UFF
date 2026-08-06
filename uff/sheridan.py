"""Public API and CLI for the UFF Sheridan Crucible siege engine."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .sheridan_artifacts import (
    SHERIDAN_ALGORITHM_ID,
    SHERIDAN_SOFTWARE_VERSION,
    SheridanVerificationReport,
    run_sheridan_analysis,
    verify_sheridan_bundle,
    write_sheridan_bundle,
)
from .sheridan_contract import (
    SHERIDAN_CONTRACT_SCHEMA,
    DensityConfig,
    InjectionConfig,
    ModelConfig,
    SheridanContract,
    SheridanDecision,
    SurveyConfig,
    fibonacci_support,
    load_sheridan_contract,
    load_sheridan_contract_payload,
)
from .sheridan_density import DensityFit, fit_density, score_nodes, vmf_kernel
from .sheridan_models import run_injection_recovery, run_model_comparison
from .sky_contract import AuditError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uff-sheridan",
        description=(
            "Run a survey-aware, preregistered sky-lattice crucible with spherical "
            "density reconstruction, nuisance-model comparison, and injection calibration."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the frozen Sheridan contract")
    run.add_argument("--catalogue", "--catalog", dest="catalogue", required=True, type=Path)
    run.add_argument("--support", required=True, type=Path)
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify", help="verify integrity and optional replay")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--catalogue", "--catalog", dest="catalogue", type=Path)
    verify.add_argument("--support", type=Path)

    support = subparsers.add_parser(
        "support-grid",
        help="write a deterministic equal-area full-sky Fibonacci support grid",
    )
    support.add_argument("--points", type=int, default=4096)
    support.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            manifest = write_sheridan_bundle(
                args.out,
                catalogue_path=args.catalogue,
                support_path=args.support,
                contract_path=args.contract,
                force=args.force,
            )
            print(f"[OK] Sheridan manifest: {manifest}")
            return 0
        if args.command == "support-grid":
            frame = fibonacci_support(args.points)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.out, index=False, float_format="%.17g", lineterminator="\n")
            print(f"[OK] support grid: {args.out}")
            return 0
        report = verify_sheridan_bundle(
            args.manifest,
            catalogue_path=args.catalogue,
            support_path=args.support,
        )
        for check in report.checks:
            print(f"[OK] {check}")
        for error in report.errors:
            print(f"[ERROR] {error}")
        return 0 if report.passed else 2
    except (AuditError, OSError, ValueError) as exc:
        print(f"uff-sheridan: error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DensityConfig",
    "DensityFit",
    "InjectionConfig",
    "ModelConfig",
    "SHERIDAN_ALGORITHM_ID",
    "SHERIDAN_CONTRACT_SCHEMA",
    "SHERIDAN_SOFTWARE_VERSION",
    "SheridanContract",
    "SheridanDecision",
    "SheridanVerificationReport",
    "SurveyConfig",
    "fibonacci_support",
    "fit_density",
    "load_sheridan_contract",
    "load_sheridan_contract_payload",
    "main",
    "run_injection_recovery",
    "run_model_comparison",
    "run_sheridan_analysis",
    "score_nodes",
    "verify_sheridan_bundle",
    "vmf_kernel",
    "write_sheridan_bundle",
]
