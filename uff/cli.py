"""Command-line interface for reproducible UFF galaxy and SMBH analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .compact import compact_object_report
from .data import GalaxyData
from .diagnostics import (
    comparison_diagnostics,
    plot_model_comparison,
    plot_posterior_corner,
    plot_posterior_predictive,
    save_rotation_sonification,
)
from .fitting import comparison_records, fit_models
from .models import ModelOptions, available_models, build_model
from .sampling import sample_posterior


DEFAULT_MODELS = "nfw,burkert,mond-rar,uff-empirical"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or "galaxy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_models(value: str) -> list[str]:
    names: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise argparse.ArgumentTypeError("at least one model is required")
    return names


def _model_options(args: argparse.Namespace) -> ModelOptions:
    return ModelOptions(
        h0_km_s_mpc=args.h0,
        a0_m_s2=args.a0,
        fit_a0=args.fit_a0,
        disk_mass_to_light=args.disk_ml,
        bulge_mass_to_light=args.bulge_ml,
        fit_stellar_mass_to_light=not args.fixed_stellar_ml,
        distance_scale=args.distance_scale,
        fit_distance_scale=args.fit_distance,
        reference_inclination_deg=args.inclination_deg,
        fit_inclination=args.fit_inclination,
        smbh_mass_msun=args.smbh_mass_msun,
        fit_smbh=args.fit_smbh,
        external_field_a0=args.external_field_a0,
        external_field_angle_deg=args.external_field_angle_deg,
    )


def _fit_one(path: Path, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    data = GalaxyData.from_csv(path, name=args.galaxy_name)
    names = _parse_models(args.models)
    options = _model_options(args)
    models = [build_model(name, data, options) for name in names]
    canonical_names = [model.name for model in models]
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError(
            "model list contains duplicate aliases for the same canonical model"
        )
    results = fit_models(
        models,
        data,
        restarts=args.restarts,
        random_state=args.seed,
        systematic_kms=args.systematic_kms,
        max_nfev=args.max_nfev,
    )
    best = results[0]

    output_root.mkdir(parents=True, exist_ok=True)
    prefix = _slug(data.name)
    comparison_path = output_root / f"{prefix}_comparison.csv"
    plot_path = output_root / f"{prefix}_models.png"
    summary_path = output_root / f"{prefix}_summary.json"

    records = comparison_records(results)
    pd.DataFrame.from_records(records).to_csv(comparison_path, index=False)
    plot_model_comparison(data, models, results, plot_path)

    generated: dict[str, str] = {
        "comparison_table": str(comparison_path),
        "comparison_plot": str(plot_path),
        "summary": str(summary_path),
    }
    if args.sonify:
        model_lookup = {model.name: model for model in models}
        radius_grid = np.geomspace(data.radius_kpc.min(), data.radius_kpc.max(), 500)
        velocity_grid = model_lookup[best.model_name].predict(radius_grid, best.theta)
        audio_path = output_root / f"{prefix}_{best.model_name}_phase_glyph.wav"
        save_rotation_sonification(radius_grid, velocity_grid, audio_path)
        generated["sonification"] = str(audio_path)

    if args.e8:
        try:
            from e8_visualization import plot_e8_projection

            e8_path = output_root / f"{prefix}_e8_reference.png"
            plot_e8_projection(str(e8_path))
            generated["e8_reference"] = str(e8_path)
        except Exception as exc:  # visual hook must never invalidate a physics fit
            print(f"[WARN] E8 reference plot failed: {exc}", file=sys.stderr)

    posterior_summary: dict[str, object] | None = None
    if args.mcmc_steps > 0:
        model_lookup = {model.name: model for model in models}
        result_lookup = {result.model_name: result for result in results}
        selected_name = (
            best.model_name if args.mcmc_model == "best" else args.mcmc_model
        )
        if selected_name not in model_lookup:
            raise ValueError(
                f"MCMC model {selected_name!r} is not in this run's canonical model list: "
                f"{', '.join(model_lookup)}"
            )
        burn = args.mcmc_burn
        if burn is None:
            burn = max(1, int(round(args.mcmc_steps * 0.3)))
        posterior = sample_posterior(
            model_lookup[selected_name],
            data,
            result_lookup[selected_name],
            steps=args.mcmc_steps,
            burn=burn,
            thin=args.mcmc_thin,
            n_chains=args.mcmc_chains,
            seed=args.seed,
            systematic_kms=args.systematic_kms,
        )
        chain_path = output_root / f"{prefix}_{selected_name}_posterior.npz"
        np.savez_compressed(
            chain_path,
            samples=posterior.samples,
            log_likelihood=posterior.log_likelihood,
            parameter_names=np.asarray(posterior.parameter_names),
            acceptance_rates=posterior.acceptance_rates,
            rhat=posterior.rhat,
            effective_sample_size=posterior.effective_sample_size,
        )
        generated["posterior_samples"] = str(chain_path)
        if args.corner:
            corner_path = output_root / f"{prefix}_{selected_name}_corner.png"
            plot_posterior_corner(
                posterior.samples, posterior.parameter_names, corner_path
            )
            generated["posterior_corner"] = str(corner_path)
        if args.postpred:
            predictive_path = output_root / f"{prefix}_{selected_name}_postpred.png"
            plot_posterior_predictive(
                data,
                model_lookup[selected_name],
                posterior.samples,
                predictive_path,
                seed=args.seed,
            )
            generated["posterior_predictive"] = str(predictive_path)
        posterior_summary = posterior.to_dict()
    elif args.corner or args.postpred:
        raise ValueError("--corner and --postpred require --mcmc-steps")

    warnings: list[str] = []
    missing = data.metadata.get("missing_components", [])
    if missing:
        warnings.append(
            f"missing baryonic components were set to zero: {', '.join(missing)}"
        )
    for result in results:
        if result.bound_hits:
            warnings.append(
                f"{result.model_name} touches bounds: {', '.join(result.bound_hits)}"
            )
        if result.aicc == float("inf"):
            warnings.append(
                f"{result.model_name} has too few data points for finite AICc; compare BIC/AIC cautiously"
            )
    if args.fit_smbh:
        warnings.append(
            "SMBH mass was fitted from a rotation curve; verify that the innermost radii "
            "resolve its sphere of influence"
        )
    if posterior_summary is not None:
        for chain_index, acceptance in enumerate(
            posterior_summary["acceptance_rates"], start=1
        ):
            if acceptance < 0.15 or acceptance > 0.55:
                warnings.append(
                    f"posterior chain {chain_index} acceptance is {acceptance:.3f}; inspect proposal mixing"
                )
        for parameter, rhat in posterior_summary["rhat"].items():
            if rhat is not None and rhat > 1.05:
                warnings.append(
                    f"posterior R-hat for {parameter} is {rhat:.3f}; run longer chains"
                )
        for parameter, ess in posterior_summary["effective_sample_size"].items():
            if ess is not None and ess < 400:
                warnings.append(
                    f"posterior ESS for {parameter} is {ess:.0f}; run longer chains"
                )

    summary: dict[str, Any] = {
        "schema": "uff.rotation-curve-summary.v4",
        "software_version": __version__,
        "galaxy": data.name,
        "input": {
            "path": str(path),
            "sha256": _sha256(path),
            "n_points": data.n_points,
            "radius_range_kpc": [
                float(data.radius_kpc.min()),
                float(data.radius_kpc.max()),
            ],
            "metadata": data.metadata,
        },
        "configuration": {
            "models": names,
            "H0_km_s_Mpc": args.h0,
            "a0_m_s2": args.a0,
            "fit_a0": args.fit_a0,
            "disk_mass_to_light": args.disk_ml,
            "bulge_mass_to_light": args.bulge_ml,
            "fit_stellar_mass_to_light": not args.fixed_stellar_ml,
            "distance_scale": args.distance_scale,
            "fit_distance_scale": args.fit_distance,
            "reference_inclination_deg": args.inclination_deg,
            "fit_inclination": args.fit_inclination,
            "smbh_mass_msun": args.smbh_mass_msun,
            "fit_smbh": args.fit_smbh,
            "external_field_a0": args.external_field_a0,
            "external_field_angle_deg": args.external_field_angle_deg,
            "systematic_kms": args.systematic_kms,
            "restarts": args.restarts,
            "seed": args.seed,
            "mcmc_steps": args.mcmc_steps,
            "mcmc_burn": args.mcmc_burn,
            "mcmc_thin": args.mcmc_thin,
            "mcmc_chains": args.mcmc_chains,
            "mcmc_model": args.mcmc_model,
        },
        "ranking_criterion": "BIC",
        "best_model": best.model_name,
        "comparison": records,
        "fits": {
            result.model_name: result.to_dict(include_arrays=True) for result in results
        },
        "diagnostics": comparison_diagnostics(results),
        "posterior": posterior_summary,
        "generated_files": generated,
        "warnings": warnings,
        "scientific_scope": {
            "SMBH": "weak-field central point mass in galaxy fits; Kerr scales are a separate command",
            "MOND": "algebraic phenomenology; EFE option is an explicitly approximate sensitivity proxy",
            "LQG": "not used in the galaxy likelihood; compact-object command reports scale diagnostics only",
            "UFF": "repository-specific empirical cored law; no claim of a covariant field derivation",
        },
    }
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(_json_safe(summary), output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"[OK] {data.name}: best={best.model_name} "
        f"BIC={best.bic:.2f} chi2/dof={best.reduced_chi_squared:.3g}"
    )
    print(f" -> {summary_path}")
    print(f" -> {plot_path}")
    return summary


def _run_fit(args: argparse.Namespace) -> int:
    output = Path(args.out)
    if args.csv:
        _fit_one(Path(args.csv), args, output)
        return 0

    batch_dir = Path(args.batch)
    paths = sorted(batch_dir.glob(args.pattern))
    if not paths:
        raise ValueError(f"no files matched {args.pattern!r} in {batch_dir}")
    summaries: list[dict[str, Any]] = []
    for path in paths:
        # File stem is the correct default in batch mode, even if --galaxy-name
        # was accidentally inherited from a scripted invocation.
        original_name = args.galaxy_name
        args.galaxy_name = None
        try:
            summaries.append(_fit_one(path, args, output))
        finally:
            args.galaxy_name = original_name
    batch_path = output / "batch_summary.csv"
    rows = [
        {
            "galaxy": summary["galaxy"],
            "best_model": summary["best_model"],
            "best_bic": summary["comparison"][0]["bic"],
            "best_reduced_chi_squared": summary["comparison"][0]["reduced_chi_squared"],
            "input_sha256": summary["input"]["sha256"],
        }
        for summary in summaries
    ]
    pd.DataFrame.from_records(rows).to_csv(batch_path, index=False)
    print(f"[OK] batch summary -> {batch_path}")
    return 0


def _run_compact(args: argparse.Namespace) -> int:
    report = compact_object_report(
        args.mass_msun,
        dimensionless_spin=args.spin,
        probe_radius_kpc=args.probe_radius_kpc,
        velocity_dispersion_kms=args.velocity_dispersion_kms,
        barbero_immirzi=args.barbero_immirzi,
    )
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(f"[OK] compact-object report -> {destination}")
    else:
        print(payload, end="")
    return 0


def _run_models(args: argparse.Namespace) -> int:
    descriptions = {
        "baryons": "Newtonian SPARC baryons with optional central SMBH",
        "nfw": "M200/c200 NFW halo plus baryons",
        "burkert": "cored Burkert halo plus baryons",
        "mond-rar": "empirical exponential radial-acceleration relation",
        "mond-simple": "simple MOND interpolating function",
        "mond-standard": "standard MOND interpolating function",
        "mond-efe": "RAR with an approximate algebraic external-field proxy",
        "uff-empirical": "bounded repository-specific empirical UFF law",
    }
    for name in available_models():
        print(f"{name:15s} {descriptions[name]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uff",
        description="Transparent galaxy rotation-curve and compact-object model laboratory",
    )
    parser.add_argument("--version", action="version", version=f"UFF {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit", help="fit and compare galaxy rotation-curve models"
    )
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="canonical or SPARC-style CSV file")
    source.add_argument("--batch", help="directory of CSV files")
    fit.add_argument(
        "--pattern", default="*.csv", help="batch glob pattern (default: *.csv)"
    )
    fit.add_argument(
        "--gal", "--galaxy-name", dest="galaxy_name", help="output galaxy name"
    )
    fit.add_argument("--out", default="outputs", help="output directory")
    fit.add_argument(
        "--models", default=DEFAULT_MODELS, help="comma-separated model names"
    )
    fit.add_argument("--h0", type=float, default=70.0, help="H0 [km/s/Mpc]")
    fit.add_argument("--a0", type=float, default=1.2e-10, help="MOND a0 [m/s^2]")
    fit.add_argument("--fit-a0", action="store_true", help="fit a0 for MOND candidates")
    fit.add_argument(
        "--disk-ml", type=float, default=0.5, help="fixed/initial disk M/L"
    )
    fit.add_argument(
        "--bulge-ml", type=float, default=0.7, help="fixed/initial bulge M/L"
    )
    fit.add_argument(
        "--fixed-stellar-ml",
        action="store_true",
        help="do not fit stellar mass-to-light ratios",
    )
    fit.add_argument(
        "--distance-scale",
        type=float,
        default=1.0,
        help="fixed/initial distance ratio D/D_ref",
    )
    fit.add_argument(
        "--fit-distance",
        action="store_true",
        help="fit D/D_ref in [0.5,1.5] with SPARC component rescaling",
    )
    fit.add_argument(
        "--inclination-deg",
        type=float,
        help="reference inclination (defaults to INC_deg metadata)",
    )
    fit.add_argument(
        "--fit-inclination",
        action="store_true",
        help="fit inclination within +/-15 degrees of its reference",
    )
    fit.add_argument(
        "--smbh-mass-msun", type=float, default=0.0, help="fixed central SMBH mass"
    )
    fit.add_argument(
        "--fit-smbh",
        action="store_true",
        help="fit log10 SMBH mass (resolved cores only)",
    )
    fit.add_argument(
        "--external-field-a0",
        type=float,
        default=0.0,
        help="external field in units of a0 (used only by mond-efe)",
    )
    fit.add_argument(
        "--external-field-angle-deg",
        type=float,
        default=0.0,
        help="angle for the algebraic MOND EFE proxy",
    )
    fit.add_argument(
        "--systematic-kms", type=float, default=0.0, help="error floor in quadrature"
    )
    fit.add_argument(
        "--restarts", type=int, default=12, help="deterministic optimizer starts"
    )
    fit.add_argument(
        "--max-nfev", type=int, default=20_000, help="maximum evaluations per start"
    )
    fit.add_argument("--seed", type=int, default=42, help="deterministic random seed")
    fit.add_argument(
        "--mcmc-steps",
        type=int,
        default=0,
        help="opt-in Metropolis steps per chain (0 disables posterior sampling)",
    )
    fit.add_argument(
        "--mcmc-burn",
        type=int,
        help="burn-in steps (default: 30%% of --mcmc-steps)",
    )
    fit.add_argument(
        "--mcmc-thin", type=int, default=5, help="posterior thinning interval"
    )
    fit.add_argument(
        "--mcmc-chains", type=int, default=4, help="number of independent chains"
    )
    fit.add_argument(
        "--mcmc-model",
        default="best",
        help="canonical candidate name to sample, or 'best'",
    )
    fit.add_argument(
        "--corner", action="store_true", help="plot retained posterior draws"
    )
    fit.add_argument(
        "--postpred", action="store_true", help="plot posterior curve bands"
    )
    fit.add_argument(
        "--sonify", action="store_true", help="export best-fit phase-quadrature WAV"
    )
    fit.add_argument(
        "--e8", action="store_true", help="export the legacy E8 reference visualization"
    )
    fit.add_argument("--compare", action="store_true", help=argparse.SUPPRESS)
    fit.set_defaults(handler=_run_fit)

    compact = subparsers.add_parser(
        "compact-object", help="report Kerr SMBH scales and LQG area-gap suppression"
    )
    compact.add_argument(
        "--mass-msun", type=float, required=True, help="black-hole mass [M_sun]"
    )
    compact.add_argument(
        "--spin", type=float, default=0.0, help="signed Kerr a* in [-1,1]"
    )
    compact.add_argument(
        "--probe-radius-kpc", type=float, help="radius for LQG scale report"
    )
    compact.add_argument(
        "--velocity-dispersion-kms", type=float, help="sigma for influence radius"
    )
    compact.add_argument(
        "--barbero-immirzi", type=float, default=0.2375, help="LQG gamma convention"
    )
    compact.add_argument("--out", help="optional JSON destination")
    compact.set_defaults(handler=_run_compact)

    models = subparsers.add_parser(
        "models", help="list canonical rotation-curve model names"
    )
    models.set_defaults(handler=_run_models)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Backward-compatible entry: ``python analyze_sparc.py --csv ...``.
    if arguments and arguments[0] not in {
        "fit",
        "compact-object",
        "models",
        "-h",
        "--help",
        "--version",
    }:
        arguments.insert(0, "fit")
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


__all__ = ["DEFAULT_MODELS", "build_parser", "main"]
