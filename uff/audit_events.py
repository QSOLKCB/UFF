"""Receiver-neutral deterministic audit telemetry for UFF.

This is a deliberately non-authoritative SONIFICATION-inspired layer. It turns
QEC boundary outcomes and already-recorded UFF scientific outputs into a stable
event document that external audio/visual receivers may inspect.

The event stream never changes a UFF decision and never participates in bundle
admission. Generate it outside the evidence bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .qec_gate import canonical_json_bytes, verify_boundary


EVENT_SCHEMA = "uff.audit-event-stream.v1"
PROFILE = "uff-qec-sonification-telemetry-v1"
_DOMAIN = b"UFF/AUDIT-EVENT-STREAM/v1\0"


class TelemetryError(RuntimeError):
    """Raised when deterministic audit telemetry cannot be constructed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TelemetryError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TelemetryError(f"JSON root must be an object: {path}")
    return value


def _ppm(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(round(number * 1_000_000.0))


def _event(
    events: list[dict[str, Any]],
    *,
    channel: str,
    code: str,
    state: str,
    polarity: int,
    authority: str,
    value_ppm: int | None = None,
    value_int: int | None = None,
) -> None:
    events.append(
        {
            "index": len(events),
            "channel": channel,
            "code": code,
            "state": state,
            "polarity": polarity,
            "authority": authority,
            "value_ppm": value_ppm,
            "value_int": value_int,
        }
    )


def _append_slfa(events: list[dict[str, Any]], base: Path, result: str) -> None:
    observations = _load_json(base / "observations.json")
    global_test = observations.get("global_test")
    node_test = observations.get("node_test")
    if not isinstance(global_test, dict) or not isinstance(node_test, dict):
        raise TelemetryError("SLFA observations are missing global_test or node_test")
    _event(
        events,
        channel="scientific-decision",
        code="SLFA_DECISION",
        state=result,
        polarity=0,
        authority="reported-scientific-output",
    )
    _event(
        events,
        channel="metric",
        code="SLFA_GLOBAL_EMPIRICAL_P",
        state="OBSERVED",
        polarity=0,
        authority="reported-scientific-output",
        value_ppm=_ppm(global_test.get("empirical_p")),
    )
    _event(
        events,
        channel="metric",
        code="SLFA_GLOBAL_RATE_CONTRAST",
        state="OBSERVED",
        polarity=0,
        authority="reported-scientific-output",
        value_ppm=_ppm(global_test.get("rate_contrast")),
    )
    supported = node_test.get("supported_nodes")
    required = node_test.get("required_supported_nodes")
    _event(
        events,
        channel="metric",
        code="SLFA_SUPPORTED_NODES",
        state="OBSERVED",
        polarity=0,
        authority="reported-scientific-output",
        value_int=supported if type(supported) is int else None,
    )
    _event(
        events,
        channel="metric",
        code="SLFA_REQUIRED_SUPPORTED_NODES",
        state="FROZEN_RULE",
        polarity=0,
        authority="contract",
        value_int=required if type(required) is int else None,
    )


def _append_sheridan(events: list[dict[str, Any]], base: Path, result: str) -> None:
    decision = _load_json(base / "decision.json")
    _event(
        events,
        channel="scientific-decision",
        code="SHERIDAN_DECISION",
        state=result,
        polarity=0,
        authority="reported-scientific-output",
    )
    component_passes = decision.get("component_passes")
    if isinstance(component_passes, dict):
        for name in sorted(component_passes):
            value = component_passes[name]
            if isinstance(value, bool):
                _event(
                    events,
                    channel="component",
                    code=f"SHERIDAN_{name.upper()}_COMPONENT",
                    state="MET" if value else "NOT_MET",
                    polarity=1 if value else -1,
                    authority="reported-scientific-output",
                )
    density = _load_json(base / "density.json")
    global_test = density.get("global_test")
    if isinstance(global_test, dict):
        for key, code in (
            ("empirical_p", "SHERIDAN_DENSITY_EMPIRICAL_P"),
            ("mean_overdensity", "SHERIDAN_MEAN_OVERDENSITY"),
        ):
            if key in global_test:
                _event(
                    events,
                    channel="metric",
                    code=code,
                    state="OBSERVED",
                    polarity=0,
                    authority="reported-scientific-output",
                    value_ppm=_ppm(global_test.get(key)),
                )


def build_event_stream(
    manifest_path: Path,
    *,
    catalogue_path: Path | None = None,
    support_path: Path | None = None,
    expected_root: str | None = None,
) -> dict[str, Any]:
    """Build deterministic read-only telemetry from a live boundary verification."""

    manifest_path = Path(manifest_path)
    gate = verify_boundary(
        manifest_path,
        catalogue_path=Path(catalogue_path) if catalogue_path is not None else None,
        support_path=Path(support_path) if support_path is not None else None,
        expected_root=expected_root,
        require_replay=True,
    )
    events: list[dict[str, Any]] = []
    _event(
        events,
        channel="trust-boundary",
        code="INTEGRITY",
        state="PASS" if gate.integrity_passed else "FAIL",
        polarity=1 if gate.integrity_passed else -1,
        authority="qec-boundary",
    )
    replay_state = (
        "PASS"
        if gate.replay_passed is True
        else "FAIL"
        if gate.replay_passed is False
        else "ABSENT"
    )
    _event(
        events,
        channel="trust-boundary",
        code="REPLAY",
        state=replay_state,
        polarity=1 if gate.replay_passed is True else -1 if gate.replay_passed is False else 0,
        authority="qec-boundary",
    )
    _event(
        events,
        channel="trust-boundary",
        code="ADMISSION",
        state="ADMIT" if gate.admitted else "REJECT",
        polarity=1 if gate.admitted else -1,
        authority="qec-boundary",
    )
    if expected_root is not None:
        anchor_ok = gate.root_sha256 == expected_root and not any(
            "trust anchor" in item for item in gate.errors
        )
        _event(
            events,
            channel="trust-anchor",
            code="BUNDLE_ROOT_ANCHOR",
            state="MATCH" if anchor_ok else "MISMATCH",
            polarity=1 if anchor_ok else -1,
            authority="external-anchor",
        )

    manifest = _load_json(manifest_path)
    if gate.integrity_passed:
        result = str(manifest.get("result", "UNKNOWN"))
        if gate.profile == "uff-slfa-qec-gate-v1":
            _append_slfa(events, manifest_path.parent, result)
        elif gate.profile == "uff-sheridan-qec-gate-v1":
            _append_sheridan(events, manifest_path.parent, result)

    core = {
        "schema": EVENT_SCHEMA,
        "profile": PROFILE,
        "source": {
            "bundle_root_sha256": gate.root_sha256,
            "gate_profile": gate.profile,
            "gate_assurance": gate.assurance,
        },
        "events": events,
        "receiver_contract": {
            "canonical": [
                "event order",
                "channel",
                "code",
                "state",
                "polarity",
                "authority",
                "value_ppm",
                "value_int",
            ],
            "noncanonical": [
                "tempo",
                "hertz",
                "MIDI_note",
                "timbre",
                "loudness",
                "stereo_position",
                "waveform",
                "rendered_audio",
            ],
            "rule": (
                "Receivers may sonify or visualize this event stream, but receiver output "
                "has no authority over UFF bundle admission or scientific decisions."
            ),
        },
        "boundary": (
            "Telemetry is an observation aid. It may expose patterns or failures to a human "
            "operator, but hearing a pattern is not statistical evidence and silence is not proof."
        ),
    }
    return {
        **core,
        "event_stream_sha256": _sha256_bytes(_DOMAIN + canonical_json_bytes(core)),
        "self_hash_exclusion": "event_stream_sha256",
    }


def write_event_stream(
    output_path: Path,
    manifest_path: Path,
    *,
    catalogue_path: Path | None = None,
    support_path: Path | None = None,
    expected_root: str | None = None,
) -> Path:
    output_path = Path(output_path)
    bundle_dir = Path(manifest_path).resolve().parent
    resolved_output = output_path.resolve()
    if resolved_output.is_relative_to(bundle_dir):
        raise TelemetryError("audit telemetry must be written outside the closed evidence bundle")
    document = build_event_stream(
        manifest_path,
        catalogue_path=catalogue_path,
        support_path=support_path,
        expected_root=expected_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(document))
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uff.audit_events",
        description="Generate receiver-neutral deterministic telemetry from a UFF QEC gate run.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--expected-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = write_event_stream(
            args.out,
            args.manifest,
            catalogue_path=args.catalogue,
            support_path=args.support,
            expected_root=args.expected_root,
        )
    except TelemetryError as exc:
        print(f"ERROR: {exc}")
        return 2
    document = _load_json(output)
    print(document["event_stream_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
