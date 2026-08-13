# Runtime correspondence

The v5.2.0 formal layer is a specification model, not a code extractor and not a verified compiler target.

| Formal module | UFF runtime/document surface | Correspondence |
|---|---|---|
| `UFF.Assurance` | `uff.qec_gate`, `docs/STATISTICAL_MECHANICS_GUARDRAIL.md` | Replay/admission is separated from future ensemble calibration and external scientific judgement. |
| `UFF.Identity` | frozen SLFA/Sheridan contracts, witness binding, claim provenance | Identity-bearing recipe and claim-boundary changes are modeled as identity changes. |
| `UFF.Transaction` | deterministic bundle construction and fail-closed execution | Cancelled work produces no modeled archival bundle. |
| `UFF.Observation` | sonification and `uff.audit_events` | Observation/telemetry is one-way and has no evidence-admission authority. |
| `UFF.Manifest` | manifest-core/envelope and QEC receipt construction | The digest function consumes the core before the envelope carries the digest. |
| `UFF.Replay` | SLFA/Sheridan replay and runtime fingerprints | Replay is a pure function of frozen recipe + runtime contract in the formal model. |
| `UFF.Assumptions` | scientific-boundary documentation | Nonclaims are machine-visible scope markers, not hidden prose exceptions. |

## NEXUS lineage

The formalization pattern is informed by the archived QSOL-NEXUS v1.0.0 deterministic workbench contracts, especially its frozen-recipe transaction model, claim-boundary identity, one-way sonification, replay semantics, and no-self-hash architecture. The source archive is preserved under `QSOLKCB/QSOL-NEXUS/archives/v1.0.0`.

The NEXUS repository later developed its own Lean 4 constitutional formalization; UFF v5.2.0 reuses the audit discipline while keeping the theorem content specific to UFF evidence assurance.
