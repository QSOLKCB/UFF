import UFF.Basic

namespace UFFFormal

structure NumericalResult where
  payload : String
  deriving Repr, DecidableEq

structure AudioArtifact where
  payload : String
  deriving Repr, DecidableEq

/-- Sonification is represented as a pure post-processing map over a completed numerical result. -/
def sonify (result : NumericalResult) : AudioArtifact :=
  { payload := "sonification:" ++ result.payload }

/-- Attach an observation artifact without replacing the underlying numerical result. -/
def attachSonification (result : NumericalResult) : NumericalResult × AudioArtifact :=
  (result, sonify result)

/-- Receiver-neutral telemetry observes evidence and has no admission authority. -/
def applyTelemetryToEvidence (evidence : EvidenceState) (_event : String) : EvidenceState :=
  evidence

theorem sonification_preserves_numerical_result (result : NumericalResult) :
    (attachSonification result).1 = result := rfl

theorem telemetry_has_zero_admission_authority (evidence : EvidenceState) (event : String) :
    applyTelemetryToEvidence evidence event = evidence := rfl

end UFFFormal
