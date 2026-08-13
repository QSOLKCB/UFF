import UFF.Basic

namespace UFFFormal

/-- Construct a formal evidence state without changing external scientific judgement. -/
def mkEvidence (level : AssuranceLevel) (external : Bool) : EvidenceState :=
  { assurance := level, externalScientificJudgement := external }

/--
Raise assurance to at least replay verification without discarding a higher,
separately earned assurance level.
-/
def replayPromotedAssurance : AssuranceLevel → AssuranceLevel
  | .inputsCommitted => .replayVerified
  | .integrityVerified => .replayVerified
  | .replayVerified => .replayVerified
  | .ensembleCalibrated => .ensembleCalibrated
  | .scientificallyDefensible => .scientificallyDefensible

/-- Computational replay never manufactures scientific judgement or lowers assurance. -/
def promoteToReplay (state : EvidenceState) : EvidenceState :=
  { state with assurance := replayPromotedAssurance state.assurance }

theorem replay_preserves_external_scientific_judgement (state : EvidenceState) :
    (promoteToReplay state).externalScientificJudgement = state.externalScientificJudgement := rfl

theorem replay_promotion_does_not_lower_assurance (state : EvidenceState) :
    state.assurance.rank ≤ (promoteToReplay state).assurance.rank := by
  have monotone : ∀ level : AssuranceLevel,
      level.rank ≤ (replayPromotedAssurance level).rank := by
    intro level
    cases level <;> decide
  exact monotone state.assurance

theorem replay_verified_is_not_ensemble_calibrated :
    ¬ hasAssurance (mkEvidence .replayVerified false) .ensembleCalibrated := by
  simp [hasAssurance, mkEvidence, AssuranceLevel.rank]

theorem ensemble_calibrated_is_not_scientifically_defensible :
    ¬ hasAssurance (mkEvidence .ensembleCalibrated false) .scientificallyDefensible := by
  simp [hasAssurance, mkEvidence, AssuranceLevel.rank]

theorem ensemble_calibrated_includes_replay (external : Bool) :
    hasAssurance (mkEvidence .ensembleCalibrated external) .replayVerified := by
  simp [hasAssurance, mkEvidence, AssuranceLevel.rank]

end UFFFormal
