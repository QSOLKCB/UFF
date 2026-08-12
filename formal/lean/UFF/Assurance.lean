import UFF.Basic

namespace UFFFormal

/-- Construct a formal evidence state without changing external scientific judgement. -/
def mkEvidence (level : AssuranceLevel) (external : Bool) : EvidenceState :=
  { assurance := level, externalScientificJudgement := external }

/-- Computational replay changes the assurance label but does not manufacture scientific judgement. -/
def promoteToReplay (state : EvidenceState) : EvidenceState :=
  { state with assurance := .replayVerified }

theorem replay_preserves_external_scientific_judgement (state : EvidenceState) :
    (promoteToReplay state).externalScientificJudgement = state.externalScientificJudgement := rfl

theorem replay_verified_is_not_ensemble_calibrated :
    ¬ hasAssurance (mkEvidence .replayVerified false) .ensembleCalibrated := by
  decide

theorem ensemble_calibrated_is_not_scientifically_defensible :
    ¬ hasAssurance (mkEvidence .ensembleCalibrated false) .scientificallyDefensible := by
  decide

theorem ensemble_calibrated_includes_replay (external : Bool) :
    hasAssurance (mkEvidence .ensembleCalibrated external) .replayVerified := by
  decide

end UFFFormal
