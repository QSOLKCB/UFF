namespace UFFFormal

/-- Ordered assurance levels used by UFF. Higher levels require separately earned evidence. -/
inductive AssuranceLevel where
  | inputsCommitted
  | integrityVerified
  | replayVerified
  | ensembleCalibrated
  | scientificallyDefensible
  deriving Repr, DecidableEq

/-- A compact rank used only for the formal assurance-order model. -/
def AssuranceLevel.rank : AssuranceLevel → Nat
  | .inputsCommitted => 0
  | .integrityVerified => 1
  | .replayVerified => 2
  | .ensembleCalibrated => 3
  | .scientificallyDefensible => 4

/-- External scientific judgement is carried explicitly and is not derived from replay. -/
structure EvidenceState where
  assurance : AssuranceLevel
  externalScientificJudgement : Bool
  deriving Repr, DecidableEq

/-- `hasAssurance s required` means that `s` is at or above `required` in the formal ladder. -/
def hasAssurance (s : EvidenceState) (required : AssuranceLevel) : Prop :=
  required.rank ≤ s.assurance.rank

/-- Identity-bearing experiment inputs. Digests are abstract strings; Lean does not prove SHA-256 security. -/
structure FrozenRecipe where
  contractDigest : String
  catalogueDigest : String
  supportDigest : Option String
  engineVersion : String
  deriving Repr, DecidableEq

/-- Runtime identity used where replay is scoped to a recorded numerical runtime class. -/
structure RuntimeContract where
  fingerprint : String
  deriving Repr, DecidableEq

end UFFFormal
