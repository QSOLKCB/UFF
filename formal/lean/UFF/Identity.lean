import UFF.Basic

namespace UFFFormal

/-- Claim-boundary fields are identity-bearing because changing their meaning changes the contract. -/
structure ClaimBoundary where
  claimClass : String
  claimStatus : String
  boundary : String
  doesNotClaim : List String
  deriving Repr, DecidableEq

structure ExperimentIdentity where
  recipe : FrozenRecipe
  claims : ClaimBoundary
  deriving Repr, DecidableEq

theorem changed_claim_boundary_changes_identity
    (recipe : FrozenRecipe) (left right : ClaimBoundary) (h : left ≠ right) :
    ({ recipe := recipe, claims := left } : ExperimentIdentity) ≠
      ({ recipe := recipe, claims := right } : ExperimentIdentity) := by
  intro hEq
  exact h (congrArg ExperimentIdentity.claims hEq)

theorem changed_frozen_recipe_changes_identity
    (left right : FrozenRecipe) (claims : ClaimBoundary) (h : left ≠ right) :
    ({ recipe := left, claims := claims } : ExperimentIdentity) ≠
      ({ recipe := right, claims := claims } : ExperimentIdentity) := by
  intro hEq
  exact h (congrArg ExperimentIdentity.recipe hEq)

end UFFFormal
