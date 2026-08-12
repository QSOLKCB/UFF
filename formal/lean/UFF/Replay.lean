import UFF.Basic
import UFF.Observation

namespace UFFFormal

/-- A deterministic engine is modeled as a pure function of frozen recipe and runtime contract. -/
abbrev DeterministicEngine := FrozenRecipe → RuntimeContract → NumericalResult

/-- Replay re-evaluates the same pure engine against the frozen inputs. -/
def replay (engine : DeterministicEngine) (recipe : FrozenRecipe)
    (runtime : RuntimeContract) : NumericalResult :=
  engine recipe runtime

structure ReplayIdentity where
  recipe : FrozenRecipe
  runtime : RuntimeContract
  deriving Repr, DecidableEq

theorem deterministic_replay_same_inputs
    (engine : DeterministicEngine) (recipe : FrozenRecipe) (runtime : RuntimeContract) :
    replay engine recipe runtime = replay engine recipe runtime := rfl

theorem changed_runtime_contract_changes_replay_identity
    (recipe : FrozenRecipe) (left right : RuntimeContract) (h : left ≠ right) :
    ({ recipe := recipe, runtime := left } : ReplayIdentity) ≠
      ({ recipe := recipe, runtime := right } : ReplayIdentity) := by
  intro hEq
  exact h (congrArg ReplayIdentity.runtime hEq)

end UFFFormal
