import UFF.Basic

namespace UFFFormal

inductive TransactionStatus where
  | pending
  | completed
  | cancelled
  deriving Repr, DecidableEq

structure Transaction where
  status : TransactionStatus
  bundle : Option String
  deriving Repr, DecidableEq

/-- Only completed transactions carrying a bundle are exportable in the model. -/
def Exportable (transaction : Transaction) : Prop :=
  transaction.status = .completed ∧ ∃ bundle, transaction.bundle = some bundle

/-- Cancellation discards the pending archival result. -/
def cancel (_transaction : Transaction) : Transaction :=
  { status := .cancelled, bundle := none }

theorem cancellation_produces_no_bundle (transaction : Transaction) :
    (cancel transaction).bundle = none := rfl

theorem cancelled_transaction_not_exportable (transaction : Transaction) :
    ¬ Exportable (cancel transaction) := by
  simp [Exportable, cancel]

end UFFFormal
