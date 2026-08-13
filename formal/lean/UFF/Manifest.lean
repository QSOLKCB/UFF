import UFF.Basic

namespace UFFFormal

/-- The manifest core excludes the digest of the envelope that will later contain it. -/
structure ManifestCore where
  artifactDigests : List String
  contractDigest : String
  recipeDigest : String
  deriving Repr, DecidableEq

/-- The outer envelope may carry a digest computed from the already-formed core. -/
structure ManifestEnvelope where
  core : ManifestCore
  coreDigest : String
  deriving Repr, DecidableEq

/-- `digestCore` consumes only `ManifestCore`, making the self-hash exclusion explicit in the type. -/
def sealManifest (digestCore : ManifestCore → String) (core : ManifestCore) : ManifestEnvelope :=
  { core := core, coreDigest := digestCore core }

theorem sealing_does_not_modify_manifest_core
    (digestCore : ManifestCore → String) (core : ManifestCore) :
    (sealManifest digestCore core).core = core := rfl

theorem manifest_digest_is_computed_from_core_only
    (digestCore : ManifestCore → String) (core : ManifestCore) :
    (sealManifest digestCore core).coreDigest = digestCore core := rfl

end UFFFormal
