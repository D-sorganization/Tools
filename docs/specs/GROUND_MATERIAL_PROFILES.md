# Ground Material Profile Contracts

## Status and scope

This document is the scientific and persistence authority for the bounded
issue #4272 contract slice. It defines strict, versioned SI material profiles,
qualification evidence, explicit operating-condition binding, fail-closed
local persistence, and a one-way neutral snapshot adapter for selected
UpstreamDrift terrain fields.

This slice does **not** provide production presets, claim that any material is
calibrated, model changing normals or regional boundaries, add profile-editing
UI, or deliver TypeScript/Rust/PyO3/WASM/UpstreamDrift consumer parity. Those
remain release criteria in epic #4267 and issues #4272-#4276.

## Canonical documents

`ground-material-profile/v1` carries:

- a stable profile identifier and revision;
- exactly eleven solver-facing SI parameters, in canonical identifier order;
- standard uncertainty plus evidence-linked lower and upper SI validity bounds
  for every parameter;
- immutable evidence records with source identity, SHA-256, parameter
  coverage, source kind, and rights declaration;
- bounded temperature, moisture, and surface-class applicability;
- a required calibration-assessment record whose parameter and evidence
  dependencies are explicit; illustrative status records still carry this
  assessment and are never presented as calibrated;
- producer/revision/source-digest provenance; and
- seven derived qualification gates plus their coherent aggregate status; and
- a separate derived scientific use status: `calibrated` only when all
  scientific gates except reuse rights pass and referenced calibration evidence
  includes measured data, otherwise `illustrative`.

`ground-profile-library/v1` is a deterministic, sorted collection of unique
profile revisions with its own provenance. Document identities use canonical
numeric JSON and SHA-256. Raw unsafe integers, non-finite values, duplicate
JSON keys, unknown or omitted fields, control/surrogate text, subclasses, and
noncanonical encodings fail closed before they can acquire a canonical
identity.

The generated Draft 2020-12 schemas express structural constraints only.
Cross-record invariants—including exact parameter ordering, evidence coverage,
calibration dependencies, qualification coherence, and applicability—are
owned by `validate_profile_payload` and `validate_library_payload`. Schema
acceptance is therefore necessary and deliberately not sufficient.

## Qualification and binding

Qualification is derived, never self-declared. A profile is qualified only
when all seven stable gates pass: complete parameter evidence, traceable
validity bounds, adequate rights, declared uncertainty, calibration,
applicability, and reproducible provenance. Each validity limit must satisfy
the parameter's physical domain, enclose its point value, and reference
evidence that covers that exact parameter. A calibration gate may use only
evidence referenced by the
calibration record, and that evidence must cover every calibrated parameter.
The declared calibration moisture point must lie within the profile's
applicability range.

Binding requires exact `GroundMaterialProfile`, `SurfacePlacement`, and
`ProfileOperatingCondition` records. Surface class, temperature, and moisture
must be inside the declared applicability envelope. All eleven values then map
without unit conversion into one explicit solver `GroundSurfaceProfile`.
Unqualified profiles remain usable only with a typed
`GROUND_PROFILE_UNQUALIFIED` warning; scientifically illustrative profiles add
the distinct `GROUND_PROFILE_ILLUSTRATIVE` warning. Qualification, scientific
use status, profile identity, applicability, and operating state remain bound
to the solver output so no caller can silently promote or forge them.

## Persistent storage boundary

`GroundProfileLibraryStore` owns one fixed file inside one caller-selected,
existing, absolute, real directory. It provides:

- bounded reads of canonical JSON;
- exact-document validation before acceptance;
- exclusive writer locking;
- create and SHA-256 compare-and-swap updates;
- same-directory temporary files, file flush/fsync, Windows write-through
  atomic replacement, and POSIX directory sync;
- a last-known-good backup; and
- explicit backup recovery bound to both the observed primary digest and the
  selected backup digest.

The store never migrates, repairs, or recovers automatically. It rejects
symlink/reparse-point roots or files, unsafe filenames, replaced root
identities, oversized
or corrupt bytes, stale compare-and-swap identities, and concurrent writers
with typed errors. If atomic replacement succeeds but directory durability
cannot be confirmed, it raises `ProfileStoreIndeterminateCommitError` with the
destination and committed-byte digest; callers must inspect durable state
rather than retry blindly.

This is a cooperative, single-principal local persistence contract, not an
adversarial filesystem sandbox. Paths are revalidated before operations, but
path-based Python I/O cannot eliminate every same-user time-of-check/time-of-use
race. Callers must not place the store in a directory writable by untrusted
principals.

## Neutral UpstreamDrift terrain adapter

Tools does not import UpstreamDrift classes. A caller first constructs exact
canonical `UpstreamTerrainSnapshot`, `FrameTransform`, and
`TerrainAdapterInterpretation` records. The adapter:

- applies only a caller-supplied proper rigid transform;
- preserves separate terrain and material identifiers/revisions, display name,
  source point, normal, tangential surface velocity, material scalars, source
  digest, source frame, adapter version, and transform;
- requires the caller to document the nonunique friction split and firmness
  interpretation;
- hashes the snapshot, transform, interpretation, and combined adapter input;
  and
- returns a field-by-field disposition report distinguishing exact retention,
  explicit interpretation, local linearization, and unrepresented regional
  topology.

The result is one local tangent plane. It must not be represented as regional
terrain parity or evidence that changing normals, boundaries, deformation, or
terrain evolution are implemented.

## Verification boundary

The qualified local slice requires the full ground test suite on CPython 3.11
and isolated real CPython 3.10.20, pinned Ruff and MyPy checks, exact public API
tests, structural budgets, campaign-manifest validation, documentation
governance, and independent adversarial review. Protected CI, review, ordinary
dependency integration, UI, compiled runtimes, production-source review, and
downstream parity remain release gates.
