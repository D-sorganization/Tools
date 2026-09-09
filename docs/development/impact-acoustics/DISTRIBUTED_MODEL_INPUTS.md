# Explicit distributed model inputs

Tools #5072, parent #5068. This is a coefficient interchange boundary for the
existing finite-rotation section model. It does not complete T3, qualify an
impact-frequency model or supply measured shaft/hand parameters.

## Inventory and integration decision

`golf_club.shaft_serialization` already owns `golf_club.shaft_profile/1`,
including cut/trim/insertion semantics, EI, GJ, density and declared provenance.
`ShaftRodProperties` separately requires axial rigidity and polar inertia for
the stationary Euler–Bernoulli path. Neither record contains the complete
coupled six-axis section law and distributed COM inertia needed by the
finite-rotation chain. No automatic conversion is provided: missing shear,
axial, coupling or rotary terms cannot be inferred from those profile fields.

`golf_club.shaft_model_data` adds `golf_club.distributed_shaft/1`. It reuses
`SectionElement`, `SectionInertia`, `InertiaSample`, `ComponentMassProperties`
and existing duplicate-key/mass parsing. `_shaft_model_provider` composes the
loaded records into `RotatingSectionChain`; the existing moving-chain solver
then consumes that chain. Integration tests preserve the original synthetic
rotating fixture's twist rates, energies and anchor power exactly after JSON
round-trip. This is integration equivalence, not independent physical evidence.

The new public module is additive. The JSON format is the portable input
boundary; private solver compositions remain implementation details. Existing
shaft profiles, assembly wires and mechanics are unchanged. The current study
wire's source labels and evidence tier are not used to promote this model.

## Physical convention

Sections are ordered: record i connects nodes i and i+1. Each contains its
reference length L, reference relative twist d0, symmetric positive-definite
6×6 constitutive matrix and an explicit integrated inertia quadrature.
The existing law uses strain `(Log(H_left^-1 H_right) - d0) / L`, with
linear-first body coordinates. Translation entries of d0 are metres; rotation
entries are radians. Strain entries are dimensionless shear/extension followed
by curvature in 1/m. Constitutive blocks have units N, N m and N m².
Finite rotations do not remove the small-material-strain assumption.

Every inertia sample supplies a section fraction, mass in kg, material COM
offset in m and full COM tensor in kg m². Mass and inertia already include the
quadrature length and weight; compilation applies neither a second time.
The COM offset is relative to the interpolated section origin. Centerline
spread is represented by sample positions and must not also be added to the
section COM tensor. Existing physical tensor checks apply, including their
documented numerical tolerances. Sparse quadrature can still be singular.

All sample frames must match the declared material-axis convention. The
observer frame, its motion, loads, poses, head/grip choices, controls and
operating history are separate inputs. The coefficient digest alone therefore
does not identify a complete simulation. This version contains no inferred
damping, constitutive nonlinearity, contact or radiation law.

## Source declarations and identity

Each section references separate coefficient and inertia source IDs. The
source table must contain exactly the referenced IDs, without duplicates.
Each source declares a method, uncertainty note, data license, artifact SHA256
and kind: `synthetic`, `analytical` or `measurement-derived`. The last kind
requires a calibration artifact SHA256. Other kinds use an explicit null when
no calibration artifact is declared. An uncertainty note is descriptive;
`unquantified` is not zero and does not supply coefficient error bounds.

`verify_shaft_source_bytes(model, blobs)` checks exactly the supplied byte
artifacts against all declared data/calibration digests. It rejects missing,
extra, non-byte or mismatched entries before returning sorted matched digests.
There is no implicit filesystem or network access. Hash agreement does not
authenticate authorship, validate a calibration, demonstrate that coefficients
were derived correctly or establish a physical validity domain. The model's
status remains `unqualified`, including after successful byte verification.

`shaft_model_digest` hashes the canonical UTF-8 JSON, including all coefficients,
sample locations, mass properties and source declarations. Keys and source IDs
are sorted; section and sample ordering is preserved. Whitespace of the original
JSON is not part of this interpreted-model identity. This is the version-one
serializer convention, not a claim of conformance to a separate universal JSON
canonicalization standard. Preserve original input bytes separately when needed.

The parser refuses unknown or missing fields, duplicate keys, unsupported
versions, unresolved source references, frame mismatches and invalid physical
inputs. It refuses boolean/string coercion in numeric arrays, even though the
legacy assembly mass loader permits some coercions. The legacy format is
unchanged. There is no partial-model fallback.

## Validation and remaining work

TDD started with missing wire/provider modules. A further four-case RED exposed
three silent mass-array coercions and one coercion reaching a later physical
tensor error. Existing strict numeric-array validation now rejects them at the
new boundary. The initial 39 new controls pass on Windows in 7.16 seconds;
final regression and publication results are recorded in PROGRESS.md.

Source declarations and artifact identity are now representable. Measured
coefficient identification, uncertainty propagation, grip/FRF acquisition and
calibration, qualified frequency/strain domains, versioned full study inputs,
flexible contact (#5073), exact-pin UpstreamDrift consumers and physical/blinded
acoustic validation remain separate requirements. No new experimental source
or scientific literature claim is introduced by this interchange implementation.
