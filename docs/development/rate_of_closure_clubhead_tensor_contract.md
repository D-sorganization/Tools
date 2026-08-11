# Rate of Closure Clubhead Inertia-Tensor Contract

## Decision

The selected `ClubSpec` currently uses the shared impact model's scalar path.
That path is mathematically equivalent to assuming an isotropic tensor about
the head CG. The value supplied by `ClubSpec`, however, is sourced as a moment
about the shaft axis. The current calculation is therefore an explicitly
retained compatibility approximation with an axis/reference mismatch—not a
resolved CG inertia model. The current data cannot produce an authoritative
full 3×3 tensor without inventing mass distribution or attitude information.

This decision does not limit the already-supported expert path:
`PreImpactState.clubhead_moi_tensor` may still receive an independently sourced
tensor expressed about the head CG in the same frame as the impact vectors.

## Why `ClubSpec` Is Insufficient

`ClubSpec` supplies head mass, two CG coordinates, a single scalar moment about
the shaft axis, and representative envelope geometry. A symmetric tensor has
six independent components. Neither one scalar moment nor the CG location
determines the remaining components or products of inertia. Many physically
different internal mass distributions satisfy the current fields.

The parametric mesh is suitable for rendering, geometric volume/centroid
inspection, and STL export. Its constant-density size scaling does not assert
that the finished head has uniform density. Treating the shell as a uniform
solid would silently discard real sole, perimeter, face, hosel, and internal
weighting. It would also disagree with the spec CG whenever the uniform-volume
centroid differs from the published CG.

Finally, the impact state currently carries a face-normal vector, not a complete
world-from-head rotation. A body-frame tensor requires all three attitude
degrees of freedom before it can be transformed as `I_world = R I_body R^T`.
Rotation about the face normal cannot be recovered from the normal alone.

## Required Production Contract

A future selected-club tensor path must provide all of the following:

1. A finite, symmetric, positive-definite `3×3` tensor in kg·m² about a stated
   head CG, with physical principal-moment triangle inequalities validated.
2. Provenance stating whether the tensor is measured, manufacturer supplied,
   CAD-integrated with an explicit material-density field, or an opt-in labeled
   approximation. Representative render geometry alone is not provenance.
3. A documented right-handed head body frame and a complete world-from-head
   attitude at impact. The tensor supplied to the shared impact model must be
   transformed into the same frame as velocity, face normal, and impact offset.
4. CG coordinates in all three body axes, including heel-toe position, and an
   explicit relation between the CG datum and the mesh/body-frame origin.
5. Reconciliation checks against head mass, the published scalar shaft-axis
   MOI, and any CAD volume/density integration; disagreement must be reported,
   never silently rescaled.
6. Tests for isotropic/scalar equivalence, anisotropic off-center response,
   frame-rotation covariance, symmetry/positive-definiteness rejection, and
   serialization round trips.

Until that contract exists, `simulation.pipeline._solve_hit` passes
`ClubSpec.moi_about_shaft_kg_m2` through the legacy scalar argument as an
isotropic-equivalent approximation and does not populate
`clubhead_moi_tensor`. The shaft-axis source and CG-centered scalar equation do
not share an exact reference; consumers must not present this path as measured
tensor physics.

## STL Boundary Delivered for #4111

The STL path is independent of the unresolved physical tensor. Only the
mesh-defining `ClubSpec` subset—club type/style, head mass, loft, and optional
bulge/roll radii—drives the representative geometry. Name, length, lie, CG, and
MOI do not drive the mesh. The generator computes internally in SI metres, then
the serializer writes millimetre coordinates because STL is unitless and mm is
the conventional CAD interchange assumption. The binary header and UI tooltip
state `units=mm` and the canonical axes (x target, y up, z toe).

The PyQt export writes a same-directory temporary file, flushes it, and
atomically replaces the user-selected destination. Serialization, write, or
replace failures preserve an existing destination where supported and remove
the temporary artifact where the operating system permits. The React action
serializes the same selected mesh in the browser, creates one local
`model/stl` object URL, initiates the download, and releases the URL after both
successful and failed click attempts. Filename defaults on both surfaces fall
back safely for Unicode-only names, avoid Windows reserved device stems, and
bound the generated stem length. Neither surface presents this representative
mesh as tensor-derived or production CAD geometry.

## Versioned Engineering Sidecar Delivered for #4111

Both selected-club panels now export
`rate_of_closure.clubhead_engineering/1` JSON. The sidecar hashes the exact
companion binary STL with SHA-256, records its portable companion filename,
byte length, and mesh-defining inputs, and declares the right-handed head frame
plus the identity/1000× head-to-STL coordinate transform. The STL byte contract
itself is unchanged.

The selected `ClubSpec` supplies an application-authoritative representative
head mass. Its two datum-relative CG offsets and one shaft-axis scalar moment
are retained only as `evidence_only`; neither unavailable record contains a
substitutable `value`. Capability entries explicitly mark the complete head CG,
full symmetric CG tensor, world-from-head attitude, and assembly mass
properties unavailable. The browser rejects an invalid SHA-256 result, and
both surfaces validate the selected inputs before serialization.

The shared `golf_club.ClubAssembly` domain carries validated component and
assembly mass, CG, full tensors, frames, and component transforms. The strict
`rate_of_closure.club_assembly_binding/1` import contract now binds one such
assembly to one exact selected `ClubSpec` identity. Both identities are SHA-256
digests of deterministic, cross-language canonical bytes; a golden vector pins
the Python and browser encoders. The binding also identifies the unique head
component, supplies the explicit selected-head-to-component transform, declares
SI units, and preserves a measured, manufacturer, CAD-integrated, or qualified-
analysis source-authority record. Duplicate JSON fields, unknown or absent
fields, payloads over 4 MiB, unsupported authority classes, nonphysical tensors,
head-mass mismatch, frame mismatch, and either identity mismatch fail closed.

After that complete validation, PyQt and React engineering-sidecar export can
mark the head CG, head full CG tensor, and assembled-club mass properties
available in their explicitly named frames. Any identity-defining Club-panel
edit clears the retained binding. The source declaration is preserved rather
than independently certified, and the included driver fixture is synthetic
qualified-analysis test data—not a manufacturer measurement or production
club definition. No default shaft/grip assembly, uniform-density mesh tensor,
CAD density, or missing transform is inferred.

This binding does not satisfy the dynamic world-attitude requirement. The
sidecar continues to mark `world_from_head` unavailable, and the simulation
does not populate `clubhead_moi_tensor`, until a separately validated complete
world-from-head orientation is supplied at impact.
