# Rate of Closure Clubhead Inertia-Tensor Contract

## Decision

The selected `ClubSpec` must continue to use the shared impact model's scalar
MOI path. The current data is not sufficient to derive an authoritative full
3×3 clubhead inertia tensor without inventing a mass distribution or an
orientation degree of freedom.

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

Until that contract exists, `simulation.pipeline._solve_hit` intentionally
passes `ClubSpec.moi_about_shaft_kg_m2` through the legacy scalar argument and
does not populate `clubhead_moi_tensor`.

## STL Boundary Delivered for #4111

The STL path is independent of the unresolved physical tensor. The selected
and user-edited `ClubSpec` deterministically generates the existing parametric
mesh, then serializes it as binary STL in the canonical head frame. Coordinates
are metres; STL carries no unit metadata, so the export action states the unit
contract in its tooltip and embeds it in the binary header. The UI writes only
to the path explicitly selected by the user.
