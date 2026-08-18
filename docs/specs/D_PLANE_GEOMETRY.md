# Comprehensive 3D D-Plane Geometry

## Scope and Frame

The canonical app frame is right-handed: `x` points along the target line,
`y` points up, and `z` points right. Club path and face angle are positive to
the right; attack angle is positive upward. The D-plane kernel also accepts an
arbitrary orthogonal target/up pair and derives its positive-right axis as
`target x up`.

The kernel is geometric. It does not independently predict ball launch,
spin, gear effect, or curvature. Those require a declared collision and flight
model with impact location, friction, contact interval, head inertia, and
aerodynamic inputs.

## Exact Definitions

For nonzero face-center velocity `v` and unit face normal `n`:

```text
v_hat                = v / |v|
spin_loft_3D         = acos(clamp(v_hat dot n, -1, 1))
D_plane_normal       = unit(v_hat x n)
club_path            = atan2(v_hat dot right, v_hat dot target)
attack_angle         = atan2(v_hat dot up, |horizontal(v_hat)|)
face_angle           = atan2(n dot right, n dot target)
dynamic_loft         = atan2(n dot up, |horizontal(n)|)
planar_spin_loft     = |dynamic_loft - attack_angle|
approximation_error  = spin_loft_3D - planar_spin_loft
```

The D-plane inclination to the ground is `acos(|n_D dot up|)`. Its signed
normal tilt is `atan2(-(n_D dot up), |horizontal(n_D)|)`; positive is
face-right. That is fade-side only under the app's current right-handed display
convention; the geometric kernel does not itself predict curvature. Face-to-path
is the wrapped signed difference between the two horizontal headings.

## Reference Points

Impact inspection reports three distinct analyses:

- tracked head-reference travel versus the face-center normal;
- face-center rigid-body travel, including `omega x r_face_center`, versus the
  face-center normal;
- actual contact-point rigid-body travel versus the location-dependent face
  normal, including driver bulge and roll when configured.

The face-center analysis is the default visual comparison. These quantities
must not be compared to a device value unless reference point and event time
are compatible or a declared point/time transformation is applied.

## Singular States

- Zero travel has no travel direction, spin loft, or D-plane.
- Parallel vectors have zero spin loft but no unique D-plane normal.
- Antiparallel vectors have 180-degree spin loft but no unique D-plane normal.
- Vertical vectors retain elevation but have no horizontal heading.
- A D-plane parallel to the ground has no unique ground-intersection heading.

The data contract preserves these states and emits `null` for unavailable
quantities. Renderers omit unavailable arrows and sectors; they never install a
plausible-looking fallback axis as if it were measured geometry.

## Verification

Python and TypeScript consume the same versioned golden fixture and pin analytic
square and horizontally mismatched deliveries, zero/parallel/antiparallel cases,
vertical projection singularity,
shortest-arc sector membership, proper-rotation equivariance, right/left
reflection signs, rigid-body face-center velocity, curved-face contact normals,
strict JSON, interactive controls, preference persistence, and SVG/PyQt scene
artifacts.
