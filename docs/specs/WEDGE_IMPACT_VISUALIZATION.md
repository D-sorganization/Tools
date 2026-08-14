# Exact-Event Wedge Impact Visualization

## Purpose

The impact inspector makes one retained swing state auditable in three
dimensions. It is intended to test questions such as whether rotation about
the physical shaft line makes the declared contact point descend more steeply
than the shaft-axis datum. It is not a claim that more shaft rotation is always
beneficial, nor a fitted model of one player's technique.

## Event and Frame Contract

The scene uses `app_frame:x_target,y_up,z_right`. Positive vertical velocity is
upward, so a descending angle of attack is negative. A hit is inspected at its
continuous impact time. A miss is inspected at its explicitly labeled closest
approach and is never relabeled as impact.

Retained translations, twists, and articulated wrist positions are linearly
interpolated to the event time. Retained rotations use shortest-arc quaternion
interpolation. The nearest retained sample index is preserved only for audit
and stable finite-difference calculations.

## Rigid-Body Decomposition

For contact point `c`, twist reference `r`, a point `s` on the physical shaft
line, shaft unit vector `ŝ`, and rigid-body angular velocity `ω`:

```text
v_axis  = v_r + ω × (s - r)
ω_shaft = (ω · ŝ) ŝ
ω_other = ω - ω_shaft
v_shaft = ω_shaft × (c - s)
v_other = ω_other × (c - s)
v_contact = v_axis + v_shaft + v_other
```

The implementation and tests enforce this vector identity. “AoA without shaft
rotation” removes `v_shaft` and recomputes AoA. “Shaft AoA contribution” is the
difference between the full and counterfactual AoA. Because `atan2` is
nonlinear, this quantity is not an additive Euler-angle component. The shared
analysis also exposes an order-independent two-factor Shapley attribution.

The face-normal and leading-edge 3D rates are `|ω × n|` and `|ω × e|`. Ground
heading rates are unavailable when the projected direction is singular. The
instantaneous screw axis is unavailable at zero angular speed.

## Scene Contents

Both interfaces show the same engineering entities:

- representative wedge face and body at the exact event pose;
- physical shaft datum and declared face contact point;
- ball center and locked-scale ground grid/plane;
- face normal, leading edge, and arc tangent;
- rigid-body face-center travel (`v_reference + omega x r_face_center`), the
  face-center normal, D-plane normal, ground-projected path, and the exact
  shaded 3D spin-loft sector;
- finite instantaneous screw axis when available;
- total contact velocity, shaft-axis translation, shaft-rotation velocity,
  other-rotation velocity, and the no-shaft counterfactual.

Vector arrows share one scale within a scene so relative magnitudes remain
meaningful. Geometry uses equal physical scaling across axes; resizing the view
does not distort the wedge or trajectory directions.

## Controls and Exports

The React still supports pointer orbit, wheel zoom, keyboard arrow-key orbit,
isometric/face-on/down-the-line views, reset, and individual velocity toggles.
It exports a device-resolution PNG, true SVG primitives, and the complete scene
as JSON. The PyQt6 swing view remains freely rotatable, adds the same named
camera presets and vector layer, and exports 300-DPI PNG, Matplotlib vector SVG,
or `rate-of-closure.impact-scene/v2` JSON. Layer visibility is persisted as a
user preference; PNG/SVG geometry honors the active layers, and the complete
JSON scene records the active layers and camera for reproducible rendering.
The interactive view consumes the shared canonical camera contract in
`docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md`; snap orientation is explicit
and does not infer handedness or frame convention from the club pose.

Every metric card is keyboard-focusable and visibly labeled “Click for
Definition.” Its expanded content gives the equation, frame, units,
assumptions, and reason for unavailability. The still does not auto-animate, so
it honors reduced-motion preferences without a separate mode.

## Scientific Boundaries

- Manual prescribed twists treat the shaft as a rigid datum through the tracked
  head reference. They do not include shaft bending or torsional compliance.
- Current articulated pendulum sources provide a measured wrist-to-head shaft
  line but no independent shaft-twist degree of freedom.
- The still is instantaneous kinematics. It does not feed the compliant turf
  wrench back into the swing trajectory or predict injury, technique quality,
  launch benefit, or “best bounce.”
- Generic wedge and turf presets are illustrative. Calibrated turf profiles and
  validation data are required before ranking designs or making forgiveness
  claims.

## Verification

Python tests cover off-grid event interpolation, vector reconstruction,
orthogonal orientation datums, strict JSON, PyQt geometry labels, locked aspect
ratios, and SVG/data export. TypeScript tests independently cover off-grid
interpolation, the vector identity, orientation orthogonality, semantic SVG
primitives, accessibility, and the integrated simulation panel. Production
build, type, lint, and full application suites remain protected merge gates.
