# Screw-Axis Analysis and Motion Glyphs

## Purpose

The Rate of Closure workbench uses instantaneous screw theory to explain how a
club or an articulated joint moves through three-dimensional space. A screw axis
is not an extra fitted golf variable. It is a geometric decomposition of the
same rigid-body angular and linear velocity already produced by the simulation.
The implementation follows the twist and screw-axis convention described in
Lynch and Park's *Modern Robotics*, Chapter 3.3.2 ([Northwestern video and
transcript](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-2-twists-part-1-of-2/),
[book preprint](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)).

## Frame and Twist Contract

All displayed values use the app/world frame:

- `x`: toward the target;
- `y`: up;
- `z`: right of the target line when looking downrange.

At a named reference point \(P\), the sampled twist is

\[
\mathcal{V}_P = (\boldsymbol{\omega}, \mathbf{v}_P),
\]

where \(\boldsymbol{\omega}\) is in rad/s and \(\mathbf{v}_P\) is the linear
velocity at \(P\) in m/s. This reference-point convention must not be confused
with a spatial twist whose linear term is defined at the world origin.

For \(\lVert\boldsymbol{\omega}\rVert > 0\), the finite instantaneous screw is

\[
\hat{\mathbf{s}} = \frac{\boldsymbol{\omega}}{\lVert\boldsymbol{\omega}\rVert},
\qquad
h = \frac{\boldsymbol{\omega}\cdot\mathbf{v}_P}
         {\lVert\boldsymbol{\omega}\rVert^2},
\]

\[
\mathbf{q} = \mathbf{P} +
\frac{\boldsymbol{\omega}\times\mathbf{v}_P}
     {\lVert\boldsymbol{\omega}\rVert^2}.
\]

Here \(\mathbf{q}\) is the closest point on the axis to \(P\), \(h\) is pitch
in m/rad, and the signed axial speed is
\(h\lVert\boldsymbol{\omega}\rVert\). The reference velocity is reconstructed
exactly as

\[
\mathbf{v}_P =
\underbrace{\boldsymbol{\omega}\times(\mathbf{P}-\mathbf{q})}_{\text{orbital}}
+
\underbrace{h\boldsymbol{\omega}}_{\text{axial}}.
\]

The software reports the numerical norm of the reconstruction error. A finite
axis is invariant to the chosen point on the same rigid body; only the named
point's radius and velocity change.

## Degenerate Motion States

The interface deliberately distinguishes three states:

1. **Finite Screw:** nonzero angular velocity; axis point, pitch, radius, and
   helical glyph are defined.
2. **Pure Translation:** zero angular velocity and nonzero linear velocity; the
   screw axis is at infinity. The scene draws a translation arrow and never a
   finite axis.
3. **Stationary:** both velocities are zero; no instantaneous axis exists.

This distinction prevents the visually attractive but physically incorrect
practice of drawing an arbitrary axis during straight translation or rest.

## Graphic Legend

- **Magenta main line and tip:** the directed instantaneous screw axis. The tip
  follows the right-hand-rule direction of \(\boldsymbol{\omega}\).
- **Orange wrapped curve:** a bounded engineering helix. Its handedness shows
  rotational direction. Its axial length grows monotonically with angular rate
  but is capped by the current swing-scale scene; it is an explanatory glyph,
  not a literal material trajectory.
- **Cyan dotted radius:** \(R_{ISA}\), from the selected reference point to the
  closest point on the axis.
- **White/purple skeleton:** physical segment geometry sampled from the same
  integrated joint state used by the clubhead path.

The glyph radius and displayed axis length are normalized for readability.
Numerical rate, pitch, axial speed, and \(R_{ISA}\) in the adjacent readout are
the quantitative values; screen length must not be measured as data.

## Club and Joint Views

**Club** uses the simulator's directly sampled club pose, angular velocity, and
reference-point velocity. It applies to every selected club specification,
including wedges, because screw geometry depends on the sampled rigid-body
motion rather than the head category.

**Joint** views reconstruct the instantaneous angular velocity of each sampled
link from its direction derivative. A joint's relative angular velocity is the
distal link angular velocity minus the proximal link angular velocity. Its
contribution at the clubhead is

\[
\mathbf{v}_{P,j} = \boldsymbol{\omega}_j \times
(\mathbf{P}-\mathbf{q}_j).
\]

The sum is compared with the numerical clubhead endpoint velocity, and the
residual is displayed. This residual exposes sampling, numerical differentiation,
base-motion, or model-contract disagreement rather than hiding it.

## Delivery Projections

Total, orbital, axial, and joint-contribution velocities can be projected onto
named unit directions using \(v_d=\mathbf{v}\cdot\hat{\mathbf{d}}\). The UI
reports target-line, vertical/AoA, lateral/path, and nominal face-normal
projections. It also reports the exact direction angles of the selected velocity:

\[
\mathrm{AoA}=\operatorname{atan2}(v_y,\sqrt{v_x^2+v_z^2}),
\qquad
\mathrm{Path}=\operatorname{atan2}(v_z,v_x).
\]

Orbital and axial vectors are additive; their AoA and path angles are not.
Component direction angles therefore remain labeled as diagnostics rather than
being summed into a fabricated total angle.

## Limitations

- The result is instantaneous and inherits the simulator's rigid-body and joint
  model assumptions; it does not prove causality in a measured golfer.
- Joint reconstruction assumes revolute links represented by nonzero sampled
  segment vectors. Translating base joints require an additional base-motion
  term.
- Numerical joint derivatives are most reliable on the simulator's uniform
  1 ms grid. The residual should be inspected before interpreting small
  differences.
- The nominal face-normal direction uses the configured loft and square-face
  delivery convention. Face curvature and local strike normals belong to the
  impact model and should be selected explicitly for off-center contact studies.
- Near-zero angular rates make a finite ISA ill-conditioned; such samples are
  classified by the documented tolerance instead of amplifying noise.

Implementation and delivery are tracked in
[Tools issue #4168](https://github.com/D-sorganization/Tools/issues/4168).
