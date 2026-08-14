# Golf Club Wedge Impact Kinematics

## Purpose and Scope

This UI-independent module evaluates one instantaneous rigid-body state at a
declared wedge contact point. It is the canonical calculation seam for the
PyQt and React impact inspector, delivery metrics, and variability studies.
It does not simulate impact forces, turf reaction, ball flight, or player
intent.

The primary question is measurable rather than assumed: how much does angular
velocity parallel to the physical shaft axis change contact-point angle of
attack (AoA) for the supplied pose, twist, shaft line, and contact point?

## Frame and Sign Contract

Every vector is expressed in one caller-declared inertial frame. Positions are
metres, translational velocities metres per second, angular velocities radians
per second, and direction vectors dimensionless. The caller supplies a unit
ground-up vector, so the calculation is not tied to one application axis order.

AoA is

`atan2(v_contact . ground_up, |v_contact - (v_contact . ground_up) ground_up|)`.

Descending motion is negative. AoA is undefined when horizontal speed is zero;
the API returns `None` rather than inventing a direction or clamping a result.

The face normal and leading-edge tangent must be orthogonal unit vectors. The
arc tangent is a unit vector and its time derivative must be orthogonal to it.
These are Design-by-Contract checks, not best-effort normalization.

## Exact Velocity Decomposition

The input twist is the reference-point velocity `v_R`, angular velocity
`omega`, and reference position `R`. A declared point `A` lies on the physical
shaft centerline, the shaft direction is the unit vector `a`, and `P` is the
contact point.

The velocity at the shaft datum is

`v_A = v_R + omega x (A - R)`.

Angular velocity is separated by vector projection:

- `omega_shaft = (omega . a) a`; and
- `omega_other = omega - omega_shaft`.

The contact velocity identity is then

`v_P = v_A + omega_shaft x (P - A) + omega_other x (P - A)`.

The three reported terms are shaft-axis-datum translation, shaft rotation, and
non-shaft rotation. They sum exactly and remain invariant if the twist is
re-expressed at another valid reference point. `A` must remain a physical shaft
datum; replacing it with a convenient clubhead origin changes the question.

## AoA Attribution

Two complementary shaft metrics are reported:

- direct counterfactual: `AoA(v_P) - AoA(v_P - v_shaft)`; and
- two-factor Shapley attribution of shaft and non-shaft rotation relative to
  `v_A`.

AoA is nonlinear, so component angles cannot simply be added. The Shapley
metric averages both factor orders and closes exactly to the difference between
total AoA and shaft-datum-translation AoA whenever all four AoAs are defined.
The direct counterfactual is easier to communicate; the Shapley result is the
order-independent attribution for multi-component comparisons.

The signed vertical velocity share is also reported. It can exceed 100%, be
negative, or be undefined when total vertical speed is zero. Those cases are
physical consequences of opposing terms and must not be clipped for display.

## Orientation and Axis Metrics

For any body-fixed unit direction `d`, its inertial derivative is
`d_dot = omega x d`. The module reports:

- signed shaft rotation rate `omega . a`;
- full 3D face-normal direction rate;
- full 3D leading-edge direction rate;
- leading-edge projected heading rate about ground up;
- arc-tangent projected heading rate about ground up; and
- their signed relative heading rate.

Projected heading is undefined when the direction is parallel to ground up.
The full instantaneous screw axis includes the point nearest the frame origin,
axis direction, screw pitch, and perpendicular distance from contact to that
axis. This makes the off-axis lever arm explicit instead of inferring it from a
screen projection.

## Worked Example

The regression fixture uses a right-handed target/ground frame, 64-degree lie,
15-degree forward shaft lean, a 20 mm contact offset from the shaft line,
1307 degrees per second of illustrative shaft-axis rotation, 30 mph contact
speed, and total AoA of -10 degrees.

The evaluated shaft term is approximately
`(0, -0.193183, -0.396084) m/s`. Removing it while holding the other terms fixed
produces -9.18118 degrees AoA, so the direct shaft contribution is -0.81882
degrees for this particular state. The 1307-degree-per-second value is an
illustrative driver-derived rate, not a claimed typical wedge rate.

For the simplified no-lean shaft direction
`a = (0, sin(lie), -cos(lie))` and a face-forward offset
`P - A = (d, 0, 0)`, shaft rotation contributes vertical velocity
`-omega_shaft d cos(lie)`. The sign reverses with rotation direction, handed
geometry, or offset direction. Shaft rotation therefore does not universally
steepen AoA; it steepens this convention only for the corresponding signed
rate and off-axis contact geometry.

## Verification and Release Boundary

Automated tests cover the worked example, exact vector closure, zero effect on
the shaft line, the analytical lie/offset formula, twist-reference invariance,
Shapley closure, ground/arc orientation rates, screw-axis clearance, invalid
AoA, and invalid frame geometry.

This foundation is ready for model adapters but does not yet establish that an
existing simulator's tracked point is the actual face contact point. Before a
UI may present these metrics as simulation results, its adapter must provide:

- the physical shaft line and true contact point in the same frame;
- complete angular velocity including shaft twist;
- face and leading-edge directions at that instant;
- an impact event or an explicit no-impact result; and
- the arc tangent and derivative used by the displayed comparison.

No metric in this module alone predicts launch height, descent angle, turf
clearance, bounce interaction, forgiveness, or shot outcome.
