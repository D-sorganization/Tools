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

## Synthetic Kernel Worked Example

### Declared State

The geometry-independent regression fixture uses a right-handed target/ground frame (`x` target,
`y` up, `z` right), 64-degree lie, 15-degree forward shaft lean, a 20 mm
face-forward contact offset from the physical shaft line, and 1,307 degrees per
second of positive shaft-axis rotation. The final contact velocity is declared
to be `(13.207454, -2.328830, 0) m/s`, whose magnitude is exactly 30 mph to the
reported precision and whose AoA is -10 degrees.

The 20 mm lever arm is intentionally synthetic. It proves the shared kernel's
vector closure and attribution identities; it must not be presented as the
geometry of the generated wedge head shown by either client.

The 1,307-degree-per-second input is the mean handle twist velocity measured by
Cheetham in 94 tour-professional **driver** swings (standard deviation 304,
observed range 652--2,432 degrees per second). It is an illustrative sourced
reference, not a measured or inferred typical wedge rate. No wedge-specific
impact distribution was found in the reviewed primary literature, so the UI
and this specification must not label it as one. See Cheetham (2014),
[The Relationship of Club Handle Twist Velocity to Selected Biomechanical
Characteristics of the Golf Drive](https://www.philcheetham.com/media/Phillip-Cheetham-Doctoral-Dissertation-2014.pdf),
especially Table 2.

### Translation-versus-shaft decomposition

For the declared state the exact vectors are:

| Term | Velocity in `(target, up, right)` [m/s] | Downward speed [m/s] |
| --- | ---: | ---: |
| Shaft-axis datum translation | `(13.207454, -2.135647, +0.396084)` | 2.135647 |
| Rotation about shaft | `(0, -0.193183, -0.396084)` | 0.193183 |
| Other rotation | `(0, 0, 0)` within numerical precision | 0 |
| **Contact point total** | **`(13.207454, -2.328830, 0)`** | **2.328830** |

Thus shaft-axis rotation supplies 8.2953% of the downward velocity and the
shaft-datum translation supplies 91.7047%. Removing only the shaft term while
holding the physical state otherwise fixed gives -9.18117 degrees AoA. The
direct shaft counterfactual is therefore -0.81882 degrees, or 8.19% of the
10-degree AoA magnitude. The velocity share and angular share answer different
questions because AoA is an `atan2` of vertical and horizontal speed and the
shaft term also cancels 0.396084 m/s of rightward velocity.

There is no non-shaft angular component in this intentionally isolated fixture,
so the two-factor Shapley result equals the direct shaft counterfactual. In a
measured swing with swing-plane rotation, the non-shaft term must be retained
and the reported angular contributions will generally not equal simple vector
percentages.

### Shaft-rate sensitivity with translation held fixed

The following table varies only shaft-axis rate while keeping the above
shaft-datum translation and geometry fixed. Values are regression-tested rather
than obtained from a display-layer approximation.

| Shaft rate [deg/s] | Total AoA [deg] | Direct shaft AoA delta [deg] | Shaft share of downward speed |
| ---: | ---: | ---: | ---: |
| 0 | -9.1812 | 0.0000 | 0.00% |
| 652 | -9.5911 | -0.4099 | 4.32% |
| 1,003 | -9.8106 | -0.6294 | 6.49% |
| 1,307 | -10.0000 | -0.8188 | 8.30% |
| 1,611 | -10.1887 | -1.0075 | 10.03% |
| 2,432 | -10.6946 | -1.5134 | 14.41% |

The table is a geometric sensitivity study, not a player recommendation. Rate,
offset, lie, lean, handedness, and contact location jointly determine the sign
and magnitude.

### Representative generated-head cross-check

For a geometry-specific comparison, use the Rate of Closure `Pitching Wedge`
profile's face center and hosel rather than the synthetic 20 mm lever arm. In
the unleaned head frame their face-center-minus-hosel vector is
`(2.966, -24.719, 37.573) mm`. Rotating that rigid head and its 64-degree shaft
15 degrees targetward gives:

- shaft-axis unit vector `(0.232625, 0.868168, -0.438371)`;
- hosel-axis-to-contact vector `(-3.533, -24.645, 37.573) mm`; and
- at 1,307 degrees per second, shaft-induced contact velocity
  `(+0.497660, -0.164057, -0.060817) m/s`.

If 30 mph denotes the **total contact-point speed** at exactly -10 degrees AoA,
the corresponding shaft-axis-datum translation is
`(12.709794, -2.164774, +0.060817) m/s`. Rotation supplies 7.0446% of the
downward speed. Removing only that rotation term gives -9.66594 degrees AoA,
so the counterfactual shaft contribution is -0.33406 degrees. An alternate
canonical-wedge CAD hosel datum produces about -0.223 m/s and -0.474 degrees;
that spread demonstrates why the exact physical datum must accompany any
reported attribution.

At zero forward lean, the representative Rate geometry's vertical shaft term
is only about -0.0297 m/s and its extra forward velocity can make the total AoA
shallower. The proposed steepening mechanism is therefore mechanically
plausible for the declared lean and hosel offset, but rotation rate alone does
not determine its sign or magnitude.

### Reconciliation with the requested 30-yard carry

Carry is not a kinematic input to this decomposition. It follows only after a
club/ball impact model supplies ball launch conditions and an aerodynamic model
integrates the flight. As an explicit consistency check, take a nominal
52-degree wedge and approximate 15 degrees of forward lean as 37 degrees of
dynamic loft. With 30 mph club speed, -10 degrees AoA, centered impact, the
current rigid-body impact model, and the `waterloo_penner` calm-flight model,
the model predicts:

| Quantity | Model result |
| --- | ---: |
| Ball speed | 30.449 mph |
| Launch angle | 37.000 degrees |
| Spin | 3,135.8 rpm |
| Carry | 17.566 m (19.211 yd) |

This model therefore does **not** predict 30 yards from that 30 mph delivery.
Holding AoA and dynamic loft fixed, the same impact/flight chain requires
approximately 37.887 mph club speed to carry 27.432 m (30 yd); that solution
launches the ball at 38.454 mph with approximately 3,960 rpm spin. Conversely,
a directly prescribed 37-degree, 3,136-rpm ball launch would need about
40.13 mph ball speed to carry 30 yards, which is not the ball speed produced by
the 30 mph rigid-body impact case.

These results are deterministic model outputs, not validation against turf,
grass, ball cover, groove condition, shaft compliance, or measured short-game
launch data. The 30 mph/30-yard request should be treated as a target to be
validated or solved, not as an assumption that overrides the forward model.

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

This foundation is ready for model adapters but the current manual Simulation
source cannot yet reproduce the representative generated-head cross-check. It
travels horizontally, has no explicit forward-lean/head-pose input, and places
the manual shaft axis through the tracked reference point. The articulated
double/triple-pendulum sources expose a wrist-to-reference shaft line but no
torsional shaft degree of freedom. Before a UI may present the representative
case as a simulation result, its adapter must provide:

- the physical shaft line and true contact point in the same frame;
- complete angular velocity including shaft twist;
- face and leading-edge directions at that instant;
- an impact event or an explicit no-impact result; and
- the arc tangent and derivative used by the displayed comparison.

The Flight Explorer can independently reproduce the downstream consistency
case by entering 30 mph, -10 degrees attack angle, and 37 degrees dynamic
loft. That does not back-fill the missing physical pose into the Simulation
tab.

No metric in this module alone predicts launch height, descent angle, turf
clearance, bounce interaction, forgiveness, or shot outcome.
