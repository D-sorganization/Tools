# Planar Ground Skid, Roll, and Regional Material Specification

Status: locally implemented feature-stack slice for Tools issue #4271; not a
protected or main-branch release.

## Scope and authority

This specification defines the Python reference continuation from the exact
`SETTLED_TO_SKID` handoff produced by issue #4270 through tangential skid,
pure roll, and (when physically representable) rest. It also defines the only
qualified composition of the #4270 prefix and #4271 suffix into the strict v1
`GroundSimulationResult`.

The qualified domain is one immutable plane with a request-bound base
`GroundSurfaceProfile`. A caller may add finite coplanar material overlays
along the plane's declared tangent axis. Higher unique integer precedence wins
where overlays overlap. A caller may also bound the base plane; crossing its
first finite edge terminates with exact `LEFT_SURFACE` evidence. This slice
does not claim changing surface normals, height or velocity discontinuities,
terrain deformation, grass-blade interaction, torsional spin damping,
roll-to-skid transitions, TypeScript or compiled regional physics, UI delivery,
or downstream application parity.

## Units, frames, and handoff preconditions

All calculations use target-frame SI units. Position is metres, velocity is
metres per second, angular velocity is radians per second, acceleration is
metres per second squared, force is newtons, energy is joules, and time is
seconds. Version 1 uses standard gravity exactly:

```text
g = (0, -9.80665, 0) m/s^2
```

The request, bounce prefix, resolver base surface, frame, provider identity,
and base material profile must match exactly. The prefix must terminate
`SETTLED_TO_SKID`, expose a handoff at zero signed sphere-plane gap, and have
zero relative normal speed. The handoff centre must lie within the declared
base domain and outside every material overlay, so the impact-bound request
profile owns the initial contact. Every overlay must have two finite bounds,
a unique ID and precedence, the same tangent axis/origin, and exactly the same
frame, height, normal, and surface velocity as the base. Only validated
material coefficients and identities may differ. Invalid inputs raise before
any trajectory or summary is constructed.

For ball radius `R`, mass `m`, rotational-inertia factor `k`, inertia
`I = k m R^2`, upward plane normal `n`, centre velocity `v`, angular velocity
`omega`, and surface velocity `u_s`, the tangential contact slip is:

```text
r = -R n
c = P_t[(v - u_s) + omega cross r]
P_t(a) = a - (a dot n) n
```

Normal-axis spin is retained. It does not change `c` and is not silently
damped by a tangential Coulomb or rolling-resistance law.

## Skid dynamics

When `|c|` exceeds the versioned slip tolerance, kinetic Coulomb friction
opposes the current slip direction:

```text
g_n = -(g dot n) > 0
N = m g_n
F_k = -mu_k N unit(c)
a = P_t(g) + F_k / m
alpha = (r cross F_k) / I
c_dot = a + alpha cross r
```

Within one frozen-direction substep, centre and angular motion use constant
acceleration. A collinear positive root of `c + c_dot t = 0` localizes exact
capture. For oblique slip, an adaptive characteristic-time bound limits each
closing substep to one quarter of `|c| / |c_dot|`; the direction is then
re-evaluated. This prevents a numerical step from crossing the singular
zero-slip manifold and converges to the versioned slip tolerance without
chatter. The transition occurs only if static friction can sustain the
required rolling constraint. The exact inclusive feasibility condition is:

```text
m k/(1+k) |P_t(g)| <= mu_s m g_n + 1e-12 N
```

If it cannot, the internal suffix terminates `UNSUPPORTED_SURFACE`; v1
composition rejects that outcome instead of fabricating a final result.

## Pure-roll dynamics and rest

At the transition, only round-off residual is projected onto the no-slip
manifold. Tangential centre velocity and normal-axis spin are preserved:

```text
q = P_t(v - u_s)
omega_roll = (n cross q)/R + (omega dot n)n
```

While static rolling remains feasible, gravity-driven rolling acceleration is
`P_t(g)/(1+k)`. Dimensionless rolling resistance supplies an opposing
acceleration of magnitude `mu_r g_n` along relative travel (or along the
gravity drive when starting from zero). A constant-vector root localizes zero
relative centre speed without reversal.

Rest is emitted only when all of the following are true:

- the surface is stationary;
- rolling resistance can hold the gravity drive at zero speed;
- full centre speed is within the linear tolerance; and
- full angular speed, including normal-axis spin, is within the angular
  tolerance.

A moving surface therefore cannot produce world-frame `REST`. Residual
normal-axis spin also prevents `REST`; without a calibrated torsional law the
bounded run continues until a declared limit.

## Deterministic integration, sampling, regions, and finite edges

The solver is bounded by request time/event limits plus versioned numerical
step and maximum-step limits. Cancellation is checked before each step. Invalid
numerical states raise without a result instead of preserving or inventing
unvalidated evidence. Output
grid samples remain anchored at first contact, not at the skid handoff, and
suffix samples are strictly later than the handoff. Event points replace
coincident grid points.

For a finite planar domain, the centre coordinate along its declared tangent
axis is a quadratic within each constant-acceleration step. The resolver
selects the earliest positive outward root and advances exactly to that edge.
The resulting `LEFT_SURFACE` event is a complete end of this surface model,
not evidence of rest. A consumer requiring rest-only legacy output must reject
it.

Every regional lower/upper coordinate is also an exact quadratic boundary
candidate. At a boundary the resolver samples the outgoing coordinate with the
next representable floating-point value in the direction of travel, then
selects the highest-precedence containing overlay. It splits the step only
when that selected identity differs from the active region. If a regional
boundary and the base-domain edge coincide, base-domain exit takes precedence.

A regional transition preserves time, position, centre velocity, angular
velocity, phase, and accumulated energy: no impulse or epsilon-time point is
fabricated. The active material changes only after a legal
`SURFACE_TRANSITION` event is appended. A parallel internal transition ledger
binds that event sequence/time/position to the source and destination region
and surface IDs. The request event limit includes these events, and a separate
positive `max_surface_transitions` setting terminates with typed internal
`SURFACE_TRANSITION_LIMIT` evidence before an unbounded regional sequence can
run. The existing positive maximum-step bound remains independent.

Region definitions and their source/destination identities remain execution-
scoped non-wire inputs in this slice. The strict v1 result preserves the legal
transition event and qualified-domain warning, but it does not serialize the
regional plan or the internal identity ledger. A versioned wire request/result
extension is required before regional plans can cross process boundaries.

Skid and roll distances are accumulated separately from centre speed relative
to the moving surface. Collinear constant-acceleration segments use the exact
trapezoidal speed integral; non-collinear segments use deterministic Simpson
quadrature over the bounded step. Refinement tests constrain the latter.

## Energy ledger

Every internal suffix reports translational plus isotropic rotational kinetic
energy before and after, gravity work, moving-surface work, and dissipation:

```text
K = 0.5 m |v|^2 + 0.5 I |omega|^2
W_g = sum(m g dot delta_position)
W_surface = sum((F_contact dot u_s) delta_time)
D = K_before + W_g + W_surface - K_after
```

`D` must be nonnegative within a small absolute/relative round-off allowance.
A larger violation fails closed. A stationary surface cannot add mechanical
energy; a moving surface may exchange energy only through the explicit work
term. Work, force, skid/roll path, and resistance are evaluated with the
material active over each exactly split segment; state continuity contributes
no boundary work.

## Prefix/suffix composition

`compose_ground_result` accepts only exact, identity-matched request, prefix,
and suffix records. It maps `REST` and `LEFT_SURFACE` to complete v1 results,
and time/event limits to partial, censored results. Cancellation, step, and
unsupported-surface outcomes remain internal because v1 has no honest wire
representation for them; invalid numerical states never produce a suffix.

When #4270 captures at first contact, its sole same-time `SKID` point is
reconstructed as `IMPACT` from the signed first-contact event and is not
duplicated. The suffix begins strictly later. No epsilon-time sample is
invented. An immediate same-time rest is rejected because v1 requires a
strictly increasing trajectory.

Summary definitions are:

```text
carry = hypot(first_contact.x, first_contact.z)
bounce_air = sum(#4270 horizontal airborne segment distances)
surface_path = skid + roll
total = hypot(final_position.x, final_position.z)
bounce_count = number of post-first-contact BOUNCE events
```

For a partial or edge-censored result, totals describe the observed endpoint;
they are not projected final-rest values. Results without transitions carry
the static-plane warning. Results with regional events carry
`REGIONAL_PLANAR_V1`, which states that only coplanar, equal-velocity material
changes are qualified. All results retain the undamped-axial-spin and
censored-endpoint warnings as applicable.

## Qualification evidence

The shared fixture
`src/rate_of_closure/web/src/model/__fixtures__/ground_skid_roll_golden_v1.json`
has SHA-256
`74e23ebe86c8b476a3414b0ff11e561e126810b5358337cb87bc1e35e3a1d73d`.
It pins the analytic flat-surface skid duration, transition speed and distance,
roll duration and distance, total surface path, and rest termination.

Tests cover arbitrary plane orientation, exact static-friction feasibility,
slip direction and transition roots, rolling kinematics, rolling-resistance
stop, axial-spin conservation, moving-surface relative motion and work,
passivity, finite-edge localization, output/event ordering, time/event/step
limits, cancellation, integration refinement, exact regional precedence and
boundary splitting, state continuity, transition-ledger identity, randomized
piecewise analytic rolling speed, immediate-capture composition, partial-result
censorship, bounce counts, total-distance definitions, and legacy-adapter
refusal of non-rest complete results.

Protected CI, independent review, normal parent integration, and explicit
consumer/UI work remain release gates.
