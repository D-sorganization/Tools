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

### Regional material plan wire boundary

The additive `ground-regional-material-plan-request/v1` contract carries one
finite base domain, its request-bound `GroundSurfaceProfile`, tangent axis and
origin, one or more finite material overlays, provenance, SI units, the exact
`coplanar_static_material_overlays` geometry model, and both fixed v1
limitations. Region IDs, precedence values, and base/overlay surface IDs are
unique. The region count is bounded at 4,096 and JSON documents at 1 MiB.
Every numeric coordinate is finite, every interval is nonempty and lies inside
the finite base interval, and every surface is stationary and shares the base
frame, height, and normal. Unknown, missing, or extra keys; duplicate JSON
keys; unsupported versions/units/limitations; and unbound material evidence
fail before a resolver exists.

`ground-regional-material-plan-result/v1` is validation evidence, not a ground
trajectory or a second physics model. It embeds the exact request, its
canonical SHA-256, a descending-precedence/ascending-ID copy of the request's
regions, producer provenance bound to that digest, and the same explicit
limitations. Both Python and TypeScript reject reordered or changed region or
surface records instead of accepting fabricated resolved materials. Python's
`regional_plan_to_surface_resolver` constructs the existing qualified
`SurfaceResolver` without changing its physics. TypeScript validates and
serializes the plan but does not claim to run regional dynamics.

These new schemas are deliberately separate from
`flight-to-ground-request/v1` and `flight-to-ground-result/v1`; neither frozen
contract is silently widened. The strict ground result continues to carry the
legal transition event and qualified-domain warning.

### Regional execution result boundary

`execute_regional_ground` accepts one exact ground request, settled bounce
prefix, regional plan request, and `RegionalGroundExecutionOptions`. Options
contain only bounded `SkidRollSettings`, an optional cancellation check, and
executor source revision; there is no resolver field. The executor requires
`plan.base_surface == request.surface`, validates request/prefix identities and
fingerprint, constructs the resolver solely from the plan, and calls the
existing `simulate_skid_roll` and `compose_ground_result` authorities.

The additive `ground-regional-execution-result/v1` envelope carries canonical
SHA-256 identities for both inputs, the ground request/surface and plan IDs,
the exact embedded regional plan and unchanged plan source provenance,
executor provenance bound to the joint input digest, exact model ID/version,
and the ordered internal transition ledger with from/to region and surface
IDs. The plan digest is recomputed from the embedded plan; plan ID, base
surface, and provenance must agree. Executor producer/version are the fixed v1
authority while source revision remains variable evidence. Each ledger row
must match one `SURFACE_TRANSITION` event by sequence, time, and position, and
its coordinate/from/to identities must describe a real boundary crossing in
the embedded plan. Empty ledgers still retain complete plan and executor
provenance.

Complete and partial outcomes embed an unchanged
`flight-to-ground-result/v1`. Cancellation, step/transition bounds,
unsupported surfaces, numerical failures, and composition failures cannot be
encoded honestly by the frozen base result; the envelope therefore uses typed
cancelled/failed status, reason, and `ground_result: null`. Python is the only
physics executor. TypeScript strictly parses/serializes the envelope and shared
golden fixture without implementing dynamics. Documents are capped at 8 MiB,
reject duplicate/extra/malformed data, use the same canonical safe-number,
integral JSON-number, nonblank-text, and vector policy in both runtimes, reject
same-surface transitions, and declare the same coplanar/static limitations as
the plan. The golden representable/cancelled/failed envelopes are produced by
`execute_regional_ground`; an adversarial fixture pins cross-runtime rejection,
and frozen base-result v1 compatibility remains a separate test. Null-result
cancelled/failed envelopes require an empty transition ledger because no
embedded result exists to substantiate transition evidence.

### Flight-through-regional composition boundary

The shared Python flight facade exposes `execute_regional_ground_from_flight`
as the single composition from a qualified flight result to this regional
executor. It validates exact contract types, bounce capture, and equality of
the plan base surface with the canonically derived launch-relative transfer
surface before bounce physics. One regional options record supplies both the
bounce cancellation check and the skid/roll settings/cancellation boundary.

The bounded in-memory `flight-regional-ground-pipeline/v1` result binds the
exact bounce pair, ground-request digest, repeated-bounce execution-input
digest, exact regional plan and canonical plan digest, and optional existing
regional envelope. The optional envelope must exist exactly when the bounce
prefix terminates `SETTLED_TO_SKID`; every other bounce reason forbids regional
evidence and is retained without translation. Regional cancellation and
internal failure remain represented by the existing regional envelope after
that phase begins.

This composition is not a new wire format. Existing strict bounded bounce,
plan, regional-execution, and ground-result serializers remain authoritative,
so there is no migration in this UI-neutral slice. #4271 physics qualification,
#4273 study integration, #4267 completion, clients, persistence, playback,
compiled parity, and downstream release remain open.

### Matched editor/readback boundary

The standalone PyQt6 and React applications register a `Ground Surfaces`
primary module that presents one base interval and one to eight regional
overlay rows. The editor exposes every `GroundSurfaceProfile` material field,
stable request/region/surface IDs, unique precedence, metre interval bounds,
and source revision. Geometry remains visibly fixed to the target frame,
zero-height upward-normal plane, downrange tangent axis, and zero surface
velocity. The interface labels metre, pascal, density, coefficient, and
dimensionless-fraction quantities explicitly.

Both clients load the same visibly **illustrative, unvalidated** discovery
values. They hash the actual editor draft, including its calibration
qualification, into provenance before delegating to the authoritative
`ground-regional-material-plan-request/v1` validator. The presentation layer
does not reimplement material or regional physics and does not soften unique
identity, interval, finite-number, material-range, or geometry contracts.
Errors remain associated with the editable draft; success exposes canonical
schema, unit, provenance, and request readback.

The regional v1 wire schema has no calibration record, so the editor does not
invent one or widen that schema. `unvalidated` is explicit presentation/source
qualification and is included in the source digest. PyQt6 native Open/Save As
and React browser import/download persist only the canonical regional request;
workspace model-input persistence remains a separate contract. Import is
transactional and accepts only the editor producer/provider v1, fixed qualified
axis/geometry, and editor row capacity. An unchanged import retains the exact
request and provenance; editing rebinds the draft digest. Browser downloads
cannot promise a native path, atomic replacement, or recent-file access.
Wire numbers are bounded to the shared cross-runtime safe range. Native
precedence entry preserves every nonnegative integer through
9,007,199,254,740,991 exactly, so a qualified import cannot be silently narrowed
before validation or Save As.
Native import reads one binary handle with a one-byte overflow sentinel before
strict UTF-8 decoding, so mutable files cannot bypass the 1 MiB allocation cap
between a metadata check and content parsing.
Neither client claims execution, result playback, or measured-course
calibration. Those capabilities require separate contracts and acceptance
evidence.

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

The shared regional-plan fixture
`src/rate_of_closure/web/src/model/__fixtures__/ground_regional_plan_golden_v1.json`
pins Python/TypeScript canonical request SHA-256
`a890b6fd544d73114ec5d0cd042f87aa2358d01ca85543a8c4d71ef2cb18cab1`
and result SHA-256
`8d9bc2f53897da241580f7b5fdaff7c6614077bed8a486cc6d7619d02b0e3e55`.
