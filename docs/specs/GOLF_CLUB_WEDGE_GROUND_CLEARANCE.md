# Golf Club Wedge Ground-Clearance Analysis

## Purpose and Claim Boundary

This shared, UI-independent module evaluates the swept geometric clearance of
a rigid parametric wedge against an immutable planar ground datum. It reports
the first named wedge feature to reach the plane, the event time and velocity,
ball/ground sequencing, and clearance margins. The Rate adapter consumes the
complete retained pose and twist history and passes only an actual ball-contact
time; a closest-approach sample remains a miss.

The calculation does not model turf stiffness, damping, friction, penetration,
divot formation, reaction forces, injury, or shot performance. It therefore
cannot establish that one bounce or grind is more forgiving than another.

## Frames, Units, and Geometry

The caller declares the inertial frame in `GroundPlane.frame_id`. Positions
are metres, time is seconds, translational velocity is metres per second, and
angular velocity is radians per second. A ground plane is defined by one point
and a unit normal. This supports level ground and sloped planar lies without
changing the solver.

Each retained pose maps the canonical wedge head frame into the declared
ground frame. The canonical profile is shared with the exact CAD generator so
the sole and leading-edge datums are not duplicated. Nine stable contact
candidates are evaluated:

- leading edge at heel, center, and toe;
- primary sole at heel, center, and toe; and
- trailing sole at heel, center, and toe.

Candidate names and ordering are deterministic. They represent auditable
engineering datums, not a continuous collision mesh or pressure patch.

## Swept Event Detection

Every retained simulation interval is subdivided into eight audit intervals.
Translation and twist are linearly interpolated. Orientation uses shortest-arc
unit-quaternion spherical interpolation (SLERP), including stable exact
half-turn handling. When a candidate changes from positive clearance to zero
or negative clearance, 48 bisection iterations refine the first crossing. The
event velocity at candidate point `P` is

`v_P = v_R + omega x (P - R)`.

Normal velocity is the dot product with the ground normal; the remaining
vector is the tangential velocity. SLERP follows the shortest orientation arc
at a constant angular rate within each retained interval. The retained twists
remain independent reported simulation data; interpolation does not overwrite
or infer them from adjacent poses.

## Reported Metrics

The immutable result includes:

- the complete swept minimum-clearance envelope;
- first-ground-contact time, feature, point, pose, and normal/tangential
  velocity;
- explicit `ball_first`, `ground_first`, `simultaneous`, `ball_only`,
  `ground_only_miss`, or `no_contact_miss` sequencing;
- leading-edge clearance at an actual ball contact;
- minimum pre-ball named-feature clearance;
- ground-contact lead/lag relative to ball contact;
- swept low-point time, point, and feature; and
- delivered central-sole bounce relative to the supplied ground normal.

The low point and minimum clearance apply to the named candidate set and swept
audit grid. They are not an analytic minimum over a continuous B-Rep.

## Verification

Tests cover the candidate taxonomy, bounce monotonicity, analytic between-frame
crossing, nonlevel heel/toe selection, all hit/miss sequence classes, event
velocity, clearance and bounce metrics, invalid retained states, common-frame
translation invariance, time-origin invariance, and timestep refinement for a
linear crossing. Exact half-turn and endpoint/translation interpolation are
also pinned. The public facade and Rate adapter are contract-tested.

The next fidelity layers are continuous sole-patch/B-Rep collision, turf
contact mechanics with documented material parameters, uncertainty propagation,
and React/PyQt visualization of the shared payload.
