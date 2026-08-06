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

The Rate adapter does not equate its tracked reference point with the wedge
datum. It registers the requested canonical face contact point to the scenario
impact lever and shifts the retained linear twist to the new wedge datum with
`v_new = v_reference + omega x r_shift`. Ball timing and its limitations remain
those of the selected Rate contact policy; the adapter publishes that policy in
its snapshot rather than upgrading a point surrogate into face-mesh contact.

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
- sole-entry margin: the minimum primary/trailing named-sole clearance at ball
  contact;
- minimum pre-ball named-feature clearance;
- ground-contact lead/lag relative to ball contact;
- swept low-point time, point, and feature; and
- delivered central-sole bounce relative to the supplied ground normal;
- path-projected effective bounce in the vertical plane containing the
  reference-point horizontal velocity;
- reference-point AoA at ball contact; and
- bounce-utilization angle margin, defined as path-projected effective bounce
  plus signed reference AoA.

The utilization margin is positive when the delivered sole angle exceeds the
magnitude of a descending reference path under this geometric convention. It
is a kinematic angle margin, not a prediction that the sole will react, skid,
or prevent digging. Path-projected metrics return `None` when horizontal speed
is zero instead of inventing a travel direction.

The low point and minimum clearance apply to the named candidate set and swept
audit grid. They are not an analytic minimum over a continuous B-Rep.

## Shared Visualization Payload

`wedge_ground_clearance_to_json_dict` emits the versioned
`upstreamdrift.wedge-ground-clearance/v1` contract. The JSON-ready document
contains explicit units and frame ID, every envelope sample, the complete
first-contact event and transform, low-point geometry, sequence, ball time,
metrics, and limitations. Missing contacts and path-dependent metrics remain
JSON `null`; non-finite placeholders are never emitted. PyQt consumes the
Python snapshot directly. The standalone React application uses a pure
TypeScript parity port that emits the same format, frame, sequence, event,
metric, envelope, provenance, and limitation fields; presentation components
render that model payload and contain no contact calculations.

The PyQt swing view consumes the registered snapshot for wedge selections and
adds sequence, leading-edge/sole margins, delivered/effective bounce, first
ground contact, utilization margin, provenance, and model limitations to its
selectable engineering readout. Non-wedge selections do not show wedge claims.
The illustrative adapter preserves the selected Rate wedge loft, lie, and mass
but clearly labels its generic 10-degree mid-bounce sole as unmeasured.
Its rotatable 3-D scene uses that same cached snapshot to draw the sole envelope,
live clearance, ball-contact point, refined ground-contact point, and swept low
point; no duplicate scene-only contact calculation is introduced.
The React swing view applies the same wedge-only policy and presents an
accessible contact-order timeline, sequence state, and eight engineering
metrics. Its retained head orientations use shortest-arc SLERP during the
swept analysis rather than treating the reference point as an unrotated head.
The swing canvas draws the minimum-clearance sole envelope, signed live
clearance, ball-contact point, refined first-ground-contact point, and swept
low-point marker. The playback controls include a direct Jump to Impact action,
so the annotated impact state can be inspected without manual scrubbing.

## Verification

Tests cover the candidate taxonomy, bounce monotonicity, analytic between-frame
crossing, nonlevel heel/toe selection, all hit/miss sequence classes, event
velocity, clearance and bounce metrics, invalid retained states, common-frame
translation and rotation behavior, handedness mirroring, time-origin
invariance, and timestep refinement for a linear crossing. Exact half-turn and
endpoint/translation interpolation are also pinned. The public facade and Rate
adapter are contract-tested.
The versioned payload is checked for completeness, deterministic structure,
strict finite JSON serialization, and missing-contact preservation.
The web parity suite pins the representative 30 mph pitching-wedge event time,
leading-edge clearance, ground-contact lag, effective bounce, reference AoA,
and 481-sample envelope against Python results. React component tests cover
the wedge-only rendering boundary, sequence, metrics, provenance, and limits.

The next fidelity layers are continuous sole-patch/B-Rep collision, turf
contact mechanics with documented material parameters, and uncertainty
propagation.
