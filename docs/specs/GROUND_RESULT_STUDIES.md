# Qualified Ground-Result Study Projection

## Scope

`ground-study-projection/v1` is the typed boundary between a canonical
`GroundSimulationResult` and later target, variation, dispersion, inverse, and
capability analyses. It projects evidence already present in the ground result;
it does not run physics, infer a rest state, qualify a material profile, or
replace a missing value.

The projection records the exact caller-supplied request digest and SHA-256
identity of the canonical source result, complete solver surface, ball radius,
frame, model identity, result-model calibration and provenance,
termination, optional evidence-bearing material profile and operating
condition, typed result warnings, profile qualification warnings, and typed
unavailable fields. A request/result identity mismatch, incompatible bound
profile, malformed target, first-contact endpoint off the plane, or complete
terminal endpoint off the plane fails before a study record is constructed. A
partial terminal endpoint must contact the plane or have strictly positive
clearance; penetration fails. The result digest identifies
the exact source bytes; it is provenance, not a digital signature. The current
`ground-result/v1` contract does not carry the request fingerprint that created
it, so this projection cannot attest that the source result was produced by the
recorded request digest. It checks the shared request ID, surface/frame,
calibration, and provenance and labels the digest as request context. Exact
request/result attribution requires a future result-side request fingerprint.

## Metric semantics

Numeric metrics exist only for `complete` and `partial` ground results. The
embedded `GroundSummary` is preserved exactly and keeps these quantities
distinct:

- carry distance from the launch origin to first contact;
- post-contact airborne distance accumulated between bounces;
- skid and pure-roll displacement;
- accumulated surface path;
- final launch-to-endpoint horizontal distance and final offline displacement;
- bounce count, excluding the initial contact event.

Total distance is the launch-to-endpoint displacement already calculated by
the ground result. It must never be reconstructed as carry plus roll because
bounce-air motion, skid, lateral displacement, and non-collinear paths make
that identity invalid. A censored run reports its final observed endpoint and
remains ineligible for objective solving. If a valid time/event-limited result
ends while the ball is airborne, its numeric final observation is retained but
its landing-area miss is absent with typed `endpoint_airborne` unavailability;
the system does not project that point onto the surface or infer a landing.

## Surface and target geometry

Target evaluation uses the sphere/surface contact point, not the ball centre.
For a plane with upward unit normal `n` and ball radius `r`, the contact point is
`q = c - r n`, with only tolerance-scale residual correction after the ball
centre is proven to lie one radius above the plane.

Ground studies accept only `landing_area` targets with a surface-circle or
surface-corridor tolerance. The target `ground_source` must equal the bound
surface identity and its centre must lie on that plane. Miss distance is
computed in an intrinsic orthonormal basis of the arbitrary plane, so tilted
surfaces are not silently flattened into the application's horizontal x-z
plane. A target miss is a valid objective value; it does not itself disqualify
an otherwise eligible result.

## Status and solver eligibility

| Evidence | Study status | Objective eligible |
| --- | --- | --- |
| Complete result, rest termination, qualified calibrated bound profile, measured/literature result calibration with positive confidence | `complete` | Yes |
| Partial surface or airborne terminal observation | `censored` | No |
| Failed result | `failed` | No |
| Required contact evidence unavailable | `unavailable` | No |
| Missing profile binding, unqualified profile, or illustrative-only profile | Result-derived status | No |

Eligibility is derived from evidence and encoded with canonical reasons; a
caller cannot forge it independently. The complete strict material profile is
revalidated against its evidence-derived qualification/calibration status and
the embedded solver surface; a detached digest or aggregate status is not
trusted. Result-model calibration is separately preserved: `estimated` or
`unvalidated` calibration kinds, and zero-confidence calibration, are never
objective eligible. Result provenance is retained verbatim for audit; the v1
contract does not attempt to certify a producer allowlist. Failed and
unavailable studies contain no numeric metrics or target
evaluations. Typed unavailable-field provenance is retained verbatim.

## Persistence and validation

The strict v1 JSON contract rejects missing or unknown fields, duplicate keys,
non-finite numbers, unsupported enum values, noncanonical eligibility, and
incoherent nested records. Calibration and provenance use the exact canonical
ground contract records. On every construction, endpoint geometry is checked
against carry/total/downrange/offline summary values, surface path is checked
against skid plus roll, sphere/plane contacts are re-derived, and intrinsic
target misses are recomputed from the embedded target and plane. All numeric
values, including spatial-target coordinates, cross the shared canonical
numeric boundary before the immutable projection is returned. Serialization is
therefore deterministic and an encode/decode round trip is value-equal even
when source calculations contain ordinary binary floating-point artifacts.

Result warnings preserve their typed code, severity, and message. Profile
qualification warnings remain a separate part of the embedded profile binding,
so neither evidence channel is collapsed into code-only strings.

## Explicit limitations

This foundation does not yet connect the projection to ensemble runners,
variation/dispersion plots, wind studies, inverse or capability optimizers,
PyQt6/React controls, compiled runtimes, or UpstreamDrift consumers. It does
not implement piecewise terrain or changing normals. Cancellation and
pre-contact/no-contact workflows require a higher-level adapter that produces
the existing typed unavailable result; this projection does not invent a new
solver terminal state.

The two digests must not be presented as an attested pair: the request digest
is exact caller context, while the result digest is exact source-result
identity. Compatibility checks reduce accidental mismatch but cannot replace a
result-carried request fingerprint.

The earlier direct `GroundSimulationResult` compatibility adapter cannot prove
material-profile qualification. It is deprecated and no longer exported from
the ground package facade; qualification-sensitive consumers must enter
through a self-validating study adapter.
