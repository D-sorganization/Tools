# Golf Club Shaft Profile and Reference Solver Specification

## Scope

This module is the UI-independent shaft contract for the shared Golf Club
Builder. It represents measured or otherwise traceable station data without
deriving composite stiffness from outside diameter alone. The same objects can
be used by a standalone builder, the Rate of Closure application, and
UpstreamDrift adapters.

The current reference implementation covers:

- station geometry, mass, directional bending stiffness, torsional stiffness,
  damping, and spine orientation;
- butt/tip trimming and hosel insertion depth;
- strict versioned JSON and self-contained CSV interchange;
- explicit immutable parameter scaling for what-if studies;
- static transverse and torsional tip response;
- undamped Euler-Bernoulli bending modes from a consistent-mass beam model; and
- integration into the canonical club-component mass-property contract.

## Coordinates and Units

All stored values use SI units. Station position starts at the raw butt and
increases toward the raw tip. Within a shaft-local component frame:

- local `z` is the shaft axis from cut butt toward cut tip;
- local `x` and `y` are transverse directions;
- `EI_x` resists bending about local `x`, producing transverse `y` motion;
- `EI_y` resists bending about local `y`, producing transverse `x` motion; and
- positive torsion follows the right-hand rule about the shaft axis.

The caller supplies `frame_id`; downstream placement into a club frame requires
an explicit `RigidTransform`. No global golf, launch, or rendering frame is
silently assumed.

## Profile Contract

Each `ShaftStation` stores:

| Field | Meaning |
| --- | --- |
| `position_m` | Raw-butt station coordinate |
| `outer_diameter_m`, `inner_diameter_m` | Annular section diameters |
| `linear_density_kg_m` | Measured mass per unit length |
| `ei_about_x_n_m2`, `ei_about_y_n_m2` | Engineering bending stiffnesses |
| `gj_n_m2` | Engineering torsional stiffness |
| `damping_ratio` | Dimensionless local damping ratio |
| `spine_angle_rad` | Unwrapped local stiffness-axis orientation |

Stations must be strictly ordered and include both raw endpoints. Geometry,
density, and stiffness must be positive; inner diameter must remain below outer
diameter; damping is constrained to `[0, 1)`; and every number must be finite.
Interpolation is linear and extrapolation must be explicitly rejected or
clamped.

The cut-length invariant is:

```text
cut length = raw length - butt trim - tip trim
```

Insertion depth reduces the exposed flexible span but does not remove the
inserted material from cut-shaft mass.

## Provenance and Derived Profiles

Every profile requires a source name, measurement method, and uncertainty note.
An optional source URI and data license travel with JSON and CSV exports.

`ShaftProfileScaling` independently scales linear density, both bending axes,
torsional stiffness, and damping. Scaling returns a new immutable profile and
marks its provenance as a deterministic what-if transformation. It does not
claim that the scaled profile was measured or that the factors correspond to a
specific layup or manufacturing process.

## Persistence

The JSON schema identifier is `golf_club.shaft_profile/1`. Serialization is
deterministic, rejects non-finite JSON extensions, duplicate keys, unknown
fields, missing required values, and unsupported schema versions.

The canonical CSV repeats the profile metadata on every station row. This makes
one CSV file self-contained and easy to inspect. The loader requires the exact
ordered SI headers and identical metadata on every row, preventing accidental
concatenation of stations from different shafts.

## Static Reference

`solve_cantilever_tip_response` treats the trimmed butt as fixed and the
exposed tip as free. Station-varying compliance is integrated with Gaussian
quadrature. For transverse tip force `F`, exposed coordinate `s`, span `L`, and
appropriate directional stiffness `EI(s)`:

```text
tip rotation   = F integral((L-s) / EI(s), ds)
tip deflection = F integral((L-s)^2 / EI(s), ds)
```

For shaft-axis tip torque `T`:

```text
tip twist = T integral(1 / GJ(s), ds)
```

This is a small-deflection reference. It intentionally omits shear deformation,
geometric nonlinearity, joint compliance, dynamic inertia, and contact.

## Modal Finite-Element Reference

`solve_shaft_bending_modes` assembles standard two-node Euler-Bernoulli beam
elements with consistent distributed mass. Station density and directional EI
are interpolated at each element midpoint. The trimmed butt is clamped and the
exposed tip is free. The generalized eigenproblem is transformed by Cholesky
factorization before a symmetric eigenvalue solve.

Returned frequencies are undamped. Stored damping is retained for future
calibrated transient models but is not silently converted into modal damping.
The solver reports its model identifier, mesh size, exposed length, and
assumptions. Regression tests compare both axes with the exact uniform
cantilever frequency:

```text
f_n = beta_n^2 / (2 pi L^2) sqrt(EI / linear_density)
```

The current reference excludes Timoshenko shear effects, rotary inertia,
composite laminate coupling, head/grip dynamic boundary conditions, large
deflection, aerodynamic loads, and swing/contact transients.

## Assembly Coupling

`shaft_component_mass_properties` integrates the complete cut shaft, including
the inserted portion, into `ComponentMassProperties`. The local center of mass
is placed on the shaft `z` axis. Transverse inertia includes the distributed
length term and local annular-section inertia; polar inertia uses the measured
linear density and station diameters. This component can be placed beside head
and grip components with explicit transforms and combined by the existing club
assembly solver.

The adapter does not infer elastic properties from mass geometry. It produces a
rigid mass-property representation for whole-club balance and inertia while the
shaft profile remains the authoritative flexible representation.

## Minimal Example

```python
from shared.python.golf_club import (
    ShaftModalSettings,
    shaft_component_mass_properties,
    shaft_profile_from_json,
    solve_shaft_bending_modes,
)

profile = shaft_profile_from_json(saved_profile_json)
shaft_component = shaft_component_mass_properties(profile)
modes = solve_shaft_bending_modes(
    profile,
    ShaftModalSettings(element_count=24, mode_count=3),
)
```

## Release Boundary

This foundation is suitable for persistence, deterministic engineering
calculations, and downstream UI adapters. It is not yet a validated prediction
of shaft behavior during a full golf swing. Production claims about a specific
commercial shaft require traceable station measurements, uncertainty, boundary
calibration, and comparison to independent dynamic test data.
