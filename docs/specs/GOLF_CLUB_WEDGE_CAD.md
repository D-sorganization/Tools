# Golf Club Modern-Wedge CAD Foundation

## Purpose and Scope

This module is the first exact-solid head family for the shared Golf Club
Builder. It replaces the Rate of Closure application's wedge-shaped
superellipse envelope with a reusable, provenance-bearing engineering contract
and an OpenCascade B-Rep.

The current slice provides a central modern-wedge body, smooth rear muscle,
rounded leading edge, hollow hosel, strict persistence, measured solid metrics,
and deterministic CAD/mesh export. It establishes the canonical seam for later
camber, relief, grind, cavity, scoreline, weight-port, and optimization work.

## Kernel Spike and Packaging Decision

The repository already pins `build123d==0.11.1` in its optional `cad` extra and
uses build123d in Vessel Drafter. The local spike verified:

- exact loft, cylinder, boolean, fillet, and B-Rep validity operations;
- STEP, BREP, and tolerance-controlled STL export;
- STEP re-import with preserved validity and volume; and
- Windows CPython 3.11 support through `cadquery-ocp-novtk` 7.9.3.1.1.

Installed package metadata declares Apache-2.0 for build123d,
cadquery-ocp-novtk, and cadquery-ocp-proxy. The dependency remains optional so
importing the shared golf-club contracts does not require OpenCascade. CAD
functions import build123d lazily at execution boundaries.

## Frames and Datums

The public contract uses metres, kilograms, and degrees. OpenCascade model
coordinates use millimetres only inside the CAD adapter.

The wedge head frame is right-handed:

- `+x`: face-forward / target direction;
- `+y`: up, normal to the ground plane;
- `+z`: toward the toe for a right-handed head; and
- ground plane: `y = 0`.

The leading-edge center is the face/sole reference. Loft is the angle of the
outward face normal above `+x`. Bounce is the central sole tangent angle above
ground from leading edge toward trailing edge. Lie is the shaft-axis elevation
above the ground plane. Left-handed geometry mirrors heel and hosel placement
through the center plane without changing signed loft, bounce, or lie values.

Face height is measured along the face plane. Face length is the heel-to-toe
datum span. The physical leading-edge radius is an exact B-Rep fillet. The hosel
is an annular solid with a validated minimum wall and an axis set by lie and
handedness.

## Parameter Contract

`WedgeHeadParameters` validates a topology-safe domain for:

- head identifier and handedness;
- loft, lie, and central bounce;
- face length and face-plane height;
- sole width, topline thickness, and leading-edge radius;
- rear-curve depth and face progression;
- hosel outer diameter, bore, length, and minimum wall;
- material density and target mass; and
- required geometry provenance and uncertainty/claim boundary.

Every numeric field must be finite and within a documented supported interval.
The intervals are implementation topology limits, not rules limits and not
claims that every combination represents a good golf club.

The low-, mid-, and high-bounce presets are clearly labeled generic
illustrations. They use 6, 10, and 14 degrees of central bounce respectively.
They do not reproduce or imply dimensional identity with a commercial head.

## Solid Construction

The body begins with identical heel and toe profile wires. Each profile uses:

- a planar lofted face;
- an explicit topline thickness;
- a cubic Bezier rear muscle controlled by rear depth;
- a planar central bounce surface; and
- an exact leading-edge fillet.

The body is fused to a hollow cylindrical hosel whose axis follows the requested
lie and handedness. The operation must return exactly one valid solid; failed
booleans or invalid topology raise an actionable error rather than returning a
partial mesh.

## Independent B-Rep Measurements

The generator does not simply echo requested angles as measured results.
Planar B-Rep faces are selected by their expected datum normals, and the
following values are recovered from the final fused solid:

- face-normal loft;
- sole-normal bounce;
- hosel-cap normal lie;
- planar face heel-to-toe span;
- exact solid volume;
- density-derived mass; and
- target-mass residual.

The reference tests require the three requested angles to recover to numerical
precision. Presets generate one valid solid with finite positive volume and
mass. At least one B-spline surface is required so a faceted polygonal rear
cannot silently replace the intended smooth muscle.

## Persistence and Export

The parameter schema identifier is `golf_club.wedge_parameters/1`. JSON is
deterministic and rejects duplicate keys, unknown fields, unsupported versions,
non-finite values, and invalid nested provenance.

`export_wedge_artifacts` supports:

- STEP for neutral exact-solid interchange;
- native BREP for OpenCascade-fidelity interchange; and
- binary STL with explicit linear and angular tessellation tolerances.

Each export set includes `golf_club.wedge_export/2` JSON recording SI units,
kernel/model units, complete requested parameters and provenance, a SHA-256 of
the canonical parameter document, measured values, tessellation tolerances,
artifact filenames, byte sizes, artifact SHA-256 digests, and post-export
validation evidence. STEP timestamps are fixed so identical geometry and
requests produce byte-identical artifacts and manifests.

STEP and BREP are reopened with their build123d/OpenCascade readers and must
recover one valid solid whose volume and axis-aligned bounds agree with the
source B-Rep. Binary STL is parsed independently without repair. Every triangle
must be finite and nondegenerate, its stored normal must agree with its winding,
every undirected edge must have exactly two oppositely directed uses, all faces
must form one connected component, and signed volume must indicate outward
orientation. Mesh volume and bounds must remain within explicit limits derived
from the requested chord tolerance. A corrupt or out-of-tolerance artifact
aborts the export before its manifest is written.

### Export-manifest migration

`golf_club.wedge_export/2` supersedes `/1`; current producers emit only `/2`.
The repository has no manifest-import API, so this is an output-contract
migration rather than a silent in-memory upgrade. Historical `/1` JSON remains
readable as archival JSON, but it has no source/artifact digests or complete
post-export validation evidence and must not be relabeled as a validated `/2`
export. A downstream archive that needs current evidence must retain its
original `/1` record and regenerate artifacts plus a `/2` manifest from the
canonical `golf_club.wedge_parameters/1` project. Consumers that index both
versions must branch on the exact `format` field and treat `/1` validation
status as unavailable, never as passing.

## Verification and Visual QA

Automated coverage includes:

- parameter and hosel-wall contracts;
- preset identity and provenance;
- versioned serialization and corruption rejection;
- solid validity and single-solid topology;
- recovered loft, lie, bounce, face span, volume, mass, and residual;
- curved rear-surface presence;
- safe export paths and tolerances;
- deterministic STEP/BREP/STL and manifests;
- STEP and BREP round-trip validity, solid count, bounds, and volume; and
- independent binary-STL triangle, watertightness, manifoldness, winding,
  connectedness, outward-orientation, bounds, and volume validation.

Rendered engineering QA must accompany geometry changes because watertightness
alone does not establish a professional or recognizable head shape.

## Current Release Boundary

This foundation is intentionally not the completed wedge family. Before the
parent CAD and wedge issues can close, the exact solid must add independently
editable:

- front-to-back and heel-to-toe camber;
- heel, toe, and trailing-edge relief and blended grind patches;
- blade/cavity/specialty back variants;
- scorelines, transition radii, and weight ports;
- requested-versus-measured sole width and effective-bounce datums;
- solid-derived CG and full inertia coupled to the club assembly;
- constrained mass/CG/inertia and silhouette optimization; and
- shared PyQt/React editing, preview, section, and dimension adapters.

No current output predicts turf reaction, impact performance, forgiveness, or
commercial-head equivalence.

The STL validator establishes topological and geometric consistency with the
generated B-Rep; it is not a print-process qualification, metrology report,
material certification, minimum-wall analysis, or guarantee that a slicer or
machine will accept the part. ASCII STL and automatic mesh repair are not
supported by this controlled export path.
