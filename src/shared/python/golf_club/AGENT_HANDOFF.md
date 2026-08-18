# AGENT_HANDOFF — shared golf_club

> Update this file in every implementation commit that changes this package.
> Last updated: 2026-08-18

## Stack and Integration Position

- Epic #4146 owns the reusable Golf Club Builder.
- #4147 / `feat/4147-club-builder-core` is the assembly-property foundation.
- #4148 / `feat/4148-shaft-profiles` adds measured shaft contracts and validated
  static/modal reference models.
- #4149 / `feat/4149-cad-families` is PR #4171, stacked on #4148. Its current
  exact head scope is a generic modern wedge, not the six-family completion.
- Rate of Closure and UpstreamDrift must consume this public facade through
  thin adapters after the provider stack lands; do not copy the calculations.

## Club Fitting Tester Epic (#4549)

This package is the **shared-first** home for the clubfitting epic's physics
and wires (design contract: `docs/specs/CLUB_FITTING_TESTER.md`).
`mesh_mass_properties.py` (C1, #4550) is the mesh mass-property authority —
watertightness, volume, centroid, and the full divergence-theorem inertia
tensor with analytic cube/box/sphere gates; `rate_of_closure.club.volumetrics`
delegates to it **lazily** (a module-scope import of this package reaches
SciPy through the turf chain and breaks the Morris UI import contract).
C2 (`shaft_delivery`) and C3 (`fitting_document`) land here next; tool-local
packages hold only UI/pipeline bindings.

## Heavy Hit Epic (#4562)

`impact_coupling.py` (H1/H3) quantifies hand/body influence at impact: a
ball-head-hands KV chain with **upper-bound semantics** (rigid-shaft worst
case) and the tau-squared decoupling law verified by gate; physiological
hands change driver ball speed by <1%. `swing_sim/model_interchange/` (H2)
imports golfer models from MJCF/URDF/.osim (MuJoCo, Drake, Pinocchio,
OpenSim) runtime-free and reduces a named hand selection to `GripBoundary`.
Contract: `docs/specs/HEAVY_HIT_COUPLING.md`.

## Current CAD and Export Contract

`wedge_parameters.py` and `wedge_serialization.py` own the immutable SI,
frame-explicit, provenance-bearing `golf_club.wedge_parameters/1` input.
`wedge_cad.py` lazily invokes the optional pinned build123d/OpenCascade kernel
and returns one exact solid plus independently recovered datum measurements.

`wedge_export.py` owns deterministic `golf_club.wedge_export/2` exports:

- STEP and native BREP are reopened with build123d and must recover one valid
  solid with source-bounded volume and axis-aligned bounds.
- Binary STL is parsed by `stl_validation.py`; no renderer, mesh repair, or
  optional trimesh dependency is trusted for release validation.
- Each triangle must be finite and nondegenerate, stored normals must agree
  with winding, each undirected edge must have two opposite uses, all faces
  must form one component, and signed volume must prove outward orientation.
- Bounds and volume must remain within limits derived from the requested chord
  tolerance. Any failed check aborts before manifest publication.
- The manifest records the canonical parameter SHA-256 plus each artifact's
  SHA-256, byte size, reader, checks, measured values, and limits.

`golf_club.wedge_export/2` supersedes `/1`. There is no manifest reader or
silent migration path: retain historical `/1` JSON as unvalidated archive
evidence, and regenerate a `/2` export from canonical wedge-parameter JSON when
current validation evidence is required. Never infer that a `/1` artifact
passed checks which did not exist in its schema.

These checks establish deterministic file and topology evidence. They do not
qualify minimum wall/feature size, machining, additive processing, materials,
metrology, turf interaction, impact performance, or commercial equivalence.

## Focused Verification

From the repository root with the branch's `.venv`:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest `
  tests\shared\python\golf_club -q -p no:xdist -o addopts=''
ruff check src\shared\python\golf_club tests\shared\python\golf_club
ruff format --check src\shared\python\golf_club tests\shared\python\golf_club
mypy src\shared\python\golf_club --ignore-missing-imports
```

The environment currently lacks some pytest plugins declared by root config;
the focused command therefore disables plugin autoload and clears `addopts`.
Unknown-config warnings are environment evidence, not test failures.

Latest local evidence on 2026-08-09:

- all `tests/shared/python/golf_club`: 121 passed;
- Ruff check and format: 31 package/test files passed;
- mypy: 17 package files passed with no issues;
- Black check: seven changed Python/test files unchanged (the global Python
  3.13 Black executable emitted its known target-parser warning);
- 400-line package module/file budgets, documentation governance, changed-test
  assertion policy, and `git diff --check`: passed.

## Residual #4149 / #4146 Scope

- Implement versioned Driver/Wood, Hybrid, Iron, Blade Putter, and Mallet
  Putter family graphs; expand the wedge beyond its central foundation.
- Add editable sections, camber/relief/grinds, cavity/back variants, scorelines,
  transition radii, wall thickness, and weight ports with minimum-feature and
  self-intersection validation.
- Derive CG and full inertia from the exact solid and couple them to assembly
  properties.
- Add bound-constrained multistart shape optimization, infeasibility/tradeoff
  reporting, professional preview contracts, golden/property tests, and visual
  engineering QA.
- Complete additional C4 formats (3MF, OBJ/PLY/glTF/GLB, DXF/SVG) only with
  qualified readers and truthful round-trip evidence. Do not claim STEP beyond
  the build123d/OpenCascade path already validated here.
