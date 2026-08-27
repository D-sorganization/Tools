# AGENT_HANDOFF — shared golf_club

> Update this file in every implementation commit that changes this package.
> Last updated: 2026-08-27

## Stack and Integration Position

- Epic #4146 owns the reusable Golf Club Builder.
- #4147 / `feat/4147-club-builder-core` is the assembly-property foundation.
- #4148 / `feat/4148-shaft-profiles` adds measured shaft contracts and validated
  static/modal reference models.
- #4149 / `feat/4149-cad-families` is PR #4171, stacked on #4148. Its current
  exact head scope is a generic modern wedge, not the six-family completion.
- Rate of Closure and UpstreamDrift must consume this public facade through
  thin adapters after the provider stack lands; do not copy the calculations.

## Club Fitting Tester Epic (#4549) — COMPLETED (#4557, #4577)

This package is the **shared-first** home for the clubfitting epic's physics and
wires (contract: `docs/specs/CLUB_FITTING_TESTER.md`). **C1–C7 all merged in
#4557 and #4577** (including PyQt6 Club Tester tab and React panel).

| Module                    | Owns                                                                                                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mesh_mass_properties.py` | C1 — watertightness, volume, centroid, full divergence-theorem inertia tensor, with analytic cube/box/sphere gates. `rate_of_closure.club.volumetrics` delegates here.                 |
| `shaft_delivery.py`       | C2 — `quasi_static_centrifugal_alignment/1`: tension stiffening `1 + N/P_cr`, alignment restoring `k_θ·θ = F_c·(d − cg_drop·θ)`; **refuses beyond β = 0.8** rather than extrapolating. |
| `fitting_document.py`     | C3 — `golf_club.fitting_document/1`, the OEM interchange wire.                                                                                                                         |
| `fitting_engine.py`       | C4 — `CounterfactualSpec`, `compare_counterfactuals`, `golf_club.fitting_report/1`.                                                                                                    |

C5's delivery-trajectory interchange and its Drake/MuJoCo/OpenSim adapters live
next door in `shared/python/swing_sim/delivery_interchange/`.

**Import trap, load-bearing:** `rate_of_closure.club.volumetrics` delegates to
`mesh_mass_properties` **lazily, inside the function body**. A module-scope
import of this package executes `golf_club/__init__`, whose eager surface reaches
SciPy through the turf chain and breaks the Morris UI import contract. Keep it
lazy, and do not add eager imports to `__init__` that reach `swing_sim.variation`.

## Heavy Hit Epic (#4562) — COMPLETED (#4568, #4577)

`impact_coupling.py` (H1/H3) quantifies hand/body influence at impact: a
ball–head–hands Kelvin-Voigt chain integrated in the **body frame** (fixed grip
anchor, ball approaches — the anchor does no work, so energy accounting is
exact). Physiological hands (3 kg, 5e4 N/m) change driver ball speed by **<1%**;
the model always reports the **rigid-shaft upper bound** alongside, because any
lumped `k_s` only approximates contact-timescale impedance. H4 GUI panels
landed in PR #4577. Contract: `docs/specs/HEAVY_HIT_COUPLING.md`.

`swing_sim/model_interchange/` (H2) imports golfer models from MJCF, URDF and
`.osim` — MuJoCo, Drake, Pinocchio, OpenSim — by **runtime-free XML parsing**
(no engine imports), and reduces an **explicitly named** hand-body selection to
`GripBoundary`; nothing is guessed from body names. URDF carries no joint
stiffness, so an explicit override is the sanctioned supply path there.

**Two physics facts learned by probe — do not "simplify" them back:**

1. The τ² decoupling law holds at _finite_ shaft stiffness (4× contact time →
   16.0× influence, measured), but a **rigid** shaft's coupling is quasi-static
   added mass and therefore τ-independent. The first gate draft assumed the
   τ-law for the rigid case and was wrong.
2. Kelvin-Voigt restitution is **reduced-mass dependent** (ζ = c/2√(kμ)), so
   welding the head to a large mass legitimately makes it bouncier than
   `(1+e_free)·v₀`. Cross-case ceilings from a fixed `e` are invalid — bound
   with the elastic `2·v₀` instead.

Bandit flags stdlib `ElementTree` (B405/B314) in the parsers; the repo's
convention is `# nosec` with a written justification, not `defusedxml`.

## Putting Epic #4800 — P3 putter head (this package's slice)

`putter_head.py` owns `golf_club.putter_head/1`: PutterSpec **v2** = the P1
v1 spec + CG + full inertia tensor + provenance, built from an STL through
`mesh_mass_properties` (C1 is the only mesh pipeline;
`stl_validation.read_binary_stl` is the promoted public reader) or from a
club-library putter — resolving the `PutterSpec` reconciliation: a library
head carries **no tensor** and strikes bit-identically to P1's
`head_moi_kg_m2=None` default (exact-equality gate). Quasi-static twist
`theta = J r tau_c/(2I)` per axis (toe→I_yy opens the face, high→I_zz adds
loft, tau_c = 0.5 ms); `head_moi_for_strike` feeds P1's explicit hook. Head
frame: x = target line, y = up, z = toe. TS twins: `putterHead.ts` +
`putterHeadWire.ts` (P2 wire-split precedent) + `volumetrics.meshInertia`.

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

The environment lacks some root-config pytest plugins, so the focused command
disables plugin autoload and clears `addopts`; unknown-config warnings are
environment evidence, not failures.

Latest local evidence, 2026-08-27 (after the P3 putter-head merge):
`tests/shared/python/golf_club` + `swing_sim/putting/tests`: **333 passed,
2 skipped** (the pre-existing `test_turf_variation` seeded-plan failure is a
local duplicate-module environment artifact, present on clean main). The C1/H1
physics gates plus P3's box-inertia, twist, and fallback-equality gates are
inside that run; ruff and the CI-faithful mypy batch (numpy < 2.4 — numpy 2.4+
stubs crash mypy 1.13's cache serializer) are clean.

## Residual #4149 / #4146 Scope

- Implement versioned Driver/Wood, Hybrid, Iron, and Blade/Mallet Putter
  family graphs; expand the wedge beyond its central foundation.
- Add editable sections, camber/relief/grinds, cavity/back variants,
  scorelines, wall thickness, and weight ports with feature validation.
- Derive CG/full inertia from the exact solid and couple them to assemblies.
- Add bound-constrained multistart shape optimization, infeasibility/tradeoff
  reporting, preview contracts, golden/property tests, and visual QA.
- Complete additional C4 formats (3MF, OBJ/PLY/glTF/GLB, DXF/SVG) only with
  qualified readers and truthful round-trip evidence; claim STEP only via the
  validated build123d/OpenCascade path.
