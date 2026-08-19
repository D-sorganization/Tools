# Club Fitting Tester — Design Contract

Status: **active** · Epic: club-tester (see the GitHub epic issue) · Owner: `rate_of_closure`
Related contracts: `FLIGHT_TO_GROUND_CONTRACT.md`, `GOLF_CLUB_WEDGE_CAD.md`,
`CAPABILITY_OPTIMIZATION.md`

## 1. Purpose

Extend the impact-zone model into a professional clubfitting instrument:

1. **Twist counterfactuals** — how head mass properties (mass, CG, inertia tensor)
   change the face-closure response and gear-effect spin for off-center strikes.
2. **Shaft forward dynamics** — how bending/torsional stiffness distributions change
   the *delivered* club state (dynamic loft, face closure, droop, speed) for the same
   swing input.
3. **CAD-driven heads** — mass properties derived from watertight meshes so an OEM
   head design drives the impact model directly.
4. **Biomechanics delivery sources** — a neutral delivery-trajectory contract with
   adapters for Drake, MuJoCo, and OpenSim exports, so full-body models feed the
   same impact + flight pipeline.
5. **Fitting metrics** — per-parameter sensitivities and counterfactual deltas on
   ball-flight outcomes, reported in a versioned, OEM-consumable document.

OEM manufacturers are an explicit audience: every boundary in this design is a
versioned, SI-unit, provenance-carrying JSON wire with a documented schema, and
every physics claim is either literature-anchored or verified against an analytic
limit in tests.

## 2. What already exists (measured 2026-08-18, not assumed)

| Capability | Where | State |
| --- | --- | --- |
| Rigid club assembly, mass-property composition, provenance binding | `golf_club/assembly.py`, `rate_of_closure/club/assembly_binding.py` | shipped |
| Measured shaft profiles (per-station EI/GJ/density/spine) | `golf_club/shaft_profile.py` | shipped |
| Shaft statics (cantilever tip response) and modal FE (bending modes) | `golf_club/shaft_statics.py`, `shaft_dynamics.py` | shipped |
| Mesh volume + centroid (divergence theorem), watertightness | `rate_of_closure/club/volumetrics.py` | shipped |
| CAD/STL validation, parametric heads, wedge geometry | `golf_club/cad_validation.py`, `stl_validation.py`, `club/parametric_head.py`, `wedge_*` | shipped |
| Impact models (rigid COR, spring-damper, finite-time), gear effect, D-plane | `swing_sim/impact/` | shipped |
| Head-CG tensor passthrough into the impact solver (`clubhead_moi_tensor`) | `impact/solver.py`, `club/simulation_adapter.py` | shipped |
| Delivery front-end (launch-monitor numbers → impact inputs) | `impact/delivery.py` | shipped |
| Monte Carlo + Morris sensitivity, registry with extension seam | `swing_sim/variation/` | shipped |
| Ball flight, ground, twist (closure-rate) derivation, both GUIs | `swing_sim/flight`, `ground`, `rate_of_closure/derivation*.py` | shipped |

**Interpretation discipline:** `adapt_club_assembly_for_impact`'s "never use assembly
inertia" refers to *whole-club* inertia; the validated **head-CG** tensor is rotated
into the app frame and consumed when a binding + attitude exist. The epic builds on
that path; it does not replace it.

## 3. Gaps this epic closes

| # | Gap | Child |
| --- | --- | --- |
| G1 | No full **inertia tensor from a watertight mesh** (volumetrics stops at volume + centroid); CAD heads cannot drive the MOI path | C1 |
| G2 | No **forward coupling** from shaft stiffness to delivered state — statics and modal frequencies exist, but nothing produces dynamic-loft/closure/droop/speed deltas from a swing input | C2 |
| G3 | No single **OEM interchange document** bundling assembly + shaft profile + face geometry + mesh reference + provenance in one versioned wire | C3 |
| G4 | No **counterfactual engine**: apply club-parameter deltas, rerun delivery→impact→flight, and report outcome deltas + sensitivities | C4 |
| G5 | No **delivery-trajectory contract** or Drake/MuJoCo/OpenSim adapters | C5 |
| G6 | No **Club Tester surface** in either GUI | C6/C7 |

## 4. Architecture

```
L0 club data      ClubAssembly · ShaftProfile · STL mesh · ClubFittingDocument (C3)
L1 derived        volumetrics (+ mesh inertia tensor, C1) · composite inertia · modal shaft
L2 delivery       DeliveryParameters ← launch monitor | DeliveryTrajectory (C5) | parametric
                  + ShaftDeliveryDeltas (C2): stiffness → dynamic loft/closure/droop/speed
L3 impact/flight  existing solvers, head-CG tensor passthrough, gear effect, D-plane, flight, ground
L4 fitting        counterfactual engine (C4) · Morris sensitivities (reused) · FittingReport wire
L5 surfaces       PyQt6 Club Tester tab (C6) · React parity (C7) · JSON reports
```

Data flows one way, L0→L5. Every L2–L4 artifact carries the identity of the club
document and delivery source that produced it (`assembly_binding` pattern), so a
fitting report is traceable to exact inputs — the property OEM workflows need.

### 4.1 C1 — Mesh inertia tensor (`shared/python/golf_club/mesh_mass_properties.py`)

Full inertia tensor about the mesh CG for a **watertight, consistently wound**
triangle mesh via the divergence theorem (canonical polyhedral mass-property
integrals). Uniform density supplied by caller or solved from a target head mass.
Gates: analytic sphere/box/cylinder fixtures to ≤1e-6 relative error at fine
tessellation; translation/rotation covariance checks; degenerate meshes fail closed
through the existing `is_watertight`/orientation validation.

### 4.2 C2 — Shaft delivery deltas (`golf_club/shaft_delivery.py`)

Quasi-static + first-mode transient model of delivered-state change from shaft
compliance, anchored to Milne & Davis (1992) and MacKenzie & Sprigings (2009):
lead/lag deflection → dynamic-loft delta, toe-down (droop) → lie/face deltas,
torsional windup (GJ) → face-closure delta, kick timing → speed scale. Inputs: a
`GripKinematics` record (angular velocity/acceleration profile near impact) and a
`ShaftProfile`. Output: `ShaftDeliveryDeltas` applied to `DeliveryParameters`.
Gates: rigid-shaft limit → exactly zero deltas; static limit reproduces
`solve_cantilever_tip_response`; first natural frequency consistent with
`solve_shaft_bending_modes`; all magnitudes within published driver ranges
(≈1–5° dynamic loft add, ≈0.5–3° droop) for representative profiles.

### 4.3 C3 — Club fitting document (`golf_club/fitting_document.py`)

One versioned wire (`golf-club/fitting-document/v1`) bundling: assembly, shaft
profile, face geometry (loft/lie/bulge/roll), optional mesh reference
(path + sha256 + declared density or target mass), grip, and provenance
(source kind: `oem_export | measured | parametric | cad_derived`, tool, date).
Canonical JSON via `canonical_numeric_json`; parse/serialize round-trip is
byte-exact; unknown fields rejected (fail closed). A human-readable schema
document (`docs/specs/CLUB_FITTING_DOCUMENT.md`) is the OEM-facing reference.

### 4.4 C4 — Counterfactual engine (`rate_of_closure/clubfitting/`)

`CounterfactualSpec`: bounded deltas over {head mass, CG offset (3), inertia
diagonal scale, loft, lie, shaft EI scale, GJ scale, length, swingweight-preserving
mass redistribution}. Engine: baseline document + delivery source → apply delta →
L2 (C2 deltas) → L3 → outcome record {launch, spin, closure-rate trace summary,
carry, lateral, dispersion under the existing variation engine}. Sensitivities:
Morris reuse through the registry **extension seam** (the
`register_ground_variation_variables` pattern — never edits the shared registry
directly). Output: `FittingReport` wire (`golf-club/fitting-report/v1`) with
baseline, per-counterfactual deltas, elementary-effect rankings, and full input
identities. Determinism: fixed seeds; the report is reproducible byte-for-byte.

### 4.5 C5 — Delivery trajectory interchange (`swing_sim/delivery_interchange/`)

`DeliveryTrajectory`: monotone time-stamped samples of the **grip frame** (butt)
pose + linear/angular velocity in a declared world frame, SI, with frame
conventions documented against `swing_sim/conventions`. Wire:
`swing-sim/delivery-trajectory/v1`. Adapters (format-level, engine runtimes NOT
required — each parses that engine's standard export and is fixture-tested):

- **OpenSim**: `.sto`/`.mot` state tables (header + tab-separated columns).
- **MuJoCo**: JSON export of site pose + `sensordata` velocity streams
  (documented minimal export snippet included for model owners).
- **Drake**: JSON serialization of a body pose/spatial-velocity trajectory
  (documented `MultibodyPlant` export snippet included).

From a trajectory: derive `DeliveryParameters` at the impact sample and the
`GripKinematics` window that drives C2 — one seam, three engines, and any OEM
motion source that can write the neutral wire.

### 4.6 C6/C7 — Surfaces

PyQt6 "Club Tester" tab: baseline club from the existing library, counterfactual
controls, run, side-by-side baseline/counterfactual metrics (closure-rate trace,
launch, spin, carry, dispersion) with the capability/evidence presentation patterns
the other tabs use. React parity starts with the model layer (fitting document +
report parsing, golden fixtures shared with Python under
`web/src/model/__fixtures__/`), then the panel. GUI slices follow the repo's
500-LOC and accessibility-manifest gates.

## 4.7 Placement policy and cross-repo sharing (binding)

**Shared-first.** Physics, wire formats, and CAD math land in the shared layer
(`src/shared/python/golf_club/`, `src/shared/python/swing_sim/`) — never
tool-locally — so UpstreamDrift reaches one implementation through
`vendor/ud-tools`. Tool-local packages (`rate_of_closure/*`) hold only what
binds shared physics to that tool's UI/simulation pipeline. Where a tool-local
module predates this rule (e.g. `club/volumetrics.py`), the shared module
becomes the authority and the tool-local module delegates with its public API
unchanged.

Cross-repo children:

- **C8 — UpstreamDrift**: bump `vendor/ud-tools` after each landed slice batch;
  extend the launcher-manifest smoke test if a new tool entry appears (the Club
  Tester tab lives inside the existing Rate of Closure entry).
- **C9 — AffineDrift**: publication content for the clubfitting capability
  (technology section article referencing the fitting-document and
  fitting-report wires), authored once C1–C5 are stable.

## 5. Engineering standards (binding for every child)

- SI units and explicit frames everywhere; DbC validation on every public function
  (`TypeError`/`ValueError` per repo standard); frozen dataclasses; `__all__`.
- Every wire is versioned, canonical-JSON, fail-closed on unknown fields, and
  round-trip tested; Python↔TS parity pinned by shared golden fixtures.
- Physics gates: analytic limits (rigid shaft, point mass, sphere/box tensors),
  covariance checks, and literature-range assertions — not just regression pins.
- Each child lands as one PR with tests, a SPEC §12 row, CI-batch mypy clean,
  file-size budget respected, and the handoff doc updated.

## 6. Non-goals (v1)

- Running Drake/MuJoCo/OpenSim **in-process** — adapters consume exports; a live
  co-simulation bridge is a follow-on epic.
- Finite-element face/CT-dependent COR modeling — face compliance stays at the
  spring-damper/COR level; per-point CT maps are a documented extension point.
- Full shaft FE in the impact loop — C2's modal/quasi-static model is the v1
  fidelity; the modal solver remains the reference for upgrades.
