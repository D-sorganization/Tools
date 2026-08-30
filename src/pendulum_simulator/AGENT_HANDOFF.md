# AGENT_HANDOFF — pendulum_simulator

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-30

## Where This Tool Is Headed

Double Pendulum Golf Swing Simulator: multi-platform (PyQt6 desktop,
React/Tauri web, Rust kernel `pendulum-core` via PyO3/WASM, JAX/GPU batch
optimization) exploration tool for multi-body kinematic chains — double
pendulum, triple pendulum, and an 8-DOF closed-loop golfer upper-body model.
See `src/pendulum_simulator/README.md` for the model topology writeup and
`src/pendulum_simulator/FEATURES.md` for the feature inventory.

Epic **#4698** is the active source-attribution extension. It introduces the
shared `force-attribution/v1` contract in `shared.python.swing_sim`, separating
cross-speed Coriolis, squared-speed, gravity, damping, control, and residual
terms with exact closure and rank-aware force-only hand-path mapping. The
triple and golfer tiers remain unavailable until they supply compatible mass-
matrix derivatives and endpoint semantics; do not synthesize those providers.

Transfer-diagnostics issue **#4406** (UpstreamDrift epic **#8551**) is **closed**,
landed via consolidation **#4450**. It shipped the model-neutral transfer
contract, exact double-pendulum drift/control grip-force attribution,
braking/work/impulse metrics, Pareto ranking, and the PyQt **Drift Transfer**
tab. The branch `research/shoulder-velocity-drift-transfer` is spent — do not
resume it.

Companion issue **#4430** is active on
merged PR **#4618** (`87ff0ea8c`). Tools `main` now has the source-pinned
18-case authority, exact qualified rotating-base physics, a registered
full-resolution provider, asynchronous PyQt surface, and React/Tauri evidence
browser. The web surface consumes the digest-pinned complete 18-run catalog
and renders five reviewer trace groups. Cross-platform scalar matching allows
solver roundoff while retaining stricter constraint and power-identity gates.

**The scientific boundary shipped with it and still binds.** The widget fails
closed for the triple and golfer tiers. Do not relabel the double model's
proximal link rate as anatomical shoulder or thorax velocity, and do not
unlock the higher tiers in the GUI until a model tier exposes an unambiguous
bilateral hand-force allocation and a rotating base. Upstream has since
qualified those pathways separately (UpstreamDrift epic #8684: distributed
grip #8696, passive shaft #8715, finite ground #8719) — and the ground tier's
preregistered screen admitted **0 of 384** cells, so a moving base is not a
free upgrade. Check the current UpstreamDrift qualification state before
widening any claim here.

Apart from that, this tool's primary current relevance is as an **upstream
physics source for rate_of_closure**: epic #4103 Phase 1 integrates this tool's
double/triple pendulum models as `SwingSource` implementations, and epic #4120
V3's variation engine explicitly reuses this tool's `perturbation_analysis`
machinery as one of three "how parameters vary" precedents (alongside
UpstreamDrift's `EnhancedBallFlightSimulator` and `movement_optimizer`'s
parallel-start machinery) rather than reimplementing Monte Carlo sampling from
scratch. If you're working in `swing_sim/variation/`, read this tool's
perturbation analysis code first.

## Active Epics — #4766 objectives, #4775 actuation and realism

#4766 shipped the mechanism-vs-outcome comparison through #4774; #4775 owns
actuation and realism (`docs/specs/SWING_ACTUATION_AND_REALISM.md`). The React
lab's `force-source-comparison/v2` artifact now binds initial state, model,
constraints, integration, robustness, and search depth to one contract ID.
Changed inputs discard stale rows. Every objective is cross-evaluated on every
displayed winner, and imports fail if any own-objective row loses. The bundled
research contract uses 512 candidates, 1 ms integration, 0.5 N m wrist steps,
and 25 robustness trials. Fixed-hub mode shows only physical markers in one
frame; impact alignment alone shows a labelled camera crosshair. Design
contract: `docs/specs/FORCE_SOURCE_WEB_LAB.md`.

**Five results that bind future work here** (all regression-pinned):

1. `P_coriolis_to_distal = 2 * P_centrifugal_to_distal` identically. Both
   energy rows are therefore one functional; centrifugal impulse remains the
   independent squared-speed objective.
2. The energy-optimal hand speed at impact is `L1*[I2 - m2*r2*(L2-r2)]`,
   identically zero for a point-mass clubhead. That is an _unconstrained ideal_,
   not a prediction about where a constrained optimum lands — a distinction that
   was got wrong once, see 3.
3. **The club preset was 2.1x too heavy at the tip (#4785).** In a
   point-mass-at-tip model you must match the real club's inertia about the
   _wrist_, not its mass: `me = 0.238 kg` for a driver, not 0.50. The earlier
   value doubled the arm/club coupling and forced the optimizer to stop the
   hands; that artifact was published as a structural limit before being caught.
   Use `club_equivalence.equivalent_tip_mass` for any new club.
4. **Corrected, the model is golf-like**: 49.7 m/s clubhead, 7.26 m/s hands,
   club/arm 3.46, five of six measured observables inside their bands, with no
   hand-speed floor. A moving hub is an improvement, not a prerequisite.
5. **The browser ranking is now self-consistent.** Its certified speed winner is
   38.33 m/s versus 38.03 m/s for hand-path impulse; Coriolis impulse and both
   energy rows select the speed candidate, while centrifugal impulse reaches
   30.93 m/s. This certifies displayed candidates, not a global optimum.

Next open item is the **late release**: every objective releases at ~90% of the
downswing against a measured 55-80%. A moving hub (`physics_triple.py`) is the
most likely route. Hill-type actuation (`actuation.py`) is built and tested.

## Must-Read Architecture Pointers

1. `src/pendulum_simulator/README.md` — model topology (double/triple
   pendulum, 8-DOF golfer upper-body closed loop with 4 holonomic
   constraints).
2. `src/pendulum_simulator/pendulum-core/` — shared Rust physics engine
   (PyO3 native + wasm-bindgen for web); this is the pattern epic #4103's
   `swing-core` crate follows.
3. `src/pendulum_simulator/pendulum-web/` — React/Tauri web mirror
   (`src/optimizer.ts` Nelder-Mead simplex — recently perf-hardened, keep
   allocation-free in the hot loop).
4. `src/pendulum_simulator/AUDIT_TDD_DBC_DRY.md` / `DEEP_REVIEW.md` — prior
   audit findings; check before assuming an area is unaudited.
5. Perturbation/Monte Carlo analysis code (search for `perturbation_analysis`
   under `src/pendulum_simulator/`) — the precedent epic #4120 V3 builds on.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/ -k pendulum_simulator -n auto --timeout=60
python3 -m pytest src/pendulum_simulator -n auto --timeout=60
cd src/pendulum_simulator/pendulum-web && npm run test && npx tsc --noEmit
cargo test -p pendulum-core
python3 -m ruff check src/pendulum_simulator
```

Note: `src/pendulum_simulator/tests/` is bridged into the top-level `tests/`
tree (see SPEC.md changelog entry on embedded-suite discovery) — running
top-level `pytest tests/` already includes pendulum coverage; don't
double-collect by also passing the embedded path.

## Do-Not List

- Do not duplicate the perturbation/Monte Carlo machinery in
  `swing_sim/variation` — reuse or adapt this tool's `perturbation_analysis`
  per epic #4120 V3's explicit DRY instruction.
- Do not reintroduce `Array.prototype.sort()` with a comparator in the web
  Nelder-Mead hot loop — it was deliberately replaced with an allocation-free
  insertion sort for the small fixed-size simplex.
- Do not change the golfer upper-body model's holonomic-constraint topology
  without updating both the Rust kernel and its Python/web bindings in the
  same PR (single-source-physics discipline, same as rate_of_closure).
- Do not widen the Drift Transfer tab beyond the double-pendulum tier, and do
  not present its proximal link rate as an anatomical or coaching quantity.
  The fail-closed behaviour for the triple and golfer tiers is deliberate.
- Do not resume `research/shoulder-velocity-drift-transfer`; #4406 is closed
  and shipped via #4450.

## Roadmap (ordered)

1. Finish #4430's UpstreamDrift consumer pin; preserve all promotion boundaries.
2. Coordinate any #4103 Phase 1 double/triple `SwingSource` bindings here.
3. Review #4120 V3 `perturbation_analysis` reuse for API-stability impact.
