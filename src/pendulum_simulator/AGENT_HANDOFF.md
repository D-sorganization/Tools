# AGENT_HANDOFF — pendulum_simulator

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-21

## Where This Tool Is Headed

Double Pendulum Golf Swing Simulator: multi-platform (PyQt6 desktop,
React/Tauri web, Rust kernel `pendulum-core` via PyO3/WASM, JAX/GPU batch
optimization) exploration tool for multi-body kinematic chains — double
pendulum, triple pendulum, and an 8-DOF closed-loop golfer upper-body model.
See `src/pendulum_simulator/README.md` for the model topology writeup and
`src/pendulum_simulator/FEATURES.md` for the feature inventory.

Transfer-diagnostics issue **#4406**, under UpstreamDrift epic **#8551**, is
**closed**. It landed on main via consolidation **#4450** (`8f654b3a1`,
2026-08-14). The branch `research/shoulder-velocity-drift-transfer` is spent —
do not resume work on it.

What shipped: a model-neutral transfer contract, exact double-pendulum
drift/control grip-force attribution, integrated braking/work/impulse metrics,
Pareto ranking, and a PyQt **Drift Transfer** analysis tab.

Companion issue **#4430** is active on
PR **#4618** (`feat/4430-rotating-base-companion`). The branch has the source-pinned
18-case authority, exact qualified rotating-base physics, a registered
full-resolution provider, asynchronous PyQt surface, and React/Tauri evidence
browser. The web surface consumes the digest-pinned complete 18-run catalog
and renders five reviewer trace groups. Protected squash auto-merge is enabled;
the active CI fix allows platform roundoff but retains strict residual gates.

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

## Recent Activity (grounding — `git log --oneline -15 -- src/pendulum_simulator`)

Latest substantive change is the #4450 consolidation that landed the drift-
transfer diagnostics. Before that, work was performance/hardening/accessibility
maintenance rather than feature growth: Bolt perf passes on the web
Nelder-Mead simplex sort, a #3745 GUI/error-handling cleanup, a 20-PR
fleet-CI-relief consolidation, physics hot-loop allocation removal, and earlier
`pendulum-core` maturin/pyo3 packaging + import-canonicalization work. No open
PR #4618 owns #4430 delivery; do not duplicate its active branch. Its catalog
digest is `66493b833955c6492a00eae4a600df795df60a6f473f9a11c403084b58e51678`.

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

1. Finish #4430's full repository gates, protected PR, and UpstreamDrift
   consumer pin for the completed PyQt/React surfaces; preserve every
   scientific-promotion boundary and adverse row.
2. If/when epic #4103 Phase 1 lands the double/triple pendulum
   `SwingSource` integration, expect a coordinated PR here exposing any
   additional bindings rate_of_closure needs.
3. Watch for epic #4120 V3 PRs reusing `perturbation_analysis` — review for
   API-stability impact on this tool's own callers.
