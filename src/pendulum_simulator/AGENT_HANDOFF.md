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

## Active Epic — #4766 Swing Objective Comparison

Mechanism-vs-outcome capability on this tool's existing physics: centrifugal/
Coriolis split (#4767), five objectives (#4768), slew-limited collocation
(#4769), cross-evaluation + wire (#4770), Lab surface (#4771), UpstreamDrift
tile (#4772). Contract: `docs/specs/SWING_OBJECTIVE_COMPARISON.md`.

**Nothing here re-derives physics.** `physics.py` already had the equations the
research prototype `Double-Pendulum-Optimization` reached independently; that
repo stays the notebook home and cross-check, not a vendored dependency.

**Three regression-pinned findings that bind future work here:**

1. `P_coriolis_hub = -2 * P_centrifugal_wrist` identically, so centrifugal and
   Coriolis _work_ are one functional. The centrifugal objective is an angular
   impulse; do not "simplify" it back to work.
2. The NLP needs non-dimensional variables and a tight `ftol`. Unscaled it leaves
   defects near 1e-1; at SciPy's default it returns the initial guess and reports
   success.
3. **A comparison can be degenerate.** Near the minimum downswing duration the
   constraints pin the trajectory and every objective returns the same swing, so
   the matrix fills with 100% entries that read as agreement but are a
   configuration artifact. Check `SwingComparison.is_degenerate` first; the
   preset (0.36 s, 250 N·m) carries slack for this. Whether the mechanism
   objectives track clubhead speed is configuration-dependent, not a result.

## Recent Activity (grounding — `git log --oneline -15 -- src/pendulum_simulator`)

Latest substantive change is the #4450 consolidation that landed the drift-
transfer diagnostics. Before that, work was performance/hardening/accessibility
maintenance rather than feature growth: Bolt perf passes on the web
Nelder-Mead simplex sort, a #3745 GUI/error-handling cleanup, a 20-PR
fleet-CI-relief consolidation, physics hot-loop allocation removal, and earlier
`pendulum-core` maturin/pyo3 packaging + import-canonicalization work. No open
PR #4618 owns the merged Tools delivery; do not duplicate it. Its catalog
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

1. Finish #4430's UpstreamDrift consumer pin for the merged PyQt/React
   surfaces and close its remaining gates; preserve every
   scientific-promotion boundary and adverse row.
2. If/when epic #4103 Phase 1 lands the double/triple pendulum
   `SwingSource` integration, expect a coordinated PR here exposing any
   additional bindings rate_of_closure needs.
3. Watch for epic #4120 V3 PRs reusing `perturbation_analysis` — review for
   API-stability impact on this tool's own callers.
