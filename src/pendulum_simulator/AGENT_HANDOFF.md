# AGENT_HANDOFF — pendulum_simulator

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-12

## Where This Tool Is Headed

Double Pendulum Golf Swing Simulator: multi-platform (PyQt6 desktop,
React/Tauri web, Rust kernel `pendulum-core` via PyO3/WASM, JAX/GPU batch
optimization) exploration tool for multi-body kinematic chains — double
pendulum, triple pendulum, and an 8-DOF closed-loop golfer upper-body model.
See `src/pendulum_simulator/README.md` for the model topology writeup and
`src/pendulum_simulator/FEATURES.md` for the feature inventory.

UpstreamDrift epic
[#8511](https://github.com/D-sorganization/UpstreamDrift/issues/8511) makes
this tool the canonical interactive companion for the proximal–distal article
and book. The active Tools branch is `feat/proximal-distal-workbench`: it adds
one shared experiment/glossary catalog consumed by PyQt6 and React/Tauri,
without importing either downstream publication repository.

Dedicated transfer-diagnostics issue **#4406** is active under UpstreamDrift
epic **#8551** on branch `research/shoulder-velocity-drift-transfer`. The first
TDD slice adds a model-neutral transfer contract, exact double-pendulum
drift/control grip-force attribution, integrated braking/work/impulse metrics,
Pareto ranking, and a PyQt **Drift Transfer** analysis tab. The widget fails
closed for triple and golfer tiers; do not relabel the double model's proximal
link rate as anatomical shoulder or thorax velocity. The next qualified model
tier must expose an unambiguous bilateral hand-force allocation and a rotating
base before those claims enter the GUI.

Apart from that issue, this tool's primary
current relevance is as an **upstream physics source for rate_of_closure**:
epic #4103 Phase 1 integrates this tool's double/triple pendulum models as
`SwingSource` implementations, and epic #4120 V3's variation engine
explicitly reuses this tool's `perturbation_analysis` machinery as one of
three "how parameters vary" precedents (alongside UpstreamDrift's
`EnhancedBallFlightSimulator` and `movement_optimizer`'s parallel-start
machinery) rather than reimplementing Monte Carlo sampling from scratch. If
you're working in `swing_sim/variation/`, read this tool's perturbation
analysis code first.

## Recent Activity (grounding — `git log --oneline -15 -- src/pendulum_simulator`)

Most recent work is performance/hardening/accessibility maintenance, not
feature growth: Bolt perf passes on the web Nelder-Mead simplex sort, a #3745
GUI/error-handling cleanup, a 20-PR fleet-CI-relief consolidation, physics
hot-loop allocation removal, and earlier `pendulum-core` maturin/pyo3
packaging + import-canonicalization work. No open PRs currently target this
tool directly.

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

## Roadmap (ordered)

1. Land the shared companion catalog and both UI consumers for UpstreamDrift
   #8511 while keeping the existing physics and `pendulum-core` API stable.
2. If/when epic #4103 Phase 1 lands the double/triple pendulum
   `SwingSource` integration, expect a coordinated PR here exposing any
   additional bindings rate_of_closure needs.
3. Watch for epic #4120 V3 PRs reusing `perturbation_analysis` — review for
   API-stability impact on this tool's own callers.
