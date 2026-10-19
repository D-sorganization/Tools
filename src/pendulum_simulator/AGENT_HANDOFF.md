# AGENT_HANDOFF — pendulum_simulator

> **Update this file with every PR and every push to main.**
> Last updated: 2026-09-07

## Where This Tool Is Headed

Double Pendulum Golf Swing Simulator: multi-platform (PyQt6 desktop, React/Tauri web, Rust kernel `pendulum-core` via PyO3/WASM, JAX/GPU batch optimization) exploration tool for multi-body kinematic chains — double pendulum, triple pendulum, and 8-DOF closed-loop golfer upper-body model. See `src/pendulum_simulator/README.md` and `FEATURES.md`.

Epic **#4698** is the active source-attribution extension introducing shared `force-attribution/v1` in `shared.python.swing_sim`. Transfer-diagnostics #4406 (UpstreamDrift #8551) is closed via #4450. Rotating-base companion #4430 is active on merged PR #4618 (`87ff0ea8c`).

**The scientific boundary shipped with it and still binds.** The widget fails closed for the triple and golfer tiers. Do not relabel proximal link rate as anatomical shoulder or thorax velocity, and do not unlock higher tiers in the GUI until a model tier exposes unambiguous bilateral hand-force allocation and a rotating base. Primary current relevance is as an **upstream physics source for rate_of_closure** (#4103 Phase 1 `SwingSource` implementations, #4120 V3 `perturbation_analysis` reuse).

## Active Epics — #4766 objectives, #4775 actuation and realism

#4766 shipped mechanism-vs-outcome comparison through #4774; #4775 owns actuation and realism (`docs/specs/SWING_ACTUATION_AND_REALISM.md`). React lab's `force-source-comparison/v4` binds initial state, model, constraints, integration, robustness, and search depth to one contract ID.

**Five results that bind future work here** (regression-pinned):

1. `P_coriolis_to_distal = 2 * P_centrifugal_to_distal` identically.
2. Energy-optimal hand speed at impact is `L1*[I2 - m2*r2*(L2-r2)]`, zero only for a tip-concentrated club. The shipped model is distributed (#5431): coefficient -0.0217 (hands move backward at 0.36% of clubhead speed). #5523 re-ran the conclusions built on the old zero.
3. Canonical club (#5431): uniform shaft `m2` plus head point mass `mClub`; `Mc`, `s`, `Jw` come from `club_inertia.distal_inertia` (Rust/TS mirrors, shared JSON fixture). Driver preset `m2=0.1078`, `mClub=0.2022` (supersedes the #4785 tip-mass `me = 0.238`). Force-source JSON artifacts are STALE until #5553; `base_force` fix is #5522.
4. Shipped distributed club (re-run #5523, `scripts/rerun_distributed_club.py`): 51.4 m/s clubhead, 8.02 m/s hands, club/arm 3.20, 14% hub braking (peak 45.6 N·m); hub-reversal artifact is lumped-club only; centrifugal (not Coriolis) is the dissenting objective; all exploratory.
5. Browser separates equal-output efficiency from equal-input capacity.

Web lab reports wrist reversal time, transition duration, torque slew, work, impulse, activation, utilization, 17 sampled signals, and Pareto matrix. Moving hub (`physics_triple.py`) remains next fidelity step.

## Must-Read Architecture Pointers

1. `src/pendulum_simulator/README.md` — model topology (double/triple pendulum, 8-DOF golfer closed loop).
2. `src/pendulum_simulator/pendulum-core/` — shared Rust physics engine (PyO3 native + wasm-bindgen).
3. `src/pendulum_simulator/pendulum-web/` — React/Tauri web mirror (`src/optimizer.ts` Nelder-Mead simplex).
4. `src/pendulum_simulator/AUDIT_TDD_DBC_DRY.md` / `DEEP_REVIEW.md` — prior audit findings.
5. Perturbation/Monte Carlo analysis code under `src/pendulum_simulator/` — precedent for #4120 V3.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/ -k pendulum_simulator -n auto --timeout=60
python3 -m pytest src/pendulum_simulator -n auto --timeout=60
cd src/pendulum_simulator/pendulum-web && npm run test && npx tsc --noEmit
cargo test -p pendulum-core
python3 -m ruff check src/pendulum_simulator
```

## Do-Not List

- Do not duplicate perturbation/Monte Carlo machinery in `swing_sim/variation` — reuse `perturbation_analysis`.
- Do not reintroduce `Array.prototype.sort()` with comparator in web Nelder-Mead hot loop.
- Do not change golfer model's holonomic-constraint topology without updating Rust kernel and Python/web bindings.
- The Rust and Python golfer solvers are one solver (#5638): change closure equations, `INDEPENDENT_CLOSURE_ROWS`, Baumgarte form or Coriolis in both, and keep `tests/test_golfer_rust_python_parity.py` green (needs `maturin develop` in `pendulum-core`). Coriolis is the Christoffel contraction `C = Mdot qdot - 0.5 grad_q(qdot^T M qdot)` (#5640 restored the third term `dM_jk/dq_i` = `dM.transpose(2, 0, 1)`); the one definition is `christoffel.christoffel_first_kind` / Rust `golfer::christoffel_contraction`, and `coriolis_jax` must match. The oracle is `tests/test_golfer_coriolis_oracle.py` (Euler-Lagrange); note `constraint_solver.coriolis_matrix` is an alias of `analytical_coriolis`, so it is not an independent reference.
- Do not widen Drift Transfer tab beyond double-pendulum tier, and do not present proximal link rate as coaching quantity.
- Do not resume `research/shoulder-velocity-drift-transfer`; #4406 is closed and shipped via #4450.

## Roadmap (ordered)

1. Finish #4430's UpstreamDrift consumer pin; preserve promotion boundaries.
2. Coordinate #4103 Phase 1 double/triple `SwingSource` bindings here.
3. Review #4120 V3 `perturbation_analysis` reuse for API-stability impact.
