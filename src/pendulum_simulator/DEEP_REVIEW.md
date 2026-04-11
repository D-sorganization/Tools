# Pendulum Simulator — Deep Dive Technical Review

**Date**: 2026-03-08
**Scope**: Optimization, DRY, DbC, TDD, Vectorization, UI/UX, Changeability
**Codebase**: ~17,700 lines across Python (PyQt6), TypeScript (React/Tauri), Rust (physics kernel)
**Test baseline**: 156 tests

---

## Summary Scorecard

| Dimension     | Grade | Key Issue                                                        |
|---------------|-------|------------------------------------------------------------------|
| Optimized     | B+    | 8-DOF Coriolis is O(N³) Python loops; duplicated KKT solve      |
| DRY           | B-    | 3 physics engines, 3 SimResult classes, JAX copy-paste of NumPy  |
| DbC           | A-    | Excellent preconditions; `assert` in GUI should be `raise`       |
| TDD           | B+    | 156 tests, property-based; no GUI tests, thin web tests          |
| Vectorized    | C+    | Coriolis should use `einsum`; SimResult accessors are scalar-only |
| UI/UX         | B     | Inconsistent tab paradigms; no validation feedback; web behind   |
| Changeable    | B-    | Adding a model requires ~3000 lines of boilerplate               |

## Tracking Status (Updated 2026-03-09)

The review findings that were already implemented have now been normalized into
GitHub issue tracking, so every addressed pendulum follow-up item has a
corresponding issue.

| Review item | Issue | Status |
|-------------|-------|--------|
| Triple pendulum friction model and dissipation UI | #1048 | Closed |
| TripleSimulationResult friction and total torque accessors | #1049 | Closed |
| Generic `make_polynomial_torque` extraction | #1041 | Closed |
| Elbow torque import support in `FunctionGeneratorDialog` | #1050 | Closed |
| `PlotThemeManager` integration for torque history plots | #1051 | Closed |
| Theme-derived pendulum styling cleanup | #1042 | Closed |
| Typing import modernization in pendulum physics modules | #1043 | Closed |
| DbC postconditions for gravity and friction helpers | #1044 | Closed |
| Triple pendulum friction regression coverage | #1052 | Closed |
| Repo-root discovery instead of fragile `sys.path` climbing | #1053 | Closed |
| Function Generator button on triple pendulum controls | #1054 | Closed |
| Removal of pendulum GUI `type: ignore` debt | #1045 | Closed |

- `PR #982` implemented the main pendulum follow-up wave on `main`.
- `PR #1046` performed cleanup and closed `#1041`, `#1042`, and `#1044`.
- The PR description mirrors this tracker so issue coverage is visible both in
  GitHub and in-repo.

---

## 1. Optimization — Grade: B+

### Strengths
- 2/3-DOF physics use closed-form analytical Lagrangian mechanics — O(1) per evaluation
- 8-DOF golfer upgraded from numerical to analytical Jacobians (14.7× speedup)
- `scipy.solve_ivp` with adaptive RK45 is the right integrator choice
- Rust core provides compiled Dormand-Prince RK45 for native/WASM

### Critical Bottlenecks

#### 1.1 Coriolis Triple Loop (physics_golfer.py:953-997)
`analytical_coriolis` computes `dM/dq_k` via 8 finite-difference mass-matrix calls, then
runs a **triple-nested Python loop** (8×8×8 = 512 iterations) for Christoffel symbols.
This is the hottest function in the ODE RHS, running pure interpreted Python.

```python
# Current: O(N³) Python loop
for i in range(N_DOF):
    for j in range(N_DOF):
        christoffel = 0.0
        for k in range(N_DOF):
            christoffel += 0.5 * (dM[i,j,k] + dM[i,k,j] - dM[j,k,i]) * qdot[k]
        C_qdot[i] += christoffel * qdot[j]

# Fix: single vectorized einsum
C_qdot = np.einsum('ijk,j,k->i',
    0.5*(dM + dM.transpose(0,2,1) - dM.transpose(2,1,0)), qdot, qdot)
```

#### 1.2 Duplicated KKT Solve (constraint_solver.py:135-188)
`constraint_forces()` rebuilds the entire 12×12 KKT system identically to
`constrained_accelerations()` just to extract `sol[n:]`. Both should return
`(qddot, lambda)` from a single solve.

#### 1.3 Identical "Analytical" Bias Functions (constraint_solver.py:191-230)
`_constraint_acceleration_bias` and `analytical_constraint_acceleration_bias` are
byte-for-byte identical. The "analytical" version is a dead misnomer using the same
finite-difference approach.

#### 1.4 ODE RHS Cost Per Step (~13 FK evaluations)
Each RHS evaluation requires: 8 for Coriolis FD + 1 bias FD + 1 mass + 1 gravity +
2 constraint = ~13 forward-kinematics evaluations.

### Speed-Up Recommendations (ranked by impact)
1. **`einsum` for Coriolis** — 50-100× faster for 8-DOF
2. **Analytical `dM/dq`** — chain rule on J^T J gives closed-form derivatives; eliminates 8 mass-matrix evals/step
3. **Merge `constrained_accelerations` + `constraint_forces`** — 2× fewer evals when both needed
4. **JAX `jax.jacfwd(mass_matrix_jax)`** — auto-diff replaces finite differences
5. **Pre-allocate KKT matrix** — fill in-place rather than `np.zeros` each call

---

## 2. DRY — Grade: B-

### Strengths
- `_m2eff()` helper avoids repeating `m2 + mClub`
- `controls_utils.py` consolidates parse helpers and style constants
- `LabeledInput` defined once, reused by all control panels
- `ellipsoid_from_jacobian` shared across 2/3-DOF models

### Violations

#### 2.1 Three Independent Physics Engines
`physics.py`, `physics_triple.py`, `physics_golfer.py` each implement:
`mass_matrix`, `coriolis_vector`, `gravity_vector`, `forward_kinematics`,
`kinetic_energy`, `potential_energy`, `total_energy`, `equations_of_motion`,
`linear_accelerations`, `net_joint_forces`, `friction_torque_vector`.

The structure is identical; only DOF count and formulas differ. A generic N-DOF
Lagrangian base could eliminate ~60% of this code.

#### 2.2 JAX Module is Verbatim Copy (~600 lines)
`physics_golfer_jax.py` duplicates `physics_golfer.py` with `np` → `jnp` and
`.at[].set()` syntax. Fix: backend-agnostic approach (pass `xp` module) or thin adapter.

#### 2.3 Three SimulationResult Classes
`SimulationResult`, `TripleSimulationResult`, `GolferSimulationResult` have identical
accessor patterns. Should share a base class.

#### 2.4 Three `make_polynomial_torque` Functions
`simulation.py:43-65`, `simulation_triple.py`, `simulation_golfer.py:44-75` — identical
logic, different arity.

#### 2.5 GUI Stylesheet Duplication
- `controls_widget_triple.py:317-322` defines `_STYLE_CHECK` locally, duplicating
  `STYLE_CHECK` from `controls_utils.py`
- `controls_widget_golfer.py:383-389` does the same

#### 2.6 Playback/Export Boilerplate
All three control widgets copy-paste: `_on_play_toggled`, `set_slider_range`,
`set_slider_value`, `stop_playback`, and identical signal declarations. A `PlaybackMixin`
would eliminate this.

---

## 3. Design by Contract — Grade: A-

### Strengths (strongest dimension)
- Every `@dataclass(frozen=True)` has `__post_init__` with comprehensive precondition
  assertions
- Every physics function asserts finiteness of inputs and outputs
- Postconditions explicitly checked: mass matrix symmetry, state_dot finiteness
- Shape assertions consistent throughout: `(4,)`, `(6,)`, `(16,)`
- Docstrings document Pre/Post contracts

### Gaps
1. **`assert` used for GUI validation** (`controls_widget.py:495-499`) — disabled by
   Python `-O`. Should be `raise ValueError(...)`.
2. **No invariant on `SimulationResult`** — no validation of `states.shape[1]` vs DOF
   count, or `t` monotonicity.
3. **`project_to_constraints`** silently returns the last iterate even if Newton's method
   didn't converge.

---

## 4. TDD — Grade: B+

### Strengths
- 156 tests across 14 test files
- Property-based: symmetry, PSD, energy conservation, velocity scaling, singularity
- Parametric fixtures (`conftest.py`)
- Test classes organized by physical property
- Analytical vs. numerical parity validation (26 tests)
- GPU optimizer validation (10 tests)

### Gaps
1. **No GUI tests** — 14 widgets (~200KB) are untested
2. **No performance regression tests** — `perf_test.py` not in CI
3. **No constraint solver edge cases** — no near-singular KKT, drift, or gain sensitivity tests
4. **Web tests are thin** — single `physics.test.ts` file

---

## 5. Vectorization — Grade: C+

### Strengths
- 2/3-DOF inherently scalar; NumPy vectorizes at BLAS level
- JAX module with `vmap`-compatible pure functions
- GPU batch simulation via JAX/diffrax

### Weaknesses

#### 5.1 Coriolis Triple Loop
Both NumPy and JAX versions use O(N³) Python loops instead of `einsum`.

#### 5.2 Scalar-Only SimulationResult
All accessors (`mass_matrix_at`, `positions_at`, `energy_at`) are single-index.
No batch trajectory computation.

#### 5.3 Numerical Mass Matrix Fallback
`numerical_mass_matrix` runs 7 mass points × 8 DOFs = 56 FK evaluations. Catastrophically
slow if called.

#### 5.4 JAX Python Loops
`physics_golfer_jax.py:465-470` uses Python `for` loops inside JIT-intended code.
Should use `jnp.einsum` or `jax.lax.fori_loop`.

---

## 6. UI/UX — Grade: B

### Strengths
- Cohesive dark theme
- Comprehensive desktop feature set: animation, matrices, ellipsoids, torque history
- Good presets for each model
- Tooltips on all inputs
- Export (CSV, video)
- Scrollable controls for 8-DOF

### Weaknesses

#### 6.1 Inconsistent Tab Paradigms
- Double: toolstrip-driven playback (hidden compat widgets)
- Triple/Golfer: embedded Run/Reset + inline playback controls
- Users get different interaction patterns across tabs

#### 6.2 No Input Validation Feedback
Entering a negative mass produces a Python `AssertionError` traceback. No inline error
highlighting, red borders, or status bar messages.

#### 6.3 Web Version Far Behind
No triple pendulum, no golfer, no force ellipsoids, no function generator, no export.
Two app files (`App.tsx`, `AppNew.tsx`) suggest incomplete migration.

#### 6.4 Missing Features
- No undo/redo for parameter changes
- Torque preview only on double/triple tabs, not golfer
- Preset format inconsistent (tuples vs dicts)

---

## 7. Changeability — Grade: B-

### Strengths
- `frozen=True` dataclasses — immutable params safe to pass around
- Pure-function physics (no global state)
- Clean separation: physics → simulation → GUI
- Well-defined Rust API boundary

### Weaknesses

#### 7.1 High Cost to Add Models
Adding a 4th model requires ~3000+ lines across:
- `physics_*.py` (~500 lines)
- `simulation_*.py` (~250 lines)
- `controls_widget_*.py` (~500 lines)
- `pendulum_widget_*.py` (~500 lines)
- `matrix_widget_*.py` (~300 lines)
- test file (~400 lines)
- `main_window.py` modifications

An N-DOF generalized architecture + model registry would reduce this to ~500 lines.

#### 7.2 No Plugin/Registry Pattern
Models hardcoded into main window tab structure. A factory pattern would allow declarative
model registration.

#### 7.3 Fragile Module-Level Aliasing (physics_golfer.py:1131-1134)
```python
mass_matrix = analytical_mass_matrix
```
Shadows the function name, preventing debugging access to numerical version.

#### 7.4 Zero Shared Physics Code (Web/Desktop)
TypeScript and Python physics are independent reimplementations. Rust core exists to bridge
this but isn't integrated into Python desktop app.

---

## Top 5 Actionable Items

1. **Vectorize Coriolis with `einsum`** — biggest single performance win, minimal code change
2. **Extract `LagrangianModel` base class** — unify physics/simulation/result across all DOF counts
3. **Replace GUI `assert` with `raise ValueError`** — production safety
4. **Add batch trajectory accessors** — `result.all_energies()`, `result.all_positions()`
5. **Unify web/desktop via Rust core** — single source of truth for physics
