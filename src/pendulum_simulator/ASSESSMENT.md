# Pendulum Simulator — Comprehensive Code Assessment

**Date**: 2026-03-07
**Scope**: Full codebase review covering Python (PyQt6), TypeScript (React/Tauri), physics engines, GUI, themes, function generator integration
**Test baseline**: 134/134 tests passing

---

## Executive Summary

The pendulum simulator is a well-architected physics simulation tool with solid Lagrangian mechanics implementations, good Design by Contract practices, and an attractive dark-themed UI with trace-style plots. However, there are significant **version parity gaps**, **DRY violations**, **missing theme system integration**, and **friction model gaps** in the triple pendulum that need attention.

---

## 1. Version Parity Issues (Python ↔ TypeScript)

### 1.1 Triple Pendulum Missing from Web Version — CRITICAL
The web version (`pendulum-web/src/physics.ts`) implements ONLY the double pendulum. The Python version has complete double AND triple pendulum implementations. The React UI (`App.tsx`) has no tab or mode for triple pendulum simulation.

**Impact**: Users of the web/Tauri version have no access to triple pendulum simulation.

### 1.2 Integrator Quality Mismatch — MODERATE
- **Python**: Uses `scipy.solve_ivp` with adaptive RK45 (double) and DOP853 8th-order (triple), with `rtol=1e-8, atol=1e-10`.
- **TypeScript**: Uses fixed-step RK4 with no adaptive stepping or error control.

**Impact**: The web version will accumulate numerical errors faster, especially for chaotic configurations. Energy conservation will be worse.

### 1.3 Missing Features in Web Version — MODERATE
Features present in Python but absent from TypeScript:
- Jacobian-based manipulability/force ellipsoids
- Counterfactual zero-torque analysis
- Linear acceleration and net joint force computation
- Torque history decomposition (drive vs. friction vs. total)
- CSV/video export
- Force vector overlays
- Zoom/pan controls on canvas

### 1.4 SimulationResult Interface Mismatch — LOW
The TypeScript `SimulationResult` doesn't store `torque_func`, so post-simulation torque analysis (friction decomposition, total torques) cannot be reconstructed.

---

## 2. Physics Accuracy Issues

### 2.1 Triple Pendulum Missing Friction Model — CRITICAL
`TriplePendulumParams` has NO friction parameters (no `b1,b2,b3` or `mu1,mu2,mu3`). The triple pendulum `equations_of_motion` uses only `M * qddot = tau - C - G` with no dissipation term. This is a **functional parity gap** — the double pendulum has full viscous + Coulomb friction.

**Impact**: Triple pendulum simulations cannot model energy dissipation, making them unrealistic for golf swing and many other physical scenarios.

### 2.2 TripleSimulationResult Missing Friction Accessors — MODERATE
`TripleSimulationResult` is missing `friction_torques_at()` and `total_torques_at()` methods that exist in the double pendulum's `SimulationResult`.

### 2.3 Coriolis Vector Triple Implementation — LOW RISK
The triple pendulum Coriolis implementation uses a manual derivation. The terms have been verified by the test suite (velocity-scaling tests pass), but the formulation differs from standard textbook Christoffel-symbol form. The tests adequately cover the key properties (zero at rest, quadratic velocity scaling).

### 2.4 Web Version Polynomial Evaluation Convention — VERIFIED CORRECT
Both Python and TypeScript use the same convention: `tau(t) = c0 + c1*t + c2*t^2 + ...` where coefficients are stored `[c0, c1, c2, ...]`. Python reverses for `np.polyval` (highest-first), TypeScript evaluates directly with `reduce`. Both produce identical results.

---

## 3. Theme and Plot Theme Management

### 3.1 Hardcoded Stylesheets — CRITICAL DRY VIOLATION
Three separate files contain hardcoded dark theme colors:

| File | Issue |
|------|-------|
| `main_window.py` | `_PENDULUM_DARK_STYLE` (30 lines of hardcoded QSS) |
| `controls_utils.py` | `STYLE_GROUP`, `STYLE_EDIT`, `STYLE_LABEL`, etc. |
| `function_generator_dialog.py` | `_STYLE_LABEL`, `_STYLE_EDIT`, `_STYLE_BTN`, `_STYLE_BTN_IMPORT` |

The repo has a comprehensive shared theme system (`shared/python/theme/`) with ThemeManager, create_theme_menu, ThemedWindowMixin, and 15 built-in themes — but the pendulum simulator barely uses it. The ThemeManager is connected in `_setup_theme()` but the hardcoded styles override theme-applied stylesheets.

### 3.2 No PlotThemeManager Integration — MODERATE
The `torque_history_widget.py` uses pyqtgraph for plotting but doesn't use the shared `PlotThemeManager` (`shared/python/plot_theme/`). This means plot colors don't respond to theme changes, breaking the fleet-wide theming contract.

### 3.3 Fragile sys.path Manipulation — MODERATE
`main_window.py` line 58 does `Path(__file__).parents[7] / "shared" / "python"` which is extremely fragile — any directory restructuring breaks the import. Same pattern in `function_generator_dialog.py` line 45.

**Recommendation**: Use the repo's standard import pattern (relative imports or installed packages).

### 3.4 Web Version Has No Theme Support — LOW
`App.css` uses CSS custom properties (`--bg`, `--surface`, etc.) which is good architecture, but there's no runtime theme switching. The web version could read from `themes.json` to achieve parity.

---

## 4. Function Generator Integration

### 4.1 Missing "Import as Elbow Torque" for Triple Pendulum — MODERATE
The `FunctionGeneratorDialog` only offers "Import → Shoulder Torque" and "Import → Wrist Torque" buttons. The triple pendulum has THREE joints (shoulder, elbow, wrist) but the dialog cannot import to the elbow joint.

### 4.2 Fragile Path Resolution — MODERATE
Line 45: `_fg_root = Path(__file__).parents[7] / "function_generator" / "python"` — same fragility issue as theme imports.

### 4.3 No Integration in Web Version — LOW
The web version has no function generator integration. Users must manually enter polynomial coefficients.

---

## 5. DRY Violations

### 5.1 Duplicate Polynomial Torque Builders — HIGH
`simulation.py::make_polynomial_torque()` and `simulation_triple.py::make_polynomial_torque()` are nearly identical — they differ only in arity (2 vs 3 joint coefficients). This should be a single generic function.

### 5.2 Duplicate Result Accessor Patterns — MODERATE
`SimulationResult` and `TripleSimulationResult` share identical patterns for `energy_at()`, `n_steps`, and similar accessor logic. A base class could eliminate duplication.

### 5.3 Duplicate Style Constants — HIGH
Hardcoded style strings appear in 3 files (see §3.1). These should derive from the theme system.

### 5.4 Duplicate Physics Function Signatures — LOW
`kinetic_energy`, `potential_energy`, `total_energy` follow the same pattern in both `physics.py` and `physics_triple.py`. This is acceptable given the different parameter types, but the total_energy = KE + PE pattern could be shared.

---

## 6. Design by Contract (DbC) Assessment

### 6.1 Strong Points
- All `PendulumParams` / `TriplePendulumParams` fields validated in `__post_init__`
- `mass_matrix()` checks finiteness pre and symmetry post
- `equations_of_motion()` validates state shape and finiteness
- `run_simulation()` validates all inputs and outputs
- TypeScript mirrors Python contracts with `assertFinite`, `assertPositive`, `assertNonNeg`

### 6.2 Gaps
- `coriolis_vector` in physics_triple.py doesn't check `dphi2` finiteness individually
- `gravity_vector` in both modules lacks postcondition (should assert finite output)
- `friction_torque_vector` postcondition (opposing sign) is documented but not asserted
- `TriplePendulumParams` doesn't document the absence of friction parameters as a contract limitation
- `mass_matrix_components()` lacks postconditions (M_full should be PD)

---

## 7. TDD Assessment

### 7.1 Current Coverage (134 tests, all passing)
| Module | Tests | Quality |
|--------|-------|---------|
| test_physics.py | 17 | Excellent — covers symmetry, PD, coupling, kinematics, energy, DbC |
| test_physics_triple.py | 13 | Good — covers key properties |
| test_simulation.py | 9 | Good — covers polynomial torques, energy conservation, accessors |
| test_simulation_triple.py | 6 | Adequate — basic coverage |
| test_friction.py | 11 | Excellent — thorough friction model validation |
| test_jacobians.py | 19 | Excellent — finite difference validation, singularity handling |
| test_counterfactual.py | 6 | Good — static and dynamic verification |

### 7.2 Missing Tests
- **No friction tests for triple pendulum** (because friction isn't implemented)
- **No GUI widget tests** — zero test coverage for all 10+ GUI modules
- **No function_generator_dialog test** — polynomial fitting is untested
- **No web version tests** — `pendulum-web/` has no test files
- **No integration tests** — no end-to-end simulation-to-GUI pipeline tests

---

## 8. UI/UX Assessment

### 8.1 Strong Points
- Beautiful dark theme with trace-style plots (the signature aesthetic)
- Interactive zoom/pan with mouse (scroll wheel, drag, double-click reset)
- Overlay controls (force vectors, ellipsoids) with scale sliders
- Toolstrip with persistent controls above scrollable content
- QSettings persistence for geometry and splitters
- Preset system with named configurations
- Real-time mass matrix visualization with color-coded coupling

### 8.2 Issues
- **controls_widget_triple.py missing Function Generator button**: Double pendulum controls have it, triple does not.
- **No keyboard shortcuts**: No hotkeys for Run, Reset, Play/Pause, Speed changes.
- **Status bar underutilized**: Shows "Ready" message but doesn't update during simulation.
- **No undo/redo for parameter changes**: Common modern UX expectation.
- **Web version is significantly feature-poor** relative to PyQt6 version.

---

## 9. Code Quality

### 9.1 Import Style Inconsistency
- `physics.py` uses `from typing import Callable, Tuple` (deprecated in 3.10+)
- `physics_triple.py` uses `from collections.abc import Callable` (modern)
- Should standardize on modern `collections.abc` and builtin `tuple`

### 9.2 Type Ignore Comments
`main_window.py` has 5x `# type: ignore[arg-type]` suggesting interface mismatches between SimulationPanel constructor and the widgets passed to it.

### 9.3 Inconsistent Naming
- Double: `phi` (single relative angle)
- Triple: `phi1, phi2` (two relative angles)
- Controls: `phi_rad` vs `phi1_rad, phi2_rad`
This is acceptable but could be more explicit.

---

## 10. Priority Fix List

| Priority | Issue | Files Affected |
|----------|-------|----------------|
| P0 | Add friction to triple pendulum | physics_triple.py, simulation_triple.py |
| P0 | Add friction accessors to TripleSimulationResult | simulation_triple.py |
| P1 | Extract shared polynomial torque builder (DRY) | simulation.py, simulation_triple.py |
| P1 | Add elbow torque import to FunctionGeneratorDialog | function_generator_dialog.py |
| P1 | Integrate PlotThemeManager for pyqtgraph plots | torque_history_widget.py |
| P1 | Replace hardcoded styles with theme-derived colors | controls_utils.py, function_generator_dialog.py |
| P2 | Modernize imports (typing → builtins) | physics.py, simulation.py |
| P2 | Add DbC postconditions to gravity/friction functions | physics.py, physics_triple.py |
| P2 | Add tests for triple pendulum friction | tests/test_friction.py |
| P2 | Fix fragile sys.path manipulation | main_window.py, function_generator_dialog.py |
| P3 | Add Function Generator button to triple controls | controls_widget_triple.py |
| P3 | Resolve type: ignore comments | main_window.py, simulation_panel.py |

---

## 11. Fixes Applied

### P0 — Critical Parity Fixes
- **✅ Triple pendulum friction model**: Added complete viscous damping (b1, b2, b3) and Coulomb friction (mu1, mu2, mu3) to `physics_triple.py` and `simulation_triple.py`, achieving full parity with double pendulum friction implementation.
- **✅ Triple pendulum friction accessors**: Added `friction_torques_at()` and `total_torques_at()` methods to `TripleSimulationResult`.
- **✅ Triple pendulum dissipation UI**: Added dissipation section (b1, b2, b3, μ1, μ2, μ3) to `controls_widget_triple.py` and wired through `main_window.py` build_params.
- **✅ Function Generator for triple**: Added "Function Generator…" button to `controls_widget_triple.py` with shoulder/elbow/wrist import support.
- **✅ Elbow import button**: Added "Import → Elbow Torque" to `function_generator_dialog.py`.

### P1 — Design Quality
- **✅ DRY style constants**: Consolidated duplicate STYLE_BTN and STYLE_BTN_IMPORT into `controls_utils.py`; removed duplicates from `function_generator_dialog.py`.
- **✅ DbC postconditions**: Added finiteness assertions to `gravity_vector()`, `coriolis_vector()`, and `friction_torque_vector()` in both physics modules.
- **✅ Modern imports**: Updated `physics.py` and `simulation.py` from deprecated `typing.Callable/Tuple/Optional` to `collections.abc.Callable`, `tuple[...]`, and `X | None`.
- **✅ Fragile sys.path**: Replaced `Path(__file__).parents[7]` with upward-walking directory search in `main_window.py` and `function_generator_dialog.py`.

### P1 — Theme Integration
- **✅ PlotThemeManager integration**: `torque_history_widget.py` now sources background/text/grid colors from the shared `PlotThemeManager` while preserving signature trace colors.
- **✅ Plot Theme menu**: Added Plot Theme submenu to View menu in `main_window.py` via `create_plot_theme_menu()`.
- **✅ Theme change callback**: `TorqueHistoryWidget` registers for `_on_plot_theme_changed` to update pyqtgraph backgrounds dynamically.

### P2 — Testing
- **✅ Triple friction tests**: Created `tests/test_friction_triple.py` with 22 new tests covering parameter contracts, friction torque computation, EOM integration with dissipation, and SimulationResult friction accessors.
- **✅ All 156 tests passing** (up from 134 original).

### Plot Appearance Preservation
- **✅ Trace colours preserved**: All six signature torque history colours (warm orange, cool blue, red, teal, gold, pale green) are unchanged.
- **✅ Torque preview colours preserved**: Shoulder (230, 120, 50), Elbow (120, 200, 140), Wrist (120, 180, 230) in both control widgets.

### Remaining (Future Work)
- Web version (TypeScript/React) parity gap remains significant (single pendulum only, no friction, no theme system, RK4 vs adaptive).
- `_PENDULUM_DARK_STYLE` in main_window.py remains as the fallback dark theme (functional but not derived from theme system colors).
- GUI widget tests (PyQt6 interaction tests) not yet written.
