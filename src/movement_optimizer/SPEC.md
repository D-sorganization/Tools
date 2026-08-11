# SPEC.md -- Movement-Optimizer Repository Specification

## 1. Identity

| Field            | Value                                                   |
| ---------------- | ------------------------------------------------------- |
| Repository Name  | `Movement-Optimizer`                                    |
| GitHub URL       | `https://github.com/D-sorganization/Movement-Optimizer` |
| Owner            | D-sorganization                                         |
| Primary Language | Python 3.10+                                            |
| License          | MIT                                                     |
| Package Name     | `movement-optimizer`                                    |
| Current Version  | `1.0.0`                                                 |
| Spec Version     | `1.0.37`                                                |
| Last Spec Update | 2026-08-10                                              |

## 2. Purpose

Movement-Optimizer is a biomechanics trajectory optimizer for barbell exercises. It models the body as a sagittal-plane planar chain, computes trajectories with Lagrangian inverse dynamics, and exposes both a GUI workflow and a headless CLI for batch optimisation.

## 3. Scope

### 2026-06-18 Update

- Animation playback helpers now declare the small mixin contract they need
  instead of annotating every method as `MainWindow`, removing module-level
  mypy suppression and per-method override ignores while preserving the
  existing Qt/MainWindow dispatch path.
- Optimization controller helpers now declare their own mixin contract for
  optimizer state, Qt signals, sidebar/control facades, and playback handoff
  methods instead of annotating every method as `MainWindow`, removing the
  module-level mypy suppression and the remaining QMessageBox arg-type ignore.
- Sidebar builder and state helpers now use declared `ParameterSidebar` and
  protocol contracts for dynamically created widgets, progress labels,
  convergence plots, and body-model sliders instead of implicit untyped
  sidebar access.
- Swingset and Chain Dynamics analysis legends are docked into reserved
  legend rows owned by `MotionAnalysisPanel`, with a larger data-to-legend
  gap, compact multi-column legends, and minimum scrollable plot-panel sizing
  so visible legends identify the plotted series without covering plot data,
  neighboring subplots, axis labels, or figure edges on compact panes.
- Swingset and Chain Dynamics analysis panels collect each plot's labeled
  series into that plot's reserved legend row instead of using one shared
  figure-level footer, so each legend remains visible and local to its plot
  without covering or crowding plot contents.
- Compact plot panes reserve enough vertical gap between data axes and docked
  legend rows for x-axis labels, y-axis labels, and titles to remain readable
  without being obscured by legends.
- Docked Swingset and Chain Dynamics plot legends are centered inside taller
  reserved legend rows, with rendered padding coverage so multi-row joint
  legends cannot crowd the plots or appear clipped on compact panes.
- Docked Swingset and Chain Dynamics plot legends now clip the legend container
  and every child artist to the reserved legend row, preventing renderer- or
  backend-specific text/handle overflow from painting over plot contents.
- Swingset and Chain Dynamics analysis panels now require wider scrollable plot
  columns and anchor dense docked legends to the full reserved legend strip, so
  the live Swingset `3 x 2` plot layout preserves usable data-axis width while
  legends remain outside plotted curves.
- Swingset and Chain Dynamics analysis legends now dock into dedicated bands
  beneath each plot instead of narrow side strips, preserving full plot width
  while keeping legends clear of curves, tick labels, and axis labels.
- Swingset and Chain Dynamics panel renderers now suppress transient data-axis
  legends while preserving series labels for `MotionAnalysisPanel` to dock in
  reserved legend rows, so visible legends cannot appear over the plotted
  curves between plot rendering and panel draw.
- Rendered `MotionAnalysisPanel` legend layout regressions now live in
  `tests/test_motion_analysis_panel_legends.py`, keeping the broad panel test
  module under the changed-file size budget while retaining the same
  no-overlap pixel assertions.
- `MotionAnalysisPanel.draw()` enforces the panel's legend-safe minimum figure
  dimensions before docking legends, so compact split-pane or backend resize
  paths cannot squeeze legend strips into axis labels or plotted curves.
- The shared exercise/swing plot-grid legend helper now anchors legends lower
  and the exercise analysis grid reserves a larger vertical inter-row band, so
  legends clear axis tick labels and x/y labels instead of merely clearing the
  plotted data rectangle.
- The Swingset policy optimization trace legend now measures and wraps entries
  by widget width, and the trace series uses the wrapped legend height as its
  top inset so the legend does not obscure optimizer telemetry in narrow panes.
- The Swingset policy optimization trace canvas now exposes a width-aware
  minimum height that includes wrapped legend rows plus a readable telemetry
  band, preventing narrow panes from compressing the plot under the legend.

### 2026-06-17 Update

- The canonical Tools implementation now adds a shared bottom-bar Autoplay
  control for completed barbell optimizations, while retaining the Swingset
  and Chain Dynamics tab-level autoplay controls for local simulations.
- Chain Dynamics gravity response now accounts for downstream link load and
  effective inertia, so top joints in a multi-link chain no longer accelerate
  like independent single rods; regression tests pin the single-link slender
  rod case and multi-link downstream-load scaling.
- Swingset force recovery now differentiates center-of-mass position twice to
  estimate acceleration before recovering chain tension, avoiding velocity-as-
  acceleration force artifacts.
- The Swingset optimizer command is kept as a larger sticky primary action
  above the scrollable settings panel so policy optimization remains visually
  prominent.
- Motion-tab slider/text controls and scroll-panel construction now live in
  `movement_optimizer.gui.motion_controls`, keeping the tab modules under the
  enforced source-size budget without changing the Swingset or Chain Dynamics
  interaction contract.

### 2026-06-16 Update

- Swingset policy search now precomputes cyclic control matrices with vectorized
  NumPy trigonometry before the necessarily sequential state rollout, reducing
  optimizer callback overhead without changing policy values.
- Chain Dynamics now uses torque-based bend stiffness/damping over rod-link
  inertia, validates single-link gravity against a slender pendulum, and
  initializes kick velocity as a tip-weighted profile instead of a mid-chain
  sine wave.
- Swingset and Chain Dynamics analysis tabs include default-on autoplay
  checkboxes so optimized or simulated motion starts playing as soon as results
  are ready.
- Analysis-tab playback now has explicit responsiveness contracts: switching
  from a playing barbell exercise into Swingset Model or Chain Dynamics stops
  the barbell animation timer before the shared playback controls retarget.
- Swingset and chain overlays cache rollout-wide force fields, avoiding
  repeated finite-difference and torque recomputation during every animation
  frame.
- Numeric sliders no longer emit continuous drag-time refreshes, and the
  Swingset optimizer action is styled as the primary command.
- The Swingset and Chain Dynamics tabs now expose per-element animation
  visibility: a "Show in animation" checklist toggles each MotionCanvas
  layer (grid/chain/rider/markers/forces) independently, on top of the
  existing force-vector filters.
- Each tab splits into Animation and Plots sub-tabs so the analysis plots
  get a roomy dedicated area; a "Show plot legends" control and a
  toggleable, top-strip-reserving policy-trace legend keep legends from
  obscuring the plotted curves.

### In Scope

- Sagittal-plane movement optimisation for barbell exercises
- Body and dynamics modelling
- Trajectory optimisation and result persistence
- GUI visualization and comparison tooling
- Export helpers for plots and animation artifacts
- Optional Rust acceleration in `rust_core/`

### Out Of Scope

- Full 3D biomechanics simulation
- Networked services or remote orchestration
- Non-barbell exercise domains unless they fit the current factory model

## 4. Architecture

### Package Layout

```text
src/movement_optimizer/
├── __main__.py          # GUI entrypoint for `python -m movement_optimizer`
├── cli.py               # Headless batch CLI
├── backend.py           # Physics backend interface
├── config.py            # Runtime configuration and state paths
├── constants.py         # Physical constants and tuning values
├── comparison.py        # Trial comparison helpers
├── export.py            # CSV/PNG/PDF/GIF export helpers
├── persistence.py       # JSON save/load for sessions and results
├── rendering.py         # Matplotlib rendering helpers
├── spine_loads.py       # Spine load analysis
├── strength.py          # Torque and load-capacity helpers
├── models/              # Body model and Lagrangian dynamics
├── exercises/           # Exercise configuration factories
├── trajectory/          # Optimizer, cache, result, tuning types
└── gui/                 # PyQt6 windows, tabs, widgets, and dialogs
rust_core/                # Optional PyO3/maturin hot path accelerator
tests/                   # Pytest suite
```

### Key Boundaries

- `models/` owns body geometry, exercise configs, and analytical dynamics.
- `trajectory/` owns optimisation orchestration, cache handling, and result types.
- `exercises/` owns exercise-specific configuration factories.
- `gui/` owns all PyQt6 presentation and interaction code.
- `cli.py` owns the headless batch interface and JSON output shaping.
- `__main__.py` owns the GUI startup path.
- Sidebar and playback GUI widgets expose facade methods for state changes,
  signal binding, and summary values so main-window mixins do not traverse into
  child widget internals.

## 5. Entry Points

- `movement-optimizer` console script maps to `movement_optimizer.__main__:main`.
- `python -m movement_optimizer` launches the GUI.
- `python -m movement_optimizer.cli` runs headless optimisation.
- `run.py` and the platform launch scripts are convenience wrappers around the package entrypoints.

## 6. Runtime Contract

- The default body model is a 3-link planar sagittal chain.
- `models/` provides the primary squat, full squat, deadlift, and bench press configuration factories alongside the body and dynamics types.
- `exercises/` provides supplemental factories for clean, jerk, snatch, gait, and sit-to-stand flows.
- Optimisation uses multi-start search and SciPy-based solvers.
- GUI state is stored locally and does not require external services.
- Optional Rust acceleration is an implementation detail, not a hard dependency.
  When the compiled `rust_core` extension is absent the dynamics fall back to an
  equivalent NumPy path, so results are identical and only performance differs.

## 7. Data And Configuration

### Inputs

- Body parameters such as mass, height, and segment multipliers
- Barbell mass and exercise-specific configuration values
- Optional runtime state directory via `MOVEMENT_OPTIMIZER_STATE_DIR`

### Outputs

- Optimisation summaries and detailed JSON results
- Matplotlib figures and exported plots
- GIF/PNG/PDF artifacts
- Persisted session state

## 8. Testing And CI

### Test Strategy

- `pytest` is the canonical test framework.
- Tests live in `tests/` and use shared fixtures from `tests/conftest.py`.
- Unit tests should cover model, trajectory, GUI helper, and export behavior.
- Property-based tests use Hypothesis where parameter-space coverage matters.

### Canonical Commands

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=movement_optimizer --cov-report=term-missing
ruff check src/ tests/
ruff format src/ tests/
mypy --ignore-missing-imports src/movement_optimizer/
```

### Quality Expectations

- Public APIs must be type-hinted.
- Preconditions should be checked early with `ValueError` or `TypeError`.
- `src/` code should use logging rather than `print`.
- Tests should remain deterministic and avoid network access.

## 9. Change Log

| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-10 | 1.0.37  | Compacted `gui/motion_tabs.py` comments, docstrings, and Ruff call layout from 1,216 to 1,152 lines without changing runtime behavior, restoring compliance with the monorepo's 1,200-line protected module budget; 80 focused motion-tab, theming, and vector-overlay tests pass.                                                                                                                           |
| 2026-06-19 | 1.0.36  | Split rendered MotionAnalysisPanel legend layout regressions into `tests/test_motion_analysis_panel_legends.py`, preserving the Swingset/Chain no-overlap assertions while keeping changed test files below the size budget.                                                                                                                                                                                  |
| 2026-06-19 | 1.0.35  | Replaced per-plot Swingset and Chain Dynamics analysis legend strips with one reserved figure-level legend footer, reducing vertical clutter while preserving rendered regression coverage that the legend cannot overlap plot data, tick labels, or axis labels.                                                                                                                                             |
| 2026-06-19 | 1.0.33  | Moved Swingset and Chain Dynamics analysis legends into right-docked per-plot strips with wider scrollable plot columns, so legends remain visible beside each plot instead of sitting between rows or covering plotted curves and neighboring axis labels.                                                                                                                                                   |
| 2026-06-19 | 1.0.32  | Widened scrollable Swingset and Chain Dynamics analysis plot columns and anchored dense docked legends to their full reserved strips, so the live Swingset 3x2 plot layout keeps usable data-axis width without legend overlap.                                                                                                                                                                               |
| 2026-06-19 | 1.0.29  | Enforced `MotionAnalysisPanel`'s legend-safe minimum render dimensions during draw before docking Swingset and Chain Dynamics legends, so compact panes and backend resize edge cases cannot squeeze legends into axis labels, tick labels, titles, or plotted curves.                                                                                                                                        |
| 2026-06-19 | 1.0.28  | Rendered Swingset and Chain Dynamics panel plots without transient data-axis legends while preserving labeled artists for `MotionAnalysisPanel` docked legend strips, so visible legends cannot obscure plot contents between renderer and panel draw steps.                                                                                                                                                  |
| 2026-06-19 | 1.0.27  | Centered docked Swingset and Chain Dynamics plot legends inside taller reserved legend rows and added rendered padding coverage, so multi-row joint legends cannot crowd plots or appear clipped on compact panes.                                                                                                                                                                                            |
| 2026-06-19 | 1.0.26  | Reserved a bottom axis-label band in the Swingset policy optimization trace canvas and included that band in the width-aware minimum-height contract, so the `iteration` label stays below the plotted optimizer traces instead of overlaying telemetry at narrow widths.                                                                                                                                     |
| 2026-06-19 | 1.0.25  | Made the Swingset policy optimization trace canvas advertise a width-aware minimum height from its wrapped legend rows plus a minimum telemetry band, so narrow optimizer panes scroll or expand instead of letting the legend dominate the plot area.                                                                                                                                                        |
| 2026-06-18 | 1.0.24  | Prevented compact Swingset and Chain Dynamics plot panes from crushing docked legends back into the plots by giving `MotionAnalysisPanel` a grid-aware minimum canvas size, hosting plot panels in scroll areas, and using up to three docked legend columns for joint-series plots.                                                                                                                          |
| 2026-06-18 | 1.0.22  | Typed sidebar builder/state helper surfaces against declared `ParameterSidebar` and protocol contracts, covering progress widgets, result widgets, convergence plots, and body-model sliders while preserving existing sidebar behavior.                                                                                                                                                                      |
| 2026-06-18 | 1.0.21  | Declared the optimization controller mixin's narrow MainWindow contract in `optimization_mixin.py`, removing module-level mypy suppression and the remaining QMessageBox arg-type ignore while preserving optimizer worker, signal, and autoplay behavior.                                                                                                                                                    |
| 2026-06-18 | 1.0.20  | Declared the animation playback mixin's narrow MainWindow contract in `animation_control.py`, removing module-level mypy suppression and per-method override ignores while preserving runtime dispatch through the concrete `MainWindow` wrappers.                                                                                                                                                            |
| 2026-06-18 | 1.0.19  | Wrapped the Swingset policy optimization trace legend by measured widget width and derives the trace top inset from the wrapped legend band, so optimizer telemetry lines start below all legend rows even in narrow panes. Added regression coverage for the narrow-width legend layout and updated the trace legend tooltip to describe the above-plot behavior.                                            |
| 2026-06-18 | 1.0.17  | Replaced below-axis Swingset and Chain Dynamics analysis legends with `MotionAnalysisPanel`-owned legend rows; legends are removed from data axes and rebuilt in reserved strips during panel draw, with rendered-bounding-box regression tests proving they stay inside the figure and do not overlap any plot axis.                                                                                         |
| 2026-06-18 | 1.0.15  | Anchored Swingset and Chain Dynamics analysis legends below each subplot via a shared renderer helper so visible legends identify torque, power, angle, COM, energy, tension, curvature, and tip-speed series without obscuring plotted data; added regression tests for the outside-plot legend contract.                                                                                                    |
| 2026-06-16 | 1.0.14  | Added per-element animation layer toggles (grid/chain/rider/markers/forces) to the Swingset and Chain Dynamics tabs via a shared `_MotionViewMixin`, split each tab into Animation/Plots sub-tabs, and made plot/policy-trace legends toggleable so they no longer obscure the plotted data. `MotionAnalysisPanel.set_legends_visible`/`has_legends` encapsulate legend control (LoD).                        |
| 2026-06-15 | 1.0.12  | Lifted the legacy `scipy<1.16` ceiling after verifying current SciPy imports `CubicSpline` cleanly, and added a dependency-contract regression so the stale cap cannot return silently.                                                                                                                                                                                                                       |
| 2026-05-16 | 1.0.11  | Isolated nightly workflow installs into a dedicated .nightly-venv virtual environment with PIP_NO_CACHE_DIR=1 to avoid shared runner cache corruption that caused ImportError: cannot import name '\_spropack' from scipy.sparse.linalg.\_propack (#462).                                                                                                                                                     |
| 2026-04-22 | 1.0.10  | Added GUI sidebar/playback facade methods and routed main-window mixins through them to reduce deep object traversal in animation, comparison, cancellation, and signal binding code (#272).                                                                                                                                                                                                                  |
| 2026-04-16 | 1.0.9   | Extracted spline-building responsibility from `TrajectoryOptimizer` into `optimizer_spline.py` (`build_splines`, `eval_trajectory`); extracted `_compute_bench_bar_cost` private helper from `_compute_cost`; exported new functions from `trajectory/__init__.py`; added 14 characterization/unit tests in `test_issue_247_split_optimizer.py` (#247).                                                       |
| 2026-04-14 | 1.0.7   | Split `gui/widgets.py` (489 LOC) into three focused modules (`labelled_slider.py`, `parameter_sidebar.py`, `playback_controls.py`) and decomposed `models/lagrangian_dynamics.py` (463 LOC) by extracting `LagrangianKinematicsMixin` into `lagrangian_kinematics.py` and balance helpers into `lagrangian_balance.py`. Each resulting module is ≤300 LOC; `widgets.py` becomes a thin re-export shim (#218). |
| 2026-04-14 | 1.0.6   | Added NaN/infinite input validation to `HillTorqueModel` constructor and key methods (`torque_angle_factor`, `torque_velocity_factor`, `available_torque`). All seven numeric constructor parameters are now checked with `math.isfinite`; NaN or infinite values raise `ValueError` immediately rather than propagating silently (#236).                                                                     |
| 2026-04-11 | 1.0.5   | Split `tests/test_trajectory.py` (678 LOC) into three focused modules — `test_trajectory_generation.py`, `test_trajectory_optimization.py`, and `test_trajectory_validation.py` — and promoted the `squat_optimizer` / `full_squat_optimizer` fixtures to `conftest.py` for shared reuse (#211).                                                                                                              |
| 2026-04-11 | 1.0.4   | Decomposed `TrajectoryOptimizer.optimize()` and `_package_results()` into thin orchestrators backed by focused helpers (`_optimize_single_start`, `_optimize_parallel_starts`, `_collect_future_results`, `_finalize_parallel_results`, `_evaluate_solution`, `_validate_solution`, `_build_result_object`) to satisfy the Function Size target (#214).                                                       |
| 2026-04-11 | 1.0.3   | Added a stable public API to `ProgressTracker` (`cost_history`, `iteration_count`, `elapsed()`, `lock()`) and refactored `TrajectoryOptimizer` to stop reaching into its private attributes, eliminating a cluster of Law-of-Demeter violations in the optimiser engine.                                                                                                                                      |
| 2026-04-10 | 1.0.2   | Replaced the last `print()` call in `src/` with direct stdout JSON emission in the CLI summary path and updated the CLI regression test to preserve the headless output contract without violating the no-print rule.                                                                                                                                                                                         |
| 2026-04-09 | 1.0.1   | Added a shared provider-pack manifest, validator, regression tests, and launcher icon asset so Movement-Optimizer can publish a launcher-compatible utility pack without embedding UpstreamDrift-specific path logic.                                                                                                                                                                                         |
| 2026-04-06 | 1.0.0   | Initial repository specification aligned to the current package layout, entrypoints, and test contract.                                                                                                                                                                                                                                                                                                       |
