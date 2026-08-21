# Pendulum Simulator — Feature Tracking

> **Purpose**: Single-source-of-truth for desired vs. implemented features.
> Update this document whenever a feature is proposed, started, or completed.

---

## Status Legend

| Icon               | Meaning                        |
| ------------------ | ------------------------------ |
| :white_check_mark: | Implemented and tested         |
| :construction:     | In progress                    |
| :clipboard:        | Planned / GitHub issue created |
| :bulb:             | Idea (not yet scoped)          |

---

## 1 Physics Models

| Feature                                  | Status             | Notes                                                      |
| ---------------------------------------- | ------------------ | ---------------------------------------------------------- |
| Double pendulum (2-DOF)                  | :white_check_mark: | Shoulder + wrist, clubhead point mass                      |
| Triple pendulum (3-DOF)                  | :white_check_mark: | Three cascading segments                                   |
| Golfer upper body (8-DOF, 4 constraints) | :white_check_mark: | Hub + bilateral arms + club, closed loop                   |
| Mass matrix M(q)                         | :white_check_mark: | All three models                                           |
| Coriolis vector C(q, qdot)               | :white_check_mark: | All three models                                           |
| Gravity vector G(q)                      | :white_check_mark: | All three models                                           |
| Viscous friction                         | :white_check_mark: | Per-joint damping coefficients                             |
| Joint torque limits                      | :white_check_mark: | Clamped torque bounds, all models                          |
| Constraint Jacobian (golfer)             | :white_check_mark: | 4x8 closed-loop constraint matrix                          |
| KKT constrained dynamics                 | :white_check_mark: | Baumgarte stabilisation (alpha, beta)                      |
| Constraint projection                    | :white_check_mark: | Position + velocity manifold projection                    |
| **Massless hub standoff**                | :white_check_mark: | `hub_options.py` — epsilon mass, PSD mass matrix verified  |
| **Adjustable rotational centre**         | :white_check_mark: | `hub_options.compute_system_com()`, `hub_offset_for_com()` |

## 2 Force & Moment Analysis

| Feature                                      | Status             | Notes                                                                     |
| -------------------------------------------- | ------------------ | ------------------------------------------------------------------------- |
| Net joint forces (F = ma - mg)               | :white_check_mark: | Per-joint, all segments                                                   |
| Constraint (Lagrange) forces                 | :white_check_mark: | Golfer model via KKT multipliers                                          |
| Geometric Jacobian                           | :white_check_mark: | Double, triple, golfer                                                    |
| ZTCF transfer matrix T                       | :white_check_mark: | (J M+ JT)^-1 J M+                                                         |
| DELTA matrix (M pseudoinverse)               | :white_check_mark: | Zero-velocity torque-to-accel map                                         |
| Zero-torque counterfactual forces            | :white_check_mark: | Double, triple, golfer                                                    |
| Zero-torque accelerations                    | :white_check_mark: | All models                                                                |
| Manipulability ellipsoids (mobility + force) | :white_check_mark: | SVD-based, GUI overlay                                                    |
| Manipulability index w = sqrt(det(J JT))     | :white_check_mark: | Numerical + analytical                                                    |
| **Net force & equivalent couple on club**    | :white_check_mark: | `club_forces.py` — sum of two-hand forces at user-configurable club point |
| **Moment of net force on club**              | :white_check_mark: | `club_forces.moment_of_net_force()` — 2D cross product                    |
| **Equivalent couple (two-hand action)**      | :white_check_mark: | `club_forces.equivalent_couple()` — couple matching two-hand action       |
| **ZTCF net force & couple decomposition**    | :white_check_mark: | `club_forces.ztcf_club_decomposition()`                                   |
| **DELTA net force & couple decomposition**   | :white_check_mark: | `club_forces.delta_club_decomposition()`                                  |

## 3 GUI & Visualisation

| Feature                                        | Status             | Notes                                                                 |
| ---------------------------------------------- | ------------------ | --------------------------------------------------------------------- |
| PyQt6 main window with dock architecture       | :white_check_mark: | Resizable, dockable panels                                            |
| Double / triple / golfer control panels        | :white_check_mark: | NoScroll spinboxes, UnitAwareInput                                    |
| Matrix display widgets (M, C, G)               | :white_check_mark: | Real-time, copy to clipboard                                          |
| Interactive 2D animation                       | :white_check_mark: | Zoom, pan, trails, markers                                            |
| Pseudo-3D rendering mode                       | :white_check_mark: | Isometric projection, tilt + azimuth                                  |
| Force vector overlays                          | :white_check_mark: | Net forces (lime), ZTCF (violet), ellipsoids (blue/orange)            |
| Tip trail with Catmull-Rom smoothing           | :white_check_mark: | Configurable history length                                           |
| Grid overlay (major/minor ticks)               | :white_check_mark: | Dark-theme compatible                                                 |
| Theme system (dark/light)                      | :white_check_mark: | ThemeManager from shared/                                             |
| Plot theme integration                         | :white_check_mark: | Matplotlib style sheets                                               |
| Settings persistence (QSettings)               | :white_check_mark: | Geometry, splitter, font zoom                                         |
| Keyboard shortcuts                             | :white_check_mark: | Ctrl+scroll zoom, Ctrl+E export, etc.                                 |
| Unit converter sidebar                         | :white_check_mark: | Metric / Imperial                                                     |
| **3D segment rendering (ellipses, cylinders)** | :white_check_mark: | `segment_geometry.py` — SegmentStyle enum, cross-sections, depth sort |

## 4 Analysis & Plotting

| Feature                                        | Status             | Notes                                                                       |
| ---------------------------------------------- | ------------------ | --------------------------------------------------------------------------- |
| Docked analysis panel (2D + 3D)                | :white_check_mark: | QDockWidget, model-aware                                                    |
| Data extractor registry (25+ series)           | :white_check_mark: | Dispatch-table pattern                                                      |
| 2D line plots (any X vs Y)                     | :white_check_mark: | Dark-themed matplotlib                                                      |
| 3D surface plots (parameter sweep)             | :white_check_mark: | Model-aware sweep variables                                                 |
| Surface evaluators (det, cond, PE, w)          | :white_check_mark: | Double, triple, golfer                                                      |
| Polynomial regression overlay (deg 0-10)       | :white_check_mark: | Pop-out chart + analysis tab                                                |
| Pop-out chart window                           | :white_check_mark: | Non-modal, independent                                                      |
| Equations popup (physics formulas)             | :white_check_mark: | Model-specific HTML rendering                                               |
| Jacobian equations topic                       | :white_check_mark: | Manipulability, ellipsoids                                                  |
| Constraint Jacobian topic                      | :white_check_mark: | KKT, Lagrange multipliers                                                   |
| **Random perturbation / consistency analysis** | :white_check_mark: | `perturbation_analysis.py` — Monte Carlo noise injection, variability stats |
| **Phase-resolved drift-transfer diagnostics**  | :white_check_mark: | Exact grip-force closure, drift/control work, braking work, along-path impulse, peak force, and Pareto metrics for the double-pendulum tier |
| **Drift Transfer GUI tab**                     | :white_check_mark: | User-declared time window with power, speed, work, braking, and model-boundary display; unsupported tiers fail closed |
| **Qualified Rotating-Base Study**              | :white_check_mark: | Source-pinned 18-case provider; async PyQt execution; digest-pinned React/Tauri scalar/trace parity; adverse rows, killswitches, closures, five reviewer charts, and governed full-run JSON exports |

## 5 Signal Generation & Torque Profiles

| Feature                                          | Status             | Notes                               |
| ------------------------------------------------ | ------------------ | ----------------------------------- |
| Polynomial torque builder (N-joint)              | :white_check_mark: | Generic, per-joint coefficients     |
| Polynomial generator widget (draw/type)          | :white_check_mark: | Freehand + equation input           |
| Signal toolkit waveforms (sin, square, tri, saw) | :white_check_mark: | Configurable freq, amplitude, phase |
| Noise injection (white, pink, brown)             | :white_check_mark: | Via signal toolkit                  |
| Digital filtering (LP, HP, BP)                   | :white_check_mark: | Butterworth filters                 |
| Function generator dialog (dual-tab)             | :white_check_mark: | Polynomial + signal tabs            |

## 6 Optimisation

| Feature                                         | Status             | Notes                         |
| ----------------------------------------------- | ------------------ | ----------------------------- |
| CMA-ES (Rust + Python)                          | :white_check_mark: | Population-based evolutionary |
| Batch evaluation (Rust rayon)                   | :white_check_mark: | Parallel candidate eval       |
| Warm-start strategy                             | :white_check_mark: | Reuse previous best           |
| Multi-objective (speed, efficiency, smoothness) | :white_check_mark: | Configurable weights          |
| Joint limit enforcement                         | :white_check_mark: | Hard bounds                   |
| Torque bound enforcement                        | :white_check_mark: | Clamp to limits               |
| Progress bar + convergence detection            | :white_check_mark: | Real-time GUI feedback        |
| Early stopping (fitness tolerance)              | :white_check_mark: | Configurable plateau          |

## 7 Integration & Backends

| Feature                    | Status             | Notes                      |
| -------------------------- | ------------------ | -------------------------- |
| scipy.integrate.solve_ivp  | :white_check_mark: | RK45, DOP853               |
| Adaptive time stepping     | :white_check_mark: | rtol / atol control        |
| Rust native backend (PyO3) | :white_check_mark: | All three models           |
| Automatic Python fallback  | :white_check_mark: | If Rust unavailable        |
| JAX backend (experimental) | :white_check_mark: | GPU-capable golfer physics |
| WASM compilation support   | :white_check_mark: | Browser deployment         |
| Model registry pattern     | :white_check_mark: | Dynamic model lookup       |

## 8 Data Export

| Feature                      | Status             | Notes               |
| ---------------------------- | ------------------ | ------------------- |
| Plot export (PNG, SVG, PDF)  | :white_check_mark: | Matplotlib savefig  |
| Trajectory CSV / HDF5 export | :white_check_mark: | Via pandas          |
| Configuration export         | :white_check_mark: | Serialisable params |
| Clipboard (matrix, params)   | :white_check_mark: | Copy/paste support  |

## 9 Testing & Quality

| Feature                       | Status             | Notes                                   |
| ----------------------------- | ------------------ | --------------------------------------- |
| pytest suite (590+ tests)     | :white_check_mark: | Headless-compatible                     |
| CI/CD (GitHub Actions)        | :white_check_mark: | Ruff + mypy + pytest, Py 3.10-3.12      |
| Ruff linting (clean)          | :white_check_mark: | Zero warnings                           |
| Design by Contract assertions | :white_check_mark: | Preconditions, postconditions           |
| DRY compliance                | :white_check_mark: | Shared base classes, mixins, registries |
| TDD workflow                  | :white_check_mark: | Tests first for all new features        |

---

## Planned Features (Backlog)

### P1 — Net Force & Equivalent Couple on Club

**GitHub Issue**: TBD
Calculate the resultant of the two hand forces on the club at a user-configurable
action point (default: midpoint between grip_right and grip_left). Decompose
into net force vector, moment of net force, and equivalent couple. Provide this
decomposition for overall forces, ZTCF forces, and DELTA forces. Display as
vector overlays on the animation widget.

### P2 — Random Perturbation / Consistency Analysis

**GitHub Issue**: TBD
Add noise to all joint torque profiles using the signal generator, run N
Monte Carlo simulations, and statistically analyse the variability of velocity
and position outputs. Identify swing patterns that are more/less consistent.
GUI: batch run dialog, summary statistics table, variability plots.

### P3 — Massless Hub Standoff & Adjustable Rotation Centre

**GitHub Issue**: TBD
Make the hub standoff (connecting rotation centre to scapula origin) effectively
massless in the Lagrangian. Add option to position the rotation centre at the
system centre of mass, yielding zero hub reaction force in the zero-gravity case.
Provide user controls for manual hub origin offset.

### P4 — 3D Segment Rendering Options

**GitHub Issue**: TBD
Extend the rendering pipeline to support multiple segment representations:
3D ellipses (inertia-scaled), cylinders (constant cross-section), tapered
cylinders, and the current line default. User selects via dropdown or per-segment
override. Requires depth sorting for correct occlusion.

### P5 — Feature Tracking Document

**GitHub Issue**: N/A (this document)
This file serves as the living feature tracker. Keep it updated as features
are proposed, implemented, and tested.

---

_Last updated: 2026-03-12_
