# Changelog

All notable changes to the Pendulum Simulator will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Qualified rotating-base two-hand compliant-club study with a single packaged
  18-case authority, asynchronous PyQt execution, React/Tauri evidence parity,
  a digest-pinned full-resolution 18-run trace catalog, five reviewer chart
  groups, retained adverse rows, exact same-state killswitches, closure
  diagnostics, and source-pinned governed JSON exports (#4430)
- Centralized theme module (`gui/theme.py`) for DRY stylesheet management (#1197)
- Advanced diagnostics tracker with JSONL persistence and viewer dialog
- Golfer topology validation tests — 18 tests covering mass distribution, FK, and DbC contracts (#1204)
- Mouse wheel blocking on all input widgets to prevent accidental value changes (#1193)
- Playback controls: slider scrubbing, loop toggle, play/pause state management
- Animation trail cleanup on slider interaction
- **Torque Vectors** checkbox in toolbar overlay — red arrows at each joint (#1208)
- **Moment of Force** checkbox in toolbar overlay — blue arrows, proximal-on-distal (#1208)
- **Sum of Moments** checkbox in toolbar overlay — green resultant arrows (#1208)
- Playback slider now has "Playback:" label, 200px min-width, 10px groove, glowing handle (#1207)
- 21 new toolstrip element tests verifying slider, checkboxes, and signals permanently exist
- `--version` CLI flag for quick version check (#1201)
- About dialog (Help → About) with version and credits (#1206)
- Keyboard shortcuts: Ctrl+R (run), Space (play/pause), Ctrl+E (export), Escape (reset) (#1206)

### Changed

- Golfer model topology corrected: standoff is now massless (0.001 kg default), upper body segments carry ~7 kg each (#1195)
- UI labels updated: Hub → Standoff, R/L Scap → R/L UBody
- Unit dropdown boxes standardized to fixed width (60px)
- Signal toolkit error messages are now selectable/copyable (QTextEdit)
- Signal toolkit import fixed: both `shared/python` and `src/` added to sys.path
- Diagnostics viewer now uses centralized theme module
- Type hint coverage improved from 81% → 99.7% (#1198)
- Assessment score increased from 7.2 → 8.2 / 10

### Removed

- Redundant local CI workflow (`src/pendulum_simulator/.github/workflows/ci.yml`) — CI managed by top-level `ci-standard.yml` (#1205)
- **Gravity checkbox permanently removed** from toolstrip AND controls panel — gravity is always on (#1209)
- Gravity toggle signal removed from toolstrip
- 43 of 55 pendulum-simulator GitHub issues closed and verified

### Fixed

- Golfer simulation not starting when clicking Run button
- Animation frame slider trace persistence across scrub operations
- `eventFilter` signature mismatch with PyQt6 type stubs
- `horizontalHeader()` None check ordering in diagnostics viewer
- `setattr` in list comprehension causing mypy `func-returns-value` error
- Playback slider was nearly invisible with range(0,0) on dark background (#1207)

## [0.1.0] - 2026-03-01

### Added

- Double pendulum (2-DOF) model with Lagrangian dynamics
- Triple pendulum (3-DOF) model
- Golfer upper-body (8-DOF) model with closed kinematic loop
- Analytical FK Jacobians with 14.7× speedup over numerical
- Constraint solver with Baumgarte stabilization and KKT formulation
- GPU batch optimization via JAX/diffrax/optax
- Rust kernel with PyO3 bindings and WASM target
- PyQt6 dark-themed GUI with real-time animation
- Analysis tab with live physics readouts
- Equations popup with styled HTML LaTeX rendering
- Model registry for clean model switching
- 630+ tests with hypothesis-based property testing
- CLI entry point (`pendulum-golf`)
