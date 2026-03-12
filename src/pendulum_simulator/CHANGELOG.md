# Changelog

All notable changes to the Pendulum Simulator will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized theme module (`gui/theme.py`) for DRY stylesheet management (#1197)
- Advanced diagnostics tracker with JSONL persistence and viewer dialog
- Golfer topology validation tests — 18 tests covering mass distribution, FK, and DbC contracts (#1204)
- Mouse wheel blocking on all input widgets to prevent accidental value changes (#1193)
- Playback controls: slider scrubbing, loop toggle, play/pause state management
- Animation trail cleanup on slider interaction

### Changed
- Golfer model topology corrected: standoff is now massless (0.001 kg default), upper body segments carry ~7 kg each (#1195)
- UI labels updated: Hub → Standoff, R/L Scap → R/L UBody
- Unit dropdown boxes standardized to fixed width (60px)
- Signal toolkit error messages are now selectable/copyable (QTextEdit)
- Signal toolkit import fixed: both `shared/python` and `src/` added to sys.path
- Diagnostics viewer now uses centralized theme module

### Removed
- Redundant local CI workflow (`src/pendulum_simulator/.github/workflows/ci.yml`) — CI managed by top-level `ci-standard.yml` (#1205)

### Fixed
- Golfer simulation not starting when clicking Run button
- Animation frame slider trace persistence across scrub operations
- `eventFilter` signature mismatch with PyQt6 type stubs
- `horizontalHeader()` None check ordering in diagnostics viewer
- `setattr` in list comprehension causing mypy `func-returns-value` error

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
