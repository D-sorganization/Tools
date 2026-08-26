# Double Pendulum Golf Swing Simulator

## Coordinate-Explicit Force Attribution

Provider schema `force-attribution/v1` adds a shared analytical layer for
Coriolis cross-speed, squared-speed centripetal/centrifugal interpretation,
gravity, damping, applied-control, and residual terms. It keeps impulse,
power, and work distinct and reports when a force-only hand-path mapping cannot
represent a joint couple. The detailed convention and optimization contract
are documented in [Pendulum Force Attribution and Impulse Optimization](../../docs/development/pendulum-force-attribution.md).

A multi-platform visualization and optimization tool for exploring the dynamics
of multi-body kinematic chains, from simple double pendulums to a full golfer
upper-body model with closed-loop constraints.

## Platforms

- **PyQt6 Desktop** — Full-featured desktop app with real-time animation,
  analysis suite, and matrix visualization
- **React/Tauri Web** — Cross-platform web app with all three models,
  canvas animation, and optimizer panel
- **Rust Kernel** (`pendulum-core`) — Shared physics engine compiled to
  native (PyO3 for Python) and WASM (wasm-bindgen for web)
- **JAX/GPU** — GPU-accelerated batch simulation and gradient-based
  torque profile optimization

## Models

The simulator provides three models of increasing complexity:

1. **Double Pendulum** — 2-DOF open chain (arm + club). Demonstrates passive
   energy transfer via off-diagonal mass-matrix coupling.
2. **Triple Pendulum** — 3-DOF open chain (shoulder, arm, club). Adds a
   second joint for richer dynamics and energy cascading.
3. **Golfer Upper Body** — 8-DOF closed kinematic loop. Two arm chains
   (right and left) connect through a shared club, forming 4 holonomic
   constraints that close the loop.

## Golfer Upper-Body Model

The golfer model uses an 8-DOF closed kinematic loop with the following topology:

- **Standoff** (origin → hub): Massless segment that adjusts where the center
  of rotation sits relative to the body's center of mass.
- **Upper Body segments** (hub → shoulders): Connect the hub to each shoulder
  via revolute joints. Represent the upper torso and carry significant mass
  (~2× arm mass each, ~7 kg default).
- **Arms** (shoulder → elbow → hand): Standard arm chains.
- **Club**: Shared between both hands via grip constraints.

```
           Origin (fixed pivot)
              |
              |  Standoff (massless, adjustable for COM offset, θ_hub)
              |
           [Hub]
           /    \
     R UBody     L UBody      (upper body segments, ~7 kg each)
         |           |
        RS          LS         (shoulder revolute joints)
         |           |
      R Upper     L Upper      (α_rs, α_ls — upper arms)
         |           |
      R Forearm   L Forearm    (α_re, α_le — forearms)
         |           |
        RH ─ ─ grip ─ CLUB ─ grip ─ ─ LH    (α_rh, α_lh — wrists)
                       |
                  [Clubhead]
                  (θ_club)
```

Generalized coordinates (8 DOF):

    [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club]

The right and left hands must coincide with their respective grip positions
on the club shaft, yielding 4 scalar constraint equations (2 per hand, x and
y). The system is solved with an augmented Lagrangian (KKT) formulation and
Baumgarte stabilization for numerical drift control.

### Key Physics

The constrained equations of motion are:

```
┌         ┐ ┌     ┐   ┌                        ┐
│  M   Φ_qᵀ│ │ q̈  │   │ τ − C·q̇ − G − b·q̇   │
│          │ │     │ = │                        │
│ Φ_q  0   │ │ λ   │   │ −γ − 2αΦ̇ − β²Φ      │
└         ┘ └     ┘   └                        ┘
```

where M is the 8×8 mass matrix, Φ_q the 4×8 constraint Jacobian, λ the
Lagrange multipliers, and α,β the Baumgarte gains.

### Analytical Derivatives (Phase 1 Optimization)

All physics computations use closed-form analytical derivatives instead of
numerical finite differences, achieving a **14.7x overall speedup**:

| Operation           | Numerical  | Analytical | Speedup   |
| ------------------- | ---------- | ---------- | --------- |
| Mass Matrix         | 40 ms      | 2 ms       | 23.6x     |
| Gravity Vector      | 36 ms      | 1 ms       | 35.9x     |
| Coriolis Forces     | 369 ms     | 26 ms      | 14.4x     |
| Constraint Jacobian | 8 ms       | 3 ms       | 3.1x      |
| **Total per RHS**   | **453 ms** | **31 ms**  | **14.7x** |

The analytical FK Jacobians compute 2×8 position derivatives for each of
7 mass points using direct chain-rule trig derivatives. The mass matrix is
assembled as `M = Σ mᵢ Jᵢᵀ Jᵢ`, Coriolis via Christoffel symbols using
`dJ/dq`, and gravity from `G_k = Σ mᵢ g dY_i/dq_k`.

### Analysis Suite

For every time step the simulator computes:

- 8×8 mass matrix with configuration-dependent coupling
- Coriolis/centrifugal torques (Christoffel symbols)
- Gravitational torques (potential-energy gradient)
- Constraint forces (Lagrange multipliers → physical joint reaction forces)
- Constraint violation magnitude (loop closure error)
- Task-space Jacobians for all endpoints
- Manipulability ellipsoids (mobility and force, via SVD)
- ZTCF matrix: (J M⁻¹ Jᵀ)⁻¹ J M⁻¹
- Zero-torque counterfactual accelerations and forces
- Kinetic, potential, and total energy
- Viscous dissipation at every joint
- Phase-window drift/control grip work, negative grip work, opposing
  along-path impulse, wrist-control work, peak grip force, distal energy gain,
  and mixed-objective Pareto ranking for the qualified double-pendulum tier

The desktop Analysis dock includes a **Drift Transfer** tab. It plots total,
drift, and control grip power alongside distal speed and proximal-link angular
velocity for a user-declared time window. The proximal coordinate is a model
link rate, not an anatomical shoulder or torso measurement. Triple- and
two-hand golfer-model attribution remain unavailable until their force
allocation and reaction-force contracts are independently qualified.

The separate **Rotating-Base Study** surface exposes the qualified
`planar_rotating_base_two_hand_compliant_club` tier without changing that
fixed-hub boundary. The PyQt6 surface executes a selected registered case on a
background thread through the shared Python provider. The React/Tauri surface
browses the same packaged 18-case authority rather than carrying a second set
of equations. Both retain invalid rows and exclusions, matching-rule and torso-
program selectors, transfer and closure diagnostics, exact same-state torso/
arm/wrist killswitch results, the pinned source revision, and the explicit
nonanatomical, no-human-validation, and noncoaching limitations. Both surfaces
show full-resolution contact power, force-generated couple, torso/club rates,
distal energy, and independent lead/trail grip-force magnitudes.

## Installation

```bash
cd src/pendulum_simulator
pip install -e ".[dev]"

# Optional: GPU optimization support
pip install -e ".[gpu]"   # requires JAX, diffrax, optax
```

## Usage

### Proximal–Distal Companion Guide

Both interfaces load the same canonical experiment and glossary catalog from
`src/double_pendulum_golf/resources/companion_catalog.json`. Open **Companion
Guide** in the PyQt6 toolstrip or use the guide above the React workbench. Each
experiment declares its purpose, hypothesis, observable outputs, workflow,
interpretation tips, falsifier, and limitations before the model is run.

The guide covers two-link passive transfer and distal-torque timing, a
three-link cascade, bilateral hand-force/equivalent-couple mechanics, pointwise
versus forward zero-torque counterfactuals, and a parameter-robustness envelope.
It supplements the simulator; it does not turn model output into a measured
biomechanical fact. Exported studies should retain the experiment ID, parameter
values, units, model version, and `exploratory_model_output` status through
`build_run_manifest()`.

The rotating-base authority is independently content-pinned at
`shared/python/swing_sim/rotating_base/resources/rotating_base_torso_velocity_study_v1.json`.
Desktop and web exports include the selected request, full-resolution traces, retained
validity/exclusion state, source revision, model tier, and scientific-promotion
boundaries as deterministic JSON. The complete web trace catalog is separately
pinned by SHA-256 and must retain all 18 rows, including five adverse rows; these
are evidence exports, not human biomechanics or coaching prescriptions.

### PyQt6 Desktop App

```bash
python -m double_pendulum_golf
# or
pendulum-golf
```

### React/Tauri Web App

```bash
cd pendulum-web
npm install
npm run dev        # development server
npm run tauri dev  # Tauri desktop wrapper
```

### GPU Optimization (JAX)

```python
from double_pendulum_golf.optimizer_gpu import optimize_torque_profile
from double_pendulum_golf.physics_golfer_jax import GolferParamsJAX

params = GolferParamsJAX(...)
result, history = optimize_torque_profile(
    params, initial_state, t_end=2.0,
    n_coeffs_per_joint=3, n_iterations=100
)
```

### Rust Kernel (pendulum-core)

```bash
cd pendulum-core
cargo build --lib                              # native library
maturin develop -r --features python           # Python FFI via PyO3
wasm-pack build --target web --features wasm   # WASM for React/Tauri
```

### Optional Native Backends

The desktop Python app can optionally delegate model kernels to the compiled
Rust extension. Each model remains opt-in so the pure-Python implementation
stays the default contract path.

```bash
cd pendulum-core
maturin develop -r --features python

export PENDULUM_DOUBLE_BACKEND=rust
export PENDULUM_TRIPLE_BACKEND=rust
export PENDULUM_GOLFER_BACKEND=rust
python -m double_pendulum_golf
```

Notes:

- Double and triple pendulum execution now have parity-validated Rust kernels
  for mass matrix, gravity, Coriolis, and forward kinematics behind their
  respective opt-in backend flags.
- The golfer model remains the dominant performance hotspot from the deep
  review, so it additionally exposes native constrained dynamics and
  projection helpers.
- Constraint projection also attempts the native path first, but falls back to
  the Python implementation if the projected state does not satisfy the Python
  constraint residual checks.

### Torque Polynomials

All tabs accept polynomial torque coefficients: `c0, c1, c2, ...`

Evaluates as: `τ(t) = c0 + c1·t + c2·t² + ...`

Example: `-25, 10` gives `τ(t) = -25 + 10t`

## Running Tests

```bash
pytest                # 229 tests (203 core + 26 analytical validation)
pytest --cov=double_pendulum_golf --cov-report=term-missing

# JAX tests run when JAX is installed, skip gracefully otherwise
pytest tests/test_physics_golfer_jax.py tests/test_optimizer_gpu.py
```

## Architecture

```
src/double_pendulum_golf/
├── physics.py                 # 2-DOF EOM (mass matrix, Coriolis, gravity)
├── physics_triple.py          # 3-DOF EOM
├── physics_golfer.py          # 8-DOF EOM — analytical Jacobians + numerical fallbacks
├── physics_golfer_jax.py      # 8-DOF EOM — JAX/GPU pure-function reimplementation
├── simulation.py              # 2-DOF integration engine
├── simulation_triple.py       # 3-DOF integration engine
├── simulation_golfer.py       # 8-DOF constrained integration (solve_ivp)
├── simulation_golfer_gpu.py   # 8-DOF GPU batch simulation (diffrax + vmap)
├── optimizer_gpu.py           # Gradient-based torque optimization (optax)
├── constraint_solver.py       # KKT solver, Baumgarte stabilization, projection
├── jacobians.py               # 2/3-DOF Jacobians + shared ellipsoid kernel
├── jacobians_golfer.py        # 8-DOF task-space Jacobians, ZTCF, Delta
├── counterfactual.py          # 2-DOF zero-torque analysis
├── counterfactual_triple.py   # 3-DOF zero-torque analysis
├── counterfactual_golfer.py   # 8-DOF zero-torque analysis
├── gui/
│   ├── main_window.py             # Tab orchestration, simulation panels
│   ├── pendulum_widget.py         # 2/3-DOF animation canvas
│   ├── golfer_pendulum_widget.py  # 8-DOF animation (branching topology)
│   ├── matrix_widget.py           # 2-DOF real-time matrix/energy display
│   ├── matrix_widget_triple.py    # 3-DOF matrix display
│   ├── matrix_widget_golfer.py    # 8-DOF matrix + constraints display
│   ├── controls_widget.py         # 2-DOF input panel
│   ├── controls_widget_triple.py  # 3-DOF input panel
│   ├── controls_widget_golfer.py  # 8-DOF input panel (scrollable)
│   ├── controls_utils.py          # Shared parsing/styling utilities
│   └── simulation_panel.py        # Panel builder + SimViewer protocol

pendulum-core/                     # Shared Rust physics kernel
├── Cargo.toml
├── src/
│   ├── lib.rs                     # Feature-gated PyO3 + WASM exports
│   ├── types.rs                   # Parameter structs for all 3 models
│   ├── double.rs                  # 2-DOF physics (mass matrix, Coriolis, gravity)
│   ├── triple.rs                  # 3-DOF physics
│   ├── golfer.rs                  # 8-DOF analytical physics (FK Jacobians)
│   ├── golfer_constraints.rs      # KKT solver, Baumgarte stabilization
│   └── integrator.rs              # RK45 adaptive ODE solver
└── python/
    └── physics_native.py          # Python wrapper (Rust FFI with numpy fallback)

pendulum-web/                      # React/Tauri cross-platform web app
├── src/
│   ├── App.tsx / AppNew.tsx       # Main app — model selector + tabs
│   ├── physics.ts                 # 2-DOF TypeScript physics
│   ├── physics_triple.ts          # 3-DOF TypeScript physics
│   ├── physics_golfer.ts          # 8-DOF TypeScript physics + KKT solver
│   ├── optimizer.ts               # Nelder-Mead simplex optimizer
│   ├── presets.ts                 # 2-DOF presets
│   ├── presets_triple.ts          # 3-DOF presets
│   ├── presets_golfer.ts          # 8-DOF presets
│   ├── units.ts                   # Unit conversion utilities
│   └── components/
│       ├── PendulumCanvas.tsx     # 2-DOF animation canvas
│       ├── TriplePendulumCanvas.tsx # 3-DOF animation canvas
│       ├── GolferCanvas.tsx       # 8-DOF golfer animation canvas
│       ├── AnalysisPlots.tsx      # Analysis charts
│       ├── OptimizerPanel.tsx     # Optimization UI
│       └── UnitSelector.tsx       # Unit picker

tests/
├── test_physics.py            # 2-DOF physics properties
├── test_physics_triple.py     # 3-DOF physics properties
├── test_physics_golfer.py     # 8-DOF physics (FK, mass matrix, energy)
├── test_analytical_jacobians.py # Analytical vs numerical parity (26 tests)
├── test_constraint_solver.py  # Constraint projection, KKT, Baumgarte
├── test_simulation.py         # 2-DOF integration
├── test_simulation_triple.py  # 3-DOF integration
├── test_simulation_golfer.py  # 8-DOF constrained integration
├── test_jacobians.py          # Jacobians and ellipsoids
├── test_friction.py           # 2-DOF dissipation
├── test_friction_triple.py    # 3-DOF dissipation
├── test_counterfactual.py     # Zero-torque counterfactual
├── test_physics_golfer_jax.py # JAX vs numpy parity (25 tests, skip w/o JAX)
└── test_optimizer_gpu.py      # GPU optimizer validation (10 tests, skip w/o JAX)
```

### Design Principles

- **TDD**: 229 tests covering mass-matrix symmetry/PSD, energy conservation,
  constraint satisfaction, FK consistency, analytical vs numerical parity,
  and contract violations
- **DbC** (Design by Contract): Assertions as pre/post-conditions in all
  physics functions — shape checks, finiteness, physical bounds
- **DRY**: Shared ellipsoid kernel, common parsing utilities, protocol-based
  widget interfaces; Rust kernel serves both Python and web platforms
- **Orthogonal development**: Each model (2/3/8 DOF) is self-contained with
  its own physics, simulation, GUI widgets, and tests
- **Cross-platform parity**: Rust kernel ensures identical physics across
  PyQt6 desktop, React/Tauri web, and Python scripting environments

## CI/CD

Linting is configured in `pyproject.toml`:

- **Ruff** — line length 95, Python 3.10 target
- **Black** — line length 95
- **Mypy** — strict return-type and unused-config warnings
- **cargo clippy** — Rust linting (pendulum-core)

All files pass Ruff, Black, and Mypy with zero errors.

## License

MIT
