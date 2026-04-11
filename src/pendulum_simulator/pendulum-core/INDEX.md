# Pendulum Core - Complete File Index

This document provides a complete overview of all files in the `pendulum-core` crate.

## File Manifest

### Configuration

- **Cargo.toml** (603 bytes)
  - Package metadata: `pendulum-core v0.1.0`
  - Dependencies: nalgebra 0.33, pyo3 0.22 (optional), wasm-bindgen 0.2 (optional)
  - Features: python, wasm, serde
  - Library type: cdylib, rlib (dual compilation)

### Documentation (37 KB total)

- **README.md** (7.2 KB)
  - Architecture overview
  - Module descriptions
  - Physics implementation details
  - Performance notes
  - File structure

- **API.md** (12 KB)
  - Complete type definitions
  - All function signatures
  - Python bindings reference
  - WASM bindings reference
  - Usage examples

- **QUICKSTART.md** (7.6 KB)
  - Setup instructions
  - Basic usage examples (Rust, Python, WASM)
  - Integration examples
  - Testing instructions
  - Common issues & solutions

- **IMPLEMENTATION_SUMMARY.md** (11 KB)
  - Project overview
  - Complete deliverables
  - Physics details
  - Key features
  - Build instructions
  - Integration guide

- **INDEX.md** (this file)
  - File manifest and descriptions

### Rust Source Code (src/) - 2,360 lines total

#### lib.rs (495 lines)

**Purpose:** Top-level module, exports, feature-gated bindings
**Modules exported:**

- `double::*` physics functions
- `triple::*` physics functions
- `golfer::*` physics functions
- `golfer_constraints::*` solver
- `integrator::*` ODE solver
- `types::*` parameter types

**PyO3 Bindings (feature = "python"):**

- `PyDoublePendulumParams` class
- `PyGolferParams` class
- `py_double_mass_matrix`, `py_double_gravity_vector`, `py_double_coriolis`, `py_double_forward_kinematics`
- `py_golfer_mass_matrix`, `py_golfer_gravity_vector`, `py_golfer_forward_kinematics`
- `py_golfer_constraint_vector`, `py_golfer_constraint_jacobian`
- Module: `pendulum_core`

**WASM Bindings (feature = "wasm"):**

- `WasmDoublePendulumParams` class
- `WasmGolferParams` class
- `wasm_double_mass_matrix`, `wasm_double_gravity_vector`
- `wasm_golfer_mass_matrix`, `wasm_golfer_forward_kinematics`

#### types.rs (296 lines)

**Purpose:** Parameter structs and result types
**Exported types:**

- `DoublePendulumParams` (7 fields: m1, m2, l1, l2, g, friction1, friction2)
- `TriplePendulumParams` (11 fields: masses[3], lengths[3], g, friction[3])
- `GolferParams` (18 fields: hub geometry, arm geometry, club geometry, masses)
- `DoubleFKResult` (wrist, club_tip, angles)
- `TripleFKResult` (3 joints, angles)
- `GolferFKResult` (10 positions)
- `Vec2` utility struct with methods

**Features:**

- Parameter validation methods
- Complete documentation
- Serde support (optional)

#### double.rs (179 lines)

**Purpose:** 2-DOF double pendulum physics
**Public functions:**

- `mass_matrix(q, params) → SMatrix<f64, 2, 2>`
- `coriolis(q, qdot, params) → SVector<f64, 2>`
- `gravity_vector(q, params) → SVector<f64, 2>`
- `forward_kinematics(q, params) → DoubleFKResult`
- `jacobian_wrist(q, params) → SMatrix<f64, 2, 2>`
- `jacobian_club_tip(q, params) → SMatrix<f64, 2, 2>`

**Tests:**

- Mass matrix symmetry
- Forward kinematics validation

#### triple.rs (254 lines)

**Purpose:** 3-DOF triple pendulum physics
**Public functions:**

- `mass_matrix(q, params) → SMatrix<f64, 3, 3>`
- `coriolis(q, qdot, params) → SVector<f64, 3>`
- `gravity_vector(q, params) → SVector<f64, 3>`
- `forward_kinematics(q, params) → TripleFKResult`
- `jacobian_joint1, jacobian_joint2, jacobian_joint3(q, params) → SMatrix<f64, 2, 3>`

**Tests:**

- Mass matrix symmetry
- Forward kinematics (vertical config)

#### golfer.rs (520 lines) ★ MAIN PHYSICS MODULE

**Purpose:** 8-DOF golfer upper body physics with analytical Jacobians
**Public functions:**

- `analytical_fk_jacobians(q, params) → HashMap<String, SMatrix<f64, 2, 8>>`
  - Keys: hub, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, club_com, club_tip
- `forward_kinematics(q, params) → GolferFKResult`
- `mass_matrix(q, params) → SMatrix<f64, 8, 8>` (via Jacobians)
- `coriolis(q, qdot, params) → SVector<f64, 8>` (finite-diff)
- `gravity_vector(q, params) → SVector<f64, 8>`
- `constraint_vector(q, params) → SVector<f64, 4>`
- `constraint_jacobian(q, params) → SMatrix<f64, 4, 8>`

**Tests:**

- Forward kinematics identity
- Jacobian dimensions
- Mass matrix symmetry

#### golfer_constraints.rs (288 lines)

**Purpose:** KKT constraint solver for golfer model
**Public types:**

- `BaumgarteGains` (alpha, beta fields)

**Public functions:**

- `constraint_acceleration_bias(q, qdot, params) → SVector<f64, 4>`
- `constrained_accelerations(q, qdot, tau, params, gains) → (SVector<f64, 8>, SVector<f64, 4>)`
- `project_to_constraints(q, params, max_iters, tol) → [f64; 8]`
- `project_velocity(q, qdot, params) → [f64; 8]`

**Tests:**

- Constraint acceleration bias finite-diff
- Constrained accelerations KKT system
- Finite values check

#### integrator.rs (328 lines)

**Purpose:** Dormand-Prince RK45 adaptive ODE solver
**Public types:**

- `RK45Config` (h0, h_min, h_max, rtol, atol, max_steps)
- `IntegrationStep<N>` (t, y, h)

**Public functions:**

- `integrate_rk45<F, N>(f, t0, t_end, y0, config) → Vec<IntegrationStep<N>>`
- `integrate_double_pendulum(f, t0, t_end, q0, qdot0, config) → Vec<IntegrationStep<4>>`
- `integrate_triple_pendulum(f, t0, t_end, q0, qdot0, config) → Vec<IntegrationStep<6>>`
- `integrate_golfer(f, t0, t_end, q0, qdot0, config) → Vec<IntegrationStep<16>>`

**Features:**

- 7-stage Dormand-Prince method
- Automatic step size control
- Error estimation
- Generic RHS function

**Tests:**

- Simple ODE (dy/dt = -y)
- Step ordering and timing

### Python Wrapper (python/) - 490 lines

#### physics_native.py (490 lines)

**Purpose:** Python wrapper with Rust FFI support and NumPy fallback
**Exported:**

- `HAS_NATIVE` flag
- `DoublePendulumParams` class
- `DoublePendulum` model class
- `GolferParams` class
- `Golfer` model class
- `get_native_info()` function

**Features:**

- Automatic Rust detection
- NumPy fallback for double pendulum
- Consistent interface regardless of backend
- Error handling

## Usage Summary

### Rust

```rust
use pendulum_core::{double, DoublePendulumParams};
let params = DoublePendulumParams { ... };
let m = double::mass_matrix(&q, &params);
```

### Python

```python
from physics_native import DoublePendulum
model = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)
M = model.mass_matrix(q)
```

### WASM

```javascript
import { WasmDoublePendulumParams, wasm_double_mass_matrix } from 'pendulum_core';
const params = new WasmDoublePendulumParams(...);
const M = wasm_double_mass_matrix(q, params);
```

## Line Count Summary

```
Rust Sources:
  lib.rs                495 lines
  types.rs              296 lines
  golfer.rs             520 lines ★
  integrator.rs         328 lines
  double.rs             179 lines
  triple.rs             254 lines
  golfer_constraints.rs 288 lines
  ─────────────────────────────
  Total Rust:         2,360 lines

Python:
  physics_native.py     490 lines

Documentation:
  README.md           7.2 KB
  API.md             12.0 KB
  QUICKSTART.md       7.6 KB
  IMPLEMENTATION_SUMMARY.md 11 KB
  INDEX.md (this)     ~6 KB
  ─────────────────────────────
  Total Documentation: ~44 KB

Config:
  Cargo.toml          603 bytes
```

## Build Instructions

```bash
# Library (Rust)
cargo build --lib

# Python FFI
maturin develop -r --features python

# WASM
wasm-pack build --target web --features wasm --release

# Tests
cargo test --lib
cargo test --lib --all-features
```

## Next Steps

1. Read README.md for architecture overview
2. Review API.md for complete function signatures
3. Check QUICKSTART.md for usage examples
4. Run cargo build or maturin develop as needed
5. Integrate with your application

## Key Features Checklist

- [x] Pure Rust implementation (nalgebra only)
- [x] 2-DOF double pendulum physics
- [x] 3-DOF triple pendulum physics
- [x] 8-DOF golfer model with 4 constraints
- [x] Analytical FK Jacobians (golfer)
- [x] KKT constraint solver
- [x] Baumgarte stabilization
- [x] RK45 adaptive ODE integrator
- [x] PyO3 Python bindings
- [x] WASM bindings
- [x] Python fallback wrapper
- [x] Comprehensive documentation
- [x] Unit tests
- [x] Complete feature-gating

All deliverables complete and ready for use.
