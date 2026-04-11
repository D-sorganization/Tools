# Quick Start Guide

## Setup

### Prerequisites

- Rust 1.70+ (for nalgebra 0.33)
- For Python: `pip install maturin pyo3`
- For WASM: `npm install -g wasm-pack`

### Installation

```bash
cd pendulum-core

# Build as library
cargo build --lib

# Build with Python bindings
cargo build --features python --release

# Build for WASM
wasm-pack build --target web --features wasm --release
```

## Basic Usage

### Rust

```rust
use pendulum_core::{double, DoublePendulumParams};

let params = DoublePendulumParams {
    m1: 1.0,
    m2: 1.0,
    l1: 1.0,
    l2: 1.0,
    g: 9.81,
    friction1: 0.0,
    friction2: 0.0,
};

let q = [0.1, 0.2];  // Configuration: [shoulder angle, relative club angle]
let qdot = [0.0, 0.0];

// Compute mass matrix
let m = double::mass_matrix(&q, &params);
println!("M = {:?}", m);

// Compute gravity vector
let g = double::gravity_vector(&q, &params);
println!("G = {:?}", g);

// Compute Coriolis vector
let c = double::coriolis(&q, &qdot, &params);
println!("C = {:?}", c);

// Forward kinematics
let fk = double::forward_kinematics(&q, &params);
println!("Wrist: {:?}", fk.wrist);
println!("Club tip: {:?}", fk.club_tip);
```

### Python

```python
from physics_native import DoublePendulum
import numpy as np

# Create model
model = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)

# Configuration
q = np.array([0.1, 0.2])
qdot = np.array([0.0, 0.0])

# Compute
M = model.mass_matrix(q)
G = model.gravity_vector(q)
C = model.coriolis(q, qdot)
fk = model.forward_kinematics(q)

print("M =", M)
print("G =", G)
print("Wrist:", fk["wrist_x"], fk["wrist_y"])
```

### JavaScript/WASM

```javascript
import init, {
  WasmDoublePendulumParams,
  wasm_double_mass_matrix,
} from "./pkg/pendulum_core.js";

async function main() {
  await init();

  const params = new WasmDoublePendulumParams(
    1.0,
    1.0,
    1.0,
    1.0,
    9.81,
    0.0,
    0.0,
  );
  const q = new Float64Array([0.1, 0.2]);

  const M = wasm_double_mass_matrix(q, params);
  console.log("M =", M);
}

main();
```

## Golfer Model (8-DOF)

### Rust

```rust
use pendulum_core::{golfer, GolferParams, golfer_constraints::BaumgarteGains};

let params = GolferParams {
    l_hub: 0.5,
    m_hub: 5.0,
    d_rs: 0.15,
    d_ls: 0.15,
    l_r_upper: 0.3,
    m_r_upper: 2.0,
    l_r_fore: 0.25,
    m_r_fore: 1.0,
    l_l_upper: 0.3,
    m_l_upper: 2.0,
    l_l_fore: 0.25,
    m_l_fore: 1.0,
    l_club: 1.0,
    m_club: 0.2,
    m_clubhead: 0.2,
    grip_right: 0.3,
    grip_left: 0.3,
    g: 9.81,
};

let q = [0.0; 8];  // Initial configuration
let qdot = [0.0; 8];
let tau = [0.0; 8];  // Zero torques

// Forward kinematics
let fk = golfer::forward_kinematics(&q, &params);
println!("Club tip: {:?}", fk.club_tip);

// Mass matrix (via analytical Jacobians)
let m = golfer::mass_matrix(&q, &params);
println!("M[0,0] = {}", m[(0, 0)]);

// Constraint solver
let gains = BaumgarteGains::default();
let (accel, lambdas) = golfer_constraints::constrained_accelerations(
    &q, &qdot, &tau, &params, &gains
);
println!("Accelerations: {:?}", accel);
println!("Lagrange multipliers: {:?}", lambdas);
```

### Python

```python
from physics_native import Golfer
import numpy as np

model = Golfer(
    l_hub=0.5, m_hub=5.0,
    d_rs=0.15, d_ls=0.15,
    l_r_upper=0.3, m_r_upper=2.0,
    l_r_fore=0.25, m_r_fore=1.0,
    l_l_upper=0.3, m_l_upper=2.0,
    l_l_fore=0.25, m_l_fore=1.0,
    l_club=1.0, m_club=0.2, m_clubhead=0.2,
    grip_right=0.3, grip_left=0.3,
    g=9.81
)

q = np.zeros(8)

# Forward kinematics
fk = model.forward_kinematics(q)
print("Club tip:", fk["club_tip"])

# Mass matrix
M = model.mass_matrix(q)
print("M[0,0] =", M[0, 0])

# Constraints
phi = model.constraint_vector(q)
J = model.constraint_jacobian(q)
print("Constraint errors:", phi)
print("Jacobian shape:", J.shape)
```

## Integration Example

### Rust: Double Pendulum Simulation

```rust
use pendulum_core::{double, DoublePendulumParams, integrator::{integrate_double_pendulum, RK45Config}};
use nalgebra::SVector;

let params = DoublePendulumParams {
    m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0,
    g: 9.81, friction1: 0.0, friction2: 0.0,
};

let q0 = [0.1, 0.0];
let qdot0 = [0.0, 0.0];

// RHS function: qddot = M^{-1}(-C-G)
let rhs = |_t: f64, q: &[f64; 2], qdot: &[f64; 2]| -> [f64; 2] {
    let m = double::mass_matrix(q, &params);
    let c = double::coriolis(q, qdot, &params);
    let g = double::gravity_vector(q, &params);

    let m_inv = m.try_inverse().unwrap();
    let accel = m_inv * (-c - g);

    [accel[0], accel[1]]
};

let config = RK45Config::default();
let steps = integrate_double_pendulum(rhs, 0.0, 10.0, q0, qdot0, config);

for step in steps.iter().step_by(10) {
    println!("t={:.4}, theta1={:.4}, phi={:.4}", step.t, step.y[0], step.y[1]);
}
```

## Testing

```bash
# Run all tests
cargo test --lib

# Run specific test
cargo test --lib test_mass_matrix_symmetry

# Run with all features
cargo test --lib --all-features

# Run integration tests (from main project)
cargo test --test integration_tests
```

## Performance Profiling

```bash
# Build in release mode
cargo build --lib --release

# Use cargo flamegraph (requires installation)
cargo install flamegraph
cargo flamegraph --lib -- --test-threads=1

# Benchmark with criterion (if added to Cargo.toml)
cargo bench
```

## Common Issues

### ImportError: No module named 'pendulum_core'

**Solution:** Compile with Python feature:

```bash
pip install maturin
maturin develop -r --features python
```

### Missing nalgebra dependency

**Solution:** Already included in Cargo.toml, cargo will fetch automatically:

```bash
cargo build --lib
```

### WASM module too large

**Solution:** Build in release mode and strip:

```bash
wasm-pack build --target web --features wasm --release
wasm-opt -Oz -o pkg/pendulum_core_bg.wasm pkg/pendulum_core_bg.wasm
```

## File Organization

```
pendulum-core/
├── Cargo.toml              # Dependencies and features
├── src/
│   ├── lib.rs              # Public API and bindings
│   ├── types.rs            # Parameter types
│   ├── double.rs           # 2-DOF physics
│   ├── triple.rs           # 3-DOF physics
│   ├── golfer.rs           # 8-DOF physics (main)
│   ├── golfer_constraints.rs  # KKT solver
│   └── integrator.rs       # RK45 ODE solver
└── python/
    └── physics_native.py   # Python wrapper
```

## Next Steps

1. **Review API.md** for complete function signatures
2. **Check README.md** for architecture overview
3. **Run tests**: `cargo test --lib`
4. **Build for your target**: Rust, Python, or WASM
5. **Integrate** with your physics solver
6. **Benchmark** against NumPy baseline

## Support

- Rust docs: `cargo doc --open`
- Python wrapper fallback: `HAS_NATIVE` flag in physics_native.py
- WASM errors: Check browser console for detailed JS errors

## Example: Custom ODE Solver

You can write your own ODE solver using the physics functions:

```rust
fn my_ode_solver(
    mut q: [f64; 8],
    mut qdot: [f64; 8],
    params: &GolferParams,
    dt: f64,
    n_steps: usize,
) {
    for step in 0..n_steps {
        // Compute accelerations (assuming zero torques)
        let tau = [0.0; 8];
        let gains = BaumgarteGains::default();
        let (qddot, _lambdas) = constrained_accelerations(
            &q, &qdot, &tau, params, &gains
        );

        // Simple Euler step
        for i in 0..8 {
            qdot[i] += qddot[i] * dt;
            q[i] += qdot[i] * dt;
        }

        if step % 100 == 0 {
            let fk = golfer::forward_kinematics(&q, params);
            println!("Step {}: Club tip at {:?}", step, fk.club_tip);
        }
    }
}
```

Refer to `integrator.rs` for a more sophisticated RK45 implementation with automatic step control.
