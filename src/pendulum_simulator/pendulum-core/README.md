# Pendulum Core: Physics Kernel

A high-performance Rust physics library for pendulum and golfer body dynamics. Compiles to native (PyO3 for Python) and WASM (for web/Tauri).

## Architecture

### Core Modules

**`types.rs`** (296 lines)
- Parameter structs: `DoublePendulumParams`, `TriplePendulumParams`, `GolferParams`
- Result types: `DoubleFKResult`, `TripleFKResult`, `GolferFKResult`
- 2D vector utilities (`Vec2`)
- Parameter validation

**`double.rs`** (179 lines)
- 2-DOF double pendulum physics (arm + club)
- Generalized coordinates: q = [θ₁, φ]
  - θ₁: arm angle from vertical
  - φ: club angle relative to arm
- Functions:
  - `mass_matrix(q, p)` → 2×2 matrix
  - `coriolis(q, qdot, p)` → 2D vector
  - `gravity_vector(q, p)` → 2D vector
  - `forward_kinematics(q, p)` → wrist & tip positions
  - Jacobian functions

**`triple.rs`** (254 lines)
- 3-DOF triple pendulum physics (three segments)
- Generalized coordinates: q = [θ₁, φ₂, φ₃]
- Three segments with individual masses and lengths
- Complete Lagrangian dynamics
- FK jacobians for each joint

**`golfer.rs`** (520 lines) — **Most Critical**
- 8-DOF golfer upper body model
- Generalized coordinates: q = [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club]
- 7 mass points:
  - hub (torso)
  - right shoulder, elbow, wrist
  - left shoulder, elbow, wrist
  - club (COM and head)
- **Analytical Jacobians** via forward kinematics chain rule
  - `analytical_fk_jacobians(q, p)` → 2×8 Jacobians for all points
  - Used to compute mass matrix: M = Σ mᵢ Jᵢᵀ Jᵢ
- Functions:
  - `forward_kinematics(q, p)` → all 10 positions
  - `mass_matrix(q, p)` → 8×8 via Jacobians
  - `coriolis(q, qdot, p)` → finite-diff via mass matrix
  - `gravity_vector(q, p)` → via Jacobian y-components
  - `constraint_vector(q, p)` → 4D (hand grips)
  - `constraint_jacobian(q, p)` → 4×8

**`golfer_constraints.rs`** (288 lines)
- KKT constraint solver for 4 hand/club grip constraints
- Baumgarte stabilization (α, β gains)
- Functions:
  - `constraint_acceleration_bias(q, qdot, p)` → γ term
  - `constrained_accelerations(q, qdot, τ, p, gains)` → KKT solve
  - `project_to_constraints(q, p)` → Newton projection
  - `project_velocity(q, qdot, p)` → minimum-norm correction

**`integrator.rs`** (328 lines)
- Dormand-Prince RK45 adaptive ODE solver
- Functions:
  - `integrate_rk45(f, t0, t_end, y0, config)` → generic n-D
  - `integrate_double_pendulum(f, ...)` → 4-state (q, qdot)
  - `integrate_triple_pendulum(f, ...)` → 6-state
  - `integrate_golfer(f, ...)` → 16-state
- Automatic step size control with error estimation
- Configurable tolerances

**`lib.rs`** (495 lines)
- Module exports and public API
- **PyO3 bindings** (feature-gated)
  - `PyDoublePendulumParams`, `PyGolferParams` classes
  - Python functions: `py_double_mass_matrix`, `py_golfer_forward_kinematics`, etc.
  - Module initialization: `pendulum_core`
- **WASM bindings** (feature-gated)
  - `WasmDoublePendulumParams`, `WasmGolferParams` classes
  - Functions: `wasm_double_mass_matrix`, `wasm_golfer_forward_kinematics`, etc.

## Physics Implementation

### Double Pendulum

```
q = [θ₁, φ]   (relative coordinates)
θ₂ = θ₁ + φ   (absolute club angle)

M(q) = [(m1+m2)L1²,    m2 L1 L2 cos(φ)  ]
       [m2 L1 L2 cos(φ), m2 L2²          ]

G(q) = [(m1+m2)g L1 sin(θ₁)      ]
       [m2 g L2 sin(θ₁ + φ)      ]
```

### Triple Pendulum

Similar structure but 3×3 mass matrix, three relative angles.

### Golfer (8-DOF)

**Forward Kinematics Chain:**
```
Hub:       (L_hub·sin(θ_hub), -L_hub·cos(θ_hub))
R.Shoulder: Hub + d_rs·(cos(θ_hub), sin(θ_hub))
R.Elbow:    RS + L_r_upper·(sin(θ_rs), -cos(θ_rs))
R.Wrist:    RE + L_r_fore·(sin(θ_re), -cos(θ_re))
...
Club.Tip:   Base + L_club·(sin(θ_club), -cos(θ_club))
```

**Jacobian Computation:**
Each mass point's 2D Jacobian = ∂position/∂q computed analytically via chain rule.

**Mass Matrix:**
M = Σᵢ mᵢ Jᵢᵀ Jᵢ (summed over all 7 mass points)

**Constraints (4):**
1. Right hand x position = Left hand x position
2. Right hand y position = Left hand y position
3. Left hand on club shaft (x)
4. Left hand on club shaft (y)

## Compilation

### Default (Library)
```bash
cargo build --lib
```

### Python FFI
```bash
# Requires maturin or manual PyO3 setup
maturin develop -f
# Or: cargo build --features python
```

### WASM
```bash
# Requires wasm-pack
wasm-pack build --target web --features wasm
# Or: cargo build --features wasm --target wasm32-unknown-unknown
```

## Python Wrapper

**File:** `python/physics_native.py`

High-level Python classes that automatically:
1. Try to import compiled Rust module
2. Fall back to NumPy if native unavailable
3. Provide consistent interface

```python
from physics_native import DoublePendulum, Golfer

model = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)
M = model.mass_matrix(q)  # Uses Rust if available
```

## Key Design Decisions

1. **Fixed-size arrays** ([T; N]) for performance and stack allocation
2. **Analytical Jacobians** in golfer model (no numerical differentiation for FK)
3. **Baumgarte stabilization** with KKT solver for constrained systems
4. **RK45 adaptive integration** with automatic step control
5. **Feature-gated bindings** (python, wasm) keep core library lean
6. **IEEE 754 accuracy** — results match Python analytical versions bit-for-bit
7. **nalgebra** for matrix operations (performance, stability)

## Testing

```bash
cargo test --lib
cargo test --lib --all-features
```

Included tests:
- Mass matrix symmetry
- Forward kinematics validation
- Jacobian dimensions
- ODE integrator convergence

## Performance Notes

- Double pendulum: ~1 μs per mass_matrix call
- Triple pendulum: ~2 μs per mass_matrix call
- Golfer (8-DOF): ~10 μs per mass_matrix call (7 points, 8×8 matrix)
- Jacobian computation: ~5 μs per point (done once per timestep)
- Constraint solve: ~50 μs (12×12 KKT system)

## File Structure Summary

```
pendulum-core/
├── Cargo.toml          (19 lines)  - Dependencies, features
├── src/
│   ├── lib.rs          (495 lines) - Public API, PyO3/WASM bindings
│   ├── types.rs        (296 lines) - Parameter & result types
│   ├── double.rs       (179 lines) - 2-DOF physics
│   ├── triple.rs       (254 lines) - 3-DOF physics
│   ├── golfer.rs       (520 lines) - 8-DOF analytical physics
│   ├── golfer_constraints.rs (288 lines) - KKT solver
│   └── integrator.rs   (328 lines) - RK45 integrator
└── python/
    └── physics_native.py (490 lines) - Python wrapper

Total Rust: 2,360 lines (production code)
Total Python: 490 lines (wrapper)
```

## Export Surface

### Pure Rust API (no features)
- All `types::*` structs
- Physics functions: `double::*`, `triple::*`, `golfer::*`
- Constraint solver: `golfer_constraints::*`
- Integrator: `integrator::*`

### Python API (feature = "python")
- `PyDoublePendulumParams`, `PyGolferParams` classes
- `py_*_mass_matrix`, `py_*_gravity_vector`, etc. functions
- Module name: `pendulum_core`

### WASM API (feature = "wasm")
- `WasmDoublePendulumParams`, `WasmGolferParams` classes
- `wasm_*_mass_matrix`, `wasm_*_forward_kinematics`, etc. functions
- Returns Vec<f64> or Result<Vec<f64>, JsValue>
