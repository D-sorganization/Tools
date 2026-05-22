# Pendulum Core API Reference

Complete API documentation for the Rust physics kernel.

## Types Module

### `DoublePendulumParams`

```rust
pub struct DoublePendulumParams {
    pub m1: f64,           // Arm mass (kg)
    pub m2: f64,           // Club mass (kg)
    pub l1: f64,           // Arm length (m)
    pub l2: f64,           // Club length (m)
    pub g: f64,            // Gravity (m/s²)
    pub friction1: f64,    // Joint 1 friction
    pub friction2: f64,    // Joint 2 friction
}
```

**Methods:**

- `validate() -> Result<(), String>` - Check parameter validity

### `TriplePendulumParams`

```rust
pub struct TriplePendulumParams {
    pub masses: [f64; 3],  // m₁, m₂, m₃
    pub lengths: [f64; 3], // L₁, L₂, L₃
    pub g: f64,
    pub friction: [f64; 3],
}
```

### `GolferParams`

```rust
pub struct GolferParams {
    // Hub
    pub l_hub: f64,        // Hub length
    pub m_hub: f64,        // Hub mass
    pub d_rs: f64,         // Right shoulder offset
    pub d_ls: f64,         // Left shoulder offset

    // Right arm
    pub l_r_upper: f64,    // Upper arm length
    pub m_r_upper: f64,
    pub l_r_fore: f64,     // Forearm length
    pub m_r_fore: f64,

    // Left arm (similar)
    pub l_l_upper: f64,
    pub m_l_upper: f64,
    pub l_l_fore: f64,
    pub m_l_fore: f64,

    // Club
    pub l_club: f64,
    pub m_club: f64,
    pub m_clubhead: f64,
    pub grip_right: f64,   // Right hand grip position on shaft
    pub grip_left: f64,    // Left hand grip position on shaft
    pub g: f64,
}
```

### `DoubleFKResult`

```rust
pub struct DoubleFKResult {
    pub wrist: (f64, f64),     // x, y position
    pub club_tip: (f64, f64),
    pub theta1: f64,           // Absolute arm angle
    pub theta2: f64,           // Absolute club angle
}
```

### `TripleFKResult`

```rust
pub struct TripleFKResult {
    pub joint1: (f64, f64),    // End of segment 1
    pub joint2: (f64, f64),    // End of segment 2
    pub joint3: (f64, f64),    // End of segment 3 (tip)
    pub angles: [f64; 3],      // Absolute angles
}
```

### `GolferFKResult`

```rust
pub struct GolferFKResult {
    pub hub: (f64, f64),
    pub r_shoulder: (f64, f64),
    pub r_elbow: (f64, f64),
    pub r_wrist: (f64, f64),
    pub l_shoulder: (f64, f64),
    pub l_elbow: (f64, f64),
    pub l_wrist: (f64, f64),
    pub club_base: (f64, f64),
    pub club_com: (f64, f64),  // Center of mass
    pub club_tip: (f64, f64),
}
```

### `Vec2`

```rust
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}
```

**Methods:**

- `new(x: f64, y: f64) -> Self`
- `from_polar(r: f64, theta: f64) -> Self` - Create from polar coords
- `dot(self, other: Self) -> f64`
- `cross(self, other: Self) -> f64` - 2D cross product (scalar)
- `add(self, other: Self) -> Self`
- `sub(self, other: Self) -> Self`
- `scale(self, s: f64) -> Self`

## Double Pendulum Module

### Mass Matrix

```rust
pub fn mass_matrix(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2>
```

Computes M(q) such that the kinetic energy is KE = (1/2) qᵀ M(q) q̇.

**Example:**

```rust
let q = [0.1, 0.2];  // [θ₁, φ]
let M = double::mass_matrix(&q, &params);
```

### Coriolis Vector

```rust
pub fn coriolis(q: &[f64; 2], qdot: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2>
```

Computes C(q, q̇) containing centrifugal and Coriolis terms.

### Gravity Vector

```rust
pub fn gravity_vector(q: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2>
```

Computes G(q) = -∂PE/∂q.

### Forward Kinematics

```rust
pub fn forward_kinematics(q: &[f64; 2], params: &DoublePendulumParams) -> DoubleFKResult
```

### Jacobians

```rust
pub fn jacobian_wrist(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2>
pub fn jacobian_club_tip(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2>
```

## Triple Pendulum Module

**Functions:** `mass_matrix`, `coriolis`, `gravity_vector`, `forward_kinematics`
**Jacobians:** `jacobian_joint1`, `jacobian_joint2`, `jacobian_joint3`

All return appropriate nalgebra types (3×3 matrices, 3D vectors, 2×3 Jacobians).

## Golfer Module (8-DOF)

### Analytical FK Jacobians

```rust
pub fn analytical_fk_jacobians(
    q: &[f64; 8],
    params: &GolferParams,
) -> HashMap<String, SMatrix<f64, 2, 8>>
```

Returns a map with keys:

- `"hub"`, `"r_shoulder"`, `"r_elbow"`, `"r_wrist"`
- `"l_shoulder"`, `"l_elbow"`, `"l_wrist"`
- `"club_com"`, `"club_tip"`

Each value is a 2×8 Jacobian matrix.

**Example:**

```rust
let jacs = golfer::analytical_fk_jacobians(&q, &params);
let j_club_tip = jacs.get("club_tip").unwrap();  // 2×8 matrix
```

### Mass Matrix

```rust
pub fn mass_matrix(q: &[f64; 8], params: &GolferParams) -> SMatrix<f64, 8, 8>
```

Computed via M = Σᵢ mᵢ Jᵢᵀ Jᵢ using analytical Jacobians.

### Coriolis Vector

```rust
pub fn coriolis(q: &[f64; 8], qdot: &[f64; 8], params: &GolferParams) -> SVector<f64, 8>
```

Uses finite-difference approximation of dM/dt.

### Gravity Vector

```rust
pub fn gravity_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 8>
```

Computed as G_i = -Σⱼ mⱼ g ∂yⱼ/∂qᵢ.

### Forward Kinematics

```rust
pub fn forward_kinematics(q: &[f64; 8], params: &GolferParams) -> GolferFKResult
```

### Constraints

```rust
pub fn constraint_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 4>
pub fn constraint_jacobian(q: &[f64; 8], params: &GolferParams) -> SMatrix<f64, 4, 8>
```

**Constraints:**

1. Right hand x = Left hand x
2. Right hand y = Left hand y
3. Left hand on club shaft (x component)
4. Left hand on club shaft (y component)

## Constraint Solver Module

### Baumgarte Gains

```rust
pub struct BaumgarteGains {
    pub alpha: f64,  // Velocity error correction gain (typical: 10)
    pub beta: f64,   // Position error correction gain (typical: 10)
}
```

**Method:**

- `default() -> Self` - Returns (alpha=10, beta=10)

### Constraint Acceleration Bias

```rust
pub fn constraint_acceleration_bias(
    q: &[f64; 8],
    qdot: &[f64; 8],
    params: &GolferParams,
) -> SVector<f64, 4>
```

Computes γ = -dJ/dt · q̇ - d²Φ/dq² term.

### Constrained Accelerations (KKT Solver)

```rust
pub fn constrained_accelerations(
    q: &[f64; 8],
    qdot: &[f64; 8],
    tau: &[f64; 8],              // Generalized forces
    params: &GolferParams,
    gains: &BaumgarteGains,
) -> (SVector<f64, 8>, SVector<f64, 4>)
```

**Returns:** (accelerations, Lagrange multipliers)

**System solved:**

```
M(q) a + C(q,q̇) + G(q) = τ + Jᵀ λ
J(q) a + γ(q,q̇) + α J q̇ + β Φ(q) = 0
```

### Constraint Projection

```rust
pub fn project_to_constraints(
    q: &[f64; 8],
    params: &GolferParams,
    max_iters: usize,
    tol: f64,
) -> [f64; 8]
```

Newton projection to satisfy Φ(q) = 0.

```rust
pub fn project_velocity(
    q: &[f64; 8],
    qdot: &[f64; 8],
    params: &GolferParams,
) -> [f64; 8]
```

Minimum-norm velocity correction: J q̇ = -dJ/dt q̇.

## Integrator Module

### RK45 Configuration

```rust
pub struct RK45Config {
    pub h0: f64,         // Initial step size
    pub h_min: f64,      // Minimum step size
    pub h_max: f64,      // Maximum step size
    pub rtol: f64,       // Relative tolerance
    pub atol: f64,       // Absolute tolerance
    pub max_steps: usize,
}
```

**Method:**

- `default() -> Self` - h0=0.01, h_min=1e-6, h_max=0.1, rtol=1e-6, atol=1e-9

### Generic Integration

```rust
pub fn integrate_rk45<F, const N: usize>(
    f: F,
    t0: f64,
    t_end: f64,
    y0: [f64; N],
    config: RK45Config,
) -> Vec<IntegrationStep<N>>
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
```

**RHS function signature:** `f(t, y) -> dy/dt`

### Specialized Integrators

```rust
pub fn integrate_double_pendulum<F>(
    f: F,
    t0: f64,
    t_end: f64,
    q0: [f64; 2],
    qdot0: [f64; 2],
    config: RK45Config,
) -> Vec<IntegrationStep<4>>
where
    F: Fn(f64, &[f64; 2], &[f64; 2]) -> [f64; 2],
```

**RHS function signature:** `f(t, q, qdot) -> qddot`

Similarly for `integrate_triple_pendulum` (6 state) and `integrate_golfer` (16 state).

### Integration Step Result

```rust
pub struct IntegrationStep<const N: usize> {
    pub t: f64,         // Time at this step
    pub y: [f64; N],    // State vector
    pub h: f64,         // Step size used
}
```

**Example:**

```rust
let steps = integrate_double_pendulum(
    |t, q, qdot| {
        let m = double::mass_matrix(q, &params);
        let c = double::coriolis(q, qdot, &params);
        let g = double::gravity_vector(q, &params);
        let m_inv = m.try_inverse().unwrap();
        m_inv * (-c - g)  // M⁻¹(-C-G)
    },
    0.0,
    10.0,
    [0.1, 0.0],  // q0
    [0.0, 0.0],  // qdot0
    RK45Config::default(),
);

for step in steps {
    println!("t={:.4}, q=[{:.4}, {:.4}]", step.t, step.y[0], step.y[1]);
}
```

## Python Bindings (PyO3)

### Classes

```python
class PyDoublePendulumParams:
    def __init__(self, m1, m2, l1, l2, g, friction1, friction2) -> None: ...
    def validate(self) -> None: ...

class PyGolferParams:
    def __init__(
        self, l_hub, m_hub, d_rs, d_ls,
        l_r_upper, m_r_upper, l_r_fore, m_r_fore,
        l_l_upper, m_l_upper, l_l_fore, m_l_fore,
        l_club, m_club, m_clubhead, grip_right, grip_left, g
    ) -> None: ...
    def validate(self) -> None: ...
```

### Functions

```python
def py_double_mass_matrix(q: List[float], params: PyDoublePendulumParams) -> List[List[float]]: ...
def py_double_gravity_vector(q: List[float], params: PyDoublePendulumParams) -> List[float]: ...
def py_double_coriolis(q: List[float], qdot: List[float], params: PyDoublePendulumParams) -> List[float]: ...
def py_double_forward_kinematics(q: List[float], params: PyDoublePendulumParams) -> Dict[str, float]: ...

def py_golfer_mass_matrix(q: List[float], params: PyGolferParams) -> List[List[float]]: ...
def py_golfer_gravity_vector(q: List[float], params: PyGolferParams) -> List[float]: ...
def py_golfer_forward_kinematics(q: List[float], params: PyGolferParams) -> Dict[str, List[float]]: ...
def py_golfer_constraint_vector(q: List[float], params: PyGolferParams) -> List[float]: ...
def py_golfer_constraint_jacobian(q: List[float], params: PyGolferParams) -> List[List[float]]: ...
```

## WASM Bindings

Similar to Python but with `Wasm*Params` classes and error handling via `Result<T, JsValue>`.

```javascript
const params = new WasmGolferParams(...);
const q = new Float64Array(8);
const M = wasm_golfer_mass_matrix(q, params);  // Returns Vec<f64>
```

## Re-exports (lib.rs)

The following are publicly exported from `lib.rs`:

```rust
pub use double::{
    coriolis as double_coriolis,
    forward_kinematics as double_forward_kinematics,
    gravity_vector as double_gravity_vector,
    jacobian_club_tip,
    jacobian_wrist,
    mass_matrix as double_mass_matrix,
};

pub use triple::{
    coriolis as triple_coriolis,
    forward_kinematics as triple_forward_kinematics,
    gravity_vector as triple_gravity_vector,
    jacobian_joint1,
    jacobian_joint2,
    jacobian_joint3,
    mass_matrix as triple_mass_matrix,
};

pub use golfer::{
    analytical_fk_jacobians,
    constraint_jacobian,
    constraint_vector,
    forward_kinematics as golfer_forward_kinematics,
    gravity_vector as golfer_gravity_vector,
    mass_matrix as golfer_mass_matrix,
};

pub use golfer_constraints::{
    constrained_accelerations,
    constraint_acceleration_bias,
    project_to_constraints,
    project_velocity,
    BaumgarteGains,
};

pub use integrator::{
    integrate_double_pendulum,
    integrate_golfer,
    integrate_triple_pendulum,
    RK45Config,
};

pub use types::{
    DoubleFKResult,
    DoublePendulumParams,
    GolferFKResult,
    GolferParams,
    TripleFKResult,
    TriplePendulumParams,
    Vec2,
};

pub use nalgebra::{SMatrix, SVector};
```
