# Fleetwide Rust Implementation Assessment

**Date:** March 13, 2026
**Scope:** `Tools`, `Gasification_Model`, `UpstreamDrift`

## 1. Executive Summary

To achieve an efficient fleet where parity is easily maintained, the organization must fully embrace the **"Hub and Spoke"** model for Rust compilation. Mathematical, thermodynamic, and biomechanical core logic must be strictly centralized in `.rs` crates maintained in the `Tools` repository (The Hub). The remaining flagship repositories (`UpstreamDrift`, `Gasification_Model`) should act as _consumers_ of these core binaries via PyO3 wheels, stripping away Python implementations to avoid "dual-maintenance" parity bugs.

Based on a fleetwide static analysis of computational hotspots (evaluating nested loops, high-density primitive math, and heavy numpy/scipy dependency), the following migration targets are prioritized.

## 2. Computational Hotspots

A quantitative analysis of the Python files across the flagship repositories identifies the following modules as the highest value targets for Rust translation:

| Repository             | Target Module                                                                   | Complexity Score | Priority |
| ---------------------- | ------------------------------------------------------------------------------- | ---------------- | -------- |
| **UpstreamDrift**      | `src/shared/python/spatial_algebra/pose6dof.py`                                 | 3377             | **P0**   |
| **Tools**              | `src/pendulum_simulator/.../double_pendulum_golf/physics.py`                    | 3151             | **P0**   |
| **Tools**              | `src/rotation_converter/modern_robotics.py`                                     | 2946             | **P1**   |
| **UpstreamDrift**      | `src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py`      | 1936             | **P1**   |
| **UpstreamDrift**      | `src/robotics/planning/collision/geometric_primitives.py`                       | 1209             | **P2**   |
| **Gasification_Model** | `src/integrated_process_simulator/core/energy_balance/core.py`                  | 1267             | **P2**   |
| **Gasification_Model** | `src/integrated_process_simulator/glass/calculator_widget.py` (Math extraction) | 1440             | **P3**   |

_Note: The Complexity Score heavily penalizes dense raw math operations (`+`, `-`, `_`), array/scipy allocations, and deep nested loops which are heavily bottlenecked by the Python GIL.\*

## 3. Recommended Implementation Strategy

### 3.1. UpstreamDrift: Biomechanics & Spatial Algebra (P0)

- **Target:** `pose6dof.py` and `geometric_primitives.py`
- **Assessment:** These files are heavily inundated with SE3/SO3 matrix operations and collision bounding-box math. Because these execute during every frame of physical simulation (often at 1000Hz+ stepping), Python object allocation overhead dominates the profile.
- **Action:** Move the `Pose6DOF` class and Raycasting/Collision physics into the `Tools/rust_core/math` primitives library. Export it to Python via PyO3 so `UpstreamDrift` can import the binary `from tools_core_py.math import Pose6DOF`.

### 3.2. Tools / UpstreamDrift: Double Pendulum Golf Dynamics (P0)

- **Target:** `double_pendulum_golf/physics.py` and `modern_robotics.py`
- **Assessment:** The RK4/Euler integrators and the robotic forward kinematics currently execute inside a deep Python nested loop (`golf_swing_physics_engine.py`). Worse, versions of this code are duplicated across `Tools` and `UpstreamDrift`.
- **Action:** Consolidate the forward kinematics and RK4 integration loops into a Rust crate. This will allow the entire swing trajectory (thousands of solver steps) to be computed in a single natively-compiled Rust call, eliminating the Python context-switching overhead entirely while simultaneously fixing the DRY / Parity violation between `Tools` and `UpstreamDrift`.

### 3.3. Gasification_Model: Energy Balance & Convergence Loops (P2)

- **Target:** `energy_balance/core.py`
- **Assessment:** While the Gibbs minimization solver has already been largely transitioned into `rust_core/thermo-solver`, the energy balance Newton-Raphson looping logic and adiabatic temperature convergence logic still live in Python, repeatedly crossing the FFI boundary to query properties.
- **Action:** Port the iterative Energy Balance convergence loop directly into the Rust `thermo-solver` crate. This permits the Newton-Raphson loops to execute fully in native memory without FFI transition costs for every iterative guess.

## 4. Architectural Standard: "The Hub & Spoke"

To ensure parity maintenance is effortless, the following rules must be enforced:

1. **No Duplication:** Do not build a `rust_core` inside `UpstreamDrift` if the tools are general mathematical principles. The `Tools` repository must act as the Hub.
2. **Abstract Boundaries:** Python should only be responsible for:
   - UI / Plotting (PyQt, Matplotlib, Plotly)
   - Configuration routing / Application state.
   - Orchestrating the "Large Steps" (e.g., kicking off a simulation run).
3. **Tight Loops belong in Rust:** Any code utilizing `while residual > tolerance:` or deep `for t in range(steps):` integrating mathematical states MUST be pushed to Rust.
4. **Data serialization:** All data crossing the Python/Rust boundary must be effectively serialized (such as `__getnewargs__` implementation) to support multiprocessing in Python.

## 5. Next Actions

1. Create a `math_primitives` Rust workspace inside `Tools/rust_core`.
2. Delete `vendor/*` copies of `Double Pendulum` from `UpstreamDrift` and `Gasification_Model` to force adherence to the centralized tooling.
3. Migrate `pose6dof.py` into native Rust equivalents.
4. Issue pull requests across `UpstreamDrift` to replace local mathematical integrations with `tools_core_py` imports.
