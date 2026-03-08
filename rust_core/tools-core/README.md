# tools-core

Shared simulation kernel providing math primitives, physics solvers, and thermodynamic calculators used across the organization's repositories.

## Architecture

```
tools-core/
├── src/
│   ├── lib.rs          # Crate root, re-exports, PyO3/WASM entry points
│   ├── math.rs         # Scalar math: clamp, lerp, deg_to_rad, rad_to_deg
│   ├── types.rs        # Vector3: 3D vector with full arithmetic
│   ├── matrix3.rs      # Matrix3: 3x3 matrix (rotation, determinant, inverse)
│   ├── quaternion.rs   # Quaternion: unit quaternion for 3D rotations
│   └── ball_flight.rs  # Ball flight types and RK4 trajectory simulation
├── benches/
│   └── math_benchmarks.rs  # Criterion benchmarks
└── Cargo.toml
```

## Feature Flags

| Feature  | Purpose                         | Build Command                       |
| -------- | ------------------------------- | ----------------------------------- |
| `python` | PyO3 bindings for Maturin wheel | `maturin develop --features python` |
| `wasm`   | wasm-bindgen for NPM package    | `wasm-pack build --features wasm`   |

## Quick Start

```bash
# Run tests
cargo test

# Run with Python bindings
maturin develop --features python
python -c "import tools_core; print(tools_core.Vector3(1, 2, 3))"

# Build WASM package
wasm-pack build --target web --features wasm
```

## Downstream Crates

- **upstream-physics** (UpstreamDrift): Uses `Vector3` for physics types
- **thermo-solver** (Gasification_Model): Uses `Vector3`, `clamp`, `lerp` for thermodynamic calculations

## Test Count

- 71 Rust unit tests (TDD)
- Criterion benchmarks for performance regression tracking
- cargo fmt + clippy enforced in CI

## Design Principles

- **DRY**: Single source of truth for math primitives across all repos
- **DbC**: `debug_assert!` on all function inputs (27 assertions)
- **No unsafe**: Zero `unsafe` blocks in library code
- **No unwrap()**: Library paths use `Result`/`Option` instead of panicking
