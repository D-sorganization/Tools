# Rust Development Setup

## Prerequisites

- **Rust toolchain**: Installed automatically via `rust-toolchain.toml` (stable channel).
- **Cargo**: Comes with Rust.
- **Maturin** (for Python wheels): `pip install maturin`
- **wasm-pack** (for WASM builds): `cargo install wasm-pack`

## Project Structure

```
Tools/
├── Cargo.toml               # Workspace root
├── rust-toolchain.toml       # Pins stable Rust
├── rust_core/
│   └── tools-core/
│       ├── Cargo.toml        # Crate manifest (PyO3 + WASM feature gates)
│       ├── src/
│       │   ├── lib.rs        # Entry point, module declarations, Python bindings
│       │   ├── types.rs      # Vector3, Quaternion, Matrix3 (canonical types)
│       │   └── math.rs       # lerp, clamp, angle conversions (utility functions)
│       └── benches/
│           └── math_benchmarks.rs   # Criterion benchmarks
```

## Quick Start

```bash
# Build
cargo build

# Run tests (TDD)
cargo test

# Lint (DbC enforcement)
cargo clippy --all-targets -- -D warnings

# Format check (style)
cargo fmt --all -- --check

# Benchmarks
cargo bench
```

## Feature Gates

The crate uses Cargo features to conditionally compile bindings:

| Feature   | Purpose               | Build Command                     |
| --------- | --------------------- | --------------------------------- |
| (default) | Pure Rust library     | `cargo build`                     |
| `python`  | PyO3 Python extension | `maturin build --features python` |
| `wasm`    | WASM/NPM package      | `wasm-pack build --features wasm` |

## Design Principles

### TDD (Test-Driven Development)

Every module has `#[cfg(test)] mod tests { ... }` blocks written **before** the implementation is finalized. No public function exists without test coverage.

### DbC (Design by Contract)

- All public functions validate preconditions via `debug_assert!()`.
- Functions that can fail return `Result<T, E>` instead of panicking.
- NaN values are rejected at construction boundaries.

### DRY (Don't Repeat Yourself)

- Types (Vector3, etc.) are defined once in Rust. Python and WASM consumers get the exact same compiled logic via thin bindings.
- Workspace-level dependency pinning ensures all crates use identical versions.
