# Rust Acceleration: Distribution Model and CI Gap

## Overview

Two Rust crates in this repository provide optional, high-performance acceleration
for signal processing and AI inference workloads. Both crates expose PyO3 bindings
built via [maturin](https://www.maturin.rs/) and fall back to pure-Python
implementations when the wheel is absent.

---

## Crate Inventory

### `rust_core/tools-core`

**Package name (Python):** `tools_core`

Shared simulation kernel: math primitives (Vector3, Matrix3, Quaternion),
physics solvers (RK4 ball-flight integration), and signal-processing kernels
(bilateral filter, LMS/RLS adaptive filters).

Key source files:

```
rust_core/tools-core/src/
  lib.rs           # crate root, PyO3 / WASM entry points
  math.rs          # scalar math: clamp, lerp, deg/rad conversions
  types.rs         # Vector3 with full arithmetic
  matrix3.rs       # 3x3 matrix (rotation, determinant, inverse)
  quaternion.rs    # unit quaternion for 3D rotations
  ball_flight.rs   # RK4 trajectory simulation
```

Feature flags:

| Flag     | Purpose                         | Build command                       |
| -------- | ------------------------------- | ----------------------------------- |
| `python` | PyO3 bindings for maturin wheel | `maturin develop --features python` |
| `wasm`   | wasm-bindgen for NPM package    | `wasm-pack build --features wasm`   |

### `rust_core/ai_backend`

**Package name (Python):** `ai_backend`

High-performance AI inference backend: vector-similarity search, RAG pipeline,
SQLite-backed memory manager, and async LLM streaming over SSE. Optionally
includes local ONNX embeddings via the `local-embeddings` feature (requires
`ORT_DYLIB_PATH` at runtime).

Key feature flags:

| Flag               | Purpose                                    | Build command                                        |
| ------------------ | ------------------------------------------ | ---------------------------------------------------- |
| `python`           | PyO3 bindings (required for maturin wheel) | `maturin develop --features python`                  |
| `local-embeddings` | ONNX `all-MiniLM-L6-v2` offline embeddings | `maturin develop --features python,local-embeddings` |

---

## Building Locally

Prerequisites: Rust toolchain (`rustup`), Python 3.10+, maturin.

```bash
# Install maturin into your active virtual environment
pip install maturin

# Build and install tools_core into the active interpreter
cd rust_core/tools-core
maturin develop --features python

# Build and install ai_backend (basic)
cd rust_core/ai_backend
maturin develop --features python

# Build ai_backend with local ONNX embeddings
cd rust_core/ai_backend
maturin develop --features python,local-embeddings
```

After `maturin develop`, the wheel is installed into the virtual environment
as an editable in-place extension. Subsequent `import tools_core` and
`import ai_backend` calls will use the compiled Rust code.

To verify:

```python
import tools_core
print(tools_core.Vector3(1, 2, 3))   # should print a Vector3 object

import ai_backend
print(ai_backend.__doc__)            # should print the crate doc string
```

---

## The CI Gap

**Today there is no automated wheel build in CI.**

The consequence is that `pip install tools` (or any `pip install` of downstream
packages that depend on this repo) will install the pure-Python fallback paths
only. Users who want Rust acceleration must run `maturin develop` manually.

The missing workflow would need to:

1. Build a maturin wheel for each `(OS, Python version)` combination.
2. Upload the wheels as release assets or to a private PyPI index.
3. Run the test suite against the compiled wheel (not the pure-Python path).

### Required CI Workflow Spec

A future `build-rust-wheels.yml` workflow (ops ticket, separate from
CLAUDE.md-governed files) should implement:

```
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
  python-version: ["3.10", "3.11", "3.12", "3.13"]

steps:
  - uses: actions/checkout@v4
  - uses: PyO3/maturin-action@v1          # OR use cibuildwheel
    with:
      command: build
      args: --release --features python
      working-directory: rust_core/tools-core
  # repeat for rust_core/ai_backend
  - uses: actions/upload-artifact@v4
    with:
      name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
      path: target/wheels/*.whl
```

Alternative: `cibuildwheel` with a `[tool.cibuildwheel]` section in each
crate's `pyproject.toml`. Both approaches are viable; `maturin-action` is
simpler for pure-Rust wheels without C dependencies.

---

## Performance Impact

When the Rust wheel is absent, the following modules transparently fall back to
slower pure-Python (NumPy) implementations and emit a `WARNING`-level log
message so users know they are on the slow path:

| Module                                                           | Rust symbol                                   | Fallback                   | Warning emitted      |
| ---------------------------------------------------------------- | --------------------------------------------- | -------------------------- | -------------------- |
| `src/shared/python/ai/adapters/rust_adapter.py`                  | `ai_backend.*`                                | `ImportError` at init      | Yes — on `__init__`  |
| `src/shared/python/signal_toolkit/bilateral_rust.py`             | `tools_core.signal.bilateral_filter`          | `ImportError` at call time | Yes — on import      |
| `src/shared/python/signal_toolkit/adaptive_filter.py`            | `tools_core.signal.lms_filter` / `rls_filter` | NumPy loop                 | Yes — on import      |
| `src/vessel_drafter/python/vessel_drafter/models/rust_kernel.py` | `tools_core.electrode_advisor`                | pure-Python advisor        | `DeprecationWarning` |

Typical speedups (when Rust wheel is available):

- **Bilateral filter**: 15–40x on 10 k-sample signals (eliminates Python loop).
- **LMS/RLS adaptive filter**: 20–60x on long signals (no per-sample GIL crossing).
- **AI RAG pipeline**: 3–10x on embedding search (SIMD cosine via ONNX / Rust ndarray).

---

## Opting Into the Rust Path

Set the environment variable `GAS_THERMO_BACKEND=rust` to force Rust acceleration
in the vessel drafter's `rust_kernel.py` (raises `ImportError` rather than
silently falling back when the wheel is missing). Omit the variable or set
`GAS_THERMO_BACKEND=auto` (default) to use Rust when available and Python otherwise.

---

## See Also

- `rust_core/tools-core/README.md` — crate-level quickstart and design principles
- `rust_core/ai_backend/Cargo.toml` — feature flag definitions
- `docs/development/rust-setup.md` — Rust toolchain setup for new contributors
