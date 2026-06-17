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
physics solvers (RK4 ball-flight integration), signal-processing kernels
(bilateral filter, LMS/RLS adaptive filters), and the SCADA kernel
(`tools_core.scada`: `AlarmEngine`, `moving_average`, `exponential_smoothing`).

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

## Distribution Model: build-on-install

The fleet decision (issue #3513) is to distribute the Rust extensions via
**build-on-install** plus **build-in-CI**, rather than publishing prebuilt
wheels to an external index.

### CI builds and tests against the compiled wheel

The `tests` job in `.github/workflows/ci-standard.yml` builds the
`tools_core` wheel and installs it before running pytest (issue #3514):

```yaml
- name: Build + install tools_core wheel
  run: |
    python -m pip install maturin
    maturin build --release --features python,extension-module \
      -m rust_core/tools-core/Cargo.toml
    python -m pip install --force-reinstall target/wheels/*.whl

- name: Assert tools_core imports (hard fail — no silent skip)
  run: python -c "import tools_core; from tools_core import scada"
```

Because `tools-core` is a workspace member, maturin emits the wheel into the
**workspace-root** `target/wheels/` (not `rust_core/tools-core/target/wheels`).

The hard-fail import assertion means a missing or broken extension breaks the
build instead of being silently `importorskip`-skipped. As a result the PyO3
binding tests under `tests/rust_bindings/` now run for real in CI (they are in
the always-run `core_tests` set). The separate `rust-quality-gate` job
additionally builds the wheel + WASM package whenever Rust sources change.

The `ai_backend` crate is built by `.github/workflows/maturin-ai-backend.yml`.

### Installing the Rust extension locally

`pip install ud-tools` installs the pure-Python fallback paths only — the root
package is setuptools and does not auto-build the maturin extension. To get
Rust acceleration, install the `rust` extra (which provides `maturin`) and
build the wheel:

```bash
pip install "ud-tools[rust]"
maturin build --release --features python,extension-module \
    -m rust_core/tools-core/Cargo.toml
pip install target/wheels/*.whl
# or, for an editable in-place build during development:
cd rust_core/tools-core && maturin develop --features python
```

---

## Performance Impact

When the Rust wheel is absent, the following modules transparently fall back to
slower pure-Python (NumPy) implementations and emit a `WARNING`-level log
message so users know they are on the slow path:

| Module                                                           | Rust symbol                                   | Fallback              | Warning emitted      |
| ---------------------------------------------------------------- | --------------------------------------------- | --------------------- | -------------------- |
| `src/shared/python/ai/adapters/rust_adapter.py`                  | `ai_backend.*`                                | `ImportError` at init | Yes -- on `__init__` |
| `src/shared/python/signal_toolkit/bilateral_rust.py`             | `tools_core.signal.bilateral_filter`          | `ImportError` at call | Yes -- on import     |
| `src/shared/python/signal_toolkit/adaptive_filter.py`            | `tools_core.signal.lms_filter` / `rls_filter` | NumPy loop            | Yes -- on import     |
| `src/vessel_drafter/python/vessel_drafter/models/rust_kernel.py` | `tools_core.electrode_advisor`                | pure-Python advisor   | `DeprecationWarning` |
| `src/p1am_control_system/backend/main.py`                        | `tools_core.scada.*`                          | `scada_fallback.py`   | Yes -- on import     |

Typical speedups (when Rust wheel is available):

- **Bilateral filter**: 15-40x on 10 k-sample signals (eliminates Python loop).
- **LMS/RLS adaptive filter**: 20-60x on long signals (no per-sample GIL crossing).
- **AI RAG pipeline**: 3-10x on embedding search (SIMD cosine via ONNX / Rust ndarray).

---

## Opting Into the Rust Path

Set the environment variable `GAS_THERMO_BACKEND=rust` to force Rust acceleration
in the vessel drafter's `rust_kernel.py` (raises `ImportError` rather than
silently falling back when the wheel is missing). Omit the variable or set
`GAS_THERMO_BACKEND=auto` (default) to use Rust when available and Python otherwise.

---

## See Also

- `rust_core/tools-core/README.md` -- crate-level quickstart and design principles
- `rust_core/ai_backend/Cargo.toml` -- feature flag definitions
- `docs/development/rust-setup.md` -- Rust toolchain setup for new contributors
