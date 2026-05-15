# ai_backend

High-performance Rust AI backend for the Tools monorepo. Provides vector
search, RAG pipelines, memory management, and (optionally) local ONNX-based
embeddings via the `local-embeddings` Cargo feature.

## Features

| Cargo feature      | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `python`           | PyO3 bindings — required for `maturin develop`          |
| `local-embeddings` | Offline embedding via ONNX Runtime (`all-MiniLM-L6-v2`) |

## Quick Start

```bash
# Core wheel (uses a remote embedding endpoint)
cd rust_core/ai_backend
maturin develop --features python

# With local ONNX embeddings (requires ORT_DYLIB_PATH — see below)
maturin develop --features python,local-embeddings
```

## ONNX Runtime Setup (`local-embeddings`)

The `local-embeddings` feature loads the ONNX Runtime shared library at
**run time** via the `ORT_DYLIB_PATH` environment variable. If this variable is
not set or points at an invalid file you will receive a clear error from the
Python preflight helper before any silent failures occur.

**Full setup instructions, per-OS steps, download links, and troubleshooting
are in [`docs/ai_backend_setup.md`](../../docs/ai_backend_setup.md).**

### TL;DR (Windows)

```powershell
# 1. Download onnxruntime-win-x64-1.18.1.zip from:
#    https://github.com/microsoft/onnxruntime/releases
# 2. Extract to C:\onnxruntime-win-x64-1.18.1\
$env:ORT_DYLIB_PATH = "C:\onnxruntime-win-x64-1.18.1\lib\onnxruntime.dll"
# 3. Run preflight
python -m src.shared.python.ai._onnx_preflight
```

### TL;DR (Linux)

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so
python -m src.shared.python.ai._onnx_preflight
```

### TL;DR (macOS)

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime-osx-arm64-1.18.1/lib/libonnxruntime.dylib
python -m src.shared.python.ai._onnx_preflight
```

## Why `ort = "=2.0.0-rc.10"` is pinned

`ort` 2.x is still in release-candidate with a moving API surface. We also
compile with `features = ["load-dynamic"]` to avoid the build-time
`download-binaries` path (which had a ureq/TLS failure on Windows MSVC). The
exact pin ensures every developer and CI runner uses a tested, stable API.
See [`docs/ai_backend_setup.md`](../../docs/ai_backend_setup.md) for details.

## Running Tests

```bash
# Rust unit tests (no ONNX required)
cargo test -p ai_backend

# Rust tests with local-embeddings (ORT_DYLIB_PATH must be set)
cargo test -p ai_backend --features local-embeddings

# Python adapter tests
python -m pytest tests/shared/python/ai/ -v
```
