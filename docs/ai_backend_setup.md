# AI Backend Setup Guide

This guide covers the full setup for `rust_core/ai_backend`, with particular focus on
the `local-embeddings` feature, which requires a working ONNX Runtime installation.

---

## 1. Building the Extension

```bash
# Install maturin (once)
pip install maturin

# Core wheel (no local embeddings)
cd rust_core/ai_backend
maturin develop --features python

# With local ONNX-based embeddings
maturin develop --features python,local-embeddings
```

---

## 2. `local-embeddings` and ONNX Runtime

The `local-embeddings` feature enables offline embedding via an
`all-MiniLM-L6-v2` ONNX model instead of a remote HTTP endpoint.

### Why `ort = "=2.0.0-rc.10"` is pinned

`ort` 2.x is still in release-candidate. The API surface moves between RC
releases and there was an ort/ureq/TLS breakage on Windows MSVC that made the
`download-binaries` build-time path unusable (see crate issue #5227). We pin
`ort = "=2.0.0-rc.10"` and `ort-sys = "=2.0.0-rc.10"` so:

- Every developer and CI runner uses an identical, tested crate API.
- The build does **not** try to download onnxruntime binaries at compile time.
  Instead `ort` is compiled with `features = ["load-dynamic"]`, so the native
  library is loaded at **run time** via `ORT_DYLIB_PATH`.

### Downloading ONNX Runtime

Download a pre-built release from the official Microsoft releases page:

> <https://github.com/microsoft/onnxruntime/releases>

Pick the release that matches your platform and architecture. You want the
**shared-library / dynamic-library** asset, e.g.:

| Platform     | Asset name pattern             |
| ------------ | ------------------------------ |
| Linux x86-64 | `onnxruntime-linux-x64-*.tgz`  |
| macOS x86-64 | `onnxruntime-osx-x86_64-*.tgz` |
| macOS arm64  | `onnxruntime-osx-arm64-*.tgz`  |
| Windows x64  | `onnxruntime-win-x64-*.zip`    |

Use version **1.18.x** or **1.19.x** — these are the versions validated against
`ort 2.0.0-rc.10`.

---

## 3. Per-OS Setup

### Linux

```bash
# Extract the tarball
tar -xzf onnxruntime-linux-x64-1.18.1.tgz

# Option A – point directly at the .so file
export ORT_DYLIB_PATH=/path/to/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so

# Option B – add the lib directory to the system search path
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH
```

When using Option B, `ORT_DYLIB_PATH` can be left unset; `ort` will search
`LD_LIBRARY_PATH` automatically.

### macOS

```bash
# Extract the tarball
tar -xzf onnxruntime-osx-arm64-1.18.1.tgz   # or x86_64 variant

# Point at the .dylib
export ORT_DYLIB_PATH=/path/to/onnxruntime-osx-arm64-1.18.1/lib/libonnxruntime.dylib

# macOS also checks DYLD_LIBRARY_PATH as an alternative to ORT_DYLIB_PATH
# export DYLD_LIBRARY_PATH=/path/to/onnxruntime-osx-arm64-1.18.1/lib:$DYLD_LIBRARY_PATH
```

### Windows

1. Download `onnxruntime-win-x64-1.18.1.zip` from the releases page.
2. Extract the zip, e.g. to `C:\onnxruntime-win-x64-1.18.1\`.
3. Set the environment variable in PowerShell (current session):

```powershell
$env:ORT_DYLIB_PATH = "C:\onnxruntime-win-x64-1.18.1\lib\onnxruntime.dll"
```

Or set it permanently via **System Properties → Environment Variables** so it
persists across terminal sessions.

> **Tip:** Add `C:\onnxruntime-win-x64-1.18.1\lib\` to your `PATH` as well.
> Some Windows environments need both.

---

## 4. Preflight Check

Before starting an application that uses `use_local_embeddings=True`, run the
Python preflight helper to verify the runtime is loadable:

```bash
python -m src.shared.python.ai._onnx_preflight
```

If `ORT_DYLIB_PATH` is unset or the library cannot be loaded you will see a
descriptive error message with a link to this document.

You can also call it from your own code:

```python
from src.shared.python.ai._onnx_preflight import check_ort_loadable

check_ort_loadable()  # raises RuntimeError on failure, returns None on success
```

---

## 5. Troubleshooting

### `DllNotFoundException` / `OSError: cannot open shared object file`

- Verify `ORT_DYLIB_PATH` points to the **exact DLL/SO/dylib file**, not to a
  directory.
- On Windows, make sure you extracted the zip and that the file exists at the
  path. Check with:

  ```powershell
  Test-Path $env:ORT_DYLIB_PATH
  ```

- On Linux/macOS run `ldd` / `otool -L` on the library to check its own
  dependencies are satisfied.

### Version mismatch

`ort 2.0.0-rc.10` expects an ONNX Runtime **C API** at version ≥ 1.17.
Using a library older than 1.17 will cause a version-check panic at startup.

Check the version of an extracted release:

```bash
# Linux/macOS
strings libonnxruntime.so | grep "OnnxRuntime"
# or check the extracted Release Notes
```

If you see `OrtGetApiBase` symbol errors, your ONNX Runtime library is too old.

### Silently returning empty embeddings

When `ORT_DYLIB_PATH` was not set before the old code path, the Rust library
would fall back to zero vectors without logging. That silent failure is what
issue #2777 fixed. With the current code the preflight raises immediately.

### `use_local_embeddings` flag ignored

Verify the wheel was built with the right features:

```bash
python -c "import ai_backend; print(ai_backend.__doc__)"
# look for "local-embeddings" in the output or rebuild:
maturin develop --features python,local-embeddings
```

---

## 6. Quick-Reference Checklist

- [ ] Downloaded ONNX Runtime ≥ 1.17 for my platform.
- [ ] `ORT_DYLIB_PATH` points to the shared library file.
- [ ] `ai_backend` wheel built with `--features python,local-embeddings`.
- [ ] Preflight check passes: `python -m src.shared.python.ai._onnx_preflight`.
