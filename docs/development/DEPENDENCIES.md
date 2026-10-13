# Dependency Pinning Policy

This document describes the dependency version strategy for this repository.

## Rust (Optional Native Extensions)

Two Rust crates — `rust_core/tools-core` and `rust_core/ai_backend` — provide optional
acceleration via PyO3/maturin wheels. These are **not Python dependencies** and are
therefore not listed in `requirements.txt` or `pyproject.toml`. They are purely
additive: when absent, pure-Python fallbacks activate automatically and emit a
`WARNING`-level log so users know they are on the slow path.

To build locally: `pip install maturin && cd rust_core/tools-core && maturin develop
--features python` (repeat for `rust_core/ai_backend`).

See [docs/development/rust_distribution.md](docs/development/rust_distribution.md) for
the full distribution model, the CI gap, and build instructions.

## Overview

This repo is a **shared library** consumed by UpstreamDrift and Gasification_Model.
Library packaging best practice (PEP 517) requires different strategies at different layers:

| Layer               | File                     | Strategy            | Rationale                                                            |
| ------------------- | ------------------------ | ------------------- | -------------------------------------------------------------------- |
| Library package     | `pyproject.toml`         | `>=` minimum bounds | Consumers set their own pins; over-constraining breaks compatibility |
| Dev/CI environment  | `requirements.txt`       | `>=` minimum bounds | Paired with lock file for reproducibility                            |
| Reproducible builds | `requirements-lock.txt`  | `==` exact pins     | Used in CI and for reproducing specific environments                 |
| Sub-module apps     | `src/*/requirements.txt` | `>=` minimum bounds | Each sub-module is independently deployable; they own their pins     |

## Pinning Tiers

### Tier 1 — Library metadata (`pyproject.toml`)

Runtime dependencies use **minimum version bounds** (`>=`). This is the correct approach
for a library. Downstream consumers (UpstreamDrift, Gasification_Model) apply their own
pins on top of these minimums.

Example: `"numpy>=1.24.0"` — correct, do NOT pin to `==` here.

### Tier 2 — Development/CI environment (`requirements.txt`)

The root `requirements.txt` specifies minimum bounds for the development environment.
These are the versions actually tested in CI. The companion `requirements-lock.txt`
provides exact reproducible pins.

To regenerate the lock file:

```bash
pip install -r requirements.txt
pip freeze > requirements-lock.txt
```

### Tier 3 — Reproducible builds (`requirements-lock.txt`)

Exact `==` pins for every package. Used when:

- Setting up a fresh CI environment
- Reproducing a specific build for debugging
- Ensuring bit-for-bit reproducibility across machines

**Regenerate when:** upgrading any dependency in `requirements.txt`, or at the start
of each sprint/release cycle.

### Tier 4 — Sub-module requirements (`src/*/requirements.txt`)

Individual tools and applications under `src/` maintain their own requirements files.
These should use `>=` minimum bounds at minimum. Completely unpinned dependencies
(no version specifier) are not permitted for production packages.

**Exception:** Standard library modules (e.g., `tkinter`) cannot be pip-installed
and should not have version specifiers.

## PyQt6 binary stack (coupled pins)

`PyQt6`, `PyQt6-Qt6`, and `PyQt6-sip` are **one ABI stack**. SIP is the binary shim
each PyQt6 wheel is built against, so `PyQt6-sip` must never move independently of the
pinned `PyQt6` release.

| Context | PyQt6 | PyQt6-Qt6 | PyQt6-sip |
| ------- | ----- | --------- | --------- |
| CI lock (`requirements-lock.txt`, `requirements-rate-pyqt.txt`) | 6.11.0 | 6.11.2 | 13.12.0 |
| Owner Windows dev host (MSVC runtime limit) | 6.8.1 | (wheel-bundled) | must stay compatible with 6.8.1 |

Dependabot PR #5486 proposed `PyQt6-sip` 13.13.0 alone; it was closed during the
2026-10-10 dependency consolidation (#5499) and tracked as issue #5508. Until a
**coordinated** PyQt6 upgrade is evaluated on both the CI lock and the owner 6.8.1
pin with Qt test shards green, keep `PyQt6-sip` at **13.12.0**.

Machine-readable authority: `config/pyqt6_binary_stack_policy.v1.json`, enforced by
`python scripts/check_pyqt6_binary_stack_policy.py`.

## Security Pinning

Security-sensitive packages may use exact pins in `requirements.txt` with an explanatory
comment (e.g., CVE reference). Example:

```
pygments>=2.18.0,<2.19.0  # Pinned: CVE-2026-4539 affects 2.19.x
```

## Intentionally Floating Dev Tools

Developer tools (formatters, linters, type checkers) in `[project.optional-dependencies].dev`
use `>=` minimum bounds intentionally. Developers are expected to use recent versions;
exact pinning of dev tools can create friction without meaningful stability benefit.

## Policy Compliance

New dependencies added to any requirements file MUST include a version specifier.
CI enforces this via the `ruff check` and manifest validation steps.

Completely unpinned packages (e.g., `fastapi` with no version) are not permitted.
Use at minimum `fastapi>=0.100.0` or the version you've tested against.
