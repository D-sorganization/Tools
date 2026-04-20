# Dependency Pinning Policy

This document describes the dependency version strategy for this repository.

## Overview

This repo is a **shared library** consumed by UpstreamDrift and Gasification_Model.
Library packaging best practice (PEP 517) requires different strategies at different layers:

| Layer | File | Strategy | Rationale |
|-------|------|----------|-----------|
| Library package | `pyproject.toml` | `>=` minimum bounds | Consumers set their own pins; over-constraining breaks compatibility |
| Dev/CI environment | `requirements.txt` | `>=` minimum bounds | Paired with lock file for reproducibility |
| Reproducible builds | `requirements-lock.txt` | `==` exact pins | Used in CI and for reproducing specific environments |
| Sub-module apps | `src/*/requirements.txt` | `>=` minimum bounds | Each sub-module is independently deployable; they own their pins |

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
