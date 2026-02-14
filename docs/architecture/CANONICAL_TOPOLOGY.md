# Canonical Repository Topology

This document defines the canonical top-level structure for the Tools repository.

## Canonical Top-Level Layout

- `src/`: active product code (shared libraries, tools, web apps, data/media processing)
- `tests/`: repository-level tests and integration contract tests
- `docs/`: user/dev documentation and assessments
- `scripts/`: automation and repo maintenance scripts
- `config/`: policy baselines and runtime/static configuration

## Transitional / Legacy Paths

These paths may exist for compatibility and migration but are not canonical for new work:

- `tools/` (legacy utility subtree)
- `data_processing/`, `media_processing/`, `web_applications/` at repo root
- launcher compatibility files retained for migration windows

## Deprecation Timeline

- Q2 2026: stop adding new modules to transitional paths
- Q3 2026: migrate remaining active modules into `src/` and remove dead roots

## Ownership

- Topology policy owner: repository maintainers
- CI enforcement owner: platform/quality maintainers
