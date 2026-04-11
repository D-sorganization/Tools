# CLAUDE.md — Tools

> **GAAI Fleet Member.** GAAI framework installed in `.gaai/`. Read `.gaai/core/GAAI.md` for full governance spec.
> Rules: `@.gaai/core/contexts/rules/base.rules.md` and `@.gaai/project/contexts/rules/project.rules.md`
> All work on `staging` branch. PRs target `staging`. Never push directly to `main`.

## What This Is

Shared engineering library consumed by UpstreamDrift and Gasification_Model. Monorepo
containing signal processing, URDF generation, process calculators, P&ID utilities,
and visualization themes. Every public API change here is a potential breaking change
for downstream repos.

## Key Directories

- `src/signal_processing/` — DSP utilities, filters, spectral analysis
- `src/urdf/` — URDF model generation and validation
- `src/calculators/` — process engineering calculators
- `src/pid/` — P&ID diagram utilities
- `src/themes/` — plotting and visualization themes
- `tests/` — pytest suite organized by module
- `manifests/` — package manifests (validated by CI)

## Python and Tooling

- **Python 3.10+**. Use `python3`.
- **Formatter:** Ruff format (NOT Black). 88-char line limit.
- **Linter:** Ruff check. Both are separate CI steps.

## Development Commands

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m ruff format .                          # auto-format
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m unit -n auto                 # unit only
python3 -m pytest -m contract                     # API contract tests
python3 -m pytest -m integration --timeout=60     # cross-repo integration
python3 -m pytest -m dwsim                        # DWSIM integration tests
```

## CI Requirements (All Must Pass)

1. `ruff check` — zero violations
2. `ruff format --check` — zero diffs
3. Changed-file delta checks: ruff and mypy on diff only (but full-repo checks also run)
4. Cross-repo integration tests — verify UpstreamDrift and Gasification_Model compatibility
5. Manifest validation — adding a module requires a manifest entry
6. pytest with **10% coverage minimum**, must not regress on touched files
7. No `print()` in `src/` — use logging
8. No TRACKED_TASK/TRACKED_DEFECT unless tied to a tracked GitHub issue

## Test Markers (13 Total)

`unit`, `integration`, `e2e`, `slow`, `contract`, `acceptance`, `dwsim`,
`benchmark`, `scientific`, `live_simulation`, `headless_safe`, `requires_gl`, `parity`

Key markers:
- `contract` — API surface tests that downstream repos depend on. Breaking a contract test means you broke UpstreamDrift or Gasification_Model.
- `dwsim` — requires DWSIM integration environment
- `e2e` — full pipeline tests across module boundaries

## Known Constraints

- **This is a shared library.** UpstreamDrift and Gasification_Model import from it. Any change to a public function signature, return type, or exception behavior is a breaking change.
- **Breaking changes require coordinated PRs** in downstream repos. Open them simultaneously, referencing this PR.
- **Delta CI:** CI lints/typechecks only changed files for speed, but full checks also run. A file passing delta can still fail full-repo if it introduces transitive issues.
- **Manifest discipline:** New modules require a manifest entry. CI will reject PRs that add modules without updating manifests.

## Coding Standards (Enforced by CI and QA)

- **DRY:** This repo IS the DRY layer. Duplicated logic in UpstreamDrift or Gasification_Model belongs here.
- **DbC:** Every public function validates inputs. `TypeError` for wrong types, `ValueError` for out-of-range. Document preconditions in docstrings.
- **LOD:** No method chains >2 levels. Modules must not import across package boundaries (signal_processing must not import from calculators).
- **TDD:** Every new public function needs tests. Contract tests (`-m contract`) guard the API surface downstream repos depend on.
- **Stable API:** No renaming, removing, or signature changes to public functions without a deprecation path and downstream coordination.

## Cross-Repo Dependencies

- **Downstream consumers:** UpstreamDrift, Gasification_Model (via symlink)
- **No upstream fleet dependencies.** Tools is a leaf dependency.
- When modifying public APIs: open Issues in UpstreamDrift and Gasification_Model tracking the migration. Link them in your PR description.

## Slash Commands

- `/gaai-deliver` — Run Delivery Loop for next ready backlog item
- `/gaai-status` — Show current backlog and memory state

## Specification

This repository's specification is defined in `SPEC.md` at the repo root.
Read SPEC.md before making any changes. Update it when your changes
affect documented functionality, features, or architecture.
