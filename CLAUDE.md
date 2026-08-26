# CLAUDE.md — Tools

> **GAAI Fleet Member.** GAAI framework installed in `.gaai/`. Read `.gaai/core/GAAI.md` for full governance spec.
> Rules: `@.gaai/core/contexts/rules/base.rules.md` and `@.gaai/project/contexts/rules/project.rules.md`
> All work on `main` branch. PRs target `main`.

## Engineering Design Manual Authority

`manuals/tools` QMD is the only editable engineering design-manual source.
Generated LaTeX, PDF, DOCX, and HTML are non-editable artifacts. Read
`config/design_manual_governance.json`, update the calculation registry, SPEC,
and handoff when their governed pathways change, and run
`python -m scripts.check_design_manual_governance` plus
`python -m scripts.build_tools_module_inventory --check` and
`python -m scripts.lint_tools_textbook_chapters` and
`python -m scripts.render_tools_design_manual --check`. Render only through the
pinned toolchain lock and never edit `manuals/tools/dist` directly. The module inventory
is a strict, LF-normalized, tracked-file baseline; `calculation` means a
provisional candidate and never scientific or operating approval. A successful render is not
scientific, semantic, visual, accessibility, license, or publication approval.
Registered textbook chapters must satisfy the versioned fourteen-section D3
contract; passing its linter is structural evidence, not calculation approval.
Private source material is not permitted in the public Tools manual.

## What This Is

Shared engineering library consumed by UpstreamDrift and Gasification_Model. Monorepo
containing signal processing, URDF generation, process calculators, P&ID utilities,
and visualization themes. Every public API change here is a potential breaking change
for downstream repos.

## Key Directories

- `src/signal_processing_studio/` — DSP utilities, filters, spectral analysis
- `src/urdf_builder_gui/` — URDF model generation and validation
- `src/shared/python/sidekick/process_calculators/` — process engineering calculators
- `src/pid_generator/` — P&ID diagram utilities
- `src/shared/python/plot_theme/` — plotting and visualization themes
- `tests/` — pytest suite organized by module

## Python and Tooling

**The Python floor is two-tier, and the distinction matters.**

- **Root distribution: Python 3.11+.** `pyproject.toml` declares
  `requires-python = ">=3.11"`, the classifiers list 3.11/3.12, and
  `[tool.mypy] python_version` is 3.11. Everything not owned by a sub-package —
  `src/shared/python/`, `src/p1am_control_system/`, the top-level `tests/` tree —
  is root-package code and may use 3.11-only features (`tomllib`, the 3.11
  `asyncio.wait_for`/`asyncio.timeouts` semantics, and so on).
- **Sub-packages and Rust crates: Python 3.10+.** Ten distributions declare
  `requires-python = ">=3.10"` and ship 3.10 wheels from the maturin workflows —
  `movement_optimizer`, `pendulum_simulator`, `pendulum-core`,
  `rotation_converter`, `tools-core`, `swing-core`, `ai_backend`,
  `data-processor-core`, `file_watcher`, and the movement-optimizer crate.
- **`ci-standard.yml` runs `["3.11", "3.12"]` and must not go below 3.11.** That
  job runs the root-package suite — `core_tests` is entirely `tests/**` and
  `src/shared/python/**` — none of which can execute on 3.10, so a 3.10 lane
  there collects nothing and reports configuration noise. Only **3.11** is a
  required check.
- **3.10 support is proven by the maturin workflows**, not by `ci-standard`.
  Each crate's `maturin-*.yml` runs a build + parity gate across 3.10/3.11/3.12
  that builds the wheel, installs it, and asserts the extension imports and the
  native backend is selected. If a sub-package needs a lower interpreter, gate
  it there — do not add a lane to `ci-standard`.
- `conftest.py` reads each package's own declared floor and skips collection of
  anything above the running interpreter, so root-package tests cannot run on a
  sub-floor interpreter locally either. **Do not add per-file version guards** —
  the conftest handles it, and `tests/test_python_version_contract.py` keeps
  every declaration in agreement, including that no `ci-standard` lane drops
  below the root floor and that every 3.10 claim still has a workflow behind it.
- Use `python3`.
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
8. No TODO/FIXME unless tied to a tracked GitHub issue
9. **Sidekick Per-File Coverage:** Every sidekick module under `src/shared/python/sidekick/` (specifically the six target modules: `latex_renderer.py`, `notes_store.py`, `notes_tab.py`, `selected_tab_panel.py`, `tab_context_menu.py`, `symbolic_engine.py` plus any modified sidekick files) must maintain at least **50% per-file coverage**.

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
- **LOD:** No method chains >2 levels. Modules must not import across package boundaries (`signal_processing_studio` must not import from `sidekick.process_calculators`).
- **TDD:** Every new public function needs tests. Contract tests (`-m contract`) guard the API surface downstream repos depend on.
- **Stable API:** No renaming, removing, or signature changes to public functions without a deprecation path and downstream coordination.

## Public-API Contract Policy

To ensure API stability for downstream consumers, the `sidekick` codebase adheres to a strict public-API contract policy:

1. **Explicit Exports (`__all__`):** Every module under `src/shared/python/sidekick/**.py` must define an explicit `__all__` list representing its public API surface. Private names, internal helper functions/classes, and logger/log variables must start with an underscore prefix `_` and must not be included in `__all__`.
2. **AST-Based Stability Validation:** Stability is guarded by the test `tests/test_sidekick_public_api_stability.py` which uses Python's `ast` parser to inspect all public symbols, class methods, function arguments, defaults, and return annotations.
3. **API Baseline:** Collected signatures are compared against `tests/sidekick_api_baseline.json`. The test fails if any public name is removed, or if its signature changes.
4. **Regeneration:** If an API change is planned and coordinated with downstream repositories, the baseline can be regenerated by running:
   ```bash
   python3 -m pytest tests/test_sidekick_public_api_stability.py --regenerate-api-baseline
   ```

## Cross-Repo Dependencies

- **Downstream consumers:** UpstreamDrift, Gasification_Model (via symlink)
- **No upstream fleet dependencies.** Tools is a leaf dependency.
- When modifying public APIs: open Issues in UpstreamDrift and Gasification_Model tracking the migration. Link them in your PR description.

## Rust Crates and Maturin (ai_backend)

The `rust_core/ai_backend/` crate provides Python bindings via
[PyO3](https://pyo3.rs) and is built into a wheel using
[maturin](https://www.maturin.rs).

### Workspace membership

`ai_backend` is listed in the root `Cargo.toml` workspace `members` list.
All workspace-level dependency pins apply (see `[workspace.dependencies]`).

### Feature flags

| Feature            | Purpose                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `python`           | Activates PyO3 `extension-module` linkage required for maturin builds. Must be enabled when building a wheel.    |
| `local-embeddings` | Opt-in ONNX-based local embeddings via `ort` + `tokenizers`. Requires a system ONNX Runtime library (see below). |

### Building a wheel locally

```bash
# Install maturin
pip install maturin

# Build wheel (python bindings only — no local embeddings)
cd rust_core/ai_backend
maturin build --release --features python

# Build wheel with local embeddings enabled
maturin build --release --features "python,local-embeddings"

# Install the built wheel into the active venv
pip install ../../target/wheels/*.whl
```

### ORT_DYLIB_PATH — required for local-embeddings

The `local-embeddings` feature uses `ort` (ONNX Runtime Rust bindings) in
`load-dynamic` mode. This means the native `onnxruntime` shared library is
**not bundled** at compile time. You must point `ort` at a system-installed
copy at runtime via the `ORT_DYLIB_PATH` environment variable.

```bash
# Example (Linux/macOS) — use the onnxruntime installed by pip
ORT_LIB=$(python -c "import onnxruntime, pathlib; \
  print(next(pathlib.Path(onnxruntime.__file__).parent.rglob('libonnxruntime*.so*'), ''))")
export ORT_DYLIB_PATH="$ORT_LIB"

# Example (Windows) — find onnxruntime.dll
$OrtLib = (python -c "import onnxruntime, pathlib; print(next(pathlib.Path(onnxruntime.__file__).parent.rglob('onnxruntime*.dll'), ''))")
$env:ORT_DYLIB_PATH = $OrtLib
```

If `ORT_DYLIB_PATH` is not set the `ai_backend` module will raise an
`OrtError` on first use of any embedding API.

**Do NOT set `ORT_LIB_DIR` or rely on the `download-binaries` ort feature** —
those paths are disabled in this project (see comment in
`rust_core/ai_backend/Cargo.toml`).

### CI — Maturin Wheel Build

The workflow `.github/workflows/maturin-ai-backend.yml` builds wheels for:

- **Platforms:** `ubuntu-latest`, `windows-latest`, `macos-latest`
- **Python versions:** 3.10, 3.11, 3.12, 3.13
- **Features:** `python` (standard) and optionally `python,local-embeddings`
  (manual dispatch only, Linux only due to build time)

The workflow triggers on push/PR to `main` whenever files under
`rust_core/ai_backend/**` or `Cargo.toml` change.

Each built wheel is uploaded as a GitHub Actions artifact
(`wheel-<os>-py<version>`) with a 14-day retention window.

## Slash Commands

- `/gaai-deliver` — Run Delivery Loop for next ready backlog item
- `/gaai-status` — Show current backlog and memory state

## Hook bypass policy

**Never use `git commit --no-verify` or `git push --no-verify` unless the hook itself is broken** (tooling not installed, hook script crashes). It is _not_ an acceptable workaround for a hook that flags real issues.

### When a hook fails on something you didn't touch

The hook is scoped to _your diff_. If `fleet-fast-guardrails` or any other guardrail reports a violation in a file you didn't change, that's a regression — file an issue against `Repository_Management`. Bypassing locally doesn't help: the same checks run in CI's `quality-gate` and will block the PR.

### When the hook is legitimately broken

Open an issue in `Repository_Management`. If you must bypass once to land an urgent fix, include the hook error in the commit body and link the tracking issue. **Do not normalize `--no-verify` as a workaround.**

### Enforcement

Branch protection requires the CI `quality-gate` check on every PR. That check runs the same lint, format, type, and security gates as the hooks. `--no-verify` only delays feedback — it cannot land code that would have failed the hook.

For the canonical hook contract, see [`Repository_Management/docs/FLEET_HOOK_STANDARDS.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/FLEET_HOOK_STANDARDS.md).

## Agent Handoff & PR Policy

Fleet-wide policy from `Repository_Management#1390`, binding in this repo:

1. **Full PRs, never drafts.** Every PR opens ready-for-review — do not use
   `gh pr create --draft`.
2. **Commit frequently.** Small, conventional commits saving progress as you
   go; never batch a day's work into one commit.
3. **Agent handoff documents, updated every PR and every push to main.**
   - Root `AGENT_HANDOFF.md` — fleet/monorepo-wide view: active epics with
     one-line status each, pointers to per-tool handoff docs, exact gate
     commands, a do-not list, and the ordered short-term roadmap.
   - Per-tool `src/<tool>/AGENT_HANDOFF.md` for actively developed tools
     (minimum: `rate_of_closure`, `pendulum_simulator`, `rotation_converter`;
     add one for any other tool once it has an active epic or sustained
     agent traffic). Same structure as the root doc, scoped to that tool.
   - New tools: copy `docs/AGENT_HANDOFF_TEMPLATE.md` to
     `src/<new_tool>/AGENT_HANDOFF.md` and fill it in from the tool's actual
     state — no placeholders.
   - All handoff docs are current-state only, ≤150 lines — history lives in
     git, not in a changelog inside the file.
4. **SPEC.md stays current.** Per the existing `spec-check.yml` gate, any
   PR touching `src/**` (including `AGENT_HANDOFF.md` files, since they live
   under `src/<tool>/`) must add a dated row to SPEC.md §12 Change Log, or
   carry the `spec-exempt` label.
