# Comprehensive Assessment Summary

**Overall Score: 72/100**
**Assessment Date: 2026-02-10**
**Assessed By: Claude Code (Automated Deep Review -- Adversarial)**
**Model: Claude Opus 4.6**

---

## Executive Summary

- **The recent decoupling wave (PRs #636-#644) was a genuine architectural inflection point.** The repository moved from sys.path spaghetti and god-file launchers to a bootstrap/gui_registration/thin-wrapper pattern that is now consistently applied across 46+ launch scripts. This is real progress, not cosmetic.
- **Test suite is healthy: 796 passing, 3 skipped, 0 failures.** Coverage of the shared modules is only 9%, however, meaning the test suite validates the *happy path* of the calc_backend and launcher infrastructure but leaves the vast majority of process calculator logic, UI code, and shared utilities unexercised.
- **Security posture has clear gaps: `eval()` with neutered `__builtins__` in two files, 10 files using `xml.etree.ElementTree` instead of `defusedxml`, 4 CORS wildcard origins, 1 `shell=True` subprocess call, and 63 production files containing `print()` statements.** None are actively exploitable in the current local-dev context, but they represent latent risk if any tool is ever exposed to untrusted input.
- **The "Jules" CI/CD automation fleet (33 of 48 workflows) is impressively ambitious but over-engineered.** Nightly agent schedules, auto-refactoring bots, and assessment generators create maintenance surface area disproportionate to the team size. Several workflows overlap in scope.
- **Legacy code remains: `Data_Processor_r0.py` at 9,000 lines, `Folders_Tool_r0.py` at 3,291 lines, and the electrode advisor main_window at 4,384 lines are unreformed monoliths.** The decoupling wave did not touch these; they remain as technical debt anchors.

---

## Score Breakdown

| Category | Assessment | Score | Weight | Weighted | Key Finding | Priority |
|:---------|:-----------|:-----:|:------:|:--------:|:------------|:---------|
| **A** Architecture & Implementation | Good modular structure post-decoupling; legacy monoliths remain | 7 | 2.0x | 14.0 | Bootstrap/gui_registration pattern well-applied | MAJOR: legacy god-files |
| **B** Code Quality & Hygiene | Ruff clean, 8 black violations, 96.9% type hint coverage | 7 | 1.5x | 10.5 | 155 files with broad `except Exception`; 63 files with print() | MAJOR |
| **C** Documentation & Comments | 90.8% docstring coverage in shared; 60 READMEs; 1,361-line user manual | 8 | 1.0x | 8.0 | Tutorials are thin (52+47 lines) | MINOR |
| **D** User Experience & Developer Journey | Unified launcher, dual PyQt6/React, consistent gui_registration | 8 | 2.0x | 16.0 | tools.json only lists 9 categories, not individual tool count | MINOR |
| **E** Performance & Scalability | RuntimeWarning overflow in syngas_water calc; no profiling infrastructure | 5 | 1.5x | 7.5 | No caching, no lazy loading, sleep(2) in web launcher | MAJOR |
| **F** Installation & Deployment | pyproject.toml with optional deps; requirements-lock.txt is incomplete | 6 | 1.5x | 9.0 | Lock file uses >= for some deps (not truly locked) | MAJOR |
| **G** Testing & Validation | 799 tests collected, 796 pass; 9% shared coverage; no UI tests | 6 | 2.0x | 12.0 | No integration tests for PyQt6; no React test suite | CRITICAL |
| **H** Error Handling & Debugging | Custom exception hierarchy in launch_utils; 155 broad except files | 6 | 1.5x | 9.0 | eval() wraps errors but suppresses stack traces | MAJOR |
| **I** Security & Input Validation | Pydantic contracts for API; path traversal protection in launch_utils | 5 | 1.5x | 7.5 | eval() in 2 files, xml.etree in 10 files, CORS * in 4 files | CRITICAL |
| **J** Extensibility & Plugin Architecture | ABC-based plugin system; gui_registration auto-discovery | 7 | 1.0x | 7.0 | Plugin system declared but no concrete plugins implemented | MINOR |
| **K** Reproducibility & Provenance | Lock file exists but improperly pinned; matrix CI across 3.10-3.12 | 5 | 1.5x | 7.5 | No Docker/containerization; env-dependent behavior | MAJOR |
| **L** Long-Term Maintainability | Consistent patterns; AGENTS.md governance; pre-commit polyglot | 7 | 1.0x | 7.0 | 48 workflow files = massive maintenance surface | MAJOR |
| **M** Educational Resources & Tutorials | Quick start + add_new_tool tutorials; help_system.py framework | 6 | 1.0x | 6.0 | Tutorials are stubs (< 60 lines each) | MINOR |
| **N** Visualization & Export | matplotlib + Three.js (GlassBath3DViewer); plot_theme system | 7 | 1.0x | 7.0 | No export-to-PDF or report generation | MINOR |
| **O** CI/CD & DevOps | Comprehensive: ruff, black, mypy, bandit, pip-audit, semgrep | 7 | 1.0x | 7.0 | mypy is continue-on-error (non-blocking) | MAJOR |

**Weighted Total: 72.0/150 normalized = 72/100**

*(Calculation: sum of weighted scores = 134.5; max possible = 150 * (10/10) = 187.5 at perfect 10s; normalized = 134.5/187.5 * 100 = 71.7, rounded to 72)*

---

## Category Group Scores

| Group | Categories | Weight | Score | Contribution |
|:------|:-----------|:------:|:-----:|:------------|
| **Core Engineering** | A, B, H | 5.0x | 6.7 avg | 33.5/50 |
| **User & Developer Experience** | C, D, M | 4.0x | 7.3 avg | 30.0/40 |
| **Reliability & Safety** | G, I, K | 6.0x | 5.3 avg | 27.0/60 |
| **Infrastructure** | E, F, L, O | 5.0x | 5.8 avg | 30.5/50 |
| **Design & Extension** | J, N | 2.0x | 7.0 avg | 14.0/20 |

---

## Top 10 Risks

| Rank | Risk | Category | Severity | Impact | Recommended Action | Effort |
|:----:|:-----|:---------|:---------|:-------|:-------------------|:-------|
| 1 | `eval()` in `signal_processing.py:401` and `Data_Processor_r0.py:2752` with neutered builtins is bypassable | I | CRITICAL | Arbitrary code execution if user-supplied formula reaches eval | Replace with `ast.literal_eval` or `simpleeval` library | 2 hours |
| 2 | 9% test coverage on shared modules; no tests for 20+ PyQt6 main_window files | G | CRITICAL | Regressions ship undetected in calculator engines and UI | Add parametric tests for each process calculator; use pytest-qt for UI | 2 weeks |
| 3 | 10 files use `xml.etree.ElementTree` for parsing (XXE vulnerability) | I | CRITICAL | XML external entity attacks on URDF/MJCF model files | Replace parsing calls with `defusedxml.ElementTree` (already a dependency) | 3 hours |
| 4 | `Data_Processor_r0.py` at 8,999 lines is an unmaintainable monolith | A | MAJOR | Any change risks cascading breakage; impossible to unit test | Extract into domain modules following the PyQt6 main_window pattern | 1 week |
| 5 | `requirements-lock.txt` uses `>=` for 4 deps -- not a real lock file | K | MAJOR | Non-reproducible builds; CI may pass with different dep versions | Use `pip freeze` or `pip-tools` to generate fully pinned lock | 1 hour |
| 6 | CORS `allow_origins=["*"]` on 4 FastAPI apps | I | MAJOR | Cross-origin attacks if any API is exposed beyond localhost | Restrict to `["http://localhost:*"]` pattern or env-configured origins | 1 hour |
| 7 | `shell=True` in `gui_launcher/launcher.py:248` for npm subprocess | I | MAJOR | Shell injection if web_path contains attacker-controlled content | Remove `shell=True`; pass command as list (already done elsewhere) | 30 min |
| 8 | 155 files catch `except Exception` broadly; many swallow or re-raise generically | H | MAJOR | Masked bugs, silent data corruption, impossible post-mortem debugging | Audit and narrow to specific exceptions; add logging before re-raise | 3 days |
| 9 | No React/TypeScript test suite (0 test files for 90 .tsx components) | G | MAJOR | Frontend regressions in 19 web apps go undetected | Add Vitest + React Testing Library for each web app | 1 week |
| 10 | 48 GitHub Actions workflow files, 33 Jules-agent workflows | L | MAJOR | Massive maintenance burden; overlapping scopes; bot PR noise | Consolidate to <10 workflows using reusable workflow pattern | 3 days |

---

## Remediation Roadmap

### Phase 1: Critical (48 hours)

**Security hardening -- zero-tolerance items:**

1. **Replace `eval()` in formula evaluation** (2 files)
   - `src/data_processing/data_processor/python/data_processor/core/signal_processing.py:401`
   - `src/data_processing/data_processor/python/data_processor/Data_Processor_r0.py:2752`
   - Action: Install `simpleeval` or use `ast.literal_eval` for numeric-only expressions. If full math expressions are needed, use `numexpr.evaluate()` which is already numpy-compatible.

2. **Replace `xml.etree.ElementTree` with `defusedxml`** (10 files)
   - Files: `urdf_generator.py`, `mjcf_converter.py`, `urdf_parser.py`, `text_editor.py`, `model_library.py`, `unified_loader.py`, `format_utils.py`, `mdl_parser.py` (all under `src/shared/python/model_generation/` and `humanoid_character_builder/`)
   - Note: `defusedxml` is already in `pyproject.toml` dependencies but not used anywhere.

3. **Remove `shell=True` from `gui_launcher/launcher.py:248`**

4. **Restrict CORS origins** in 4 FastAPI apps to localhost patterns.

### Phase 2: Major (2 weeks)

**Test coverage and code quality:**

5. **Add process calculator unit tests** -- parametric tests for each of the 15 calculator engines in `upstream_drift_tools/process_calculators/`. Target: 70% coverage on shared module.

6. **Add React component tests** -- Vitest + RTL for at least the 5 most complex components (`FunctionGenerator.tsx`, `ElectrodeAdvisorCalculator.tsx`, `GlassBath3DViewer.tsx`, `TRCVesselDesignerCalculator.tsx`, `DataChart.tsx`).

7. **Pin `requirements-lock.txt`** properly using `pip-tools compile` or `pip freeze`. Remove all `>=` entries.

8. **Narrow broad exception handlers** -- audit the 155 files with `except Exception` and replace with specific exceptions where possible.

9. **Make mypy blocking in CI** -- remove `continue-on-error: true` from `ci-standard.yml:52`.

### Phase 3: Full (6 weeks)

**Architectural debt reduction:**

10. **Decompose `Data_Processor_r0.py`** (8,999 lines) -- extract CSV parsing, filtering, plotting, and export into separate modules following the already-established `core/signal_processing.py` and `core/feature_engineering.py` pattern.

11. **Decompose `electrode_advisor/main_window.py`** (4,384 lines, 86 methods) -- extract tab logic into separate widget classes.

12. **Consolidate GitHub Actions workflows** -- merge the 33 Jules-agent workflows into 5-8 using reusable workflows and matrix strategies.

13. **Add performance benchmarks** -- address the `RuntimeWarning: overflow encountered in exp` in `syngas_water_calculator.py:276` and add `pytest-benchmark` for critical calculator paths.

14. **Implement concrete plugins** for the `ModelGenerationPlugin` ABC.

---

## Go/No-Go Assessment

| Criterion | Status | Notes |
|:----------|:------:|:------|
| Tests pass | GO | 796/799 pass (3 skipped with reason) |
| Linting clean | GO | Ruff reports 0 violations |
| Formatting | CONDITIONAL | 8 files fail `black --check` |
| Security | NO-GO for production exposure | eval(), XXE, CORS * must be fixed first |
| Local dev use | GO | All tools functional for local engineering use |
| Type safety | CONDITIONAL | 96.9% annotated but mypy is non-blocking |
| Documentation | GO | Adequate for onboarding; tutorials need depth |

**Verdict: GO for continued local development. NO-GO for any network-exposed deployment without completing Phase 1 security hardening.**

---

## Appendix: Individual Category Details

### A: Architecture & Implementation (Score: 7/10)

**Strengths:**
- The 4-wave decoupling (PRs #636-#639) established a clean architecture: `upstream_drift_tools.bootstrap.ensure_paths()` -> `gui_registration.GUI_INFO` -> `gui_launcher.launch_from_gui_info()`. Every launch script is now a 21-line thin wrapper. Evidence: `src/flare_calculator/launch_pyqt6.py` (21 lines), `src/flare_calculator/launch_web.py` (21 lines).
- Contract-first API design via Pydantic models in `src/shared/python/calc_backend/contracts/` (11 router/contract pairs, 594 lines of well-typed contracts).
- Clear separation: `src/shared/python/` for reusable libraries, `src/<tool>/python/<tool>/` for tool-specific code, `src/<tool>/web/` for React frontends.

**Weaknesses:**
- MAJOR: `Data_Processor_r0.py` (8,999 lines) is a single-file application mixing CSV parsing, Tkinter UI, matplotlib plotting, scipy filtering, and file I/O. It uses `customtkinter` (not PyQt6), making it an architectural outlier.
- MAJOR: `electrode_advisor/ui/pyqt6/main_window.py` (4,384 lines, 86 methods) is a god-class. Should be decomposed into tab widgets.
- MINOR: `tools.json` registry only covers 9 tool categories but the actual src/ directory has 34 tool directories. The registry is incomplete.

### B: Code Quality & Hygiene (Score: 7/10)

**Strengths:**
- Ruff produces 0 violations. Code is lint-clean.
- 96.9% of non-test functions have return type annotations (4,233/4,368).
- 90.8% docstring coverage in the shared module (1,598/1,759 functions).
- Pre-commit configuration is polyglot and thorough (Python, JS/TS, CSS, C++, Rust).

**Weaknesses:**
- MAJOR: 155 files use `except Exception` broadly. In `src/shared/python/calc_backend/routers/flare.py:37`, all calculator exceptions are caught and re-raised as HTTP 422, losing the original exception type.
- MAJOR: 63 non-test files contain `print()` statements despite AGENTS.md explicitly forbidding them.
- MINOR: 8 files fail `black --check` formatting (including recently-modified test files like `test_dry_compliance.py`).
- MINOR: 41 TODO/FIXME/HACK/XXX markers remain in source files.

### C: Documentation & Comments (Score: 8/10)

**Strengths:**
- `docs/USER_MANUAL.md` (1,361 lines) provides comprehensive end-user documentation.
- 60 README.md files across the repository provide per-tool documentation.
- `AGENTS.md` (515 lines) is an exceptionally detailed agent governance document with coding standards, workflow schedules, and emergency procedures.
- Process calculator READMEs exist for individual calculators (e.g., `flare_README.md`, `scrubber_README.md`).

**Weaknesses:**
- MINOR: `docs/tutorials/quick_start.md` (52 lines) and `docs/tutorials/add_new_tool.md` (47 lines) are stubs, not real tutorials.
- MINOR: `docs/IMPLEMENTATION_GAPS.md` (1,018 lines) documents known gaps but many entries appear stale.
- NIT: Several `docs/` files reference outdated PR numbers and counts.

### D: User Experience & Developer Journey (Score: 8/10)

**Strengths:**
- Dual-platform launch: every tool with a PyQt6 desktop app also has a React web counterpart, launched via `launch_pyqt6.py` or `launch_web.py`.
- `gui_registration.py` pattern means adding a new tool follows a documented, consistent pattern.
- Fleet-wide theme system (`src/shared/python/theme/`) with signal-based notifications and QSettings persistence.
- Plot theme system (`src/shared/python/plot_theme/`) for consistent matplotlib styling.

**Weaknesses:**
- MINOR: Web launcher uses `time.sleep(2)` as a hardcoded delay before opening the browser (`gui_launcher/launcher.py:254`). Should poll for server readiness.
- MINOR: `tools.json` registry is manually maintained and does not auto-discover tools.
- NIT: No unified CLI entry point (e.g., `ud-tools launch flare_calculator`). Only `urdf-gen` is registered as a console script.

### E: Performance & Scalability (Score: 5/10)

**Strengths:**
- Process calculators use numpy vectorized operations (not pure Python loops).
- `@dataclass` usage for lightweight data transfer objects in calculators.

**Weaknesses:**
- MAJOR: `syngas_water_calculator.py:276` triggers `RuntimeWarning: overflow encountered in exp` and `:392` triggers `RuntimeWarning: invalid value encountered in scalar subtract` during normal test execution. These are numerical stability bugs.
- MAJOR: No caching layer for any calculator. Each API call re-instantiates the calculator class (e.g., `FlareCalculator()` in `routers/flare.py:22`).
- MAJOR: No profiling infrastructure, no `pytest-benchmark`, no performance regression tests.
- MINOR: `debug_utils.py` at 1,122 lines is disproportionately large for a utility module.

### F: Installation & Deployment (Score: 6/10)

**Strengths:**
- `pyproject.toml` is well-structured with optional dependency groups (`[urdf]`, `[signal]`, `[process]`, `[gui]`, `[theme]`, `[dev]`).
- Supports Python 3.10, 3.11, 3.12 (tested in CI matrix).
- `pip install -e .` works for development.

**Weaknesses:**
- MAJOR: `requirements-lock.txt` is not a proper lock file. Lines like `pandas>=2.2.2  # Note: Exact version depends on environment` and `pytest-cov  # Version depends on pytest version` defeat the purpose entirely.
- MAJOR: No Docker/container support. No `Dockerfile`, no `docker-compose.yml`.
- MINOR: `requirements.txt` pulls in `playwright>=1.40.0` as a top-level dependency even though it is only needed for testing (should be dev-only).

### G: Testing & Validation (Score: 6/10)

**Strengths:**
- 799 tests collected, 796 pass, 3 skipped (with valid reason: missing WGS scipy dependency).
- `test_dry_compliance.py` (283 lines) structurally verifies that all launch scripts follow the thin-wrapper pattern -- excellent meta-testing.
- `test_wave1_paths.py`, `test_waves_2_3_imports.py`, `test_wave4_launcher.py` verify the decoupling architecture.
- Calc backend API tests (`tests/calc_backend/`) with 702 lines test all 11 calculator endpoints.

**Weaknesses:**
- CRITICAL: Coverage of `src/shared/` is only 9% (24,435 of 26,868 lines uncovered). Most process calculator engines have 0% coverage from the test suite.
- CRITICAL: Zero React/TypeScript tests for 90 `.tsx` files across 19 web apps.
- MAJOR: No PyQt6 GUI tests (no `pytest-qt` usage anywhere). 20+ `main_window.py` files are untested.
- MAJOR: Some test files exist inside `src/` (117 files) rather than the canonical `tests/` directory, creating confusion about what the CI actually runs.

### H: Error Handling & Debugging (Score: 6/10)

**Strengths:**
- Custom exception hierarchy in `launch_utils.py`: `LaunchError`, `ToolNotFoundError`, `SecurityError`, `PlatformError`. Well-designed with proper `from` chaining.
- Path validation with `validate_and_sanitize_path()` raises specific exceptions.
- FastAPI routers use `HTTPException` with status codes.

**Weaknesses:**
- MAJOR: 155 files catch `except Exception`. Many log and continue, masking real errors. Example: `launch_utils.py:126` catches `Exception` in stream reading, logs it, but continues silently.
- MAJOR: Calculator routers uniformly catch `except Exception as exc: raise HTTPException(status_code=422, detail=str(exc))` -- this maps *all* errors (including `KeyError`, `ZeroDivisionError`, programmer bugs) to 422 Unprocessable Entity, making debugging impossible for API consumers.
- MINOR: `calculator_state_mixin.py` (783 lines) handles state restoration errors by silently falling back to defaults.

### I: Security & Input Validation (Score: 5/10)

**Strengths:**
- Pydantic models with `Field(gt=0)` constraints validate API inputs at the boundary.
- `validate_and_sanitize_path()` in `launch_utils.py` prevents path traversal attacks.
- `.gitignore` properly excludes `.env` files.
- `pip-audit` and `bandit` in CI pipeline.

**Weaknesses:**
- CRITICAL: `eval()` with `{"__builtins__": {}}` in `signal_processing.py:401` and `Data_Processor_r0.py:2752`. The `__builtins__` restriction is trivially bypassable via `().__class__.__bases__[0].__subclasses__()` chain.
- CRITICAL: 10 files use `xml.etree.ElementTree` for XML parsing despite `defusedxml` being declared as a dependency in `pyproject.toml:29`. This is XXE-vulnerable.
- MAJOR: `allow_origins=["*"]` in 4 FastAPI middleware configurations.
- MAJOR: `shell=True` in `gui_launcher/launcher.py:248`.
- MINOR: `github_importer.py` reads `GITHUB_TOKEN` from environment without validation.

### J: Extensibility & Plugin Architecture (Score: 7/10)

**Strengths:**
- `ModelGenerationPlugin` ABC in `src/shared/python/model_generation/plugins/__init__.py` defines a clean plugin interface with `name`, `version`, `initialize()`, `shutdown()`.
- `gui_launcher/registry.py` provides `auto_discover_guis()` for runtime tool discovery.
- `gui_registration.py` per-tool pattern makes adding new tools formulaic.

**Weaknesses:**
- MINOR: `ModelGenerationPlugin` has zero concrete implementations. It is a dead interface.
- MINOR: No entry_points or setuptools plugin discovery mechanism.
- NIT: `tools.json` is manually curated rather than auto-generated from `gui_registration.py` data.

### K: Reproducibility & Provenance (Score: 5/10)

**Strengths:**
- CI tests against Python 3.10, 3.11, 3.12 matrix.
- `requirements-lock.txt` exists (intent is correct).
- `pyproject.toml` has pinned minimum versions for core deps.

**Weaknesses:**
- MAJOR: Lock file is not a real lock file. Contains comments like "Version depends on environment" and uses `>=` for several entries.
- MAJOR: No containerization. No way to reproduce exact environment.
- MINOR: No data versioning or test fixture provenance tracking.
- MINOR: Ruff version in `requirements.txt` (`ruff` unpinned) differs from CI (`ruff==0.14.10`) and pre-commit (`v0.14.10`) -- potential inconsistency.

### L: Long-Term Maintainability (Score: 7/10)

**Strengths:**
- Consistent patterns: every tool follows `gui_registration.py` + `launch_pyqt6.py` + `launch_web.py` + `python/<tool>/ui/pyqt6/main_window.py` structure.
- `AGENTS.md` governance with conventional commit messages.
- Pre-commit hooks enforce quality on every commit.

**Weaknesses:**
- MAJOR: 48 GitHub Actions workflow files (33 Jules-agent workflows) create enormous maintenance surface. Overlap between "Code Quality Reviewer", "Code Quality Fixer", "Auto-Refactor", and "DRY-Orthogonality" workflows.
- MAJOR: Legacy code (`Data_Processor_r0.py`, `Folders_Tool_r0.py`, `Data_Processor_Integrated.py`) uses different UI toolkit (customtkinter) than the rest of the repo (PyQt6), creating a fork in maintenance.
- MINOR: 91 bare `pass` statements in non-test code suggest incomplete implementations.

### M: Educational Resources & Tutorials (Score: 6/10)

**Strengths:**
- `docs/tutorials/quick_start.md` and `docs/tutorials/add_new_tool.md` exist.
- `src/python/src/help/help_system.py` provides a reusable help dialog framework.
- `docs/USER_MANUAL.md` at 1,361 lines.
- Process calculator READMEs explain the engineering context.

**Weaknesses:**
- MINOR: Both tutorials are under 60 lines -- they are outlines, not walkthrough-quality tutorials.
- MINOR: No Jupyter notebook examples for calculator APIs.
- NIT: No video or animated GIF documentation of the GUI tools.

### N: Visualization & Export (Score: 7/10)

**Strengths:**
- `plot_theme/` system (1,277 lines) provides consistent matplotlib theming across all tools.
- `plot_engine/` provides cross-platform rendering abstraction.
- `GlassBath3DViewer.tsx` (796 lines) uses Three.js for 3D electrode visualization.
- 19 React web apps with interactive charting.

**Weaknesses:**
- MINOR: No PDF/report export from calculators. Results are display-only.
- MINOR: No data export from React web apps (no CSV download buttons visible in component code).
- NIT: `VISUALIZATION_GUIDE.md` exists in docs but is not linked from main README.

### O: CI/CD & DevOps (Score: 7/10)

**Strengths:**
- `ci-standard.yml` is clean: ruff -> black -> mypy (soft) -> pip-audit -> pytest matrix.
- Concurrency groups prevent redundant CI runs.
- Pre-commit config includes bandit, semgrep, radon for security and complexity checking.
- Path-ignore patterns skip CI for docs-only changes.

**Weaknesses:**
- MAJOR: `mypy` runs with `continue-on-error: true` -- type errors do not block merges.
- MAJOR: No frontend CI (no npm test, no TypeScript compilation check, no ESLint in CI).
- MINOR: Lock file versions (ruff 0.5.0) differ from CI versions (ruff 0.14.10).
- NIT: `codeql-analysis.yml.disabled` is committed rather than deleted.
