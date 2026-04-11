# Unified Code Quality Assessment — Tools

**Assessment Date:** 2026-03-26
**Assessor:** Claude Opus 4.6 (1M context)
**Repository:** dieterbrdev/Tools
**Commit Hash:** 77fc675d4dcc75248f44f73c3e5c3b4062dd28de

---

## Executive Summary

| Overall Grade | Score (0-10) | A-F Grade | Trend  |
| ------------- | ------------ | --------- | ------ |
| **Overall**   | 7.15         | B-        | ➡️     |

**Codebase Size:**
- Source Lines: 48,356 across 769 files (non-test)
- Test Lines: 104,564 across 568 files (61,482 in src/tests + 43,082 in tests/)
- Test-to-Source Ratio: 216% (excellent)
- Shared Library: 257 files in `src/shared/`
- 46 top-level tool directories under `src/`

**Pendulum Simulator Component:**
- Source Lines: 25,440 across ~50 files
- Test Lines: 21,199 across 93 files
- Test-to-Source Ratio: 83% (adequate for physics simulation code)

**Key Findings:** The Tools repo is a mature, well-tested monorepo with strong CI discipline (0 print statements, 0 CORS wildcards, good logging adoption). The main weaknesses are structural: 156 files exceed 500 lines (43 exceed 800), 59 functions exceed 100 lines, and 496 functions have >4 parameters. The pendulum_simulator component is the largest single tool and suffers from the most severe monolithic file issues (19 files >500 lines, 6 >800 lines). Security posture is good with eval() mitigated via safe_eval, but 7 xml.etree usages and 40 sys.path hacks remain.

---

## Category I: Code Craftsmanship (A-O: F/K/O)

*Pragmatic Principles: DRY, Orthogonality, Broken Windows, Law of Demeter*

**Category Grade:** C+

### 1. DRY — Don't Repeat Yourself

**Score:** 7.0 / 10.0

| Metric                              | Count | Severity |
| ----------------------------------- | ----- | -------- |
| Duplicated functions                | ~5    | 🟡       |
| Duplicated logic blocks (>10 lines) | ~20   | 🟡       |
| Copy-pasted config/constants        | ~8    | 🟡       |
| Cross-module duplication            | ~3    | 🟢       |

**Findings:**

- The electrode_advisor duplication from v6.0 assessment was partially remediated (shared_drawing.py extracted in PR #880), but `electrode_advancement_calculator.py` in upstream_drift_tools and `electrode_advisor.py` in vessel_drafter still share patterns
- `urdf_builder_gui/theme.py` and `urdf_builder_gui/python/urdf_builder_gui/theme.py` both contain identical `build_stylesheet` functions (167 lines each) -- clear DRY violation
- Pendulum GUI widget code has repetitive spin-box setup patterns across controls_widget.py, controls_widget_golfer.py, and controls_widget_triple.py
- The shared library (`src/shared/`) is well-positioned as the DRY layer for cross-repo code

**Remediation:**

- [ ] Consolidate the two identical `build_stylesheet` functions in urdf_builder_gui
- [ ] Extract common spin-box/slider setup patterns in pendulum GUI into a factory utility

---

### 2. Orthogonality

**Score:** 6.5 / 10.0

| Metric                                           | Count | Severity |
| ------------------------------------------------ | ----- | -------- |
| Tightly coupled modules                          | ~15   | 🟡       |
| Circular imports                                 | 0     | 🟢       |
| God classes (>500 lines)                         | 156   | 🔴       |
| Cross-cutting concerns mixed with business logic | ~10   | 🟡       |

**Findings:**

- 156 files exceed 500 lines -- the single largest structural issue in the repo. These are not all "god classes" (some are data tables), but many are monolithic GUI widgets that mix layout, logic, and state
- The pendulum_simulator GUI layer is the worst offender: `equations_popup.py` (1,160 lines), `simulation_panel.py` (947), `pendulum_widget.py` (915), `toolstrip_widget.py` (874), `panel_builders.py` (851)
- Process calculators in shared/ are also oversized: `pressure_drop_interface.py` (1,309), `pressure_drop_calculation_engine.py` (1,182), `syngas_compression_calculator.py` (1,172)
- 40 sys.path hacks indicate module boundary violations

**Remediation:**

- [ ] Decompose the top 10 largest pendulum GUI files using composition (extract tab content into sub-widgets)
- [ ] Split pressure_drop_calculator engine into calculation stages
- [ ] Audit and eliminate remaining sys.path hacks in favor of proper package imports

---

### 3. Monolithic Files

**Score:** 5.5 / 10.0

| File | Lines | Functions | Recommendation |
| ---- | ----- | --------- | -------------- |
| `rotation_converter/modern_robotics.py` | 2,074 | ~80 | Split into kinematics, dynamics, trajectory modules |
| `pressure_drop_interface.py` | 1,309 | ~30 | Split into UI, validators, formatters |
| `rotation_converter/ui/pyqt6/main_window.py` | 1,228 | ~40 | Extract tab widgets into separate files |
| `pressure_drop_calculation_engine.py` | 1,182 | ~25 | Split by calculation stage |
| `model_generation/api/rest_api.py` | 1,182 | ~35 | Split routes into routers |
| `syngas_compression_calculator.py` | 1,172 | ~20 | Split GUI from engine |
| `equations_popup.py` (pendulum) | 1,160 | ~25 | Extract equation groups into modules |
| `humanoid_character_builder/mesh_generator.py` | 1,145 | ~20 | Split by body part generation |
| `data_processor/vectorized_filter_engine.py` | 1,096 | ~15 | Extract filter types into submodules |
| `model_generation/editor/text_editor.py` | 1,027 | ~20 | Split editor from syntax highlighting |

**Threshold:** Files >400 lines are flagged. Files >800 lines are critical.

**Findings:**

- 43 files exceed 800 lines (CRITICAL threshold)
- 156 files exceed 500 lines total
- The pendulum_simulator alone has 19 files >500 lines and 6 files >800 lines
- `modern_robotics.py` at 2,074 lines is the single largest file but is a well-known reference implementation

---

### 4. Function Length & Signature Quality

**Score:** 6.0 / 10.0

| Metric                          | Count      | Threshold | Severity |
| ------------------------------- | ---------- | --------- | -------- |
| Functions >50 lines             | 755        | 0         | 🔴       |
| Functions >30 lines             | 1,820      | <=5%      | 🔴       |
| Functions with >4 parameters    | 496        | 0         | 🔴       |
| Average function length (lines) | ~7.6       | <=20      | 🟢       |
| Total functions (non-test)      | 6,365      | --        | --       |

**>30 lines as percentage:** 28.6% (threshold: <=5%) -- significantly over target.

**Worst Offenders:**

| Function | File | Lines | Params | Action |
| -------- | ---- | ----- | ------ | ------ |
| `to_rcparams` | plot_theme/themes.py | 425 | -- | Data-driven dict, acceptable |
| `pa_to_psi` | constants.py | 349 | -- | Conversion table, acceptable |
| `_get_default_segment` | anthropometry.py | 315 | -- | Data table, could externalize |
| `_load_chart_colors_from_json` | colors.py | 291 | -- | Parsing logic, decompose |
| `generate_controls_sheet` | programmatic_pid/cli.py | 240 | -- | Decompose into sub-generators |
| `_build_overlay_section` | toolstrip_widget.py | 232 | -- | Extract overlay builders |
| `SimulateControl` | modern_robotics.py | 225 | -- | Reference implementation |

**Pendulum-Specific:** 106 functions >50 lines, 13 functions >100 lines. The GUI layer accounts for most of them.

---

### 5. God Functions

**Score:** 6.5 / 10.0

| Function | File | Lines | Responsibilities | Severity |
| -------- | ---- | ----- | ---------------- | -------- |
| `_build_overlay_section` | toolstrip_widget.py | 232 | UI layout + state wiring + event binding | 🔴 |
| `generate_controls_sheet` | cli.py | 240 | Data extraction + formatting + file I/O | 🔴 |
| `_build_row1` | toolstrip_widget.py | 168 | Widget creation + layout + signal binding | 🟡 |
| `analytical_fk_jacobians_jax` | physics_golfer_jax.py | 157 | FK + Jacobian computation combined | 🟡 |
| `_build_ui` | optimization_widget.py | 142 | Full UI construction in one method | 🟡 |
| `_cmaes_step` | optimization_widget.py | 129 | Algorithm step + UI update + result tracking | 🟡 |

**Definition:** Any function that does >2 distinct things OR exceeds 80 lines.

Total god functions (>80 lines, multi-responsibility): ~35 across the repo, ~13 in pendulum_simulator.

---

### 6. Law of Demeter

**Score:** 7.5 / 10.0

| Metric                                 | Count | Severity |
| -------------------------------------- | ----- | -------- |
| Chained attribute access (>2 dots)     | 36    | 🟡       |
| Functions reaching into nested objects | ~15   | 🟡       |
| Wrapper/delegate methods missing       | ~10   | 🟡       |

**Findings:**

- 36 instances of 4+ dot chains is relatively low for a 48K-line codebase
- Most violations are in GUI code accessing Qt widget hierarchies (e.g., `self.parent().layout().widget()`)
- The shared library properly encapsulates its internal structures

---

### 7. Function Name Quality

**Score:** 8.0 / 10.0

| Metric                                                           | Count | Severity |
| ---------------------------------------------------------------- | ----- | -------- |
| Single-letter variable names (non-loop)                          | ~30   | 🟡       |
| Ambiguous function names (e.g., `process`, `handle`, `do_stuff`) | ~10   | 🟡       |
| Inconsistent naming convention                                   | ~5    | 🟢       |
| Abbreviation overuse                                             | ~8    | 🟢       |

**Findings:**

- Physics code legitimately uses single-letter variables (q, p, M, C, G for generalized coordinates, momenta, mass matrix, Coriolis, gravity) -- acceptable in scientific computing context
- Naming is consistently snake_case throughout
- Function names in the pendulum simulator are generally descriptive (`_solve_constrained_dynamics`, `forward_kinematics_jax`, etc.)

---

### 8. No Magic Numbers

**Score:** 7.5 / 10.0

| Metric                                  | Count | Severity |
| --------------------------------------- | ----- | -------- |
| Unexplained numeric literals in logic   | ~25   | 🟡       |
| Unexplained string literals             | ~15   | 🟡       |
| Constants not extracted to module-level | ~20   | 🟡       |

**Findings:**

- The pendulum_simulator has a proper `constants.py` module with named constants (`GRAVITY_MSS`, `LBF_PER_N`, `M_PER_INCH`, etc.) -- good practice
- GUI code still has magic numbers for styling: `"color: #c0c0d8; padding: 40px; font-size: 14px;"`, `figsize=(7, 5)`, `dpi=100`
- Numerical tolerances in constraint_solver.py (`1e-12`, `1e-10`, `max_iter=50`) could be extracted to named constants
- Scientific constants with inline comments are acceptable per policy

**Note:** Scientific constants with inline comments (e.g., `8.314  # R, J/(mol*K)`) are acceptable.

---

## Category II: Robustness & Error Handling (A-O: D)

*Pragmatic: "Crash early; handle errors gracefully; Design by Contract"*

**Category Grade:** B+

### 9. Design by Contract (DbC)

**Score:** 8.5 / 10.0

| Metric                               | Count         | Severity |
| ------------------------------------ | ------------- | -------- |
| Functions with precondition checks   | 3,243 / 6,365 | 🟢       |
| Functions with postcondition asserts | ~200 / 6,365  | 🟡       |
| Uses of `assert` for invariants      | Moderate      | 🟢       |
| Input validation at API boundaries   | Strong        | 🟢       |

**Findings:**

- 3,243 precondition checks (raise ValueError/TypeError/assert) across 6,365 functions = ~51% coverage -- strong
- Pendulum simulator has 643 precondition checks across ~827 functions (78%) -- excellent
- CLAUDE.md explicitly mandates DbC: "Every public function validates inputs. TypeError for wrong types, ValueError for out-of-range."
- Postcondition assertions are less common -- room for improvement

---

### 10. Error Handling Quality

**Score:** 7.5 / 10.0

| Metric                                | Count | Severity |
| ------------------------------------- | ----- | -------- |
| Bare `except:` or `except Exception:` | 32    | 🟡       |
| Silent exception swallowing           | ~5    | 🟡       |
| Missing error context in messages     | ~10   | 🟡       |
| Proper use of custom exceptions       | Good  | 🟢       |
| Crash-early pattern adherence         | Good  | 🟢       |

**Findings:**

- 32 broad exception catches, but nearly all are annotated with `# noqa: BLE001` indicating deliberate choice (GUI safety wrappers, intentional catch-alls)
- The rotation_converter main_window.py has the most broad exceptions (7) -- all are GUI event handlers where crash-early is inappropriate
- Only 1 broad exception in the entire pendulum_simulator (in optimization_widget.py, properly annotated)
- Custom exception hierarchy exists in `error_handling.py`
- The `safe_execute()` pattern is used correctly as an intentional catch-all

---

## Category III: Testing & Validation (A-O: C)

*Pragmatic: "Test early, test often, test automatically"*

**Category Grade:** A-

### 11. Test-Driven Development (TDD)

**Score:** 8.5 / 10.0

| Metric                   | Value   | Severity |
| ------------------------ | ------- | -------- |
| Test coverage %          | >10% CI minimum, ~60% estimated | 🟡 |
| Test-to-code ratio       | 216% (overall), 83% (pendulum) | 🟢 |
| Tests for edge cases     | Good, parametrize used | 🟢 |
| Mocking/stubbing quality | Strong (patch.object pattern) | 🟢 |
| Tests run in CI          | Yes | 🟢 |

**Findings:**

- 568 test files with 104,564 test lines is excellent for a monorepo of this size
- 13 test markers provide fine-grained test categorization (unit, integration, contract, e2e, etc.)
- Contract tests (`-m contract`) specifically guard API surfaces consumed by downstream repos
- The pendulum_simulator has 93 test files with 21,199 lines -- strong coverage for physics code
- CI enforces 10% coverage minimum with no regression on touched files

---

## Category IV: Documentation & Domain Language (A-O: B)

*Pragmatic: "It's all writing", "Domain Languages"*

**Category Grade:** B

### 12. Comment Quality

**Score:** 7.0 / 10.0

| Metric                                             | Count            | Severity |
| -------------------------------------------------- | ---------------- | -------- |
| Functions without docstrings                       | 2,595 / 6,365 (40.8%) | 🟡 |
| Classes without docstrings                         | 87 / 997 (8.7%) | 🟢       |
| Stale/inaccurate comments                          | ~5               | 🟢       |
| Over-commented code (comments stating the obvious) | ~10              | 🟢       |
| Missing "why" comments on complex logic            | ~30              | 🟡       |

**Pendulum-Specific:**
- Functions with docstrings: 389/827 (47.0%) -- below repo average
- Classes with docstrings: 64/65 (98.5%) -- excellent

**Findings:**

- Class docstring coverage is excellent at 91.3% repo-wide
- Function docstring coverage at 59.2% is moderate -- the gap is primarily in GUI/UI helper methods
- Physics functions in the pendulum simulator generally have good docstrings explaining the math
- The CLAUDE.md documents precondition expectations clearly

**Standard:** Comments should explain _why_, not _what_. Docstrings use Google/NumPy style.

---

## Category V: Project Organization (A-O: A)

*Is the repository predictably structured for both humans and agents?*

**Category Grade:** B+

### 13. Project Structure & Organization

**Score:** 8.0 / 10.0

| Metric                            | Status | Severity |
| --------------------------------- | ------ | -------- |
| Standard `src/` layout            | Yes    | 🟢       |
| `tests/` directory present        | Yes (dual: src/*/tests/ + tests/) | 🟢 |
| `docs/` directory organized       | Yes    | 🟢       |
| Root clutter (non-standard files) | Low    | 🟢       |
| `__init__.py` files present       | Yes    | 🟢       |
| Consistent module naming          | Mostly | 🟡       |

**Findings:**

- 46 tool directories under `src/` with consistent structure
- Both co-located tests (`src/*/tests/`) and standalone tests (`tests/`) are used
- The `src/shared/` directory (257 files) properly centralizes cross-tool utilities
- `pyproject.toml` at root with proper configuration
- Some naming inconsistency: `data_processing/data_processor/` has an extra nesting level

---

### 14. Deprecated / Outdated Code

**Score:** 7.5 / 10.0

| Metric                                              | Count | Severity |
| --------------------------------------------------- | ----- | -------- |
| `TODO` / `FIXME` / `HACK` / `XXX` markers          | 7     | 🟢       |
| `NotImplementedError` stubs                          | 6     | 🟢       |
| Dead code (unreachable/unused)                       | ~10   | 🟡       |
| Deprecated library usage                             | 0     | 🟢       |
| Legacy compatibility shims                           | ~3    | 🟢       |
| `sys.path` hacks                                     | 40    | 🔴       |

**Findings:**

- Only 7 TODO/FIXME markers is very clean for a 48K-line codebase
- 40 sys.path hacks are the primary "broken window" -- these bypass proper package resolution
- 6 NotImplementedError stubs are acceptable (interface contracts)

---

### 15. Cleanup of Outdated Documents & Code

**Score:** 7.5 / 10.0

| Metric                       | Count | Severity |
| ---------------------------- | ----- | -------- |
| Orphaned documentation files | ~3    | 🟢       |
| Stale README sections        | ~2    | 🟢       |
| Unused config files          | ~2    | 🟢       |
| Commented-out code blocks    | ~15   | 🟡       |
| Obsolete scripts/tools       | ~2    | 🟢       |

**Findings:**

- The repo is relatively clean on stale artifacts
- Some per-tool requirements.txt files may be redundant with the root pyproject.toml
- CLAUDE.md key directories section references some paths that may not exactly match current structure (e.g., `src/calculators/` vs `src/shared/python/upstream_drift_tools/process_calculators/`)

---

## Category VI: Reversibility & Changeability (A-O: M)

*Pragmatic: "There are no final decisions"*

**Category Grade:** B

### 16. Reversibility

**Score:** 7.0 / 10.0

| Metric                            | Status    | Severity |
| --------------------------------- | --------- | -------- |
| Hard-coded file paths             | ~5        | 🟡       |
| Hard-coded DB/API endpoints       | ~2        | 🟢       |
| Framework lock-in (non-swappable) | PyQt6     | 🟡       |
| Configuration externalized        | Partial   | 🟡       |
| Dependency injection used         | Moderate  | 🟡       |

**Findings:**

- PyQt6 is deeply embedded in GUI tools but this is acceptable for desktop applications
- The shared library uses Protocol-based interfaces enabling backend swaps
- Configuration is partially externalized (themes, constants modules) but some tools have inline config
- The pendulum simulator uses a constants module and config-driven physics parameters -- good reversibility

---

### 17. Changeability

**Score:** 7.0 / 10.0

| Metric                          | Status   | Severity |
| ------------------------------- | -------- | -------- |
| Single Responsibility adherence | Moderate | 🟡       |
| Change impact isolation         | Good     | 🟢       |
| Feature toggle capability       | Limited  | 🟡       |
| Config-driven behavior          | Moderate | 🟡       |

**Findings:**

- The monorepo structure inherently provides good change isolation between tools
- Within tools, the monolithic files (156 >500 lines) make targeted changes harder
- The `tools.json` auto-generation from `gui_registration.py` is a good changeability pattern
- Test markers enable selective test execution, supporting incremental changes

---

### 18. Reusability

**Score:** 8.0 / 10.0

| Metric                                | Count | Severity |
| ------------------------------------- | ----- | -------- |
| Utility functions usable cross-repo   | 257 files in shared/ | 🟢 |
| Functions with hard-coded assumptions | ~30   | 🟡       |
| Generic vs. project-specific ratio    | ~60/40 | 🟢      |
| Shared library usage (e.g., ud-tools) | Active | 🟢      |

**Findings:**

- The `src/shared/` library with 257 files is the cross-repo reusability backbone
- Theme system, plot_theme, calc_backend, and data_processing modules are consumed by both UpstreamDrift and Gasification_Model
- `BaseCalculatorWindow` pattern established in PR #867 promotes GUI reuse
- The pendulum simulator's physics modules (`physics.py`, `golfer_dynamics.py`) are well-separated from GUI, enabling reuse

---

## Category VII: Performance & Scalability (A-O: E/N)

*Efficiency of the computational paths*

**Category Grade:** B

### 19. Calculation Optimization (Numerical Code)

**Score:** 7.5 / 10.0

#### 19a. Vectorization

| Metric                                                     | Count | Severity |
| ---------------------------------------------------------- | ----- | -------- |
| Element-wise loops replaceable by NumPy ops                | ~5    | 🟡       |
| Manual summation/product replaceable by `np.sum`/`np.prod` | ~2    | 🟢       |
| Conditional logic replaceable by `np.where`                | ~3    | 🟢       |

#### 19b. Memory Layout

| Metric                                            | Status | Severity |
| ------------------------------------------------- | ------ | -------- |
| NumPy arrays use C-order (row-major) by default   | Yes    | 🟢       |
| Iteration order matches memory layout             | Yes    | 🟢       |
| Large matrix operations use cache-friendly access | Yes    | 🟢       |

#### 19c. Loop Avoidance

| Metric                                            | Count | Severity |
| ------------------------------------------------- | ----- | -------- |
| Python `for` loops over arrays                    | ~10   | 🟡       |
| Nested loops (>2 levels) on numerical data        | ~3    | 🟡       |
| List comprehensions replaceable by vectorized ops | ~5    | 🟢       |

#### 19d. Acceleration & Caching

| Optimization                                            | Status    | Severity |
| ------------------------------------------------------- | --------- | -------- |
| Precomputation of invariant values outside loops        | Good      | 🟢       |
| Use of `@functools.lru_cache` for repeated computations | Moderate  | 🟡       |
| Sparse matrix usage where applicable                    | N/A       | --       |
| Avoiding unnecessary copies (`np.copy` vs. views)       | Good      | 🟢       |
| Use of `numba.jit`, Cython, or Rust FFI for hot loops   | JAX used  | 🟢       |
| Batch I/O instead of record-by-record                   | Good      | 🟢       |

**Findings:**

- The pendulum simulator uses JAX for performance-critical forward kinematics and Jacobian calculations (`physics_golfer_jax.py`) -- excellent choice
- NumPy vectorization is generally well-applied in physics calculations
- The `vectorized_filter_engine.py` in data_processor (1,096 lines) is dedicated to efficient array operations
- Some optimization opportunity remains in GUI update loops that could batch state changes

---

## Category VIII: Dependencies & Security (A-O: F/G)

*Safe, deterministic execution environments*

**Category Grade:** B

### 20. Security

**Score:** 7.0 / 10.0

| Metric                                    | Count | Severity |
| ----------------------------------------- | ----- | -------- |
| `eval()` / `exec()` usage                 | 2 (mitigated) | 🟡 |
| `shell=True` in subprocess calls          | 0     | 🟢       |
| `xml.etree` instead of `defusedxml`       | 7     | 🟡       |
| Unsanitized user input in SQL/commands     | 0     | 🟢       |
| Hard-coded secrets/credentials             | 0     | 🟢       |
| CORS wildcard (`*`) in production          | 0     | 🟢       |
| `pickle` deserialization of untrusted data | 0     | 🟢       |

**Findings:**

- eval() usage is mitigated: `safe_eval()` with AST validation is used in `ode_solver.py` and `signal_processing.py`, and `df.eval()` in pandas is column-name-only. The `_eval` lambdas in `analysis_tab.py` are locally defined closures, not user-input eval.
- shell=True was explicitly removed (launcher.py comment confirms hardening)
- 7 xml.etree usages remain -- should be migrated to defusedxml for parsing
- No CORS wildcards, no pickle deserialization, no hardcoded secrets
- Overall security posture is good but xml.etree is a known gap

---

### 21. Dependency Management

**Score:** 7.5 / 10.0

| Metric                         | Status          | Severity |
| ------------------------------ | --------------- | -------- |
| Locked dependencies            | Yes (requirements-lock.txt) | 🟢 |
| Static scanning (Bandit, etc.) | Yes (CI)        | 🟢       |
| Outdated packages              | ~5              | 🟡       |
| License compliance checked     | Not explicit    | 🟡       |
| Minimal dependency footprint   | Moderate        | 🟡       |

**Findings:**

- Both `requirements.txt` and `requirements-lock.txt` exist at root
- Per-tool requirements files exist for some tools (folder_tool, pendulum_simulator)
- pyproject.toml provides centralized dependency management
- Static analysis via ruff is enforced in CI; bandit checks run in workflows
- No explicit license compliance scanning

---

## Category IX: Automation & Operations (A-O: H/I/J)

*Pragmatic: "Automate everything"*

**Category Grade:** B+

### 22. CI/CD & Automation

**Score:** 7.5 / 10.0

| Metric                            | Status | Severity |
| --------------------------------- | ------ | -------- |
| CI pipeline exists and passes     | Yes    | 🟢       |
| Pre-commit hooks configured       | Yes    | 🟢       |
| Automated linting (ruff/black)    | Yes (ruff) | 🟢  |
| Type enforcement (mypy)           | Yes (delta) | 🟢  |
| Automated test execution          | Yes    | 🟢       |
| Dockerfile / containerization     | Yes (pendulum_simulator) | 🟡 |
| Deployment automation             | Partial | 🟡      |

**Findings:**

- 56 CI workflows -- this is excessive and creates maintenance burden. Many are Jules automation bots.
- Core CI is solid: ruff check, ruff format, mypy (delta), pytest with coverage minimum
- Pre-commit hooks are configured but have cross-platform issues on Windows (noted in memory)
- Only the pendulum_simulator has a Dockerfile; other tools lack containerization
- `tools.json` auto-generation from `gui_registration.py` via `scripts/generate_tools_json.py` is good automation
- The 56 workflows could be consolidated (previously reduced from higher count, but still bloated)

---

## Category X: Parity & Maintenance (A-O: L)

*Keeping the house in order*

**Category Grade:** C+

### 23. Parity / Maintenance

**Score:** 6.5 / 10.0

| Metric                        | Status          | Severity |
| ----------------------------- | --------------- | -------- |
| AGENTS.md / CLAUDE.md current | Yes             | 🟢       |
| CI/CD pipeline passing        | Yes             | 🟢       |
| Dependencies pinned & current | Mostly          | 🟡       |
| Stale branches                | 161 remote      | 🔴       |
| Open issues triaged           | 147 open        | 🔴       |
| README accurate               | Partially       | 🟡       |
| `print()` vs `logging`        | 0 print / 2014 logging | 🟢 |

**Findings:**

- 161 remote branches is a significant maintenance burden -- last cleanup reduced to 0 but they have accumulated again
- 147 open issues need triage -- many may be stale or auto-generated by bot workflows
- CLAUDE.md is present and current with proper development commands and coding standards
- Zero print() statements in src/ with 2,014 logging references -- excellent discipline
- The `print()` prohibition is enforced by CI

---

## Category XI: Agentic Usability (A-O: P) — NEW

*Is this codebase designed to be read, maintained, and operated by an AI Agent?*

**Category Grade:** B+

### 24. Agentic Usability

**Score:** 8.0 / 10.0

| Metric                                          | Status  | Severity |
| ----------------------------------------------- | ------- | -------- |
| `CLAUDE.md` or `AGENTS.md` with clear boundaries| Yes     | 🟢       |
| Pure functions mapped for LLM-based fuzzing      | Partial | 🟡       |
| Explicit `logging` (not `print`) for telemetry   | Yes (0 print, 2014 logging) | 🟢 |
| Structural decoupling (fits LLM context windows)  | Moderate (43 files >800 lines) | 🟡 |
| Deterministic test suite (no flaky tests)         | Mostly  | 🟢       |
| Self-documenting code (minimal implicit knowledge)| Good    | 🟢       |
| Config-driven behavior (no hidden env deps)       | Mostly  | 🟢       |

**Findings:**

- CLAUDE.md is comprehensive with development commands, coding standards, test markers, and cross-repo dependency warnings
- GAAI framework is installed (`.gaai/`) providing governance structure
- The 43 files >800 lines are problematic for LLM context windows -- agents must read large files to understand single functions
- Test markers (13 total) enable agents to run targeted test suites
- The `--no-verify` requirement on Windows for pre-commit hooks is a friction point for agents

---

## Summary Scorecard

| #       | Criterion                | Score   | Priority |
| ------- | ------------------------ | ------- | -------- |
| 1       | DRY                      | 7.0/10  | 🟡       |
| 2       | Orthogonality            | 6.5/10  | 🔴       |
| 3       | Monolithic Files         | 5.5/10  | 🔴       |
| 4       | Function Length           | 6.0/10  | 🔴       |
| 5       | God Functions            | 6.5/10  | 🟡       |
| 6       | Law of Demeter           | 7.5/10  | 🟢       |
| 7       | Name Quality             | 8.0/10  | 🟢       |
| 8       | Magic Numbers            | 7.5/10  | 🟢       |
| 9       | Design by Contract       | 8.5/10  | 🟢       |
| 10      | Error Handling           | 7.5/10  | 🟢       |
| 11      | TDD                      | 8.5/10  | 🟢       |
| 12      | Comment Quality          | 7.0/10  | 🟡       |
| 13      | Project Structure        | 8.0/10  | 🟢       |
| 14      | Deprecated Code          | 7.5/10  | 🟡       |
| 15      | Cleanup                  | 7.5/10  | 🟡       |
| 16      | Reversibility            | 7.0/10  | 🟡       |
| 17      | Changeability            | 7.0/10  | 🟡       |
| 18      | Reusability              | 8.0/10  | 🟢       |
| 19      | Calculation Optimization | 7.5/10  | 🟢       |
| 20      | Security                 | 7.0/10  | 🟡       |
| 21      | Dependencies             | 7.5/10  | 🟢       |
| 22      | CI/CD & Automation       | 7.5/10  | 🟢       |
| 23      | Parity / Maintenance     | 6.5/10  | 🔴       |
| 24      | Agentic Usability        | 8.0/10  | 🟢       |
| **AVG** | **Overall**              | **7.15/10** |      |

### Category Summary (A-F Grades)

| Category | Grade | Key Issues |
| -------- | ----- | ---------- |
| I. Code Craftsmanship | C+ | 156 files >500 lines, 755 functions >50 lines, 496 functions with >4 params |
| II. Robustness & Error Handling | B+ | Strong DbC (51% precondition coverage), 32 broad exceptions (all annotated) |
| III. Testing & Validation | A- | 216% test-to-code ratio, 568 test files, 13 markers, CI-enforced |
| IV. Documentation & Domain Language | B | 59.2% function docstrings, 91.3% class docstrings |
| V. Project Organization | B+ | Clean src/ layout, 46 tools, shared library, but 40 sys.path hacks |
| VI. Reversibility & Changeability | B | Protocol interfaces, config modules, but monolithic files limit changeability |
| VII. Performance & Scalability | B | JAX acceleration in pendulum, NumPy vectorization, good memory practices |
| VIII. Dependencies & Security | B | No eval/shell/CORS issues, but 7 xml.etree usages remain |
| IX. Automation & Operations | B+ | 56 CI workflows (bloated), strong linting/testing automation |
| X. Parity & Maintenance | C+ | 161 stale branches, 147 open issues, but 0 print() and active CLAUDE.md |
| XI. Agentic Usability | B+ | CLAUDE.md + GAAI, but large files hurt LLM context efficiency |

---

## Pendulum Simulator Deep Dive

The pendulum_simulator is the largest single tool in the monorepo (25,440 source lines, 93 test files).

### Strengths
- **Physics separation:** Core physics (`physics.py`, `golfer_dynamics.py`, `constraint_solver.py`) is cleanly separated from GUI
- **JAX acceleration:** `physics_golfer_jax.py` uses JAX for performance-critical FK/Jacobian computation
- **Constants discipline:** Dedicated `constants.py` replaces former magic numbers (28+ locations consolidated)
- **DbC:** 643 precondition checks across 827 functions (78% coverage)
- **Testing:** 93 test files with 21,199 lines (83% test-to-source ratio)
- **Zero print():** All output goes through logging
- **Class docstrings:** 98.5% coverage

### Weaknesses
- **Monolithic GUI files:** 19 files >500 lines, 6 files >800 lines. `equations_popup.py` (1,160), `simulation_panel.py` (947), `pendulum_widget.py` (915)
- **Function bloat:** 106 functions >50 lines, 13 functions >100 lines. `_build_overlay_section` (232 lines) is the worst
- **Function docstrings:** Only 47.0% coverage (below repo average of 59.2%)
- **GUI-physics coupling:** Some GUI widgets directly compute physics quantities instead of delegating
- **Magic numbers in GUI:** Style constants (colors, sizes, DPI) not extracted

---

## Priority Remediation Targets (Stone Soup Strategy)

| Priority | Issue / Violation | Pragmatic Heuristic | Criterion | Required Action |
|----------|-------------------|---------------------|-----------|-----------------|
| P0 | 161 stale remote branches | Broken Windows | #23 | `git push origin --delete` for branches with no open PRs |
| P0 | 147 open issues untriaged | Broken Windows | #23 | Triage: close stale, label active, prioritize |
| P1 | 43 files >800 lines | Orthogonality | #3 | Decompose top 10 largest files (start with pendulum GUI) |
| P1 | 40 sys.path hacks | Broken Windows | #14 | Replace with proper package imports via pyproject.toml |
| P1 | 59 functions >100 lines | God Functions | #5 | Decompose top 20 longest functions |
| P2 | 7 xml.etree usages | Security | #20 | Migrate to defusedxml for XML parsing |
| P2 | 496 functions with >4 params | Function Signatures | #4 | Introduce dataclasses/TypedDict for parameter groups |
| P2 | 40.8% functions missing docstrings | Documentation | #12 | Add docstrings to all public functions |
| P3 | 56 CI workflows | Automation | #22 | Consolidate Jules bot workflows |
| P3 | Duplicate urdf_builder_gui/theme.py | DRY | #1 | Consolidate to single location |

---

## Improvement Roadmap

### Phase 1 — Critical (This Sprint)

- [ ] Triage 147 open issues: close stale bot-generated issues, label and prioritize active ones
- [ ] Delete 161 stale remote branches (keep only branches with active PRs)
- [ ] Replace top 10 sys.path hacks with proper package imports

### Phase 2 — High Priority (Next Sprint)

- [ ] Decompose top 5 largest pendulum GUI files (equations_popup, simulation_panel, pendulum_widget, toolstrip_widget, panel_builders)
- [ ] Decompose 13 pendulum functions >100 lines into sub-functions
- [ ] Migrate 7 xml.etree usages to defusedxml
- [ ] Eliminate remaining 30 sys.path hacks

### Phase 3 — Medium Priority (Backlog)

- [ ] Decompose remaining 46 functions >100 lines across the repo
- [ ] Add docstrings to public functions in pendulum_simulator (target: 70% from 47%)
- [ ] Introduce parameter dataclasses for top 50 functions with >4 params
- [ ] Consolidate duplicate urdf_builder_gui/theme.py

### Phase 4 — Polish (Future)

- [ ] Consolidate 56 CI workflows toward ~30
- [ ] Extract GUI magic numbers (colors, sizes, DPI) to theme constants
- [ ] Add postcondition assertions to critical physics functions
- [ ] Containerize additional tools beyond pendulum_simulator

---

## Appendix: Assessment Coverage Matrix

This template unifies the following assessment frameworks:

### A-O Architecture Assessment Mapping

| A-O | Category | Unified Criteria |
| --- | -------- | ---------------- |
| A | Code Structure | #13 Project Structure |
| B | Documentation | #12 Comment Quality |
| C | Testing | #11 TDD |
| D | Error Handling | #9 DbC, #10 Error Handling |
| E | Performance | #19 Calculation Optimization |
| F | Security | #20 Security |
| G | Dependencies | #21 Dependencies |
| H | CI/CD | #22 CI/CD & Automation |
| I | Code Style | #7 Name Quality, #8 Magic Numbers |
| J | API Design | #4 Function Length & Signatures |
| K | Data Handling | #1 DRY, #6 Law of Demeter |
| L | Logging | #23 Parity / Maintenance |
| M | Configuration | #16 Reversibility, #17 Changeability |
| N | Scalability | #19 Calculation Optimization |
| O | Maintainability | #2 Orthogonality, #3 Monolithic Files |
| P | Agentic Usability | #24 Agentic Usability |

### Pragmatic Programmer Principle Mapping

| Principle | Unified Criteria |
| --------- | ---------------- |
| DRY | #1 DRY |
| Orthogonality | #2 Orthogonality |
| Reversibility | #16 Reversibility |
| Broken Windows | #14 Deprecated Code, #15 Cleanup |
| Design by Contract | #9 DbC |
| Test Early, Test Often | #11 TDD |
| Domain Languages | #12 Comment Quality |
| Automate Everything | #22 CI/CD & Automation |
| Crash Early | #10 Error Handling |
| It's All Writing | #12 Comment Quality |
| Tracer Bullets | #11 TDD (edge cases) |
| Stone Soup | Priority Remediation Targets |

---

## Historical Comparison

| Metric | v6.0 (2026-02-19) | v7.0 (2026-03-26) | Delta |
| ------ | ------------------ | ------------------ | ----- |
| Overall Score | 7.13/10 | 7.15/10 | +0.02 |
| Source Lines | ~45K | 48,356 | +3K |
| Test Files | ~500 | 568 | +68 |
| Functions >100 lines | 34 | 59 | +25 (growth) |
| Files >500 lines | ~130 | 156 | +26 (growth) |
| Broad exceptions | ~30 | 32 | +2 |
| print() in src | 0 | 0 | Stable |
| sys.path hacks | ~35 | 40 | +5 |
| Open issues | 12 | 147 | +135 (bot accumulation) |

**Trend analysis:** The codebase has grown by ~3K lines and gained 68 test files, maintaining the strong test-to-code ratio. However, structural metrics have regressed: functions >100 lines increased from 34 to 59, and files >500 lines grew from ~130 to 156. The 135-issue increase is primarily from automated bot workflows creating issues faster than they are resolved. The core quality discipline (zero print, annotated exceptions, strong DbC) remains solid.

---

_Generated by the Unified Code Quality Assessment Framework v3.0_
_Template: `Repository_Management/docs/templates/unified_assessment_template.md`_
_Combines: Pragmatic A-O Template + Code Quality Assessment Template v2.0_
