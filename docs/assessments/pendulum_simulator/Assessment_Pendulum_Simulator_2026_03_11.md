# Pendulum Simulator — Comprehensive A–O Assessment

**Date**: 2026-03-11
**Assessor**: AI Assessment Agent (Antigravity)
**Scope**: `src/pendulum_simulator/` — all source, tests, and configuration
**Context**: Post V2 feature sprint (24 issues, PR #1156)

---

## Executive Summary

The Pendulum Simulator has undergone a major V2 enhancement sprint closing 24 issues across 12 batches. The codebase demonstrates **strong physics modeling**, **comprehensive GUI integration**, and **good TDD practices** for new features. However, the rapid feature development has created pockets of **incomplete DbC coverage**, **bare except clauses**, and **GUI modules lacking dedicated test files**.

**Weighted Score: 7.8/10** (up from ~6.0 before V2 sprint)

---

## Grade Table

| Cat | Category                   | Score  | Trend | Key Evidence |
|-----|----------------------------|--------|-------|-------------|
| A   | Architecture & Structure   | 8.5/10 | ↑     | Clean separation: physics ↔ simulation ↔ GUI. 3 model variants (double/triple/golfer). New modules (unit_converter, equations_popup, popout_chart) well-isolated |
| B   | Code Quality & Hygiene     | 7.0/10 | →     | 2 print() calls remain (optimizer_gpu). 8 bare `except Exception:` clauses. 2 TODOs. No wildcard imports |
| C   | Documentation & Docstrings | 6.5/10 | ↑     | 62% public method docstring coverage (201/325). DbC docstrings (Pre/Post) on new code. Missing on many GUI event handlers |
| D   | Error Handling             | 5.5/10 | →     | 8 bare `except Exception:` in GUI widgets. GUI error handling swallows exceptions silently. `_on_sim_error` now logs properly |
| E   | Performance                | 7.5/10 | ↑     | Rust native backend available. DOP853 integration. GPU optimizer. No profiling CI gate |
| F   | Security                   | 8.0/10 | →     | No secrets. No SQL. File I/O via Qt dialogs. No network calls |
| G   | Testing & TDD              | 7.5/10 | ↑↑    | 321 tests pass. Strong physics/simulation coverage. Missing: 26 GUI modules have no dedicated tests (UI testing is inherently harder) |
| H   | CI/CD                      | 9.0/10 | →     | Ruff + Black + Mypy + Bandit + module size budget all enforced. CRLF normalization |
| I   | Code Style & Consistency   | 8.5/10 | ↑     | Ruff/Black formatted. Type hints on 98%+ of functions. Consistent naming conventions |
| J   | API / Interface Design     | 8.0/10 | ↑     | Clean dataclass params. Signal/slot GUI architecture. UnitConverter is stateless/pure |
| K   | DRY (Don't Repeat Yourself)| 7.0/10 | ↑     | Shared `controls_utils.py` for common widgets. `clamp_torque_ndof` reused across models. Some duplication in controls widget presets and style strings |
| L   | Logging                    | 8.0/10 | ↑↑    | `logging` module used throughout. No print() in physics/simulation. 2 print() remaining in GPU optimizer |
| M   | Design by Contract (DbC)   | 6.0/10 | ↑     | 18% of functions have assertions (87/483). New code has Pre/Post in docstrings. Legacy code lacks contracts |
| N   | Scalability                | 7.5/10 | →     | Supports 2/3/8-DOF models. Extensible model architecture. Graph and chart modules independent |
| O   | Maintainability            | 7.0/10 | ↑↑    | All modules under 900 lines (budget: 1500). Clear module boundaries. 2 TODOs tracked with issue numbers |

---

## Top 10 Findings

| ID | Severity | Category | Location | Issue | Remediation | Effort |
|----|----------|----------|----------|-------|-------------|--------|
| PS-B-001 | MAJOR | Quality | `optimizer_gpu.py:195,264` | 2 `print()` calls violate logging standard | Replace with `logger.info()` | S |
| PS-D-001 | MAJOR | Error | 8 GUI files | 8 bare `except Exception:` swallow errors | Catch specific exceptions, log others | M |
| PS-M-001 | MAJOR | DbC | All legacy code | Only 18% DbC assertion coverage | Add Pre/Post assertions to physics functions | L |
| PS-C-001 | MODERATE | Docs | 26 GUI modules | 124 public methods missing docstrings | Add docstrings to GUI event handlers | L |
| PS-G-001 | MODERATE | Testing | 26 modules | No dedicated test files for GUI modules | Add smoke tests for widget construction | M |
| PS-K-001 | MODERATE | DRY | controls_widget*.py | Preset dictionaries duplicated across 3 controls widgets | Extract to shared preset registry | M |
| PS-K-002 | MINOR | DRY | controls_widget*.py | Style strings for QGroupBox repeated in each widget | Already in `controls_utils.py` STYLE_GROUP — verify all usage |  S |
| PS-B-002 | MINOR | Quality | `perf_test.py` | Uses `print()` with noqa — acceptable for benchmarking script | No action needed | — |
| PS-E-001 | MINOR | Perf | physics*.py | No benchmark regression tests in CI | Add criterion-style benchmark assertions | M |
| PS-N-001 | MINOR | Scale | `popout_chart.py` | `_regression` attribute created dynamically via `self._regression = ...` | Initialize in `__init__` | S |

---

## Detailed Category Analysis

### A: Architecture (8.5/10)
**Strengths**: Clean MVC separation with physics (model), simulation (controller), GUI (view). Each model variant (double, triple, golfer) has dedicated physics, simulation, and controls modules. New V2 modules follow the same pattern.

**Weaknesses**: `main_window.py` at 896 lines is nearing the complexity limit and acts as a wiring hub for all models.

### B: Code Quality (7.0/10)
- ✅ No wildcard imports
- ✅ No bare `except:` (all have at least `Exception`)
- ⚠️ 2 `print()` calls in `optimizer_gpu.py`
- ⚠️ 8 bare `except Exception:` without logging

### G: Testing (7.5/10)
- ✅ 321 tests passing, 2 skipped
- ✅ Physics modules have excellent coverage (mass matrix, coriolis, gravity, forward kinematics)
- ✅ New V2 features all have TDD tests
- ⚠️ GUI modules lack individual test files (inherently harder to test)

### K: DRY (7.0/10)
- ✅ `controls_utils.py` provides shared labeled inputs, parse helpers
- ✅ `clamp_torque_ndof` reused across triple + golfer
- ✅ `fit_regression` isolated for reuse
- ⚠️ Preset configurations duplicated in each controls widget

### M: Design by Contract (6.0/10)
- ✅ New code (V2 batch) consistently uses Pre/Post assertions
- ✅ `TriplePendulumParams.__post_init__` validates all fields
- ⚠️ Legacy code (particularly GUI event handlers) lacks assertions
- ⚠️ Only 87/483 functions have any assertion

---

## Recommendations (Priority Order)

1. **[S] Fix print() in optimizer_gpu.py** — Replace 2 print() calls with logger
2. **[S] Initialize _regression in PopOutChart.__init__** — Avoid dynamic attribute creation
3. **[M] Narrow except clauses** — Replace 8 bare `except Exception:` with specific exceptions + logging
4. **[M] Add GUI smoke tests** — Widget construction tests for untested modules
5. **[L] Increase DbC coverage** — Add assertions to physics functions (target: 40%)
6. **[L] Add docstrings** — Focus on public API methods first
7. **[M] Extract preset registry** — DRY preset configurations

---

## Methodology
Manual code review + static analysis (AST parsing, grep). Scoring based on fleet assessment A–O template. Compared against Tools repository standards and AGENTS.md requirements.
