# Tools Repository — Comprehensive A-O + Pragmatic Programmer Assessment 2026-03-22

> Full code read assessment. Synced with staging at commit aec0117e, 2026-03-22.
> Scope: 1,120 Python files, 264,549 lines in src/
> GitHub Issue: D-sorganization/Tools#1708

---

## Score Summary

| # | Category | Score | Key Finding |
|---|----------|-------|-------------|
| A | Abstraction | 6/10 | 25 files fail format check; clean code surface mostly good |
| B | Broken Windows | 4/10 | type:ignore rose 724->1147; 6 test failures on staging |
| C | Correctness | 6/10 | 4052/4096 tests pass but 27 collection errors hide failures |
| D | DRY | 5/10 | 6 duplicate files in urdf_builder_gui; 4x calculate_pressure_drop |
| E | Entropy | 5/10 | Dual ruff configs with conflicting rules; dead Black config |
| F | File Size | 6/10 | Down from 3 to 2 oversized files; 20 in danger zone |
| G | God Functions | 5/10 | 15 functions >100 lines; max 239; complexity up to 32 |
| H | High Coupling | 5/10 | 10 god classes (25+ methods) in shared library |
| I | Integration | 7/10 | Cross-repo tests pass (38/38); gaps in Gasification coverage |
| J | Security | 7/10 | defusedxml used in most places; 6 files use stdlib xml.etree |
| K | Key Dependencies | 7/10 | cors.py import FIXED; contracts fragile but functional |
| L | LOD | 6/10 | Good package isolation; large __init__.py files (up to 611L) |
| M | Monitoring | 4/10 | CI delta-only means full-repo violations undetected |
| N | Noise | 6/10 | 304 noqa + 1137 type:ignore; T201 missing from active config |
| O | Orthogonality | 6/10 | Good module boundaries; E501 globally suppressed |
| PP1 | Broken Windows | 5/10 | Inconsistent contracts import paths (3 styles) |
| PP2 | Tracer Bullets | 6/10 | 40% coverage threshold is low for shared library |
| PP3 | Reversibility | 7/10 | Clean dependency graph; dead Black config minor |
| PP4 | Design by Contract | 7/10 | Mature DbC framework; unevenly applied to calculators |
| PP5 | Estimating | 6/10 | Budget limit 1200 (docs) vs 1500 (CI) mismatch |
| PP6 | Decoupling | 5/10 | 4 pressure_drop implementations; 7 calculate() variants |
| PP7 | Types | 6/10 | 67% files have annotations; 91% docstring coverage in shared |

**Overall: 5.8/10** (up from ~5.0 in Phase 4 assessment, 2026-03-13)

---

## Verified Key Items

| Question | Status | Details |
|----------|--------|---------|
| cors.py import regression | **FIXED** | `from contracts import require` resolves correctly |
| 3 files >1200 lines | **Improved to 2** | modern_robotics.py (2084, baselined), main_window.py (1233, NOT baselined at assessment time) |
| 724 type:ignore | **REGRESSED to 1147** | 58% increase; 1137 targeted, 4 bare |
| 92 print() calls | **Partially fixed** | 2 remain in modern_robotics.py; T201 not in active ruff config |
| Cross-repo integration tests | **PASS (38/38)** | But 3 heavy_integration tests fail (missing deps) |

---

## Assessment A: Abstraction

**Score: 6/10**

- 25 Python files fail `ruff format --check` → Issue #1685
- Code surface is mostly clean; public APIs have docstrings
- Large `__init__.py` files re-export many symbols, making API boundaries fuzzy

## Assessment B: Broken Windows

**Score: 4/10**

- `type: ignore` count rose from 724 → 1147 (58% increase) → Issue #1686
- 6 test failures on staging branch → Issue #1688
- T201 (print detection) missing from active ruff config → Issue #1684
- These are the highest-severity broken windows: failures allowed to accumulate

## Assessment C: Correctness

**Score: 6/10**

- 4,052/4,096 tests pass (98.7% pass rate)
- 27 test collection errors hide additional failures → Issue #1687
- Heavy integration tests fail on missing optional deps (trimesh, ezdxf) → Issue #1688

## Assessment D: DRY

**Score: 5/10**

- 6 duplicate files in `urdf_builder_gui/` → Issue #1693
- 4 implementations of `calculate_pressure_drop` across modules → Issue #1705
- This repo is the DRY layer for downstream consumers; duplication here is especially damaging

## Assessment E: Entropy

**Score: 5/10**

- Dual ruff configs (`ruff.toml` + `pyproject.toml`) with conflicting rules → Issue #1689
- Dead Black config in `pyproject.toml` → Issue #1703
- Single source of truth for linting config is a foundational requirement

## Assessment F: File Size

**Score: 6/10**

- Improved from 3 → 2 oversized files (>1200 lines)
- `modern_robotics.py` (2,084 lines) — baselined
- `main_window.py` (1,233 lines) — PyQt6 GUI monolith, not yet baselined at assessment time
- 20 files in danger zone (900–1200 lines) → Issue #1690

## Assessment G: God Functions

**Score: 5/10**

- 15 functions exceed 100 lines → Issue #1691
- Longest: 239 lines
- Cyclomatic complexity up to 32 on some functions
- High complexity correlates with type:ignore clusters

## Assessment H: High Coupling

**Score: 5/10**

- 10 god classes with 25+ methods in the shared library → Issue #1692
- Shared library god classes are a breaking-change risk for downstream consumers
- High method count correlates with many responsibilities (SRP violations)

## Assessment I: Integration

**Score: 7/10**

- Cross-repo integration tests: 38/38 PASS
- Gaps in Gasification_Model coverage → Issue #1699
- Heavy integration tests (trimesh, ezdxf) fail due to missing optional deps guard

## Assessment J: Security

**Score: 7/10**

- `defusedxml` used in majority of XML-parsing code
- 6 files still use stdlib `xml.etree.ElementTree` without defusedxml → Issue #1694
- No other critical security findings

## Assessment K: Key Dependencies

**Score: 7/10**

- cors.py import regression: **FIXED** (from contracts import require)
- contracts import chain fragility — 3 different import styles → Issues #1695, #1700
- Downstream repos depend on this working reliably

## Assessment L: Law of Demeter

**Score: 6/10**

- Good package isolation; modules mostly don't import across package boundaries
- Large `__init__.py` files (up to 611 lines) create long re-export chains → Issue #1696
- Method chains generally within 2-level limit

## Assessment M: Monitoring

**Score: 4/10**

- CI delta-only linting/typecheck means full-repo violations accumulate undetected → Issue #1697
- No visibility into cross-module type:ignore drift
- This is the systemic cause of the 724→1147 type:ignore increase

## Assessment N: Noise

**Score: 6/10**

- 304 `# noqa` suppressions + 1,147 `# type: ignore` → Issues #1686, #1698
- T201 (print detection) missing from active config — `print()` in src/ undetected → Issue #1684
- E501 (line length) globally suppressed → Issue #1702

## Assessment O: Orthogonality

**Score: 6/10**

- Good module boundaries between signal_processing, urdf, calculators, pid, themes
- LOD violations: E501 globally suppressed undermines line-length discipline
- 4x `calculate_pressure_drop` is an orthogonality failure → Issue #1705

## Pragmatic Programmer Assessments

### PP1: Broken Windows
**Score: 5/10** — Inconsistent contracts import paths (3 styles in use) → Issue #1700

### PP2: Tracer Bullets
**Score: 6/10** — 40% test coverage threshold is low for a shared library used by 2 downstream repos → Issue #1701

### PP3: Reversibility
**Score: 7/10** — Clean dependency graph; dead Black config is minor noise → Issue #1703

### PP4: Design by Contract
**Score: 7/10** — Mature DbC framework (contracts library); unevenly applied to calculators → Issue #1704

### PP5: Estimating
**Score: 6/10** — Module size budget: docs say 1200 lines, CI enforces 1500 → Issue #1706

### PP6: Decoupling
**Score: 5/10** — 4 implementations of pressure_drop; 7 variants of calculate() → Issue #1705

### PP7: Types
**Score: 6/10** — 67% of source files have function type annotations; 91% docstring coverage in shared modules → Issue #1707

---

## Child Issues Created

| Issue | Title | Priority |
|-------|-------|----------|
| #1684 | T201 rule missing from ruff.toml | High |
| #1685 | 25 files fail ruff format check | Medium |
| #1686 | type:ignore count rose to 1147 | High |
| #1687 | 27 test collection errors | High |
| #1688 | 6 test failures on staging | High |
| #1689 | Dual ruff config conflict | High |
| #1690 | File size budget (2 over, 20 danger zone) | Medium |
| #1691 | God functions (15 over 100 lines) | Medium |
| #1692 | God classes (10 with 25+ methods) | Medium |
| #1693 | 6 duplicate files in urdf_builder_gui | Medium |
| #1694 | 6 files use unsafe xml.etree.ElementTree | Medium |
| #1695 | contracts import chain fragility | Medium |
| #1696 | Large __init__.py god modules | Low |
| #1697 | CI delta-only checking gaps | High |
| #1698 | noqa suppression audit | Low |
| #1699 | Cross-repo integration gaps | Medium |
| #1700 | Inconsistent contracts import paths | Medium |
| #1701 | Test coverage threshold too low | Medium |
| #1702 | E501 globally suppressed | Low |
| #1703 | Dead Black config in pyproject.toml | Low |
| #1704 | DbC enforcement inconsistency | Medium |
| #1705 | Duplicated pressure drop functions | Medium |
| #1706 | Budget limit docs vs CI mismatch | Medium |
| #1707 | Type annotation coverage gaps | Low |

---

## Top 5 Priority Actions

1. **Fix T201 in ruff.toml** (#1684) — broken window that undermines the entire GH1655 print-to-logging effort
2. **Fix 6 test failures on staging** (#1688) — staging must always be green
3. **Fix 27 collection errors** (#1687) — blocks `pytest -x` for all developers
4. **Add full-repo CI check** (#1697) — prevents silent regression (root cause of type:ignore drift)
5. **Consolidate ruff config** (#1689) — single source of truth for linting rules

---

## Trend Analysis

| Metric | Phase 4 (2026-03-13) | This Assessment (2026-03-22) | Trend |
|--------|---------------------|------------------------------|-------|
| Overall Score | ~5.0/10 | 5.8/10 | +0.8 ↑ |
| type:ignore | 724 | 1,147 | +58% ↓ |
| Files >1200 lines | 3 | 2 | -1 ↑ |
| print() in src/ | 92 | 2 | -95% ↑ |
| Test pass rate | ~97% | 98.7% | +1.7% ↑ |
| Cross-repo tests | unknown | 38/38 | PASS ↑ |
