# Adversarial Security & Functional Assessment — Tools Repository (Extended)

**Date**: 2026-04-22
**Assessor**: Antigravity automated adversarial review
**Branch**: `audit/adversarial-review-2026-04-22`

## Executive Summary

This extended assessment covers the comprehensive adversarial review of the
D-SOrganization Tools repository. Building on the initial 12-finding audit
(which addressed Rust `tools_core` and shared Python utilities), this extension
systematically reviews ALL remaining Python modules including:

- `calc_backend` (FastAPI REST API for calculators)
- `signal_toolkit` (signal processing library)
- `model_generation` (URDF/MJCF generation, conversion, REST API)
- `programmatic_pid` (DXF P&ID generation)
- `upstream_drift_tools` (process calculators, UI, state management)
- `scripting` (interactive console environment)
- `data_processing` (DataFrame processing)
- `cors` (shared CORS middleware)

Overall, the security posture is **strong**. The AST-hardened `safe_eval`
framework, consistent `defusedxml` usage, and DbC framework are well-applied.
The 9 new findings are primarily about exception handling hygiene, defense-in-depth
for pandas eval, and timezone-naive timestamps.

---

## New Findings (This Extension)

### Finding 13 — MEDIUM: rotation_converter.py broad `except Exception` in API

**File**: `src/shared/python/calc_backend/routers/rotation_converter.py`
**Lines**: 68, 104, 127

Three `except Exception` catches in REST API handlers mask programming errors
as HTTP 422/500 responses. All have `noqa: BLE001` suppression.

**GitHub Issue**: [#2238](https://github.com/D-sorganization/Tools/issues/2238)

---

### Finding 14 — LOW: `datetime.now()` without timezone in 14 locations

**Files**: `state_manager.py` (6), `unit_converter_widget.py` (2),
`calculator_state_mixin.py` (2), `data_processing/core.py` (1),
`text_editor.py` (1), tests (2)

All timestamps are timezone-naive, causing ambiguity in cross-timezone
comparisons and potential DST edge cases in autosave cleanup.

**GitHub Issue**: [#2239](https://github.com/D-sorganization/Tools/issues/2239)

---

### Finding 15 — LOW: unified_loader.py broad `except Exception` in conversions

**File**: `src/shared/python/model_generation/library/unified_loader.py`
**Lines**: 450, 487

`convert_to_urdf()` and `convert_to_mjcf()` catch `Exception` broadly and
re-raise as `ConversionError`. The `logger.exception()` mitigates
observability concerns.

**GitHub Issue**: [#2240](https://github.com/D-sorganization/Tools/issues/2240)

---

### Finding 16 — MEDIUM: scripting_env.py `except Exception` on library load

**File**: `src/shared/python/scripting/scripting_env.py`
**Line**: 155

The `refresh_user_functions()` method catches `Exception` (including
`SystemExit`, `KeyboardInterrupt`) and writes to stderr, which may not be
visible in a GUI context.

**GitHub Issue**: [#2241](https://github.com/D-sorganization/Tools/issues/2241)

---

### Finding 17 — LOW: DataFrame.eval() without explicit engine guard

**Files**: `data_processing/processor.py` (line 385),
`upstream_drift_tools/data_processing/core.py` (line 260)

Both `DataFrame.eval()` calls lack an explicit `engine='numexpr'` parameter.
While the default is safe, explicitly specifying it prevents future regressions.

**GitHub Issue**: [#2242](https://github.com/D-sorganization/Tools/issues/2242)

---

### Finding 18 — LOW: test_urdf_roundtrip.py uses standard ElementTree

**File**: `model_generation/tests/test_urdf_roundtrip.py`
**Line**: 216

Only test file using `xml.etree.ElementTree` directly instead of `defusedxml`.
All production parsers correctly use `defusedxml`.

**GitHub Issue**: [#2243](https://github.com/D-sorganization/Tools/issues/2243)

---

### Finding 19 — MEDIUM: REST API handle_request info leakage + temp file leak

**File**: `src/shared/python/model_generation/api/rest_api_routes.py`
**Lines**: 342 (broad exception), 763-780 (temp file leak)

1. Top-level handler exposes raw exception messages in HTTP 500 responses
2. `inertia_from_mesh` creates temp file with `delete=False` but cleanup
   not guarded by try/finally — file leaks on exception

**GitHub Issue**: [#2244](https://github.com/D-sorganization/Tools/issues/2244)

---

### Finding 20 — LOW: ODE solver `_rk4_solve` div-by-zero at num_points=1

**File**: `src/shared/python/calc_backend/routers/ode_solver.py`
**Line**: 97

`dt = (t_end - t_start) / (num_points - 1)` fails when `num_points == 1`.
The Pydantic model has `ge=2` validation, but the function has no self-guard.

**GitHub Issue**: [#2245](https://github.com/D-sorganization/Tools/issues/2245)

---

### Finding 21 — LOW: Signal generators div-by-zero at zero frequency

**File**: `src/shared/python/signal_toolkit/core.py`
**Lines**: 534, 541, 572, 600, 631

`chirp`, `sawtooth`, `triangle`, and `square` generators divide by
`frequency` or `t_end` without guarding zero values, producing
uninformative `ZeroDivisionError` instead of clear `ValueError`.

**GitHub Issue**: [#2246](https://github.com/D-sorganization/Tools/issues/2246)

---

## Positive Findings (No Action Required)

| Area | Assessment |
|------|-----------|
| `safe_eval.py` | AST-hardened evaluator with `__builtins__: {}` sandbox |
| `scripting_env.py` exec/eval | Controlled, documented, `nosec` annotated, GUI-only |
| `defusedxml` usage | All production XML parsers use `defusedxml.ElementTree` |
| `cors.py` | Centralized CORS config, no wildcard origins |
| `contracts.py` | DbC framework consistently applied |
| `signal_toolkit` | No broad exception catches, clean arithmetic |
| `programmatic_pid` | Clean DXF rendering, no injection vectors |
| `calc_backend/app.py` | No `shell=True`, proper CORS configuration |
| `bootstrap.py` | Documented single-use sys.path mutation |
| `upstream_drift_tools` | Pydantic validation on all REST contracts |
| Process constants | CODATA-sourced from centralized `unit_constants.py` |
| Security headers | REST API applies CSP, X-Frame-Options, HSTS |

---

## Combined Summary Table (All Findings)

### Original Audit (Findings 1-12, already remediated)
See [adversarial_review_2026-04-22.md](adversarial_review_2026-04-22.md)

### Extended Audit (Findings 13-21, this document)

| # | Severity | Component | Description |
|---|----------|-----------|-------------|
| 13 | 🟡 MEDIUM | `rotation_converter.py` | Broad `except Exception` in 3 API handlers |
| 14 | 🟢 LOW | 14 locations | `datetime.now()` without timezone |
| 15 | 🟢 LOW | `unified_loader.py` | Broad exception in conversion methods |
| 16 | 🟡 MEDIUM | `scripting_env.py` | Broad exception on user library load |
| 17 | 🟢 LOW | `processor.py` + `core.py` | `DataFrame.eval()` without explicit engine |
| 18 | 🟢 LOW | `test_urdf_roundtrip.py` | Standard ElementTree in test (not defusedxml) |
| 19 | 🟡 MEDIUM | `rest_api_routes.py` | Info leakage + temp file leak |
| 20 | 🟢 LOW | `ode_solver.py` | RK4 div-by-zero at num_points=1 |
| 21 | 🟢 LOW | `signal_toolkit/core.py` | Zero-frequency div-by-zero in generators |

### Totals

| Severity | Count |
|----------|-------|
| 🔴 HIGH | 0 |
| 🟡 MEDIUM | 3 |
| 🟢 LOW | 6 |
| **Total** | **9** |
