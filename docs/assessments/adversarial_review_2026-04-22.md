# Adversarial Security & Functional Assessment — Tools Repository

**Date**: 2026-04-22
**Assessor**: Antigravity automated adversarial review
**Branch**: `audit/adversarial-review-2026-04-22`

## Executive Summary

The Tools repository is a shared utility hub providing 35+ individual tools with a
Rust core (`tools-core`), Python shared libraries, and a Design-by-Contract (DbC)
framework used by the entire D-SOrganization fleet. Overall code quality is high:
the `safe_eval` module correctly restricts AST node types, subprocess calls avoid
`shell=True`, and no `pickle.load()` deserialization is present. However, several
medium- and low-severity findings were identified that impact reliability,
correctness, and maintainability.

---

## Findings

### Finding 1 — MEDIUM: `contracts.py` stale module-level binding silently bypasses `set_contract_level()`

**File**: `src/shared/python/contracts.py`
**Lines**: 81, 168, 182, 192, 202

The core contract primitives (`require`, `ensure`, `invariant`) read from the
module-level `DBC_LEVEL` variable:

```python
DBC_LEVEL: ContractLevel = _ContractState.level  # line 81
# ...
def require(condition, message, value=None):
    if DBC_LEVEL == ContractLevel.OFF:  # reads the MODULE-LEVEL copy
        return
```

`set_contract_level()` (line 85) correctly updates both `_ContractState.level` and
the module alias. **However**, any code that has already imported `DBC_LEVEL` via
`from contracts import DBC_LEVEL` holds a **stale reference** to the original
`ContractLevel` enum value. Since Python enum instances are immutable, the
rebinding at line 92 only updates `sys.modules[__name__].DBC_LEVEL` — it does not
retroactively update local names in consumer modules.

**Impact**: Callers who toggle contract levels at runtime (e.g. switching to `OFF`
for benchmarks) may find that contracts remain active in modules that imported
`DBC_LEVEL` before the toggle.

**Fix**: Change `require`/`ensure`/`invariant` to read from `_ContractState.level`
instead of the module alias, or use a function call `get_contract_level()`.

---

### Finding 2 — MEDIUM: `normalize` transform produces `NaN` / division-by-zero on constant columns

**File**: `src/shared/python/upstream_drift_tools/data_processing/core.py`
**Line**: 325

```python
"normalize": lambda: (col - col.min()) / (col.max() - col.min()),
```

When `col.max() == col.min()` (constant column), this produces a division by zero
resulting in `NaN` across the entire column — silently, with no error or warning.

**Impact**: Downstream consumers receive all-NaN data without any indication of
failure.

**Fix**: Guard against zero-range columns:

```python
range_val = col.max() - col.min()
if range_val == 0:
    raise TransformationError(f"Cannot normalize constant column '{column}'")
```

---

### Finding 3 — MEDIUM: `standardize` transform produces `NaN` on zero-variance columns

**File**: `src/shared/python/upstream_drift_tools/data_processing/core.py`
**Line**: 326

```python
"standardize": lambda: (col - col.mean()) / col.std(),
```

Same division-by-zero issue as Finding 2 when `col.std() == 0`.

---

### Finding 4 — MEDIUM: `safe_eval` allows `ast.Starred` enabling denial-of-service via `f(*range(10**8))`

**File**: `src/shared/python/safe_eval.py`
**Line**: 67

`ast.Starred` is in the allowed node types, which permits expressions like
`f(*range(10**8))` if `range` were somehow in the namespace. While `range` is not
currently exposed, allowing `Starred` widens the attack surface unnecessarily for
a math evaluator that should only handle scalar/array expressions.

**Impact**: Low risk currently, but defense-in-depth dictates removing unnecessary
AST node permissions.

**Fix**: Remove `ast.Starred` from `_ALLOWED_NODE_TYPES` unless there is a
documented use case.

---

### Finding 5 — MEDIUM: `atmosphere_at_altitude()` returns physically wrong results for negative altitudes

**File**: `rust_core/tools-core/src/atmosphere.rs`
**Lines**: 70–94

The function only checks `altitude_m.is_finite()` in a `debug_assert!`. It
happily accepts negative altitudes (e.g. −1000 m) and extrapolates the ISA
formulas below sea level, producing unrealistically high pressures and
temperatures that violate the postconditions in release mode (`debug_assert!`
is stripped).

**Impact**: Callers passing negative altitudes (e.g. Dead Sea at −430 m) get
technically incorrect but silently accepted results. In release builds, even
the postcondition checks are absent.

**Fix**: Clamp altitude to `max(0.0, altitude_m)` or explicitly validate with
`assert!` (not `debug_assert!`) that altitude ≥ 0.

---

### Finding 6 — MEDIUM: Rust `debug_assert!` contracts are stripped in release builds

**File**: `rust_core/tools-core/src/engineering.rs`, `math.rs`, `atmosphere.rs`

All Rust DbC preconditions and postconditions use `debug_assert!`, which is
removed in release builds. This means that in production, functions like
`reynolds_number(1.0, 0.0, 1000.0, 0.0)` silently divide by zero, returning
`NaN` or `Inf` instead of panicking.

**Impact**: Silent propagation of NaN/Inf through numerical pipelines.

**Fix**: For critical safety checks (division-by-zero guards), use `assert!`
instead of `debug_assert!`. Keep `debug_assert!` only for range-of-validity
checks where silent degradation is acceptable.

---

### Finding 7 — LOW: `ProcessingResult.timestamp` uses naive `datetime.now()`

**File**: `src/shared/python/upstream_drift_tools/data_processing/core.py`
**Line**: 111

```python
timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
```

Uses timezone-naive `datetime.now()`. If processing results are compared across
machines or stored in a database, timestamps may be inconsistent.

**Fix**: Use `datetime.now(datetime.timezone.utc).isoformat()`.

---

### Finding 8 — LOW: Butterworth filter hardcodes `fs=1000` in `DataProcessor`

**File**: `src/shared/python/data_processing/processor.py`
**Line**: 332

```python
b, a = butter(order, cutoff, btype="low", fs=1000)
```

The sample rate is hardcoded to 1000 Hz. If data is sampled at a different
rate (which is common for many datasets), the cutoff frequency is misinterpreted
and the filter produces incorrect results.

**Impact**: Incorrect signal filtering for any dataset not sampled at exactly 1 kHz.

**Fix**: Accept `sample_rate` as a parameter, or compute it from the time column.

---

### Finding 9 — LOW: `_undo_stack` uses `list.pop(0)` for queue eviction

**File**: `src/shared/python/upstream_drift_tools/data_processing/core.py`
**Line**: 585

```python
if len(self._undo_stack) > 50:
    self._undo_stack.pop(0)
```

`list.pop(0)` is O(n) for Python lists. For large DataFrames this is fine at n=50
entries, but the pattern is technically suboptimal. Using `collections.deque(maxlen=50)`
would handle both the size limit and O(1) eviction automatically.

---

### Finding 10 — LOW: R_GAS precision discrepancy between Rust math.rs and engineering.rs

**File**: `rust_core/tools-core/src/math.rs` vs `engineering.rs`

```rust
// math.rs
pub const R_GAS: f64 = 8.31446;

// engineering.rs
pub const R_UNIVERSAL: f64 = 8.314_462_618_153_24;
```

Two constants for the same physical quantity, at different precisions. `math.rs`
uses a truncated value. `atmosphere.rs` imports from `math.rs` (lower precision).
Engineering calculations use the full CODATA value from `engineering.rs`.

**Impact**: Calculations using different modules get subtly different results for
the same gas constant. Altitude → density calculations are affected.

**Fix**: Use one canonical constant. Either re-export `engineering::R_UNIVERSAL`
as the single source of truth, or update `math::R_GAS` to full precision.

---

### Finding 11 — LOW: `scripting_env.py` user library path is not sanitized

**File**: `src/shared/python/scripting/scripting_env.py`
**Lines**: 87, 125

```python
self._user_lib_path = os.path.expanduser(user_lib_path)
```

The path is expanded but not validated against directory traversal. Combined with
`set_user_library_path()`, a caller could set an arbitrary path. While the module
is GUI-only (no network exposure), hardening the path validation would improve
defense-in-depth.

---

### Finding 12 — LOW: `filter_data` injects `operator` directly into pandas query string

**File**: `src/shared/python/upstream_drift_tools/data_processing/core.py`
**Line**: 529

```python
self.data = self.data.query(f"{column} {operator} @value")
```

The `operator` parameter is interpolated directly into the query string without
validation. While pandas `query()` has its own safety restrictions, validating
that `operator` is one of `==`, `!=`, `>`, `>=`, `<`, `<=` would prevent
unexpected behavior from malformed operator strings.

---

## Positive Findings (No Action Required)

| Area                | Assessment                                                            |
| ------------------- | --------------------------------------------------------------------- |
| `safe_eval.py`      | Strong AST validation; blocks attribute access, import, function defs |
| `subprocess` usage  | No `shell=True` anywhere in `src/`                                    |
| `pickle`            | No `pickle.load()` found                                              |
| `launch_utils.py`   | Path sanitization prevents directory traversal                        |
| Rust `unsafe`       | Zero `unsafe` blocks in entire Rust codebase                          |
| Rust tests          | Comprehensive unit tests for engineering, atmosphere, math modules    |
| DbC framework       | Well-structured with decorator + function-call + mixin patterns       |
| `datetime.utcnow()` | Not used anywhere (clean)                                             |

---

## Summary Table

| #   | Severity  | Component                      | Description                                            |
| --- | --------- | ------------------------------ | ------------------------------------------------------ |
| 1   | 🟡 MEDIUM | `contracts.py`                 | Stale module-level DBC_LEVEL bypasses runtime toggling |
| 2   | 🟡 MEDIUM | `data_processing/core.py`      | Normalize division by zero on constant columns         |
| 3   | 🟡 MEDIUM | `data_processing/core.py`      | Standardize division by zero on zero-variance columns  |
| 4   | 🟡 MEDIUM | `safe_eval.py`                 | Unnecessary `ast.Starred` in allowed node types        |
| 5   | 🟡 MEDIUM | `atmosphere.rs`                | No validation for negative altitudes                   |
| 6   | 🟡 MEDIUM | `engineering.rs` + `math.rs`   | `debug_assert!` contracts stripped in release          |
| 7   | 🟢 LOW    | `data_processing/core.py`      | Naive `datetime.now()` in timestamps                   |
| 8   | 🟢 LOW    | `data_processing/processor.py` | Hardcoded `fs=1000` in Butterworth filter              |
| 9   | 🟢 LOW    | `data_processing/core.py`      | O(n) list pop for undo eviction                        |
| 10  | 🟢 LOW    | `math.rs` vs `engineering.rs`  | R_GAS precision discrepancy                            |
| 11  | 🟢 LOW    | `scripting_env.py`             | User lib path not validated                            |
| 12  | 🟢 LOW    | `data_processing/core.py`      | filter_data operator injection                         |

---

## Remediation Status

| #   | Issue                                                         | Status      | Commit                                                        |
| --- | ------------------------------------------------------------- | ----------- | ------------------------------------------------------------- |
| 1   | [#2217](https://github.com/D-sorganization/Tools/issues/2217) | ✅ FIXED    | `afb7dfae` — reads `_ContractState.level` directly            |
| 2   | [#2218](https://github.com/D-sorganization/Tools/issues/2218) | ✅ FIXED    | `afb7dfae` — raises `TransformationError` on constant columns |
| 3   | [#2218](https://github.com/D-sorganization/Tools/issues/2218) | ✅ FIXED    | `afb7dfae` — raises `TransformationError` on zero-variance    |
| 4   | [#2219](https://github.com/D-sorganization/Tools/issues/2219) | ✅ FIXED    | `afb7dfae` — removed `ast.Starred` from allowed nodes         |
| 5   | [#2220](https://github.com/D-sorganization/Tools/issues/2220) | 🔴 OPEN     | Requires design decision: clamp vs reject                     |
| 6   | [#2221](https://github.com/D-sorganization/Tools/issues/2221) | 🔴 OPEN     | Requires fleet-wide policy on debug_assert vs assert          |
| 7   | —                                                             | 🟡 DEFERRED | Low risk; needs timezone policy decision                      |
| 8   | [#2222](https://github.com/D-sorganization/Tools/issues/2222) | ✅ FIXED    | `afb7dfae` — `sample_rate` parameter added                    |
| 9   | —                                                             | 🟡 DEFERRED | O(n) at n=50 is negligible; improvement optional              |
| 10  | [#2223](https://github.com/D-sorganization/Tools/issues/2223) | ✅ FIXED    | `afb7dfae` — `R_GAS` unified to full CODATA precision         |
| 11  | —                                                             | 🟡 DEFERRED | GUI-only path; no network exposure                            |
| 12  | [#2224](https://github.com/D-sorganization/Tools/issues/2224) | ✅ FIXED    | `afb7dfae` — operator whitelist validation added              |
