# F-TL-016

**PyO3 bindings unwrap as_slice(): strided numpy input raises PanicException**

| Severity  | Category    | Confidence | Effort | Status   |
| --------- | ----------- | ---------- | ------ | -------- |
| 🟨 Medium | correctness | high       | S      | proposed |

**Locations:** `rust_core/tools-core/src/scada.rs:694`, `rust_core/tools-core/src/scada.rs:711`, `rust_core/tools-core/src/scada.rs:734`, `rust_core/tools-core/src/signal.rs:490`, `rust_core/tools-core/src/signal.rs:505`

## Finding

- Severity: Medium - Category: correctness
- Confidence: high
- Location: rust_core/tools-core/src/scada.rs:694, :711, :734; rust_core/tools-core/src/signal.rs:490, :505, :519, :532, :546 (21 sites total)
- Evidence: `let v = values.as_slice().unwrap().to_vec();`
- Failure scenario: Python passes `x[::2]` or a Fortran-ordered column → `as_slice()` returns `Err(NotContiguous)` → panic → `pyo3_runtime.PanicException` (a `BaseException`); the Python fallbacks in `signal_toolkit` catch `Exception`, so the GUI dies instead of falling back.
- Why a newer engineer misses this: numpy arrays usually are contiguous; the failure needs a slice someone else made.
- Proposed change: `let v = values.as_array().to_vec();` or `values.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))?`.
- Verification: `tools_core.moving_average(np.arange(10.0)[::2], 2)` returns a result or raises `ValueError`.
- Effort: S

## Draft Pull Request

**Title:** `fix(rust-core): raise ValueError on non-contiguous arrays, not panic`

### Why This Matters (The Pitch)

Hand the Rust bindings a strided numpy slice and the GUI dies instead of falling back to Python. Twenty-one PyO3 entry points call as_slice().unwrap() on their numpy inputs, so x[::2] or a Fortran-ordered column triggers a PanicException, which derives from BaseException. The Python fallbacks in signal_toolkit catch Exception, so they never see it. The fix is one line per site: take the array as an ndarray view, or map the contiguity error to a ValueError with a clear message. Anyone who passes a slice they did not create, which is most real callers, benefits. It trades a crash for either a result or a catchable error, and the verification is a single call on a strided arange that either returns or raises ValueError.

### Proposed PR Body

## Problem

`rust_core/tools-core/src/scada.rs:694, :711, :734` and `rust_core/tools-core/src/signal.rs:490, :505, :519, :532, :546` (21 sites in total) use `values.as_slice().unwrap().to_vec()`. A non-contiguous array makes `as_slice()` return `Err(NotContiguous)`, panicking with `pyo3_runtime.PanicException`, a `BaseException` that the Python fallbacks do not catch.

## Change

- [ ] Replace with `values.as_array().to_vec()` or `values.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))?` at all 21 sites

## Verification

`tools_core.moving_average(np.arange(10.0)[::2], 2)` returns a result or raises `ValueError`.

## Risk and Rollback

Rust rebuild is required; behaviour for contiguous arrays is unchanged. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-016

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
