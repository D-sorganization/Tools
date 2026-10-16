# F-TL-007

**IIR filters in (b,a) form unstable at order>=8 band-pass; non-finite clamp hides failures**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | M      | proposed |

**Locations:** `src/shared/python/signal_toolkit/filters.py:180`, `src/shared/python/signal_toolkit/filters.py:216`, `src/shared/python/signal_toolkit/filters.py:252`, `src/shared/python/signal_toolkit/filters.py:292`, `src/shared/python/signal_toolkit/filters.py:324`, `src/shared/python/signal_toolkit/filters.py:359`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/shared/python/signal_toolkit/filters.py:180, :216, :252, :292, :324, :359-370
- Evidence:

```python
b, a = butter(order, wn, btype=btype)
...
if not np.all(np.isfinite(filtered_values)):
    logging.getLogger(__name__).warning("Filter output contains non-finite values; clamping to input range.")
    filtered_values = np.nan_to_num(filtered_values, nan=0.0, posinf=np.max(signal.values), neginf=np.min(signal.values))
```

- Failure scenario (run, fs = 1 kHz, 40–60 Hz Butterworth band-pass): order 4 max|pole| 0.98 (OK); order 8 → 1.016 (unstable); order 10 → output max 6e85; order 12 → 4e198 — all _finite_, so the clamp does not fire and the caller gets garbage with no warning. Had it overflowed to inf, the clamp would have replaced it with the input's max and returned a "valid" Signal.
- Why a newer engineer misses this: SciPy recommends `output='sos'` precisely because polynomial coefficients lose precision for order > ~6; most tutorials still show `b, a`.
- Proposed change: design with `output="sos"`, apply with `sosfiltfilt`/`sosfilt`; store `sos` in `FilterSpec` (keep `b, a` as derived properties for the baseline); replace the clamp with `raise ArithmeticError("filter produced non-finite output")`.
- Verification: assert `np.max(np.abs(apply_filter(sig, spec).values)) < 10 * np.max(np.abs(sig.values))` for band-pass orders 4..16.
- Effort: M

## Draft Pull Request

**Title:** `fix(signal): design IIR filters as SOS and fail on non-finite output`

### Why This Matters (The Pitch)

A Butterworth band-pass of order 8 or higher in this toolkit is unstable, and the safety net hides it. Designed in (b, a) form at fs 1 kHz, 40 to 60 Hz, order 4 is fine with a maximum pole radius of 0.98, but order 8 reaches 1.016, order 10 returns values around 6e85 and order 12 around 4e198. Those are finite, so the non-finite clamp never fires and the caller receives garbage with no warning. Where the output does overflow to inf, the clamp replaces it with the input's maximum and returns a plausible Signal. Designing with output sos and applying sosfiltfilt is the standard fix, and replacing the clamp with an explicit error means a failed filter looks like a failure. Everyone filtering motion or sensor data with this toolkit benefits, from golf kinematics to plant instrumentation.

### Proposed PR Body

## Problem

`src/shared/python/signal_toolkit/filters.py:180, :216, :252, :292, :324, :359-370` calls `butter(order, wn, btype=btype)` returning `(b, a)`, applies `filtfilt`, and on non-finite output runs `np.nan_to_num` with the input range. At fs 1 kHz, 40-60 Hz band-pass, order 8 has max pole radius 1.016 and order 12 outputs about 4e198, both finite.

## Change

- [ ] Design with `output="sos"` and apply with `sosfiltfilt`/`sosfilt`
- [ ] Store `sos` in `FilterSpec`, keep `b, a` as derived properties for the baseline
- [ ] Replace the clamp with `raise ArithmeticError("filter produced non-finite output")`
- [ ] Add band-pass tests for orders 4 to 16

## Verification

`assert np.max(np.abs(apply_filter(sig, spec).values)) < 10 * np.max(np.abs(sig.values))` for band-pass orders 4..16.

## Risk and Rollback

Filter outputs at low orders change at rounding level; high orders change from garbage to correct. Callers that relied on the clamp will now see an exception. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-008 is in the same file.

Refs: adversarial review 2026-10-16, F-TL-007

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
