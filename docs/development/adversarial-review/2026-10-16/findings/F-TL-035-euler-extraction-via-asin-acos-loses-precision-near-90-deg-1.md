# F-TL-035

**Euler extraction via asin/acos loses precision near +/-90 deg (1e-12 round-trip; 1e-8 gimbal threshold)**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | numerics | high       | S      | proposed |

**Locations:** `src/rotation_converter/core.py:464`, `src/rotation_converter/core.py:482`, `src/rotation_converter/core.py:466`, `src/rotation_converter/core.py:484`

## Finding

- Severity: Low - Category: numerics
- Confidence: high
- Location: src/rotation_converter/core.py:463-464, :481-482
- Evidence: `b = math.acos(cb)` / `b = math.asin(sb)` with `sb = np.clip(sign * R[i, k], -1.0, 1.0)`.
- Failure scenario (run): 2,000 random round-trips per convention: max |R − R(euler(R))| 7.9e-13 (xyz), 2.9e-12 (zxy); SciPy-intrinsic agreement 8.9e-16. Near gimbal lock (|b − π/2| ≈ 1e-7) error 2.8e-8 — the 1e-8 threshold selects the degenerate branch slightly early. All 12 conventions are otherwise correct (intrinsic, Hamilton, wxyz, active).
- Proposed change: `b = math.atan2(sb, math.hypot(R[i, i], R[i, j]))` (Tait-Bryan) and `b = math.atan2(math.hypot(R[j, i], R[other, i]), R[i, i])` (proper Euler); threshold `1e-12`.
- Verification: round-trip max error < 1e-14 for 12 conventions.
- Effort: S

## Draft Pull Request

**Title:** `fix(rotation): use atan2 for Euler extraction near the singularity`

### Why This Matters (The Pitch)

The rotation kernel is correct for all 12 conventions, but extraction uses asin and acos, so round-trip error is 1e-12 to 3e-12 instead of 1e-15, and 2.8e-8 near gimbal lock. Switching to atan2 with hypot and tightening the degenerate threshold to 1e-12 brings it back to machine precision.

### Proposed PR Body

## Problem

`src/rotation_converter/core.py:463-464, :481-482` use `b = math.acos(cb)` and `b = math.asin(sb)`; 2,000 random round-trips give max error 7.9e-13 (xyz) and 2.9e-12 (zxy), and 2.8e-8 near gimbal lock.

## Change

- [ ] Tait-Bryan: `b = math.atan2(sb, math.hypot(R[i, i], R[i, j]))`
- [ ] Proper Euler: `b = math.atan2(math.hypot(R[j, i], R[other, i]), R[i, i])`
- [ ] Set the degenerate threshold to `1e-12`

## Verification

Round-trip max error is below 1e-14 for all 12 conventions.

## Risk and Rollback

Outputs change at the 1e-12 level only. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-035

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
