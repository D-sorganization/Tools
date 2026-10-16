# F-TL-037

**Rust/Python parity tests are trivial (Vector3 magnitude) or 1-2 % tolerant and skipped without the wheel**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | tests    | medium     | M      | proposed |

**Locations:** `tests/rust_bindings/test_rust_bindings.py:142`, `src/shared/python/swing_sim/flight/tests/test_parity_rust_flight.py:53`, `src/shared/python/swing_sim/flight/tests/test_parity_rust_flight.py:108`

## Finding

- Severity: Low - Category: tests
- Confidence: medium
- Location: tests/rust_bindings/test_rust_bindings.py:137-149; src/shared/python/swing_sim/flight/tests/test_parity_rust_flight.py:34-56, :106-110
- Evidence: `assert abs(v.magnitude() - python_mag) < 1e-12` (trivial) vs `assert rust.carry_distance == pytest.approx(python.carry_distance, rel=0.01)`.
- Failure scenario: a 0.9 % drift in the Rust drag law passes forever; locally the suite is skipped (no wheel), so only the CI shard that builds the wheel sees it. 1 % of carry is 2.5 m.
- Proposed change: run both integrators with the same fixed step and compare state vectors at identical times with `rtol=1e-9`; keep 1 % only for adaptive-vs-fixed.
- Effort: M

## Draft Pull Request

**Title:** `test(rust-parity): compare integrators at identical steps, rtol 1e-9`

### Why This Matters (The Pitch)

The parity tests check Vector3 magnitude against math.sqrt, which is trivial, while the flight kernel comparison accepts one percent and skips locally without the wheel. A 0.9 percent drift in the Rust drag law passes forever, and one percent of carry is 2.5 m. Run both integrators with the same fixed step and compare states at identical times with rtol 1e-9.

### Proposed PR Body

## Problem

`tests/rust_bindings/test_rust_bindings.py:137-149` asserts `abs(v.magnitude() - python_mag) < 1e-12`; `src/shared/python/swing_sim/flight/tests/test_parity_rust_flight.py:34-56, :106-110` uses `pytest.approx(python.carry_distance, rel=0.01)` and skips without the wheel.

## Change

- [ ] Run both integrators with the same fixed step
- [ ] Compare state vectors at identical times with `rtol=1e-9`
- [ ] Keep 1 % only for adaptive-versus-fixed comparisons

## Verification

The new parity test passes in the CI shard that builds the wheel.

## Risk and Rollback

May expose real drift in the Rust drag law that needs fixing first. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-037

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
