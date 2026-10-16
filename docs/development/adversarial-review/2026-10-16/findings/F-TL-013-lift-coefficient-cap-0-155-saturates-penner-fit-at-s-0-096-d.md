# F-TL-013

**Lift coefficient cap 0.155 saturates Penner fit at S>0.096; driver carry ~7 % short**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | medium     | S      | proposed |

**Locations:** `src/shared/python/swing_sim/flight/_constants.py:35`, `src/shared/python/swing_sim/flight/aerodynamics.py:11`, `src/shared/python/swing_sim/flight/models.py:366`

## Finding

- Severity: Medium - Category: numerics
- Confidence: medium
- Location: src/shared/python/swing_sim/flight/\_constants.py:35-37; src/shared/python/swing_sim/flight/aerodynamics.py:8-12; src/shared/python/swing_sim/flight/models.py:359-368
- Evidence: `MAX_GOLF_BALL_LIFT_COEFFICIENT = 0.155`; Penner params `cl1=0.70, cl2=0.645` → C_L = 0.70·S^0.645 reaches 0.155 at S = 0.096.
- Failure scenario (run): driver 72 m/s, 11°, 2500 rpm (S ≈ 0.08) → carry 235 m, apex 25 m; TrackMan-class references for 160 mph/11°/2500 are ≈ 251 m carry, ≈ 30 m apex. PW 37 m/s, 32°, 10 000 rpm → 100 m vs ≈ 115 m. Bearman & Harvey (1976) measure C_L ≈ 0.22 at S = 0.1 and ≈ 0.30 at S = 0.25; the cap removes the regime wedges live in. Both the Waterloo `cl_max` and the global cap apply (double-capped).
- Why a newer engineer misses this: a constant with a citation feels settled; 7 % short still looks like golf.
- Proposed change: raise the cap to ≈ 0.30 (or make it a model parameter) and add a calibration test against 3–4 published launch-monitor averages with ±4 % tolerance.
- Verification: driver case carry in [243, 259] m.
- Effort: S

## Draft Pull Request

**Title:** `fix(swing-sim): lift 0.155 Cl cap so the Penner fit applies`

### Why This Matters (The Pitch)

The golf-ball flight model underpredicts carry because a hard cap on lift coefficient turns the spin-ratio fit into a constant for every realistic shot. The Penner fit reaches the 0.155 cap at a spin ratio of only 0.096, so a driver at 72 m/s, 11 degrees and 2500 rpm lands at 235 m carry and 25 m apex, where TrackMan-class references give about 251 m and 30 m. A pitching wedge comes out near 100 m against about 115 m. Published measurements put Cl near 0.22 at spin ratio 0.1 and 0.30 at 0.25, so the cap removes the regime where wedges and short irons live. The Waterloo cl_max and the global cap both apply, so the shot is capped twice. Raising the cap to about 0.30, or making it a model parameter, plus a calibration test against three or four published averages, makes the simulator credible for the coaching and research uses it serves.

### Proposed PR Body

## Problem

`src/shared/python/swing_sim/flight/_constants.py:35-37` sets `MAX_GOLF_BALL_LIFT_COEFFICIENT = 0.155`; with Penner `cl1=0.70, cl2=0.645` (`aerodynamics.py:8-12`, `models.py:359-368`) C_L = 0.70\*S^0.645 hits the cap at S = 0.096. Driver 72 m/s, 11 deg, 2500 rpm gives 235 m carry, 25 m apex versus about 251 m and 30 m reference.

## Change

- [ ] Raise the cap to about 0.30 or make it a model parameter
- [ ] Remove the double cap (global cap plus Waterloo `cl_max`)
- [ ] Add a calibration test against 3-4 published launch-monitor averages with +/-4 % tolerance

## Verification

Driver case carry falls in [243, 259] m.

## Risk and Rollback

All simulated carries will rise, so stored golden values and any downstream calibration need regenerating. Rollback is a revert.

## Handoff

Follow-ups: F-TL-037 on tightening Rust/Python parity tolerances.

Refs: adversarial review 2026-10-16, F-TL-013

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
