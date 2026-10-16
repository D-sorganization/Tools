# F-TL-008

**zero_phase default squares magnitude response (-6 dB at design cutoff) and metadata records design cutoff**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/signal_toolkit/filters.py:339`, `src/shared/python/signal_toolkit/filters.py:353`, `src/shared/python/signal_toolkit/filters.py:381`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/signal_toolkit/filters.py:336-356, :377-382
- Evidence: `zero_phase: bool = True` … `filtered_values = filtfilt(filter_spec.b, filter_spec.a, signal.values)`; metadata `"cutoff": filter_spec.cutoff`.
- Failure scenario (run): 4th-order Butterworth low-pass at 50 Hz applied to a 50 Hz sine → gain 0.50 zero-phase vs 0.707 causal. Anyone comparing "cutoff 50 Hz" with a single-pass spec is 3 dB off; effective order also doubles.
- Why a newer engineer misses this: `filtfilt` is sold as "zero phase"; the |H|² cost is in the fine print.
- Proposed change: pre-warp the design to hit −3 dB after two passes when `zero_phase`, or store `"passes": 2, "cutoff_attenuation_db": -6.0` in metadata and say so in the docstring.
- Verification: assert gain at fc ≈ 0.707 after fix, or metadata documents −6 dB.
- Effort: S

## Draft Pull Request

**Title:** `fix(signal-toolkit): make zero-phase cutoff honest, -6 dB not -3 dB`

### Why This Matters (The Pitch)

Ask this toolkit for a 50 Hz low-pass and the default settings give you one that is already 6 dB down at 50 Hz. Because zero_phase defaults to true and filtfilt runs the filter twice, the magnitude response is squared: a 4th-order Butterworth at 50 Hz applied to a 50 Hz sine gives gain 0.50, not the 0.707 a single pass gives. Effective filter order doubles too, and the metadata records the design cutoff as if nothing happened. Anyone comparing the result against a causal filter or a spec sheet is 3 dB off without knowing it. The fix is either to pre-warp the design so the two-pass response hits minus 3 dB, or to record the two passes and minus 6 dB in the metadata and docstring. Both are small. It matters for biomechanics and sensor analysis, where cutoff choices get reported in papers.

### Proposed PR Body

## Problem

`src/shared/python/signal_toolkit/filters.py:336-356, :377-382` defaults `zero_phase: bool = True`, runs `filtfilt(filter_spec.b, filter_spec.a, signal.values)` and records `"cutoff": filter_spec.cutoff`. A 4th-order Butterworth low-pass at 50 Hz on a 50 Hz sine gives gain 0.50 zero-phase versus 0.707 causal.

## Change

- [ ] Pre-warp the design to hit -3 dB after two passes when `zero_phase` is set
- [ ] Or add `"passes": 2, "cutoff_attenuation_db": -6.0` to metadata and document it in the docstring

## Verification

Assert gain at fc is approximately 0.707 after the fix, or that metadata documents -6 dB.

## Risk and Rollback

Pre-warping changes zero-phase outputs for existing callers; the metadata-only option changes nothing numerically. Rollback is a revert.

## Handoff

Follow-ups: F-TL-007 is in the same file.

Refs: adversarial review 2026-10-16, F-TL-008

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
