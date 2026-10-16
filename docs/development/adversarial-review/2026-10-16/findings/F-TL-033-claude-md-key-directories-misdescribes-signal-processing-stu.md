# F-TL-033

**CLAUDE.md Key Directories misdescribes signal_processing_studio as the DSP library**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | docs     | high       | S      | proposed |

**Locations:** `CLAUDE.md:49`, `CLAUDE.md:138`

## Finding

- Severity: Low - Category: docs
- Confidence: high
- Location: CLAUDE.md:49, :138
- Evidence: "`src/signal_processing_studio/` — DSP utilities, filters, spectral analysis"; actual filters: src/shared/python/signal_toolkit/filters.py (9,067-LOC package); `signal_processing_studio` is 810 LOC including 439 test lines.
- Failure scenario: agents directed to "the DSP module" edit the wrong package.
- Proposed change: fix both lines; list `signal_toolkit`, `swing_sim`, `rotation_converter`, `rate_of_closure`, `p1am_control_system` (70 % of the code, absent from Key Directories).
- Effort: S

## Draft Pull Request

**Title:** `docs(claude-md): point Key Directories at the real DSP packages`

### Why This Matters (The Pitch)

CLAUDE.md names signal_processing_studio as the DSP library, but it is an 810-line GUI shell and the filters live in signal_toolkit. Agents sent to the DSP module edit the wrong package. Fix both lines and list the packages that hold most of the code.

### Proposed PR Body

## Problem

`CLAUDE.md:49, :138` describe `src/signal_processing_studio/` as DSP utilities, filters and spectral analysis; the filters are in `src/shared/python/signal_toolkit/filters.py` (a 9,067-LOC package) and `signal_processing_studio` is 810 LOC including 439 test lines.

## Change

- [ ] Fix both lines to point at `signal_toolkit`
- [ ] List `swing_sim`, `rotation_converter`, `rate_of_closure` and `p1am_control_system` in Key Directories

## Verification

Read-through of CLAUDE.md Key Directories against the tree.

## Risk and Rollback

Documentation only. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-033

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
