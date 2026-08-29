# Rate Visualization Rendered Review Protocol

## Purpose and Claim Boundary

This protocol records the user-oriented rendered review retained by the
visual-first epic. Automated geometry, accessibility, performance, and image
comparison checks are inputs to the review; they are not user approval. A
review is incomplete until a named reviewer inspects the normal production
surfaces, records exact identities, disposes every finding, and signs an
outcome.

## Review Identity

Complete every field before inspection:

| Field | Required Record |
| --- | --- |
| Repository commit SHA |  |
| Application/package version |  |
| Reviewer |  |
| Review date and time zone |  |
| Operating system and build |  |
| React browser and version |  |
| PyQt and Qt versions |  |
| Font package and version |  |
| React viewports | 1440x900, 1280x720, and 390x844 |
| PyQt viewport and scales | 1440x900 at 100% and 150% |
| Representative dataset identities |  |
| Baseline manifest and image identities |  |
| Automated evidence artifact URL |  |
| Review evidence directory or URL |  |
| Deviations from this protocol |  |

Use normal production routes and controls. Do not inject component props,
fabricate successful results, or relabel a preview as computed evidence.
Unavailable production states are findings unless the acceptance authority
marks them not applicable.

## Per-Tab and Per-State Inspection

For every tab in `visualization_acceptance.v1.json`, inspect each applicable
empty, loading, result, and error state at every registered reference case.
Record the following against the exact tab, state, and image identity:

1. The primary visual or honest semantic landmark intersects the initial
   viewport and satisfies its minimum geometry.
2. Narrow and 150% DPI layouts have no clipped controls, horizontal page
   overflow, unexplained overlap, or unreachable content.
3. Frame, units, sample or cohort count, provenance, missing-data treatment,
   and limitations are visible or reachable without obscuring the visual.
4. Loading, unavailable, no-impact, failure, stale, and no-result states retain
   a recognizable landmark and do not imply a successful calculation.
5. Keyboard order reaches the landmark, primary action, state/status, detailed
   disclosure, nonvisual alternative, and export where applicable.
6. Result changes preserve stable visual focus and clearly distinguish prior,
   stale, partial, excluded, and current evidence.
7. Decimation or aggregation is disclosed, deterministic, and does not change
   the underlying scientific authority.
8. Text remains readable; labels, legends, and annotations do not collide or
   overflow at the qualified sizes.

## Findings Record

Use one row per tab/state/reference case. Add rows rather than combining
different states under one judgment.

| Surface | Tab | State | Reference Case | Dataset/Scenario | Image SHA-256 | Geometry | Scientific Context | Keyboard/Nonvisual | Performance | Finding IDs | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |

Every finding requires severity, reproduction steps, expected behavior, owner,
and disposition. A diagnostic image is not an approved baseline until the
protected baseline authority and this signed review identify it.

## Completion Checks

- [ ] Every registered tab/state/reference case has one findings row.
- [ ] Exact build, dataset, manifest, image, and automated artifact identities
      are recorded.
- [ ] All critical and high findings are fixed and re-inspected.
- [ ] Medium and low exceptions have explicit acceptance, owners, and dates.
- [ ] The assistive-technology protocol is linked but remains a separate human
      qualification.
- [ ] The evidence ledger is regenerated after protected merge and retains any
      unmet human or downstream boundary as partial.

## Outcome

Select exactly one outcome:

- [ ] Approve — all required evidence and dispositions are acceptable.
- [ ] Conditional approval — controlled exceptions are attached.
- [ ] Reject — one or more required states or cases are unacceptable.

Reviewer: ____________________  Date: __________

Approver: ____________________  Date: __________
