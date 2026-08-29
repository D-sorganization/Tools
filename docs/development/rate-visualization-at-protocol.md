# Rate Visualization Assistive-Technology Qualification Protocol

## Purpose and Claim Boundary

This protocol is the human qualification step for the ten primary Rate of
Closure visualization tabs in both the React and PyQt applications. It is not
an automated test result. A release is **not** manually assistive-technology
qualified until a named evaluator completes this protocol, records the exact
build identity and environments, attaches the evidence, and signs the outcome.

The protected automated evidence is narrower:

- React: axe-core 4.13.0 checks WCAG A/AA rules through WCAG 2.2 on every
  primary tab in production Chromium.
- PyQt: every visible, enabled, focusable semantic control on every primary tab
  must expose a bounded accessible name.
- Both: the versioned accessibility manifest must exactly match the existing
  visualization-tab authority.

Automated success does not prove screen-reader announcement quality, voice
control usability, cognitive accessibility, or successful human task
completion.

## Qualification Record

Complete every field before execution:

| Field | Required record |
| --- | --- |
| Repository commit SHA |  |
| Application/package version |  |
| Evaluator |  |
| Execution date and time zone |  |
| Operating system and build |  |
| React browser and version |  |
| Screen reader and version |  |
| PyQt/Qt version |  |
| Display scale and resolution |  |
| Input methods used | Keyboard, pointer, and any qualified voice/switch input |
| Evidence directory or artifact URL |  |
| Deviations from this protocol |  |

Do not use a development state seam or injected UI props. Run the normal built
applications and normal user-visible controls. Record any dependency failure
or inaccessible state as a failure, not as skipped evidence.

## Common Acceptance Criteria

For each surface and tab:

1. Reach the tab using only the keyboard and confirm focus is visible.
2. Confirm the tab name, primary landmark, and first actionable control are
   announced with a meaningful role and name.
3. Traverse all visible controls in a logical order without a keyboard trap.
4. Operate at least one primary task using the keyboard alone. Visual-first
   tabs must include their documented keyboard interaction or an equivalent
   nonvisual data/status path.
5. Confirm changing a presentation-only control does not announce a new
   scientific result.
6. Trigger one production-reachable validation error and confirm the error is
   announced once, remains understandable in context, and does not erase the
   prior accepted result without disclosure.
7. Run a valid action where applicable and confirm loading/result/status
   announcements are concise, ordered, and do not steal focus.
8. At 200% zoom or the closest supported desktop scaling, confirm the task is
   still operable without two-dimensional page scrolling except within a data
   surface that explicitly requires it.
9. Capture the focus sequence, spoken announcement notes, screenshots, and any
   defect identifiers.

Pass requires all criteria or an approved, documented exception linked to a
tracked remediation. “Not tested,” silence, or ambiguous output is not a pass.

## React Tab Task Matrix

| Tab | Required primary task | Required announcement evidence |
| --- | --- | --- |
| Explorer | Adjust camera, reset view, and distinguish procedural/generated/imported source | Interactive clubhead name, camera status, source/error status |
| Calculation | Change one scientific input and inspect the derived explanation | Input name/unit, changed result context, semantic calculation content |
| Simulation | Run, scrub or step playback, and inspect an impact state | Run state, playback position, selected state, bounded error |
| Plots | Select/add a plot, navigate the plot inspector, and return to plot controls | Managed plot name, selected point status, processing/error state |
| Flight | Run and select a raw trajectory sample from the primary profile | Accepted-flight context, exact sample status, prior-result warning |
| Launch Monitor Analytics | Import a valid dataset and run an available analysis | Import result, analysis status, validation error |
| Neural Model Lab | Load an eligible model/input and inspect predictions and residuals | Eligibility, model-card limits, prediction status, and typed unavailable state |
| Variation | Run a study, reach Cancel/Return/Retry as applicable, inspect result | One live status owner, retained/empty error distinction, result landmark |
| Putting | Change pace input, recompute, and inspect a retained solver sample | Accepted context, exact sample status, retained-error distinction |
| Glossary | Search, select a term, and read its definition | Search name, term list, selected definition heading/content |

## PyQt Tab Task Matrix

| Tab | Required primary task | Required announcement evidence |
| --- | --- | --- |
| Clubhead | Focus canvas, use camera keys, reset view, and inspect source status | Canvas name/description, camera/source status, inline error |
| Plots | Select/add a plot and navigate the point inspector | Managed plot list, canvas name, selected point/status |
| Calculation Description | Navigate the semantic explanation and referenced controls | Tab and section headings or equivalent readable structure |
| Simulation | Run, scrub impact and playback position, inspect display controls | Both slider names/values, run state, displayed evidence |
| Flight Explorer | Run and select a raw trajectory sample | Inspector name, exact sample status, accepted/prior context |
| Launch Monitor Analytics | Import and run an available analysis | Import/analysis controls, status and validation error |
| Neural Model Lab | Load an eligible model/input and inspect predictions and residuals | Eligibility, model-card limits, prediction status, and typed unavailable state |
| Variation | Configure a variable and run/cancel/retry a study | Variable/distribution/scale names, live status, result/error state |
| Putting | Change pace input and inspect a sample | Pace control name/value, sample status, retained error |
| Glossary | Search, select a term, and read its definition | Search name, glossary list, selected definition |

## Evidence Package

The qualification package must contain:

- the completed qualification record and both task matrices;
- one timestamped transcript or structured announcement log per surface;
- screenshots showing representative focus and error/result states;
- the exact automated axe JSON attachment and PyQt semantic-audit result from
  the same commit;
- a defect list with severity, affected tab/surface, reproduction steps, and
  disposition;
- evaluator and approver signatures or an equivalent controlled approval
  record.

Diagnostic screenshots and automated attachments remain unapproved evidence
until they are included in this signed package. A later code, dependency,
browser, Qt, or assistive-technology change requires an impact review and may
require requalification.

## Outcome

Select exactly one outcome and sign it:

- [ ] Pass — all required tasks and announcements are acceptable.
- [ ] Conditional pass — approved exceptions are attached with owners/dates.
- [ ] Fail — one or more required tasks or announcements are unacceptable.

Evaluator: ____________________  Date: __________

Approver: _____________________  Date: __________
