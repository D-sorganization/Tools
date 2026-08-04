# Professional SCADA Product Epic

**Status:** approved for implementation against synthetic data and simulated
equipment only

**Scope:** `src/p1am_control_system`

**Safety boundary:** this epic does not authorize connection to or modification
of a live plant or independent protection system

## Outcome

Evolve the P1AM SCADA from a capable bench-control system into a reusable,
professional-grade supervisory platform. Preserve the existing control behavior
while adding operational trust, reusable operator workflows, controlled change,
commissioning evidence, resilience, and advisory optimization.

All demonstrations and fixtures must be synthetic. Do not add real plant tags,
addresses, limits, recipes, sequences, credentials, network details, production
data, or native control-system artifacts.

## Non-negotiable engineering rules

- **TDD:** introduce each behavior with a failing test, implement the smallest
  complete behavior, then refactor with the full relevant suite green.
- **Design by Contract:** validate inputs at API, driver, persistence, and
  control boundaries; document preconditions, postconditions, and invariants.
- **Law of Demeter:** isolate UI, domain, persistence, driver, and deployment
  responsibilities behind narrow interfaces.
- **DRY:** one canonical model for identity, signal quality, alarm state,
  configuration revision, and evidence metadata; adapters translate at edges.
- Never persist or restore an energized/running state.
- SCADA status and interlocks must not be described as independent protection.
- Every state-changing operation must be attributable and auditable.
- Bad or stale data must remain visibly bad or stale through the whole stack.
- No protected configuration may activate without validation and an auditable
  revision identity.

## Safe baseline

Before this epic branch was created, the existing Tools checkout was preserved
without changing its working tree:

- a local Git recovery ref captures the tracked working-copy state;
- a complete verified Git bundle contains that recovery ref and the captured
  remote-main baseline;
- an origin-main SCADA source archive was produced;
- both local SQLite copies were backed up and passed `PRAGMA integrity_check`;
- SHA-256 checksums and an isolated recovery procedure were recorded outside
  the repositories.

Runtime databases and the recovery package are intentionally local-only and
must never be committed or uploaded.

## Delivery phases

### Phase A — Trustworthy foundation

- [ ] F01 named-user identity, role-based access, and append-only audit trail
- [ ] F02 end-to-end signal quality and communications health
- [ ] F03 professional alarm lifecycle and performance management
- [ ] F04 versioned configuration, approval, deployment, and rollback
- [ ] F05 backup, restore, deployment identity, and system-health center
- [ ] F12 FAT/HIL scenario runner and acceptance-evidence packages

Exit criterion: the synthetic system can be operated, changed, faulted,
audited, backed up, restored, and regression-tested without ambiguity about
identity, data validity, active configuration, or evidence.

### Phase B — Professional operator experience

- [ ] F06 generic process overview and reusable high-performance faceplates
- [ ] F07 interlock, permissive, first-out, and managed-bypass view
- [ ] F08 historian context, annotations, comparisons, and reporting
- [ ] F10 asset health, calibration, and maintenance workspace
- [ ] F13 shift log, run/campaign context, and handover reporting

Exit criterion: an operator can navigate the synthetic process from overview
to cause, understand abnormal conditions, and hand off unresolved work with
traceable context.

### Phase C — Reusable control product

- [ ] F09 generic sequence/state and procedure demonstration
- [ ] F11 driver/plugin framework and device diagnostics
- [ ] F14 notification and escalation policies
- [ ] F15 high availability, time synchronization, and disaster-recovery mode

Exit criterion: a representative unit and connector can be added through
documented contracts, commissioned with scenarios, and operated through defined
infrastructure faults.

### Phase D — Advanced differentiation

- [ ] F16 advisory optimization, digital-twin, and advanced-control workspace

Exit criterion: model outputs are reproducible, versioned, uncertainty-aware,
reviewable, and unable to write authoritative commands without a separately
approved integration.

## Feature acceptance matrix

| ID | Required evidence |
| --- | --- |
| F01 | Named sessions and server-side roles; every attempted mutation records actor, action, target, before/after, reason, time, result, and revision without secrets. |
| F02 | Value, source/server timestamps, quality, diagnostic reason, and sequence propagate through driver, API, historian, alarm logic, and HMI; disconnect and stale scenarios pass. |
| F03 | Priority, lifecycle state, acknowledgment, authorized shelving with expiry, designed suppression, first-out, deadband/delay, help, and alarm-performance reports are deterministic and audited. |
| F04 | Draft, validate, diff, review, approve, activate, identify, and roll back a protected synthetic configuration; no bypass path can silently activate it. |
| F05 | A clean isolated instance restores a verified package, starts de-energized, and reports build, configuration, database, clock, storage, service, driver, and backup health. |
| F06 | One synthetic multi-area process uses reusable accessible faceplates and progressive overview-to-detail navigation with consistent quality, mode, alarm, interlock, and trend drill-down. |
| F07 | Synthetic trips capture first-out and consequences; bypasses require role, reason, expiry, banner, audit, and a non-bypassable policy; protection categories remain distinct. |
| F08 | Saved investigations reproduce query, tag metadata, transformations, charts, context, and export checksums; bad data is not silently interpolated. |
| F09 | Simulator-only start, run, hold, stop, abort, and recovery transitions are deterministic, bounded, auditable, and scenario-tested with invented parameters. |
| F10 | Calibration, drift, flatline, command/feedback mismatch, runtime/start counters, noisy signals, and device statistics generate maintainable advisory records distinct from trips. |
| F11 | A connector can fail without crashing SCADA; its tags degrade quality, commands fail closed, diagnostics identify the connector, and secrets remain redacted. |
| F12 | Declarative scenarios emit a self-contained evidence package with software/config hashes, expected states, alarms, audit events, timing windows, limitations, and sign-off fields. |
| F13 | Shift/run entries are attributable, linked to exact events/trends, searchable, and append-only after sign-off with explicit handover acknowledgment. |
| F14 | Alarm-driven notifications prove delay, suppression, escalation, acknowledgment cancellation, rate limiting, redaction, and delivery audit. |
| F15 | Fault injection proves one command authority, ordered timestamps, buffered-data reconciliation, defined recovery objectives, and safe behavior without the HMI. |
| F16 | Advisory results carry model/data provenance, constraints, confidence, replay evidence, and operator disposition; there is no authoritative write path. |

## Cross-cutting release gates

- Backend and frontend unit/integration suites pass.
- New domain behavior has explicit RED, GREEN, and REFACTOR evidence in the PR.
- Ruff, formatter, mypy, TypeScript, ESLint, and production frontend build pass
  for changed surfaces.
- Synthetic-data classification and confidentiality checks pass.
- Database migrations have forward and rollback/recovery tests.
- Safety-state and emergency-stop regression tests pass.
- Restore and fault scenarios produce evidence packages.
- Documentation, operator help, API schema, and specification match behavior.
- Each child issue is closed only by a merged PR or an approved exempt label.

## Completion rule

The epic is complete only when every feature row has direct evidence, every
phase exit criterion passes, the complete regression and recovery gates pass,
the deployed branch state is identifiable, and no open child issue or required
follow-up remains.
