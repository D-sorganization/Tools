# Professional SCADA Product Epic

**Status:** implemented on the consolidated development branch; local release
gates passed; remote protected-branch gates pending

**Scope:** `src/p1am_control_system`

**Delivery:** all phases are consolidated on GitHub PR #4091. Earlier stacked
phase PRs are superseded and must not be merged independently.

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

- [x] F01 named-user identity, role-based access, and append-only audit trail
- [x] F02 end-to-end signal quality and communications health
- [x] F03 professional alarm lifecycle and performance management
- [x] F04 versioned configuration, approval, deployment, and rollback
- [x] F05 backup, restore, deployment identity, and system-health center
- [x] F12 FAT/HIL scenario runner and acceptance-evidence packages

Exit criterion: the synthetic system can be operated, changed, faulted,
audited, backed up, restored, and regression-tested without ambiguity about
identity, data validity, active configuration, or evidence.

#### Phase A verification — 2026-08-03

| Feature | Direct evidence |
| --- | --- |
| F01 | Named principals, short-lived digest-only sessions, server-side role gates, append-only SQLite audit guards, automatic success/failure mutation capture, redaction, and paginated audit query tests. |
| F02 | Canonical qualified signal samples propagate value, source/server timestamps, quality, diagnostic, sequence, and source through poll frames, WebSocket/API schemas, historian migration/query, alarm eligibility, and HMI communications status. |
| F03 | Deterministic lifecycle domain and REST/HMI workflows cover priority, acknowledgment, timed shelving/unshelving, designed suppression, first-out, deadband/delay, help, and performance metrics. The panel is explicitly supervisory and not independent protection. |
| F04 | Immutable SQLite revisions and protected draft, validation, diff, review, approval, activation, supersession, and rollback workflows are role-gated and audited. The former direct route returns `409` without touching an adapter. Failed deployment never publishes runtime configuration. |
| F05 | Recovery archives verify package and entry SHA-256 values, exclude energized state and runtime/database data, and restore only as a draft. Identity and health report software/configuration, database, clock, storage, service, driver, primary transport, simulator, and recovery-verification status independently. |
| F12 | Machine-marked synthetic scenarios can access only the isolated in-memory adapter. Evidence archives contain scenario/software/configuration hashes, expected and observed states, synthetic alarm and audit records, timing windows/results, limitations, overall result, and blank sign-off fields. |

Phase A release-gate evidence: Ruff, formatting, and strict mypy checks for all
new foundation modules pass; the complete backend suite passes with 971 tests
and 6 CI-only dependency checks skipped; the
complete frontend suite, TypeScript build, and production bundle pass; ESLint
has zero errors and two unchanged pre-existing hook warnings. No database,
runtime recovery archive, credential, or private control artifact is committed.
Unverified clock synchronization is intentionally reported as degraded rather
than inferred from an available wall clock.

### Phase B — Professional operator experience

- [x] F06 generic process overview and reusable high-performance faceplates
- [x] F07 interlock, permissive, first-out, and managed-bypass view
- [x] F08 historian context, annotations, comparisons, and reporting
- [x] F10 asset health, calibration, and maintenance workspace
- [x] F13 shift log, run/campaign context, and handover reporting

Exit criterion: an operator can navigate the synthetic process from overview
to cause, understand abnormal conditions, and hand off unresolved work with
traceable context.

#### Phase B verification — 2026-08-03

| Feature | Direct evidence |
| --- | --- |
| F06 | The machine-marked synthetic feed, reaction, and separation areas use a reusable accessible faceplate contract with value, timestamp, quality, mode, alarm, interlock, asset-detail, and trend-drill-down context. |
| F07 | Protection definitions preserve control/interlock/independent-protection categories, deterministic group first-out and consequences, and managed bypasses with engineer role, reason, 24-hour maximum expiry, persistent banner flag, automatic expiry, audit-covered REST mutation, and a non-bypassable policy. |
| F08 | Immutable SQLite-backed saved investigations reproduce time-bounded queries, tag metadata, transformations, charts, annotations, exact events, context, and an explicit preserve-or-exclude bad-data policy. Deterministic ZIP exports carry entry and package SHA-256 values; interpolation is not an accepted policy. |
| F10 | Deterministic reports cover calibration due, drift, flatline, command/feedback mismatch, noise, runtime, starts, and device statistics. Every finding is explicitly a maintenance advisory with `authoritative_trip=false`. |
| F13 | SQLite-backed entries attribute author, shift, run, unresolved actions, exact event times, and investigation checksums; search is deterministic. Sign-off hashes the entry and installs database guards against update/delete, while handover acknowledgment is a separate attributable append. |

Phase B release-gate evidence: Ruff, formatting, and strict mypy checks for all
new domain, persistence, and API modules pass; the complete backend suite passes
with 995 tests and 6 CI-only dependency checks skipped; all 394 frontend tests,
TypeScript, and the production bundle pass; ESLint has zero errors and two
unchanged pre-existing hook warnings. The operator workspace and every new
record are explicitly synthetic and not a representation of confidential plant
logic, identifiers, limits, or operating values.

### Phase C — Reusable control product

- [x] F09 generic sequence/state and procedure demonstration
- [x] F11 driver/plugin framework and device diagnostics
- [x] F14 notification and escalation policies
- [x] F15 high availability, time synchronization, and disaster-recovery mode

Exit criterion: a representative unit and connector can be added through
documented contracts, commissioned with scenarios, and operated through defined
infrastructure faults.

#### Phase C verification — 2026-08-03

| Feature | Direct evidence |
| --- | --- |
| F09 | A simulator-only state machine deterministically covers start, run, hold, resume, stop, completion, abort, recovery, and timeout. Transitional states have explicit deadlines; invalid transitions and viewer commands fail closed; every event carries actor, reason, sequence, before/after, and synthetic/non-live markings. |
| F11 | Versioned connector descriptors declare owned read/write tags. Poll and command boundaries isolate exceptions, degrade only owned tags, reject unknown/failed commands closed, identify the responsible connector, validate finite values/tag ownership, and redact diagnostic secret fields. |
| F14 | Deterministic policy tests prove initial delay, designed suppression, escalation, acknowledgment cancellation, rate limiting, secret redaction, and an audit record for every delivery or policy outcome. The representative channel has no external delivery side effect. |
| F15 | Availability contracts enforce one command-authority lease, strictly ordered sequences/timestamps, bounded offline buffering and one-time reconciliation, clock-skew reliability, explicit RTO/RPO, and rejection of energizing commands while the HMI is unavailable. The UI states that these contracts do not claim deployed redundant hardware. |

Phase C release-gate evidence: Ruff, formatting, and strict mypy checks for all
new procedure, connector, notification, availability, composition, and API
modules pass; the complete backend suite passes with 1,010 tests and 6 CI-only
dependency checks skipped; all 394 frontend tests, TypeScript, and the production
bundle pass; ESLint has zero errors and two unchanged pre-existing hook warnings.

### Phase D — Advanced differentiation

- [x] F16 advisory optimization, digital-twin, and advanced-control workspace

Exit criterion: model outputs are reproducible, versioned, uncertainty-aware,
reviewable, and unable to write authoritative commands without a separately
approved integration.

Phase D TDD evidence: the RED run failed collection because the advisory domain
and router did not exist. GREEN added five passing domain/API contract tests for
deterministic results, model and data provenance, bounded constraints,
confidence intervals, replay checksums, attributable dispositions, invalid
input rejection, and the absence of command/write routes. REFACTOR introduced
canonical hashing, immutable contracts, retained identical evaluations, strict
dependency checks, and shared schema validation while preserving the no-write
boundary.

Phase D release-gate evidence: Ruff, formatting, and strict mypy checks pass for
the advisory domain and API; the complete backend suite passes with 1,015 tests
and 6 CI-only dependency checks skipped; all 394 frontend tests, TypeScript, and
the production bundle pass; ESLint has zero errors and two unchanged
pre-existing hook warnings. The UI and in-app help label the model and data as
synthetic, disclose that the representative linear projection is not validated
against a plant, and state that no authoritative write path exists.

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

### Consolidated single-PR evidence — 2026-08-04

- Phase A through Phase D are present together on one development branch and
  one PR, with the original pre-epic recovery ref and verified external backup
  package retained.
- The complete backend suite passes with 1,016 tests and 6 CI-only dependency
  checks skipped locally.
- All 394 frontend tests pass; ESLint reports zero errors and two unchanged
  hook warnings; TypeScript and the production Vite build pass.
- All 41 changed production Python modules pass strict mypy. The complete
  P1AM Python surface passes Ruff lint and Ruff formatting.
- The repository detect-secrets baseline contract passes all 23 tests. The two
  keyword detections are explicit synthetic redaction fixtures with line-level
  allowlist annotations; no runtime database, credential, real tag/address,
  plant limit, recipe, sequence, or native controls artifact is included.
- Focused identity, configuration, qualified-signal, alarm, connector,
  operator, reusable-product, and advisory route regressions pass after the
  final consolidation refactor.
- Black is not used to rewrite the changed files because the repository's
  authoritative Ruff formatter targets Python 3.14 and the local Black safety
  check runs under Python 3.13; Ruff formatting is the enforced project gate.

## Completion rule

The epic is complete only when every feature row has direct evidence, every
phase exit criterion passes, the complete regression and recovery gates pass,
the deployed branch state is identifiable, and no open child issue or required
follow-up remains.
