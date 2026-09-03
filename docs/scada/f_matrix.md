# P1AM SCADA requirement matrix (F01-F16, H1-H9)

**Generated:** 2026-09-03T16:16:10Z · **Base:** `origin/main` @ `90b7ff3c4639`

**Audit:** [#4912](https://github.com/D-sorganization/Tools/issues/4912) · 
**Program:** D-sorganization/Repository_Management#1505

> This file is generated. Edit `docs/scada/f_matrix.v1.json` (or the generator)
> and run `python scripts/check_scada_f_matrix.py --check`; do not hand-edit.

## Headline

0 of 16 SCADA requirements and 0 of 9 historian children are landed. The 38/38 ticked boxes on #4085-#4088 correspond to no merged code: every carrier PR was closed unmerged. Corroboration: docs/development/professional-scada-epic.md is on main with all 16 of its own feature checkboxes still unchecked.

This matrix is the tracker of record, superseding the checklists on #4085, #4086, #4087, #4088, #4089 and #4046.

## Counts

| Series | Landed | Partial | Missing | Total |
| --- | --- | --- | --- | --- |
| SCADA F01-F16 | 0 | 10 | 6 | 16 |
| Historian H1-H9 | 0 | 2 | 7 | 9 |

## Status classes

- **`landed`** - the requirement as stated is substantially implemented on main and covered by tests on main
- **`partial`** - some capability exists on main, but the requirement as stated is not met; the specific gap is recorded
- **`missing`** - nothing on main addresses the requirement

## Requirements

| ID | Requirement | Status | Tracker | Evidence PRs |
| --- | --- | --- | --- | --- |
| F01 | Named-user identity, server-side RBAC, and append-only audit trail | `partial` | #4085 | #4448, #4928 |
| F02 | End-to-end signal quality and communications health | `partial` | #4085 | #4678, #4893, #4928 |
| F03 | Professional alarm lifecycle and alarm-performance management | `partial` | #4085 | #4928 |
| F04 | Versioned configuration, approval, deployment, and rollback | `partial` | #4085 | #3912, #4928 |
| F05 | Backup, restore, deployment identity, and system-health center | `missing` | #4085 | - |
| F06 | Generic process overview and reusable high-performance faceplates | `missing` | #4086 | - |
| F07 | Interlock, permissive, first-out, and managed-bypass view | `partial` | #4086 | #4928, #3912 |
| F08 | Historian context, annotations, comparisons, and reporting | `partial` | #4086 | #4678, #4448, #3912 |
| F09 | Generic sequence/state and procedure demonstration | `partial` | #4087 | #3912 |
| F10 | Asset health, calibration, and maintenance workspace | `partial` | #4086 | #4448, #3081, #3806 |
| F11 | Driver/plugin framework and device diagnostics | `partial` | #4087 | #4448, #4893, #4928 |
| F12 | FAT/HIL scenario runner and acceptance-evidence packages | `missing` | #4085 | - |
| F13 | Shift log, run/campaign context, and handover reporting | `missing` | #4086 | - |
| F14 | Notification and escalation policies | `missing` | #4087 | - |
| F15 | High availability, time synchronization, and disaster-recovery mode | `missing` | #4087 | - |
| F16 | Advisory optimization, digital-twin, and advanced-control workspace | `partial` | #4088 | #4448, #4928, #3062 |
| H1 | Extract a HistorianSink protocol (pluggable write backend) | `missing` | #4047 | - |
| H2 | Store-and-forward shipper that cannot stall the poll loop | `partial` | #4048 | #4678, #4893 |
| H3 | TimescaleDB schema: hypertable, compression, CAGGs, retention | `missing` | #4049 | - |
| H4 | Wire dual-write behind a feature flag | `missing` | #4050 | - |
| H5 | Shipper observability: queue depth, lag, drop counters | `partial` | #4051 | #4678, #4893, #4448 |
| H6 | Grafana provisioning-as-code: datasource + dashboards in git | `missing` | #4052 | - |
| H7 | ISA-18.2 / EEMUA 191 alarm performance dashboard | `missing` | #4054 | - |
| H8 | Deployment topology: historian + Grafana off the control Pi | `missing` | #4055 | - |
| H9 | Operator/engineer documentation, runbook, and ADR | `missing` | #4056 | - |

## Detail

### F01 - Named-user identity, server-side RBAC, and append-only audit trail

**Status:** `partial` · **Tracker:** #4085

**On main:** The append-only half is real: a pure-ASGI AuditMiddleware records every POST/PUT/PATCH/DELETE into a dedicated AuditEvent table that no handler writes and clear_capture cannot erase, with redacted payload, status, client IP and credential fingerprint. Every mutating route is gated and the gate is pinned by a route matrix.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/audit.py`
- `src/p1am_control_system/backend/auth_config.py`
- `src/p1am_control_system/backend/main.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_audit_trail.py`
- `src/p1am_control_system/backend/tests/test_route_authz_matrix.py`
- `src/p1am_control_system/backend/tests/test_auth_config.py`
- `src/p1am_control_system/backend/tests/test_auth_resolution.py`
- `src/p1am_control_system/backend/tests/test_read_auth.py`
- `src/p1am_control_system/backend/tests/test_request_guard.py`

**Gaps:**

- Identity is not named-user: authorization is two shared environment keys (P1AM_API_KEY / P1AM_ADMIN_API_KEY), so the actor is a key tier plus a SHA-256 key fingerprint. No user accounts, no roles beyond the two nested tiers, no login.
- Audit rows lack three fields the requirement names: reason, before/after values, and configuration revision.
- Alarm acknowledgement takes the actor as a free-text `user` string in the request body, i.e. self-asserted and unverified.
- POST /api/estop is deliberately unauthenticated (documented at the call site) - an intentional exception to 'every mutation attributable'.

### F02 - End-to-end signal quality and communications health

**Status:** `partial` · **Tracker:** #4085

**On main:** The driver -> control -> alarm -> historian leg is genuine and tested: a held/faulted scan yields None to the control laws and is never written as a tag row (a truthful gap, not fabricated continuity), NaN/Inf classifies as BadQuality at trip severity in both the Python and Rust engines with a parity test, and source transitions emit a DATA_QUALITY event. Delivered by Phase 0 PR #4928.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/data_quality.py`
- `src/p1am_control_system/backend/poll_runtime.py`
- `src/p1am_control_system/backend/models.py`
- `src/p1am_control_system/backend/historian.py`
- `src/p1am_control_system/backend/scada_fallback.py`
- `rust_core/tools-core/src/scada.rs`
- `src/p1am_control_system/frontend/src/lib/dataAge.ts`
- `src/p1am_control_system/frontend/src/components/DataAgeIndicator.tsx`

**Tests:**

- `src/p1am_control_system/backend/tests/test_poll_data_quality.py`
- `src/p1am_control_system/backend/tests/test_alarm_nan_bad_quality.py`
- `src/p1am_control_system/backend/tests/test_historian.py`
- `src/p1am_control_system/backend/tests/test_tag_force_nonfinite.py`
- `src/p1am_control_system/frontend/src/components/DataAgeIndicator.test.tsx`

**Gaps:**

- The chain breaks on the read/HMI side: GET /api/trends returns only {timestamps, values, truncated} with no quality column, and the CSV export header is Timestamp/Tag Name/Value - stored quality is dropped on the way out.
- The React frontend contains zero references to `data_source`. The poll loop publishes it in the frame, but the HMI infers badness only from wall-clock frame age, so a `held` or `fault` scan arriving on time renders as CONNECTED.
- Quality is scan-scoped, not per-tag; there are no source/server timestamps, per-tag diagnostic reason codes, or sequence numbers.

### F03 - Professional alarm lifecycle and alarm-performance management

**Status:** `partial` · **Tracker:** #4085

**On main:** A four-band (Low/High/LoLo/HiHi) plus BadQuality classifier with a coarse 0/1/2 severity, engine-authoritative acknowledgement that survives a routing redeploy, and an acknowledge-all header.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/alarm_processing.py`
- `src/p1am_control_system/backend/scada_fallback.py`
- `rust_core/tools-core/src/scada.rs`
- `src/p1am_control_system/backend/main.py`
- `src/p1am_control_system/frontend/src/components/AlarmsHeader.tsx`

**Tests:**

- `src/p1am_control_system/backend/tests/test_alarm_ack_engine.py`
- `src/p1am_control_system/backend/tests/test_alarm_processing.py`
- `src/p1am_control_system/backend/tests/test_alarm_nan_bad_quality.py`
- `src/p1am_control_system/backend/tests/test_scada_fallback.py`
- `src/p1am_control_system/frontend/src/components/AlarmsHeader.test.tsx`

**Gaps:**

- Absent entirely: shelving with authorization and expiry, designed suppression, configurable alarm priority (severity is a hardcoded _SEVERITY_BY_STATE map), alarm deadband or on/off delay, first-out grouping, and per-alarm operator help.
- No alarm-performance KPI exists. backend/performance.py is loop/scan cadence health, not alarm-rate management.

### F04 - Versioned configuration, approval, deployment, and rollback

**Status:** `partial` · **Tracker:** #4085

**On main:** The validation clause is partly met: protected config is a Pydantic model that rejects non-finite limits, unrouted tags default to fully disabled (Phase 0 PR #4928), deployment is behind the admin key, and the mutation is audited.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/config_store.py`
- `src/p1am_control_system/backend/models.py`
- `src/p1am_control_system/backend/main.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_config_store.py`
- `src/p1am_control_system/backend/tests/test_interlock_defaults_contract.py`
- `src/p1am_control_system/backend/tests/test_validation_guards_3745.py`

**Gaps:**

- The versioning clause is not met at all. PersistedConfig is a single key -> value_json table whose writer is an upsert, so exactly one generation of each config exists: no revision id, no diff, no draft/review/approve step, no activation record, and no rollback path. The previous revision is destroyed on write.
- There is no revision identity to stamp into audit rows or historian data, which is also what blocks F01's `revision` field.

### F05 - Backup, restore, deployment identity, and system-health center

**Status:** `missing` · **Tracker:** #4085

**Gaps:**

- No health endpoint of any kind exists: /api/health, /healthz, /api/version, build_info and system_health all return zero hits across the subsystem.
- No backup, restore, integrity-check or package-verification code exists in the backend or in deploy/.
- main.py exposes no build/commit/config identity, so a running instance cannot report which revision it is.
- Adjacent but not this requirement: backend/performance.py (scan overruns and historian write failures), backend/shutdown_safety.py (bounded de-energized teardown), and tests/test_deployment_hardening.py, which asserts static properties of the installer text and exercises no restore path.

### F06 - Generic process overview and reusable high-performance faceplates

**Status:** `missing` · **Tracker:** #4086

**Gaps:**

- The HMI is eleven hardcoded device/function tabs with bespoke, non-reusable panels; PowerSupplyControl.tsx and TemperatureControl.tsx each re-implement their own state/trip/setpoint chrome instead of instantiating a shared faceplate.
- No faceplate, synoptic, mimic or P&ID component exists in the React HMI. desktop/mimic_tab.py belongs to the legacy PyQt package, not the web HMI, and is not this requirement.
- PlantHierarchy.tsx is an area/unit/equipment/tag tree browser, not a process overview graphic: it carries no quality, mode, alarm or interlock state and offers no overview-to-detail drill-down.

### F07 - Interlock, permissive, first-out, and managed-bypass view

**Status:** `partial` · **Tracker:** #4086

**On main:** Permissive and latched-trip-with-acknowledge are solidly landed and tested at both the firmware and controller layers, and Phase 0 PR #4928 wired the reset path (coil 1 -> ClearTrip()) with latch semantics. The firmware does compute a genuine first-out: trip_tag_id_ is the tag that latched the trip.

**Implementing files (verified present on main):**

- `src/p1am_control_system/firmware/SafetyInterlock.h`
- `src/p1am_control_system/firmware/SafetyInterlock.cpp`
- `src/p1am_control_system/backend/hardware.py`
- `src/p1am_control_system/backend/safety_state_machine.py`
- `src/p1am_control_system/backend/models.py`
- `src/p1am_control_system/frontend/src/components/InterlocksPanel.tsx`

**Tests:**

- `src/p1am_control_system/backend/tests/test_safety_state_machine.py`
- `src/p1am_control_system/backend/tests/test_interlock_defaults_contract.py`
- `src/p1am_control_system/backend/tests/test_backend_p1am_safety.py`
- `src/p1am_control_system/backend/tests/test_estop_clear_endpoint.py`
- `src/p1am_control_system/backend/tests/test_power_supply_runtime_safety.py`

**Gaps:**

- The first-out never reaches the operator. INTERLOCK_TRIP_TAG_REGISTER appears only as a constant and in a register-layout assertion: no Python reads it, it is not in the telemetry frame, and no component displays it.
- InterlocksPanel.tsx is a limit-entry form, not an interlock/consequence view.
- Managed bypass is entirely absent: no bypass record, role gate, reason, expiry, banner, or non-bypassable policy exists.
- rust_core/tools-core/src/scada.rs contains an InterlockMatrix, but no Python in the SCADA backend references it. It is dead relative to this subsystem and is deliberately not counted as evidence.

### F08 - Historian context, annotations, comparisons, and reporting

**Status:** `partial` · **Tracker:** #4086

**On main:** A real analysis workspace: historian-sourced datasets through a deterministic align/resample/filter/derive/trim/downsample pipeline, statistics/correlation/PCA/spectrum, several plot types, CSV/JSON export, and the important negative property that gaps and non-finite samples round-trip as None rather than being interpolated.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/historian.py`
- `src/p1am_control_system/backend/data_capture.py`
- `src/p1am_control_system/backend/data_explorer_router.py`
- `src/p1am_control_system/backend/data_explorer_service.py`
- `src/p1am_control_system/backend/data_explorer_stats.py`
- `src/p1am_control_system/backend/data_explorer_expression.py`
- `src/p1am_control_system/frontend/src/components/data_explorer/DataExplorer.tsx`

**Tests:**

- `src/p1am_control_system/backend/tests/test_data_explorer_service.py`
- `src/p1am_control_system/backend/tests/test_data_explorer_router.py`
- `src/p1am_control_system/backend/tests/test_data_explorer_stats.py`
- `src/p1am_control_system/backend/tests/test_data_explorer_expression.py`
- `src/p1am_control_system/backend/tests/test_trends_endpoint.py`
- `src/p1am_control_system/backend/tests/test_data_capture_queries.py`

**Gaps:**

- Saved investigations are stored client-side in localStorage under p1am.explorer.sessions.v1, so they are per-browser, unshareable and not immutable - the requirement's provenance half is unmet.
- No export checksums, no tag-metadata capture, and no annotation model or endpoint anywhere in the subsystem.
- No run/campaign comparison feature and no reporting output.
- Exported CSV omits the TagLog.quality column, so an investigation cannot show which samples were simulated (same root cause as F02).

### F09 - Generic sequence/state and procedure demonstration

**Status:** `partial` · **Tracker:** #4087

**On main:** A Generic[StateT] base class factoring the IDLE/ARMED/RUNNING/TRIPPED scaffolding, one-way E-stop latch, permissive toggle and trip latch/acknowledge, shared by the temperature and power-supply controllers.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/safety_state_machine.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_safety_state_machine.py`

**Gaps:**

- This is per-controller safety scaffolding, not a sequence engine: no hold, abort or recovery transition, no declarative procedure/step/phase model, and no scenario tests.
- No simulator-only gate confines procedure execution to the simulated driver, which the requirement makes a precondition.

### F10 - Asset health, calibration, and maintenance workspace

**Status:** `partial` · **Tracker:** #4086

**On main:** An offline interactive AI/AO calibration CLI over raw Modbus registers, rolling feedback-noise statistics with an arc-detection flag, and a provenance tracker that logs DataSource transitions.

**Implementing files (verified present on main):**

- `src/p1am_control_system/calibration/calibrate.py`
- `src/p1am_control_system/calibration/CALIBRATION.md`
- `src/p1am_control_system/backend/signal_stats.py`
- `src/p1am_control_system/backend/power_supply_noise.py`
- `src/p1am_control_system/backend/data_quality.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_signal_stats.py`
- `src/p1am_control_system/backend/tests/test_power_supply_noise.py`
- `src/p1am_control_system/backend/tests/test_poll_data_quality.py`

**Gaps:**

- No asset register, no calibration-due or drift tracking, no flatline or command/feedback-mismatch detection, no runtime/start counters, and no device-statistics store.
- No advisory maintenance-record type distinct from authoritative trips - which is the specific property the phase issue claimed.
- No maintenance workspace: none of the eleven HMI tabs is asset health or maintenance.

### F11 - Driver/plugin framework and device diagnostics

**Status:** `partial` · **Tracker:** #4087

**On main:** BasePLCClient is a real abc.ABC seam; the poll loop wraps each scan with exponential backoff and a degraded frame so a driver fault does not kill the backend; modbus_client forces energizing commands to zero/false while the E-stop write-seam latch is set, never blocks the de-energizing direction, and retries a failed de-energize once; audit.redact_payload masks credential-like keys recursively.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/plc_interface.py`
- `src/p1am_control_system/backend/plc_factory.py`
- `src/p1am_control_system/backend/poll_runtime.py`
- `src/p1am_control_system/backend/modbus_client.py`
- `src/p1am_control_system/backend/audit.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_plc_factory.py`
- `src/p1am_control_system/backend/tests/test_estop_write_seams.py`
- `src/p1am_control_system/backend/tests/test_audit_trail.py`
- `src/p1am_control_system/backend/tests/test_poll_data_quality.py`
- `tests/plant_simulator/test_plc_contract_identity.py`

**Gaps:**

- PLCFactory.create_client is a hardcoded if/elif over four driver strings: no plugin registry, entry points, discovery, or per-driver capability declaration.
- Only one connector exists at a time, so 'a connector can fail without crashing SCADA' is untested as a multi-connector isolation property, and diagnostics do not identify a failing connector by name.
- settings.py holds API keys as plain str (no SecretStr / repr suppression), so redaction covers audit bodies only.
- The `neural` driver does not satisfy the ABC and raises TypeError on instantiation; see the needs_owner entry for #3984.

### F12 - FAT/HIL scenario runner and acceptance-evidence packages

**Status:** `missing` · **Tracker:** #4085

**Gaps:**

- Nothing in the subsystem matches scenario, evidence, sign-off, hil or fat_. The only acceptance artifacts in the repo belong to src/rate_of_closure, a different subsystem.
- No declarative scenario format, no software/config hashing for evidence identity, and no sign-off or limitations packaging.

### F13 - Shift log, run/campaign context, and handover reporting

**Status:** `missing` · **Tracker:** #4086

**Gaps:**

- Zero matches for handover anywhere. 'Campaign' appears only as prose describing historian retention pruning.
- backend/audit.py and models.EventLog give an append-only trail, but there is no shift or run entity, no operator narrative entries, no link-to-event/trend, and no sign-off or handover acknowledgement.

### F14 - Notification and escalation policies

**Status:** `missing` · **Tracker:** #4087

**Gaps:**

- The only 'escalation' is POLL_FAILURE_ESCALATION_THRESHOLD in the poll loop (a log level plus a degraded frame), and NotificationBanner.tsx is an in-page UI banner.
- No notification channel, delay/suppression/escalation/rate-limit policy, acknowledgement cancellation, or delivery audit.

### F15 - High availability, time synchronization, and disaster-recovery mode

**Status:** `missing` · **Tracker:** #4087

**Gaps:**

- Zero matches for failover, high availability, NTP, time sync or clock skew.
- No second command authority or arbitration, no timestamp-ordering or time-sync check, no buffered-data reconciliation, no stated RPO/RTO, and no headless 'safe without HMI' mode.
- Adjacent but different: backend/shutdown_safety.py is an ordered, deadline-bounded teardown that de-energizes before joining tasks. `backup_simulator` in main.py is a simulated-PLC fallback for routing and tag endpoints, not a redundancy peer.

### F16 - Advisory optimization, digital-twin, and advanced-control workspace

**Status:** `partial` · **Tracker:** #4088

**On main:** An unconstrained DMC move solver and a PID-vs-MPC comparison exposed only as POST /api/mpc/simulate (admin-gated), so the MPC genuinely has no write path today.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/mpc.py`
- `src/p1am_control_system/backend/tuning_router.py`
- `src/p1am_control_system/backend/pid_tuning.py`
- `src/p1am_control_system/backend/plant_model.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_mpc_dmc.py`
- `src/p1am_control_system/backend/tests/test_pid_tuning_math.py`
- `src/p1am_control_system/backend/tests/test_pid_tuning_tag_guards.py`

**Gaps:**

- That absence is incidental, not an enforced review-only contract: no record states authoritative_write_available=false.
- Advisory results carry no model or data provenance, constraint set, confidence ordering, replay evidence, or operator-disposition record.
- plant_model.py is a Plant/Area/Unit/Equipment/Tag metadata hierarchy, not a digital twin.
- The same router's PID auto-tuning does write setpoints through write_pid_setpoint, so the advisory/authoritative boundary the requirement demands is not drawn as a policy.

### H1 - Extract a HistorianSink protocol (pluggable write backend)

**Status:** `missing` · **Tracker:** #4047

**Gaps:**

- No HistorianSink symbol exists. Every hit for the string is ThrottledHistorianSink, which is a throttling decorator holding a duck-typed writer - not a pluggable write-backend abstraction.
- historian.py imports sqlalchemy.insert, sqlmodel.Session and models.TagLog directly and _write_batch/_commit_once speak SQLAlchemy, so the SQLite backend is hardwired. No Protocol or ABC defines a sink contract.

### H2 - Store-and-forward shipper that cannot stall the poll loop

**Status:** `partial` · **Tracker:** #4048

**On main:** The non-stall half is real and well tested: submit() is strictly non-blocking onto a bounded asyncio.Queue (256), writes are batched (32) onto a worker thread off the event loop, OperationalError is retried, and on overflow the oldest record is evicted while its alarm/event payload is merged forward.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/historian.py`
- `src/p1am_control_system/backend/poll_runtime.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_historian_writer.py`
- `src/p1am_control_system/backend/tests/test_historian.py`

**Gaps:**

- This is an in-process write buffer, not a store-and-forward shipper: no durable on-disk spool surviving restart, no remote destination, and no forward/replay/checkpoint or at-least-once delivery.
- A full queue silently drops samples by design.

### H3 - TimescaleDB schema: hypertable, compression, CAGGs, retention

**Status:** `missing` · **Tracker:** #4049

**Gaps:**

- Zero matches for `timescale` repo-wide: no migration, DDL, create_hypertable, compression policy or continuous aggregate.
- The only retention on main is SQLite row pruning plus incremental VACUUM in backend/data_capture.py, a different mechanism.

### H4 - Wire dual-write behind a feature flag

**Status:** `missing` · **Tracker:** #4050

**Gaps:**

- settings.py has no historian-backend or dual-write flag, and main.py constructs exactly one sink over one HistorianWriter.
- Blocked by H1 and H3: with no sink protocol and no second backend there is nothing to dual-write to.

### H5 - Shipper observability: queue depth, lag, drop counters

**Status:** `partial` · **Tracker:** #4051

**On main:** Drop and write-failure counters exist (_WriterCounters: dropped_samples, write_failures, rows_written) and reach the operator via loop_diagnostics and the performance snapshot model.

**Implementing files (verified present on main):**

- `src/p1am_control_system/backend/historian.py`
- `src/p1am_control_system/backend/poll_runtime.py`
- `src/p1am_control_system/backend/performance.py`
- `src/p1am_control_system/backend/performance_models.py`

**Tests:**

- `src/p1am_control_system/backend/tests/test_historian_writer.py`
- `src/p1am_control_system/backend/tests/test_performance.py`
- `src/p1am_control_system/backend/tests/test_poll_loop_cadence.py`

**Gaps:**

- queue_depth is implemented as a property but referenced nowhere outside historian.py, so it is not actually observable.
- No lag or latency metric (no oldest-unshipped timestamp or backlog age) and no /metrics surface at all.

### H6 - Grafana provisioning-as-code: datasource + dashboards in git

**Status:** `missing` · **Tracker:** #4052

**Gaps:**

- Zero matches for `grafana` repo-wide. deploy/ contains only .gitkeep; src/p1am_control_system/deploy/ has no provisioning directory, datasource YAML or dashboard JSON.

### H7 - ISA-18.2 / EEMUA 191 alarm performance dashboard

**Status:** `missing` · **Tracker:** #4054

**Gaps:**

- Alarm machinery on main is classification and event folding only. No ISA-18.2/EEMUA-191 KPI computation (alarm rate per 10 min, flood/chattering/stale/standing counts, top-10 contributors) and no dashboard consuming them.
- Shares a root cause with F03: severity is a fixed map, so there is no priority distribution to report.

### H8 - Deployment topology: historian + Grafana off the control Pi

**Status:** `missing` · **Tracker:** #4055

**Gaps:**

- docker-compose.yml deploys backend + frontend on one host with the historian as a local SQLite file on the dcs_db_data volume, and deploy/install-services.sh installs systemd units on the same Pi.
- No separate historian/visualization host, network split, or remote-write configuration exists.

### H9 - Operator/engineer documentation, runbook, and ADR

**Status:** `missing` · **Tracker:** #4056

**Gaps:**

- docs/adr/ has 18 files, none matching histor|timescale|grafana|scada.
- USER_MANUAL.md, BENCH_HANDOFF.md and deploy/README.md mention the historian only in passing. No operating procedure, backfill/failure runbook, or architecture decision record.

## Closed carrier PRs

| PR | Claimed | Head OID | Reachable | Product files absent from main |
| --- | --- | --- | --- | --- |
| #4091 | Phase A (F01-F05, F12) | `2259f5915426` | yes | 78 |
| #4093 | Phase B (F06-F08, F10, F13) | `43c018f0b9ff` | yes | 63 |
| #4094 | Phase C (F09, F11, F14, F15) | `49fdd11abbbd` | yes | 74 |
| #4095 | Phase D (F16) | `4efb2e81c1de` | yes | 78 |
| #4449 | Consolidation of #4065 + #4091 | `7fba01f5c561` | yes | 105 |
| #4065 | Historian epic #4046 (H1-H9) | `128bd1c3b492` | yes | 25 |

- **#4091** - Nominally the Phase A carrier, but its head OID carries the *whole* four-phase stack: its set of added-and-absent product files is byte identical to #4095's. #4093 and #4094 are strict subsets of that same set (#4093 lacks 15 files, #4094 lacks 4). The four 'phases' are one nested stack, not four independent deliveries, so the recovery corpus is a single 78-file set - not 4 x ~80.
- **#4093** - Subset of #4091/#4095's file set; adds no unique product file.
- **#4094** - Subset of #4091/#4095's file set; adds no unique product file.
- **#4095** - Stack tip; file set identical to #4091's.
- **#4449** - Union of the #4091 and #4065 corpora plus exactly two unique files (backend/enum_compat.py, backend/tests/_route_inventory.py). Its closing note cites #4445, a 9-file CI-only PR with no SCADA content, so nothing it carried reached main.
- **#4065** - The only carrier of TimescaleDB/Grafana content: 6 SQL migrations, 4 Grafana dashboards, provisioning YAML, a deploy compose file, ADR-007, and historian_sink/shipper/wiring + timescale_writer with tests. Also adds dcs_scada.db, a runtime SQLite artifact that must never be committed.

## Needs an owner ruling

### Full removal of the p1am flat-import packaging (#3984)

The BasePLCClient/RoutingConfig duplicate-class hazard is fixed by aligning plant_simulator on the flat import path. Removing the flat path itself - package __init__, package-absolute imports across ~50 backend modules, and a Dockerfile whose build context is the package root rather than backend/ - changes the container layout and is not safe to fold into an audit PR.

### The `neural` PLC driver is uninstantiable (#3984)

NeuralSimulatorClient never implemented clear_estop (added to the ABC by #3415 for the E-stop reset path), and its read_tags/write_tag signatures drifted from the ABC (list[float] vs dict[str, float] | None; tag_id: int vs tag_name: str). PLCFactory therefore raises TypeError for plc_driver='neural'. Pinned by an xfail(strict=True). Owner must choose: complete the driver against the current safety contract, or withdraw the factory branch.

### TimescaleDB and Grafana licensing (#4046)

Epic #4046 flags Grafana as AGPLv3 (a network-copyleft question for customer-facing deliverables) and TimescaleDB as split-licensed (Apache-2 core, TSL for the compression and continuous aggregates the epic depends on). Re-landing #4065's H3/H6 content commits the repo to both. This must be ruled on before, not after, a re-land.

### Re-scope or withdraw the six zero-delivery requirements (#4089)

F05, F06, F12, F13, F14, F15 and seven of the nine historian children have no implementation on main at all. They should be re-scoped as fresh, sized issues or withdrawn - not carried as ticked boxes.
