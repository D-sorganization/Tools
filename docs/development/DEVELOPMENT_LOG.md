# Development Log — Tools

State table for every feature in flight in this repository. Update
entries **in place**; never append dated sections. One entry per
feature, from proposal to ship. See the `development-logs` section of
`AGENTS.md` for the binding rules and
`shared_scripts/development_log.py` for the validator.

- **Portfolio:** infra
- **WIP limit:** 4
- **Last audited:** 2026-10-14 by Night Watch (reconciled 37 stale in_review/in_progress entries to shipped; every referenced PR/issue was already merged/closed)

### DL-#4149 · Exact golf-club head CAD families (iron family slice)

- **State:** shipped
- **Owner:** claude
- **Issue:** #4149
- **Branch:** `claude/issue-4149-iron`
- **PR:** #5425 (draft)
- **Paths:** `src/shared/python/golf_club/iron_*.py`, `src/shared/python/golf_club/_head_cad.py`, `src/shared/python/golf_club/wedge_cad.py`, `src/shared/python/golf_club/wedge_export.py`, `tests/shared/python/golf_club/test_iron_*.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-05 @ 696b70a74 (base; golf_club API baseline realigned to `Any`) — test_iron_parameters + test_iron_cad pass with build123d 0.13.0; wedge CAD/export tests unchanged and green; ruff and mypy clean.
- **Summary:** Iron family following the wedge precedent: versioned `golf_club.iron_parameters/1` schema with boundary DbC, ruled-loft B-Rep with fused hollow hosel, mass properties recovered from the solid and checked against a closed-form prismatoid and the exported-mesh divergence theorem, STEP/BREP/STL export with `golf_club.iron_export/1` manifest. Shared datum-face, hosel-tube and file-export helpers extracted to `_head_cad.py`. Driver, hybrid and putter families and shape optimisation remain open under #4149.
- **Next step:** Shipped via PR #5425.

### DL-#5434 · Align Workflow Python Matrices with requires-python

- **State:** shipped
- **Owner:** claude
- **Issue:** #5434
- **Branch:** `fix/5434-matrix-python-floor`
- **PR:** #5435
- **Paths:** `.github/workflows/maturin-file-watcher.yml`, `.github/workflows/maturin-swing-core.yml`, `tests/ops/test_workflow_python_floor.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`d61e4015ecc4f52e9713b636e9840e61bf929e96`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Split `maturin-file-watcher` and `maturin-swing-core` into a `wheel-gate` (build + import, Python 3.10-3.12, proves the crates' >=3.10 claim) and the existing wrapper/parity job (root code, now 3.11-3.12 only, since conftest will not collect it below the root floor and a 3.10 leg exited 5). A policy test fails any workflow leg below the floor of the code it exercises, and any maturin crate not built on its declared floor (`rust_core/tools-core` is a listed known gap).
- **Next step:** Shipped via PR #5435.

### DL-#5432 · Fail fast on stale module inventory test

- **State:** shipped
- **Owner:** claude
- **Issue:** #5432
- **Branch:** `fix/5432-inventory-fast-fail`
- **PR:** #5433
- **Paths:** `tests/architecture/_inventory_diff.py`, `tests/architecture/test_inventory_diff_5432.py`, `tests/architecture/test_tools_module_inventory_contract.py`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`44a1ac0e3e76220177da9af93f53d4ffcbdbbd3a`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** The inventory freshness test compared two huge dicts with `assert ==`, whose quadratic pytest diff stalled a CI shard for 90 minutes. It now fails with a bounded shard-level summary; strictness is unchanged.
- **Next step:** Shipped via PR #5433.

### DL-#4464 · Fork PRs never run on the self-hosted fleet

- **State:** shipped
- **Owner:** claude
- **Issue:** #4464
- **Branch:** `fix/4464-fork-pr-self-hosted`
- **PR:** #5427
- **Paths:** `.github/workflows/`, `scripts/fork_pr_runner_guard.py`, `tests/ops/test_fork_pr_runner_guard.py`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`696b70a74739813e05fb07d330889fb8efef91e4`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Workflow-level defence in depth for #4464: every job in a PR-triggered workflow that can reach `d-sorg-fleet` skips fork PRs, the required `tests` lane routes fork PRs to a hosted runner, and a static checker enforces both. Admin settings (fork-PR approval, default token permissions, runner-group access) remain owner-only.
- **Next step:** Shipped via PR #5427; issue #4464 closed completed.

### DL-#5429 · Fix Clippy Rust 1.99 Errors in pendulum-core

- **State:** shipped
- **Owner:** claude
- **Issue:** #5429
- **Branch:** `fix/pendulum-core-clippy-f64`
- **PR:** #5430
- **Paths:** `src/pendulum_simulator/pendulum-core/src/cmaes.rs`, `src/pendulum_simulator/pendulum-core/src/dynamics.rs`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`eb4b886ad29af93ab7b22493ad9f4e4fb1bc31d8`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Removed the deprecated `use std::f64;` in `cmaes.rs` (it made `f64::INFINITY`/`NEG_INFINITY` resolve to the deprecated module constants) and replaced constant-size `chunks_exact(2)` with `as_chunks` with const parameter 2 in `dynamics.rs`, so the pendulum_core rust quality gate passes on the fleet's Rust 1.99.
- **Next step:** Shipped via PR #5430.

### DL-#5422 · Fix Clippy Deprecated f64 Constants in tools-core

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #5422
- **Branch:** `fix/issue-5422-clippy-f64-deprecated`
- **PR:** #5423
- **Paths:** `rust_core/tools-core/src/rrt.rs`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`57e90bce55e475c890a37f7dba8b53184e845497`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Removed deprecated `use std::f64;` in `rust_core/tools-core/src/rrt.rs` that shadowed primitive `f64` associated constants (`f64::NEG_INFINITY` and `f64::MAX`) causing Clippy `-D warnings` failure in `rust-quality-gate` on CI Standard.
- **Next step:** Shipped via PR #5423.

### DL-#4433 · Visual-first first-viewport geometry beyond the initial state

- **State:** shipped
- **Owner:** claude
- **Issue:** #4433
- **Branch:** `claude/issue-4433`
- **PR:** #5424 (draft), #5447
- **Paths:** `tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py`, `tests/rate_of_closure/pyqt_launch_monitor_state_probe.py`, `src/rate_of_closure/web/e2e/visualization-tab-visibility.spec.ts`, `docs/audits/rate_of_closure_visual_first_epic_4433.v1.json`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-06 (`8c3cc237`; collated from changes/4433-launch-monitor-analytics-no-longer-scrol.md)
- **Summary:** First tab pair (Launch Monitor Analytics) of the epic's state-by-state geometry gap: the primary scatter stays in the first viewport through result, error, loading and empty on React (3 reference viewports) and PyQt (1.0/1.5 DPI). Other pairs, approved baselines and the AT protocol stay open under the epic.
- **Next step:** Shipped via PR #5424 and #5447; epic issue #4433 closed completed 2026-10-14.

### DL-#5417 · ADR-008 dependency and license deny-list check

- **State:** shipped
- **Owner:** claude
- **Issue:** #5417
- **Branch:** `claude/issue-5417`
- **PR:** #5421
- **Paths:** `scripts/check_license_denylist.py`, `config/license_denylist.json`, `tests/architecture/test_license_denylist.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`d7cef753e54ce89f30eca336ebe037f8a6b9ab86`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Enforces the ADR-008 licensing rule (no FreeMoCap or SkellyCam, AGPL ids) with a static scan of pyproject.toml and requirements files plus an optional installed-closure license walk. Lists contain only what ADR-008 names. CI wiring is out of scope.
- **Next step:** Shipped via PR #5421. CI wiring remains a separate follow-up (out of scope per original issue).

### DL-#5416 · Python validator for calculation-runtime-manifest/v1

- **State:** shipped
- **Owner:** claude
- **Issue:** #5416
- **Branch:** `claude/issue-5416`
- **PR:** #5420
- **Paths:** `src/shared/python/swing_sim/runtime_manifest.py`, `tests/shared/python/test_runtime_manifest_parity_fixture.py`, `docs/specs/active/CALCULATION_RUNTIME_MANIFEST.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`f729eca97dd26fd4955bcabeb73e7083a47609bd`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** One-for-one Python port of the TypeScript `parseRuntimeManifest`, pinned by the shared `runtime_manifest_parity_v1.json` fixture.
- **Next step:** Shipped via PR #5420.

### DL-#5392 · Longitudinal regression residual variance

- **State:** shipped
- **Owner:** codex
- **Issue:** #5392
- **Branch:** `fix/longitudinal-residual-variance-20260930`
- **PR:** #5394 (draft)
- **Paths:** `src/rate_of_closure/web/src/model/launchMonitorLongitudinal.ts`, `src/rate_of_closure/web/src/model/launchMonitorLongitudinal.test.ts`, `manuals/tools/manifests/module-inventory/`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-30
- **Last verified:** 2026-09-30 — controlled c096 baseline regression RED (0 vs 0.00020716016510533694), candidate focused GREEN (3/3), and one-worker full web Vitest rerun 2361/2361 across 238 files; type-check, lint, build, and required design-manual checks run. Initial two-worker full run retained its single import-boundary timeout; isolated test passed.
- **Summary:** The merged #5389 optimization retained the covariance/sum-of-squares pass but derived residual variance by subtracting nearly equal aggregates, reporting zero standard error for a representable near-perfect regression with nonzero ordered residuals. Restored the intercept and ordered direct-residual sum of squares, with an independent public-analysis regression test. This is a focused follow-up to merged #5389.
- **Next step:** Shipped via PR #5394.

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked`
reachable from any live state and `abandoned` from `parked`.
`shipped` never returns to `in_progress`; open a new entry instead.

## Active

### DL-#5884 · Chat Dock Tests Close Leftover Parentless Docks After Each Test so a Deferred Scroll Cannot Fire on a GC-Freed Scroll Bar and Segfault the Xdist Worker

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5884
- **Branch:** brick/ci-toolchain/5884-chat-scroll-segfault
- **PR:** #5886
- **Paths:** see #5886
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`80347f1a`; collated from changes/5884-chat-dock-tests-close-leftover-parentles.md)
- **Summary:** Chat dock tests close leftover parentless docks after each test so a deferred scroll cannot fire on a GC-freed scroll bar and segfault the xdist worker
- **Next step:** Merge after acceptance review

### DL-#5646 · Adversarial Review 2026-10-16: 38 Findings With Draft PR Pitches Under Docs/Development/Adversarial-Review/

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5646
- **Branch:** claude/adversarial-review-2026-10-16
- **PR:** #5646
- **Paths:** docs/development/adversarial-review/2026-10-16
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`6d094b4b`; collated from changes/5646-adversarial-review-2026-10-16.md)
- **Summary:** Adversarial review 2026-10-16: 38 findings with draft PR pitches under docs/development/adversarial-review/
- **Next step:** Triage Critical/High finding pages; each carries a paste-ready PR body.

### DL-#5547 · \ Deployment Topology With Docker-Compose for TimescaleDB and Grafana on Internal Host\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5547
- **Branch:** feat/5547-deployment-topology
- **PR:** #5599
- **Paths:** see #5599
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6623cbd0`; collated from changes/5547-deployment-topology-with-docker-compose.md)
- **Summary:** \ Deployment topology with docker-compose for TimescaleDB and Grafana on internal host\
- **Next step:** Shipped via PR #5599.

### DL-#5543 · \ Dual-Write to SQLite and TimescaleDB Behind P1AM_HISTORIAN_BACKEND Flag\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5543
- **Branch:** feat/5543-dual-write
- **PR:** #5598
- **Paths:** see #5598
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`529c706b`; collated from changes/5543-dual-write-to-sqlite-and-timescaledb-beh.md)
- **Summary:** \ Dual-write to SQLite and TimescaleDB behind P1AM_HISTORIAN_BACKEND flag\
- **Next step:** Shipped via PR #5598.

### DL-#5546 · \ ISA-18.2 / EEMUA 191 Alarm Performance KPIs and Dashboard\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5546
- **Branch:** feat/5546-alarm-kpis
- **PR:** #5597
- **Paths:** see #5597
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`a8a74602`; collated from changes/5546-isa-18-2-eemua-191-alarm-performance-kpi.md)
- **Summary:** \ ISA-18.2 / EEMUA 191 alarm performance KPIs and dashboard\
- **Next step:** Shipped via PR #5597.

### DL-#5545 · \ Grafana Provisioning-as-Code: Datasource and Baseline Dashboards\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5545
- **Branch:** feat/5545-grafana-provisioning
- **PR:** #5593
- **Paths:** see #5593
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`cb98894c`; collated from changes/5545-grafana-provisioning-as-code-datasource.md)
- **Summary:** \ Grafana provisioning-as-code: datasource and baseline dashboards\
- **Next step:** Shipped via PR #5593.

### DL-#5544 · Historian H5: Expose Queue Depth, Lag and Drop Counters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5544
- **Branch:** feat/5544-historian-observability
- **PR:** #5592
- **Paths:** see #5592
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`fabb775b`; collated from changes/5544-historian-h5-expose-queue-depth-lag-and.md)
- **Summary:** Historian H5: expose queue depth, lag and drop counters
- **Next step:** Shipped via PR #5592.

### DL-#5541 · Historian H2: Durable On-Disk Spool so the Historian Survives Restart (Store-and-Forward)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5541
- **Branch:** feat/5541-historian-spool
- **PR:** #5584
- **Paths:** see #5584
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`807a6bd0`; collated from changes/5541-historian-h2-durable-on-disk-spool.md)
- **Summary:** Historian H2: Durable on-disk spool so the historian survives restart (store-and-forward)
- **Next step:** Shipped via PR #5584.

### DL-#5540 · Historian H1: Extract HistorianSink Protocol for Pluggable Write Backends

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5540
- **Branch:** feat/5540-historian-sink-protocol
- **PR:** #5584, #5588
- **Paths:** see #5584
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`7c44b38f`; collated from changes/5540-historian-h1-extract-historiansink-proto.md)
- **Summary:** Historian H1: Extract HistorianSink protocol for pluggable write backends
- **Next step:** Shipped via PR #5584.

### DL-#5577 · Normalize Compatible Contract-Level Enums

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5577
- **Branch:** fix/contract-enum-interop-5577-20261012
- **PR:** #5579
- **Paths:** see #5579
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`9b880daf`; collated from changes/5577-normalize-compatible-contract-level-enum.md)
- **Summary:** Normalize compatible contract-level enums
- **Next step:** Shipped via PR #5579.

### DL-#5548 · Historian H9: Historian Operator/Engineer Documentation, Runbook and ADR

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5548
- **Branch:** docs/5548-historian-adr-runbook
- **PR:** #5583
- **Paths:** see #5583
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`f71ee279`; collated from changes/5548-historian-h9-docs-runbook-adr.md)
- **Summary:** Historian H9: Historian operator/engineer documentation, runbook and ADR
- **Next step:** Shipped via PR #5583.

### DL-#5523 · Pendulum_Simulator: Re-Run Swing_Objectives/Impact_Optimality/Model_Adequacy Results on the Distributed Club (51.4 m/s, 8.02 m/s Hands, 14% Braking; Centrifugal Is the Dissenting Objective) With a Reproducible Script

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5523
- **Branch:** fix/5523-rerun-distributed-club
- **PR:** #5576
- **Paths:** see #5576
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`5c1f1657`; collated from changes/5523-pendulum-simulator-re-run-swing-objectiv.md)
- **Summary:** pendulum_simulator: re-run swing_objectives/impact_optimality/model_adequacy results on the distributed club (51.4 m/s, 8.02 m/s hands, 14% braking; centrifugal is the dissenting objective) with a reproducible script
- **Next step:** Shipped via PR #5576; exact-provider qualification keeps results exploratory.

### DL-#5530 · SCADA F09: Declarative Procedure Model With Hold/Abort, Gated to the Simulated Driver

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5530
- **Branch:** feat/5530-scada
- **PR:** #5578, #5580
- **Paths:** see #5578
- **Started:** 2026-10-12
- **Last verified:** 2026-10-13 (`98d0ea6c`; collated from changes/5530-scada-f09-matrix-row-flipped-to-landed-c.md)
- **Summary:** SCADA F09: declarative procedure model with hold/abort, gated to the simulated driver
- **Next step:** Shipped via PR #5578.

### DL-#5531 · SCADA F10: Advisory-Only Flatline and Command/Feedback-Mismatch Detectors (asset_health.py)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5531
- **Branch:** feat/5531-asset-health
- **PR:** #5559
- **Paths:** see #5559
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`86819437`; collated from changes/5531-scada-f10-advisory-only-flatline-and-com.md)
- **Summary:** SCADA F10: advisory-only flatline and command/feedback-mismatch detectors (asset_health.py)
- **Next step:** Shipped via PR #5559.

### DL-#5572 · Preserve Signal Toolkit Import Identity

- **State:** shipped
- **Owner:** codex
- **Issue:** #5572
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/**init**.py, tests/shared/python/signal_toolkit/test_import_identity.py, manuals/tools/manifests/module-inventory.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5572-signal-toolkit-import-identity.md)
- **Summary:** Bind Signal toolkit import aliases before package children capture provider classes.
- **Next step:** Shipped via PR #5573.

### DL-#5571 · Preserve Signal Fitting Consumer Contracts

- **State:** shipped
- **Owner:** codex
- **Issue:** #5571
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/fitting.py, tests/shared/python/signal_toolkit/test_fitting_consumer_contracts.py, manuals/tools/manifests/module-inventory.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5571-signal-fitting-consumer-contracts.md)
- **Summary:** Preserve Signal fitting empty-input and failed-fit ndarray identity contracts in the canonical Tools provider.
- **Next step:** Shipped via PR #5573.

### DL-#5570 · Reconcile Development Log Provenance

- **State:** shipped
- **Owner:** codex
- **Issue:** #5570
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** .secrets.baseline, docs/development/DEVELOPMENT_LOG.md, docs/development/development-log-provenance-migration.md, docs/development/provenance/tools-devlog-migration-plan.json, docs/development/provenance/tools-devlog-metadata-review.md, docs/development/provenance/tools-devlog-migration-receipt.json, shared_scripts/development_log_migration.py
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5570-governed-development-log-provenance.md)
- **Summary:** Reconcile inherited Tools development-log findings through the reviewed, lossless RM provenance migration.
- **Next step:** Shipped via PR #5573. Stale historical feature states reviewed and reconciled by Night Watch 2026-10-14 (this pass).

### DL-#5569 · Keep Signal Structural Shape Validation Active Under Optimized Python

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5569
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/core.py; tests/shared/python/signal_toolkit/test_core_structural_guards.py; manuals/tools/manifests/module-inventory.json; manuals/tools/manifests/module-inventory/entries-src-shared-python-signal-toolkit.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5569-keep-signal-structural-shape-validation.md)
- **Summary:** Keep Signal structural shape validation active under optimized Python
- **Next step:** Shipped via PR #5573. UpstreamDrift #7704 repin/mirror remains a downstream follow-up.

### DL-#5528 · SCADA F07: Publish Firmware Trip Latch Tri-State (Plc_Trip_Latch) and First-Out Tag (First_Out_Tag_Id) From One Register Read; InterlocksPanel Shows PLC Trip and an Explicit Unknown State

- **State:** shipped
- **Owner:** claude
- **Issue:** #5528
- **Branch:** feat/5528-first-out-tag
- **PR:** #5550
- **Paths:** see #5550
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`aa48c2e1`; collated from changes/5528-scada-f07-read-firmware-first-out-trip-t.md)
- **Summary:** SCADA F07: publish firmware trip latch tri-state (plc_trip_latch) and first-out tag (first_out_tag_id) from one register read; InterlocksPanel shows PLC trip and an explicit unknown state
- **Next step:** Shipped via PR #5550.

### DL-#5474 · Require Matching Versioned Actuator-Profile References for Mixed Command Evidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5474
- **Branch:** feat/5474-actuator-profile-reference-admission
- **PR:** #5475
- **Paths:** see #5475
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`86d0f28b`; collated from changes/5474-compiled-actuator-profile-evidence.md)
- **Summary:** Require matching versioned actuator-profile references for mixed command evidence
- **Next step:** Shipped via PR #5475. Profile-bytes verification against native execution in UpstreamDrift #11955 remains a downstream follow-up.

### DL-#5454 · Deterministic 1440X900 Visual-Baseline Candidates: Settle Tab-Pill Transitions Before Capture and Pin Determinism With a Two-Session Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5454
- **Branch:** fix/5454-deterministic-visual-baselines
- **PR:** #5481
- **Paths:** see #5481
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`24eede77`; collated from changes/5454-deterministic-1440x900-visual-baseline-c.md)
- **Summary:** Deterministic 1440x900 visual-baseline candidates: settle tab-pill transitions before capture and pin determinism with a two-session test
- **Next step:** Shipped via PR #5481.

### DL-#5472 · Repair Canonical Manual Source-Count Regression While Preserving Release Blockers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5472
- **Branch:** fix/canonical-manual-count-5472
- **PR:** #5473
- **Paths:** see #5473
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`25b7c1fb`; collated from changes/5472-repair-canonical-manual-source-count-reg.md)
- **Summary:** Repair canonical manual source-count regression while preserving release blockers
- **Next step:** Shipped via PR #5473.

### DL-#5470 · Add Opaque Native Restart Transport and Strict Replay Envelope With Native-Clock Integrity and Dynamic Complete Manual Page Inspection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5470
- **Branch:** feat/native-state-artifact-5470
- **PR:** #5471
- **Paths:** see #5471
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`89415ee8`; collated from changes/5470-add-opaque-native-restart-transport-and.md)
- **Summary:** Add opaque native restart transport and strict replay envelope with native-clock integrity and dynamic complete manual page inspection
- **Next step:** Shipped via PR #5471. Native R2025b restart provider integration in UpstreamDrift #11921 remains a downstream follow-up.

### DL-#5463 · Bound Replay Experiment Resources and Privacy-Safe Preview Artifacts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5463
- **Branch:** feat/5463-bounded-preview-resources
- **PR:** #5468
- **Paths:** src/shared/python/sidekick/lab/mocap,schemas/mocap/preview-manifest-v1.schema.json,tests/shared/python/sidekick/lab/mocap,test_sidekick_public_api_stability.py,docs/development/feedback_controls
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`b1480ff6`; collated from changes/5463-add-bounded-replay-resources-and-privacy.md)
- **Summary:** Add bounded replay resources and privacy-aware preview manifest contracts
- **Next step:** Shipped via PR #5468.

### DL-#5460 · Plan Shared Controlled-Matching and Muscle Replay Interchange With Private Evidence Boundaries

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5460
- **Branch:** docs/feedback-controls-5460
- **PR:** #5465
- **Paths:** docs/development/feedback_controls
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`789fd312`; collated from changes/5460-controlled-matching-contracts.md)
- **Summary:** Plan shared controlled-matching and muscle replay interchange with private evidence boundaries
- **Next step:** Shipped via PR #5465.

### DL-#5440 · Drop Faulthandler Timeout That Crashed Tests-Shared

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5440
- **Branch:** fix/5440-drop-faulthandler-timeout
- **PR:** #5443, #5442
- **Paths:** pyproject.toml,tests/architecture/test_faulthandler_timeout_contract.py
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`ead6be5f`; collated from changes/5440-variation-gui-lost-signal.md)
- **Summary:** Remove faulthandler_timeout=50 added by #5442: its stack dump fired inside legitimately long numerical tests and crashed the tests-shared shard on main (exit 249). A contract test keeps any future watchdog at or above the longest timeout marker.
- **Next step:** Shipped via PR #5443.

### DL-#5436 · Prove Tools-Core Python Floor in CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5436
- **Branch:** fix/5436-tools-core-wheel-gate
- **PR:** #5438
- **Paths:** .github/workflows/maturin-tools-core.yml,tests/ops/test_workflow_python_floor.py
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`625c4e0c`; collated from changes/5436-ci-tools-core-new-maturin-tools-core-whe.md)
- **Summary:** ci(tools-core): new maturin-tools-core wheel-gate proves the crate's >=3.10 floor on Python 3.10-3.12; floor policy test now has no exemptions
- **Next step:** Shipped via PR #5438.

### DL-#1976 · Add Collate-Changes Workflow and Wire Spec-Check to Check_Spec_Freshness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1976
- **Branch:** ci/1976-tools-collate-changes
- **PR:** #5448, #5451, #5428
- **Paths:** see #5448
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`ad530938`; collated from changes/1976-tools-change-fragments.md)
- **Summary:** Add collate-changes workflow and wire spec-check to check_spec_freshness
- **Next step:** Shipped via PR #5448.

### DL-#5385 · Propagate flight termination states and gate landing metrics

- **State:** shipped
- **Owner:** local (agy executor)
- **Issue:** #5385
- **Branch:** `fix/tools-flight-termination-5385`
- **PR:** #5391
- **Paths:** `src/shared/python/swing_sim/flight/`, `src/rate_of_closure/application/`, `tests/api_baselines/swing_sim_api_baseline.json`
- **Started:** 2026-09-30
- **Last verified:** 2026-10-12 (`6be5b5b164a64bca8a5c6bedad57f25bd11f7b91`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Propagate explicit FlightTermination enum states (LANDED, TIME_LIMIT, SOLVER_FAILED, CANCELLED) before reporting landing metrics. Gate carry_distance, landing_angle, and lateral_deviation to None on non-landed flights. Fail closed across CenteredClubDeliveryAdapter, ground simulation transfer, and flight execution profiles.
- **Next step:** Shipped via PR #5391. UpstreamDrift #11145 link remains a downstream follow-up.

### DL-#5353 · Pre-impact bundle wire, modal-state record and public validators

- **State:** shipped
- **Owner:** local (agy executor)
- **Issue:** #5353
- **Branch:** `feat/5353-pre-impact-bundle-modal-state`
- **PR:** #5368
- **Paths:** `src/shared/python/golf_club/_pre_impact_contracts.py`, `src/shared/python/golf_club/_pre_impact_serde.py`, `src/shared/python/golf_club/modal_state.py`, `src/shared/python/golf_club/pre_impact_bundle.py`, `src/shared/python/golf_club/pre_impact_frames.py`, `tests/shared/python/golf_club/`, `tests/api_baselines/golf_club_api_baseline.json`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-12 (`b61fc5b5021b34d239b738da1390bda6a67497bd`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Complete implementation of #5353: items 1-2 add the versioned pre-impact bundle wire (`PreImpactBundle`, `Provenance`, `TimeBase`, `HeadState`, `BallState`, `HandWrench`, `Pose`, `Quantity`), Plücker frame transforms with $10^{-12}$ wrench power invariance, and modal-state records with M-orthogonal projection ($0.5 \dot{q}^T M_r \dot{q} + 0.5 q^T K_r q$). Follows items 3-4 (validation and field origin) landed in #5366.
- **Next step:** Shipped via PR #5368.

### DL-#5362 · Input contracts surfaced by the UpstreamDrift shadow burn-down

- **State:** shipped
- **Owner:** claude (agy executor)
- **Issue:** #5362
- **Branch:** `claude/tools-5362-shadow-contracts`
- **PR:** #5364 (draft)
- **Paths:** `src/shared/python/signal_toolkit/`, `src/shared/python/model_generation/`, `src/shared/python/ai/gui/session_manager.py`, `src/shared/python/humanoid_character_builder/__init__.py`, `tests/shared/python/signal_toolkit/`, `tests/shared/python/model_generation/`, `tests/unit/ai/gui/test_session_manager_2872.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-12 (`3678409fc51024150ab28970b72e3b468935f345`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Preconditions on signal_toolkit smoothing/saturation/rate-limit/deadband/noise and primitive inertia mass, standard gravity from the shared constant, UTC-normalised chat session timestamps, and a corrected humanoid_character_builder docstring, so UpstreamDrift can retire its quarantined expectations by bumping the pin instead of editing its shadow copies.
- **Next step:** Shipped via PR #5364.

### DL-#5347 · Knowledge-pack golden Q&A evaluation and optional MiniLM hybrid ranking

- **State:** shipped
- **Owner:** claude
- **Issue:** #5347 (epic Repository_Management#1772 K4)
- **Branch:** `agy/issue-5347`
- **PR:** #5351
- **Paths:** `src/shared/python/ai/knowledge/`, `tests/shared/python/ai/knowledge/`, `tests/api_baselines/knowledge_api_baseline.json`, `manuals/tools/manifests/module-inventory/`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 (PR #5351 merged to main sha 1a8f012)
- **Summary:** Added `ai/knowledge/eval.py` for golden Q&A evaluation (recall@k, MRR, must-not-source violation detection, markdown report, CLI) and optional dense MiniLM embeddings with hybrid BM25 + cosine Reciprocal Rank Fusion ranking gated by `embeddings: true`. Kept stdlib-only core importable without external ML dependencies. Rebased onto main after #5350 (Sidekick Wizards) merged; `get_minilm_embedder` now logs a warning and returns `None` on import failure instead of swallowing it, and `_search_hybrid` falls back to plain BM25 with a one-time warning when no embedder is available. Fixed mypy `no-any-return` on embedder resolution and compacted `pack.py` to 496 LOC to satisfy the file-size budget gate.
- **Next step:** Shipped via PR #5351.

### DL-#5322 - Enforce the Deferred Validation Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #5322; full rollout Repository_Management#1687
- **Branch:** `chore/5322-deferred-catalog-guard`
- **PR:** #5323
- **Paths:** `shared_scripts/`, `tests/test_deferred_catalog_hook.py`, `.pre-commit-config.yaml`, `docs/development/`, `AGENT_HANDOFF.md`, `README.md`, `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (c451d0be7; complete actual pre-push hooks including unit suite, Bandit and dependency audit pass; five RED then GREEN controls; 54 catalog/inventory/handoff/SPEC tests; pinned root Ruff, 4041-file format, strict new-test mypy and all nine manual gates pass; approved policy blocks already present)
- **Summary:** Install the central validator unchanged and exercise the actual always-run hook without changing the original v1 plans or empirical acceptance gates.
- **Next step:** Shipped via PR #5323.

### DL-#5317 · Deferred Validation Project Projection

- **State:** shipped
- **Owner:** codex
- **Issue:** #5317; parents Repository_Management#1687 and Runner_Dashboard#1248
- **Branch:** `docs/deferred-project-projection`
- **PR:** https://github.com/D-sorganization/Tools/pull/5318
- **Paths:** `docs/project/CHARTER.md`, `docs/project/STATUS.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (c720785aeb3485e543022b23a90d2d29dedd59ec; merged #5318; central catalog valid, actual parsers agree on 18 features/13 decisions, all original features/decisions and planning bytes preserved; normal documentation commit/pre-push hooks pass)
- **Summary:** Preserve the existing portfolio and Board questions while exposing all three owner plans through the shared Projects contract. Historical observations remain explicitly dated; missing evidence stays missing.
- **Next step:** Projection published and its three parked rows/owner links verified in the dashboard; Board decisions remain pending. Enforcement is DL-#5322.

### DL-#5068-planning · Deferred Physical Validation Plans

- **State:** shipped
- **Owner:** codex (session `codex-validation-planning-20260922-tools`)
- **Issue:** #5068; #4729; #4267
- **Branch:** `docs/deferred-validation-planning`
- **PR:** #5306
- **Paths:** `docs/development/planning/`, `docs/development/HANDOFF.md`, `AGENT_HANDOFF.md`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 (source issue bodies unchanged; strict three-plan catalog, design-manual governance and full commit hook stage pass; pre-push environment install failed with disk-full Errno 28)
- **Summary:** Preserve unavailable experiments for Board review without closing mixed software epics or supplying fabricated measurements.
- **Next step:** Shipped via PR #5306.

### DL-#5160 · Objective Coupled Friction Trajectory

- **State:** shipped
- **Owner:** codex
- **Issue:** [#5160](https://github.com/D-sorganization/Tools/issues/5160)
- **PR:** https://github.com/D-sorganization/Tools/pull/5162 (open; do-not-merge)
- **Branch:** feat/5073-friction-trajectory
- **Paths:** src/shared/python/swing_sim/impact, src/shared/python/golf_club, tests/shared/python/golf_club, docs/development/impact-acoustics/FRICTION_TRAJECTORY.md, SPEC.md and turnover/inventory
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (d4323c14ad364640b3f371ba8bed606ab958c990; turnover SELF):1657 local coverage tests pass/2 optional skips/93.80%; manual34496105215 all14 shards and both aggregates pass. PR34496053564 attempt2 times out at unchanged60s: Python3.12 job102940510943 entry60 and Python3.11 job102940510985 entry240. Both outcomes remain recorded; no runtime reliability or merge qualification is claimed.
- **Summary:** Parent #5073. Coupled endpoint mechanics and objective elastic/Coulomb history with explicit transport, nonlinear solve criteria and disjoint energy accounting.
- **Next step:** Shipped via PR #5162.

### DL-#5114 · Rate Shard Scientific Test Isolation

- **State:** shipped
- **Owner:** codex
- **Issue:** #5114
- **PR:** #5158 (merged as 25367070f)
- **Branch:** fix/issue-5114-rate-isolation
- **Paths:** `scripts/ci_test_shards.py`, `tests/ops/test_ci_test_shards.py`
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10; four new regression cases fail before isolation; all 16 shard contracts and the 1,684-file partition pass afterward. All three unchanged Club Tester tests pass serially with coverage and the existing deadline (45.04 seconds total). Both Linux rate shards and aggregate gates pass at a7d025264; all nine manual governance checks and normal hooks pass. Windows retains two independently reproduced, unchanged-source GUI limitations recorded in PR #5158.
- **Summary:** Reuse serial scientific invocation support for the entire Club Tester file while retaining parallel execution elsewhere, separate coverage outputs and failure propagation. Full-suite worker exit is reproduced; exact root cause remains unproven.
- **Next step:** Qualify the combined context provider in PR #5141.

### DL-#5137 · Identified Moving Reference Placements

- **State:** shipped
- **Owner:** codex
- **Issue:** #5137; consumer UpstreamDrift#9899
- **Branch:** feat/5137-reference-placements
- **PR:** #5140 (open)
- **Paths:** src/shared/python/sidekick/lab/mocap/reference_placements.py, placement_solver.py, calibration_numerics.py; reference tests and manual inventory
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`d129737273800c8cb3c31e28c38c421f05993a95`; 24 numerical/reference tests pass on OpenCV 4.13 and 5.0; all101 mocap/authority/API tests pass in17.03s; scoped lint/format/mypy and governance gates pass)
- **Summary:** Labelled target geometry, immutable per-camera/profile observations, connected pose initialization and joint camera/target fitting with a fixed anchor and independent held-out views; explicit unsupported/ambiguous geometry and cancellation outcomes.
- **Next step:** Shipped via PR #5140.

- **Main integration:** repair45f3bd8b9/main2c9a8d6c9 retains impact work.101 mocap/authority/API tests and24 OpenCV5 numerical/placement checks pass; inventory/handoff gates pass. Normal publication checks pending.

### DL-#5132 · Calibration Numerical Recovery

- **State:** parked
- **Owner:** codex
- **Issue:** #5132; consumer UpstreamDrift#9897/#9899
- **Branch:** fix/5132-calibration-numerics
- **PR:** https://github.com/D-sorganization/Tools/pull/5136
- **Paths:** src/shared/python/sidekick/lab/mocap/calibration.py, extrinsics.py, calibration_numerics.py; numerical tests and inventory
- **Started:** 2026-09-09
- **Last verified:** 2026-10-12 (`0a561daff18302ee143b214fe6a6138455d8c542`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Real distortion-aware pose recovery replaces silent fabricated poses and unchanged refinement results.
- **Next step:** PR #5136 closed without merge; issue #5132 closed. Parked pending decision to retry or abandon.
- **Parked:** PR #5136 closed without merge; issue #5132 closed. Await a governed decision to retry or abandon the calibration work.

### DL-0056 · Distributed Shaft Prestress and Grip

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5072
- **PR:** https://github.com/D-sorganization/Tools/pull/5133
- **Branch:** feat/5072-prestressed-shaft
- **Paths:** `src/shared/python/golf_club/*shaft*`, `src/shared/python/golf_club/_beam_fem.py`, `src/shared/python/golf_club/*grip*`, `src/shared/python/golf_club/*rotating_body*`, `tests/shared/python/golf_club/test_shaft*.py`, `tests/shared/python/golf_club/test_rotating_body.py`, `tests/shared/python/golf_club/test_grip_impedance.py`, `docs/development/impact-acoustics/*`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-09 (published 00d17e7f9; base 421889407; oracle repair SELF; 17 Linux 3.11 and 36 Linux 3.12 coverage checks pass)
- **Summary:** Review #5130 covers the finite-rotation shaft and explicit input wire; #5072 retains physical/FRF qualification. First CI repairs reproduce and correct NumPy typing, fixture typing and scientific deadline failures through exact reuse and isolated resource ownership. Full serial golf coverage: 1,167 passed, two optional CAD skips; 58.78 s slowest retains a narrow CI margin. UpstreamDrift #9912 verifies the candidate vendor and installed wheel; final reviewed pin and protected CI remain. Actual incremental mypy now passes 14 files after a tuple-return annotation; 71 affected tests pass. See CI_REPAIR_RESULTS.json and PROGRESS.md for controls, including failed runs.
- **Next step:** Shipped via PR #5133.

### DL-0055 · Qualified Lumped Impact Dynamics

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5071
- **PR:** https://github.com/D-sorganization/Tools/pull/5082
- **Branch:** fix/5071-qualified-impact-coupling
- **Paths:** `src/shared/python/golf_club/*coupl*`, `tests/shared/python/golf_club/test*coupl*`, `docs/specs/HEAVY_HIT_COUPLING.md`, `docs/development/impact-acoustics/PROGRESS.md`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (#5082 merged as 80d580d57; #5071 closed; golf source/tests equal reviewed e47fde4e and prior 476eaa98)
- **Summary:** RED-to-GREEN termination, passive energy and scaling gates; complete epic scope remains active.
- **Next step:** Delivery complete for this slice; integrate current main into T3 and continue distributed milestone #5072. The full scientific/acoustic program remains open.

### DL-0001 · Backup Tools 3300 Pyo3 Split

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`df269c251`)
- **Summary:** Seeded from local branch `backup/tools-3300-pyo3-split`, which is
  4 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0002 · Bolt Optimize Isnan Usedataprocessor 13774140709308323057

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`cdc092247`)
- **Summary:** Seeded from local branch `bolt/optimize-isnan-useDataProcessor-13774140709308323057`, which is
  5 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0003 · Claude 4624 Mirror Freshness

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`81816df91`)
- **Summary:** Seeded from local branch `claude/4624-mirror-freshness`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0004 · Codex Rebase 2852

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`d4b0abe3a`)
- **Summary:** Seeded from local branch `codex-rebase-2852`, which is
  1397 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0005 · Codex Issue 7249 Tools Sidekick Agent Local

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`49c4ad734`)
- **Summary:** Seeded from local branch `codex/issue-7249-tools-sidekick-agent-local`, which is
  5 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0006 · Codex Movement Optimizer Swingset Chain

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`bf98a547a`)
- **Summary:** Seeded from local branch `codex/movement-optimizer-swingset-chain`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0007 · Codex Pr 2658 Spec

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`d620f5d1e`)
- **Summary:** Seeded from local branch `codex/pr-2658-spec`, which is
  1298 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0008 · Codex Pr 2658 Spec On Merge

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`0a896d6a2`)
- **Summary:** Seeded from local branch `codex/pr-2658-spec-on-merge`, which is
  1299 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0009 · Codex Pr 3062

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`1e72b4d67`)
- **Summary:** Seeded from local branch `codex/pr-3062`, which is
  57 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0010 · Codex Pr 3108 Ci Fix

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`d2fa3bd75`)
- **Summary:** Seeded from local branch `codex/pr-3108-ci-fix`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0011 · Codex Pr2635 Fix

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3ae9fdf41`)
- **Summary:** Seeded from local branch `codex/pr2635-fix`, which is
  1282 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0012 · Codex Pr2635 Live

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`8d5f4e014`)
- **Summary:** Seeded from local branch `codex/pr2635-live`, which is
  1281 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0013 · Codex Pr2635 Skip Guard

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3ae9fdf41`)
- **Summary:** Seeded from local branch `codex/pr2635-skip-guard`, which is
  1282 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0014 · Codex Sidekick Canonical Runtime Final

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`b5e5af078`)
- **Summary:** Seeded from local branch `codex/sidekick-canonical-runtime-final`, which is
  20 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0015 · Consolidate Open Prs 20260727

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`77e81a36d`)
- **Summary:** Seeded from local branch `consolidate/open-prs-20260727`, which is
  94 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0016 · Feat Sidekick Gui Thread Marshalling

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`618807020`)
- **Summary:** Seeded from local branch `feat/sidekick-gui-thread-marshalling`, which is
  7 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0017 · Fix 3296 Anti Phantom Checkout

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`064520b1d`)
- **Summary:** Seeded from local branch `fix/3296-anti-phantom-checkout`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0018 · Fix 3298 Ci

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`f60f53844`)
- **Summary:** Seeded from local branch `fix/3298-ci`, which is
  7 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0019 · Fix 3300 Ci

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`df269c251`)
- **Summary:** Seeded from local branch `fix/3300-ci`, which is
  4 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0020 · Fix 3300 Ci Integrated

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`db9d3dd06`)
- **Summary:** Seeded from local branch `fix/3300-ci-integrated`, which is
  22 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0021 · Fix 3936 Sidekick Chat Websocket

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`6abe0c181`)
- **Summary:** Seeded from local branch `fix/3936-sidekick-chat-websocket`, which is
  6 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0022 · Fix 3937 Python Floor Followup

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3e6dab0f2`)
- **Summary:** Seeded from local branch `fix/3937-python-floor-followup`, which is
  7 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0023 · Fix 8198 Pyqt Submodule Skips

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`16ae46e69`)
- **Summary:** Seeded from local branch `fix/8198-pyqt-submodule-skips`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0024 · Fix 8199 Humanoid Preview Interface

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`6ea520b0e`)
- **Summary:** Seeded from local branch `fix/8199-humanoid-preview-interface`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0025 · Fix Bolt Detect Secrets 2986

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`ed327939a`)
- **Summary:** Seeded from local branch `fix/bolt-detect-secrets-2986`, which is
  1471 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0026 · Fix C3D Missing Point Units

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3d4cabe94`)
- **Summary:** Seeded from local branch `fix/c3d-missing-point-units`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0027 · Fix Ci Bugs 3291 3294 3295 3296 3284

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`aa274a8c8`)
- **Summary:** Seeded from local branch `fix/ci-bugs-3291-3294-3295-3296-3284`, which is
  8 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0028 · Fix Issue 2943 Maxwell Ruff Timeout Investigation

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`4bd90f13c`)
- **Summary:** Seeded from local branch `fix/issue-2943-maxwell-ruff-timeout-investigation`, which is
  1454 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0029 · Fix Rotation Converter Unroll Arrays

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`1315ecbcc`)
- **Summary:** Seeded from local branch `fix/rotation-converter-unroll-arrays`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0030 · Fix Sidekick C3D Header Validation

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`b80d18548`)
- **Summary:** Seeded from local branch `fix/sidekick-c3d-header-validation`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0031 · Fix Sidekick Corrupt Json State

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`66fed724b`)
- **Summary:** Seeded from local branch `fix/sidekick-corrupt-json-state`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0032 · Fix Sidekick Standard Response Import

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`a41897b3f`)
- **Summary:** Seeded from local branch `fix/sidekick-standard-response-import`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0033 · Integration Remediate T1 2026 07 26

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`aadcf0be9`)
- **Summary:** Seeded from local branch `integration/remediate-t1-2026-07-26`, which is
  32 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0034 · Pr 2702

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`801753d5b`)
- **Summary:** Seeded from local branch `pr-2702`, which is
  1314 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0035 · Pr 2703

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`89644f600`)
- **Summary:** Seeded from local branch `pr-2703`, which is
  1311 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0036 · Pr 2716 Symbolic

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`1faced323`)
- **Summary:** Seeded from local branch `pr-2716-symbolic`, which is
  1323 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0037 · Pr 2717 Dtype

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`0f9297e9b`)
- **Summary:** Seeded from local branch `pr-2717-dtype`, which is
  1322 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0038 · Pr 4687 Head

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`a3111eb30`)
- **Summary:** Seeded from local branch `pr-4687-head`, which is
  3 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0039 · Rebase Pr4692 V2

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`b6fd9d5f4`)
- **Summary:** Seeded from local branch `rebase-pr4692-v2`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0040 · Rebase Pr4696

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3df117dce`)
- **Summary:** Seeded from local branch `rebase-pr4696`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0041 · Rebase Pr4697

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`39b04177c`)
- **Summary:** Seeded from local branch `rebase-pr4697`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0042 · Rebase Pr4746

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`426de9fc9`)
- **Summary:** Seeded from local branch `rebase-pr4746`, which is
  5 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0043 · Rebase Pr4749

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`06bf6ed89`)
- **Summary:** Seeded from local branch `rebase-pr4749`, which is
  3 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0044 · Rebase Pr4750

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`6b0049f57`)
- **Summary:** Seeded from local branch `rebase-pr4750`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0045 · Rebase Pr4788

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`9b84f1353`)
- **Summary:** Seeded from local branch `rebase-pr4788`, which is
  3 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0046 · Rebase Pr4789

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`a98ccf7e2`)
- **Summary:** Seeded from local branch `rebase-pr4789`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0047 · Rebase Pr4790

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`3d864eff2`)
- **Summary:** Seeded from local branch `rebase-pr4790`, which is
  4 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0048 · Rebase Pr4798

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`5b518afdf`)
- **Summary:** Seeded from local branch `rebase-pr4798`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0049 · Test Morris Explicit Seam And Normalized Step 4462 4461

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`bb0ec562b`)
- **Summary:** Seeded from local branch `test/morris-explicit-seam-and-normalized-step-4462-4461`, which is
  2 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0050 · Test Morris Scale Sensitivity 4455

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`000998479`)
- **Summary:** Seeded from local branch `test/morris-scale-sensitivity-4455`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0051 · Worker Tools Audit Mypy Autofix

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`627ec2638`)
- **Summary:** Seeded from local branch `worker-tools-audit-mypy-autofix`, which is
  1 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0052 · Workerb Pr3059

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`554c78c39`)
- **Summary:** Seeded from local branch `workerB-pr3059`, which is
  13 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0053 · Worktree Agent A48Ec9727E03Fb564

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`0df1c65ec`)
- **Summary:** Seeded from local branch `worktree-agent-a48ec9727e03fb564`, which is
  1336 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

## Shipped (Last 90 Days)

### DL-#5911 · Nightly Full Test Suite Provisions Bundled Chromium for the Unit-Converter Playwright Regressions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5911
- **Branch:** merged via #5912
- **PR:** #5912
- **Paths:** see #5912
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`abce3d7b`; collated from changes/5911-nightly-full-test-suite-provisions-bundl.md)
- **Summary:** Nightly Full Test Suite provisions bundled Chromium for the unit-converter Playwright regressions
- **Next step:** Shipped in PR #5912.

### DL-#5905 · Rate Canvas Pins Its Owning Window During paintEvent so Cyclic GC in a Pending-Draw Flush Cannot Delete It Mid-Paint and Segfault Qt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5905
- **Branch:** merged via #5906
- **PR:** #5906
- **Paths:** see #5906
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`1df9764d`; collated from changes/5905-rate-canvas-pins-its-owning-window-durin.md)
- **Summary:** Rate canvas pins its owning window during paintEvent so cyclic GC in a pending-draw flush cannot delete it mid-paint and segfault Qt
- **Next step:** Shipped in PR #5906.

### DL-#5894 · Skills_Manager Reads Workflows Only From .agents/workflows; the Legacy .agent/workflows Fallback Is Removed Now That Every Consumer Has Migrated (RM #2567)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5894
- **Branch:** merged via #5910
- **PR:** #5910
- **Paths:** see #5910
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`0b5a1eba`; collated from changes/5894-skills-manager-reads-workflows-only-from.md)
- **Summary:** skills_manager reads workflows only from .agents/workflows; the legacy .agent/workflows fallback is removed now that every consumer has migrated (RM #2567)
- **Next step:** Shipped in PR #5910.

### DL-#5907 · Mypy.Ini Skips PyQt6 Imports and Stubs so the PyQt6 6.12.0 QtGui.Pyi Syntax Error No Longer Aborts the Changed-Files Mypy Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5907
- **Branch:** merged via #5908
- **PR:** #5908
- **Paths:** see #5908
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`84a9ccc3`; collated from changes/5907-mypy-ini-skips-pyqt6-imports-and-stubs-s.md)
- **Summary:** mypy.ini skips PyQt6 imports and stubs so the PyQt6 6.12.0 QtGui.pyi syntax error no longer aborts the changed-files mypy gate
- **Next step:** Shipped in PR #5908.

### DL-#5902 · StrEnum Fallback Test Loads a Fresh Compatibility Module Instead of Reloading an Import-Alias Module, so It No Longer Depends on Test Order

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5902
- **Branch:** merged via #5903
- **PR:** #5903
- **Paths:** see #5903
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`e56406fb`; collated from changes/5902-strenum-fallback-test-loads-a-fresh-comp.md)
- **Summary:** StrEnum fallback test loads a fresh compatibility module instead of reloading an import-alias module, so it no longer depends on test order
- **Next step:** Shipped in PR #5903.

### DL-#5899 · CI Shards Fail Fast When an Xdist Worker Crashes (--Max-Worker-Restart=0) and the Test Step Has a 45-Minute Deadline

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5899
- **Branch:** merged via #5900
- **PR:** #5900
- **Paths:** see #5900
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`cdb41a8a`; collated from changes/5899-ci-shards-fail-fast-when-an-xdist-worker.md)
- **Summary:** CI shards fail fast when an xdist worker crashes (--max-worker-restart=0) and the test step has a 45-minute deadline
- **Next step:** Shipped in PR #5900.

### DL-#5896 · Test_Cmd_Validate_Execution Patches the Camera on the Sys.Modules Entry Cmd_Validate Imports, so a Stale Shared.Python.Camera Package Attribute Left by Other Suite Tests No Longer Lets the Real Ffmpeg Source Run (Nightly Red)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5896
- **Branch:** merged via #5897
- **PR:** #5897
- **Paths:** see #5897
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`5a6c24c9`; collated from changes/5896-test-cmd-validate-execution-patches-the.md)
- **Summary:** test_cmd_validate_execution patches the camera on the sys.modules entry cmd_validate imports, so a stale shared.python.camera package attribute left by other suite tests no longer lets the real ffmpeg source run (nightly red)
- **Next step:** Shipped in PR #5897.

### DL-#5893 · Delete Legacy .agent/ Copies and Fence GAAI Ci-Watch-and-Fix Behind the Fleet PR Lifecycle (RM #2567 Tools Slice)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5893
- **Branch:** merged via #5904
- **PR:** #5904
- **Paths:** see #5904
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`ae979ad1`; collated from changes/5893-delete-legacy-agent-copies-and-fence-gaa.md)
- **Summary:** Delete legacy .agent/ copies and fence GAAI ci-watch-and-fix behind the fleet PR lifecycle (RM #2567 Tools slice)
- **Next step:** Shipped in PR #5904.

### DL-#5889 · Docs: Root REVIEW.md With Tools Review Rules (RM #2565 Rollout)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5889
- **Branch:** merged via #5890
- **PR:** #5890
- **Paths:** see #5890
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`77172306`; collated from changes/5889-review-md.md)
- **Summary:** docs: root REVIEW.md with Tools review rules (RM #2565 rollout)
- **Next step:** Shipped in PR #5890.

### DL-#5881 · Heavy Integration Installs Only Requirements-Heavy.Txt (~174 MB, Was ~1.07 GB) and Keys Its Pip Cache on It, so Slow Runners No Longer Time Out

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5881
- **Branch:** brick/ci-toolchain/5881-heavy-slim-install
- **PR:** #5883
- **Paths:** see #5883
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`9ba0f9fa`; collated from changes/5881-heavy-integration-installs-only-requirem.md)
- **Summary:** Heavy Integration installs only requirements-heavy.txt (~174 MB, was ~1.07 GB) and keys its pip cache on it, so slow runners no longer time out
- **Next step:** Compare Heavy Integration run timings on OGLaptop and nvme after merge

### DL-#5874 · Skip Vessel_Drafter Real-CAD Tests When Build123d Is Only the Tests/Vessel_Drafter MagicMock Stub

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5874
- **Branch:** claude/5874-vessel-drafter-stub-skip
- **PR:** #5875
- **Paths:** see #5875
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`1b9d047d`; collated from changes/5874-skip-vessel-drafter-real-cad-tests-when.md)
- **Summary:** Skip vessel_drafter real-CAD tests when build123d is only the tests/vessel_drafter MagicMock stub
- **Next step:** Merge after required checks; confirm on the next nightly Full Test Suite.

### DL-#5878 · Chat History-Browser Test Puts the Repo Root, Not Tests/, on Sys.Path so Optimizer_Gui Is No Longer Shadowed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5878
- **Branch:** claude/5878-chat-test-sys-path-root
- **PR:** #5879
- **Paths:** see #5879
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`327a6398`; collated from changes/5878-chat-history-browser-test-puts-the-repo.md)
- **Summary:** Chat history-browser test puts the repo root, not tests/, on sys.path so optimizer_gui is no longer shadowed
- **Next step:** Merge after required checks; confirm on the next nightly.

### DL-#5876 · Wave_Solver Main() Tests Pass an Explicit WaveConfig Instead of Parsing the Test Runner's Argv

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5876
- **Branch:** claude/5876-wave-solver-test-argv
- **PR:** #5877
- **Paths:** see #5877
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`dece4bb4`; collated from changes/5876-wave-solver-main-tests-pass-an-explicit.md)
- **Summary:** wave_solver main() tests pass an explicit WaveConfig instead of parsing the test runner's argv
- **Next step:** Merge after required checks; confirm on the next nightly.

### DL-#5871 · Run the Scipy.Signal Eager-Import Boundary Check in a Fresh Interpreter so It No Longer Pops Scipy.Signal From Sys.Modules and Breaks 17 Nightly Tests With _Signal_Api NameError

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5871
- **Branch:** merged via #5872
- **PR:** #5872
- **Paths:** see #5872
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`e019d2a2`; collated from changes/5871-run-the-scipy-signal-eager-import-bounda.md)
- **Summary:** Run the scipy.signal eager-import boundary check in a fresh interpreter so it no longer pops scipy.signal from sys.modules and breaks 17 nightly tests with _signal_api NameError
- **Next step:** Shipped in PR #5872.

### DL-#5868 · Stop Verification Test Playwright Stubs Leaking Into Pytest Collection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5868
- **Branch:** merged via #5869
- **PR:** #5869
- **Paths:** see #5869
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`cd85e62e`; collated from changes/5868-playwright-import-isolation.md)
- **Summary:** Stop verification test Playwright stubs leaking into pytest collection
- **Next step:** Shipped in PR #5869.

### DL-#5858 · Preserve Pip Cache in Heavy Integration Workflows and Raise Install Timeout

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5858
- **Branch:** merged via #5862
- **PR:** #5862
- **Paths:** see #5862
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`59972cb0`; collated from changes/5858-heavy-integration-pip-cache.md)
- **Summary:** preserve pip cache in heavy integration workflows and raise install timeout
- **Next step:** Shipped in PR #5862.

### DL-#5859 · Expose Launch Angle and Spin Rate on ToolsDeliveryEstimate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5859
- **Branch:** merged via #5860
- **PR:** #5860
- **Paths:** see #5860
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`79c4bc71`; collated from changes/5859-expose-launch-angle-and-spin-rate-on-too.md)
- **Summary:** expose launch angle and spin rate on ToolsDeliveryEstimate
- **Next step:** Shipped in PR #5860.

### DL-#5851 · Evaluate Aqueous Acid Dewpoint for HCl-H2O Systems via Kiang Correlation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5851
- **Branch:** merged via #5853
- **PR:** #5853
- **Paths:** see #5853
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`6f637141`; collated from changes/5851-acid-gas-dewpoint-kiang.md)
- **Summary:** evaluate aqueous acid dewpoint for HCl-H2O systems via Kiang correlation
- **Next step:** Shipped in PR #5853.

### DL-#5852 · Refine Gas Viscosity Calculation to Avoid Non-Physical (MW/MW_Air)^0.25 Scaling (TOOLS-21)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5852
- **Branch:** merged via #5854
- **PR:** #5854
- **Paths:** see #5854
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`f3989e07`; collated from changes/5852-scrubber-gas-viscosity-ordering.md)
- **Summary:** refine gas viscosity calculation to avoid non-physical (MW/MW_air)^0.25 scaling (TOOLS-21)
- **Next step:** Shipped in PR #5854.

### DL-#5849 · Validate Triangle Inequality on Principal Moments of Inertia (TOOLS-17)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5849
- **Branch:** fix/5849-inertia-principal-triangle-inequality
- **PR:** #5850
- **Paths:** see #5850
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`cf72fa6d`; collated from changes/5849-inertia-principal-triangle-inequality.md)
- **Summary:** Validate triangle inequality on principal moments of inertia (TOOLS-17)
- **Next step:** Merge PR

### DL-#5847 · Do Not Divide Operating_Current_a by Electrode Count in Electrode Advisor Layout (TOOLS-09)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5847
- **Branch:** fix/5847-electrode-operating-current
- **PR:** #5848
- **Paths:** see #5848
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`438e29a7`; collated from changes/5847-electrode-operating-current.md)
- **Summary:** Do not divide operating_current_a by electrode count in electrode advisor layout (TOOLS-09)
- **Next step:** Merge PR

### DL-#5844 · Delete Dead Unreachable HAS_THERMO Branch and Update Docstrings in Baghouse_Calculator (TOOLS-25)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5844
- **Branch:** fix/5844-baghouse-dead-thermo-branch
- **PR:** #5845
- **Paths:** see #5845
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`83258d3a`; collated from changes/5844-baghouse-dead-thermo-branch.md)
- **Summary:** Delete dead unreachable HAS_THERMO branch and update docstrings in baghouse_calculator (TOOLS-25)
- **Next step:** Merge PR

### DL-#5842 · Support Configurable Standard Temperature Reference Basis for SCFM Calculation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5842
- **Branch:** fix/5842-baghouse-scfm-standard-basis
- **PR:** #5843
- **Paths:** see #5843
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`f8a0705e`; collated from changes/5842-baghouse-scfm-standard-basis.md)
- **Summary:** Support configurable standard temperature reference basis for SCFM calculation
- **Next step:** Merge the PR into main.

### DL-#5840 · Correct Colebrook Friction Factor Solver Method From Newton-Raphson to Fixed-Point Picard Iteration and Document Transitional Regime (TOOLS-13)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5840
- **Branch:** fix/5840-colebrook-fixed-point-docstring
- **PR:** #5841
- **Paths:** see #5841
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`81d3338b`; collated from changes/5840-colebrook-fixed-point-docstring.md)
- **Summary:** Correct Colebrook friction factor solver method from Newton-Raphson to fixed-point Picard iteration and document transitional regime (TOOLS-13)
- **Next step:** Merge the PR.

### DL-#5837 · Evaluate Glass Conductivity Strictly at Glass Temperature Without Uncited Power Density Bias in Sidekick Glass_Interface (GLASS-07)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5837
- **Branch:** fix/5837-glass-conductivity-uncited-bias
- **PR:** #5846
- **Paths:** see #5846
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`fce2acd5`; collated from changes/5837-glass-conductivity-uncited-bias.md)
- **Summary:** Evaluate glass conductivity strictly at glass temperature without uncited power density bias in sidekick glass_interface (GLASS-07)
- **Next step:** Merge PR

### DL-#5832 · Use Atan2 and Hypot for Euler Angle Extraction Near Singularity (TOOLS-28)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5832
- **Branch:** fix/5832-rotation-euler-atan2-precision
- **PR:** #5833
- **Paths:** see #5833
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`17af7269`; collated from changes/5832-rotation-euler-atan2-precision.md)
- **Summary:** Use atan2 and hypot for Euler angle extraction near singularity (TOOLS-28)
- **Next step:** Merge the PR.

### DL-#5799 · Evaluate Water Knockout at Cooler Temperature and Re-Normalize Composition (TOOLS-05)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5799
- **Branch:** fix/5799-syngas-water-dropout-cooler
- **PR:** #5818
- **Paths:** see #5818
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`5975d65d`; collated from changes/5799-syngas-water-dropout-cooler.md)
- **Summary:** Evaluate water knockout at cooler temperature and re-normalize composition (TOOLS-05)
- **Next step:** Merge the PR into main.

### DL-#5836 · Correct KG_TO_MG Constant From 1000.0 to 1E6 in Scrubber Calculator (TOOLS-22)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5836
- **Branch:** fix/5836-scrubber-kg-to-mg-constant
- **PR:** #5838
- **Paths:** see #5838
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`f1050edd`; collated from changes/5836-scrubber-kg-to-mg-constant.md)
- **Summary:** Correct KG_TO_MG constant from 1000.0 to 1e6 in scrubber calculator (TOOLS-22)
- **Next step:** Merge the PR.

### DL-#5830 · Include Elevation Pressure Drop in Compressible Flow Calculations (TOOLS-12)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5830
- **Branch:** fix/5830-pressure-drop-compressible-elevation
- **PR:** #5831
- **Paths:** see #5831
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`fb8648a4`; collated from changes/5830-pressure-drop-compressible-elevation.md)
- **Summary:** Include elevation pressure drop in compressible flow calculations (TOOLS-12)
- **Next step:** Merge the PR.

### DL-#5828 · Sort Nominal Pipe Sizes by True Physical Diameter Instead of String Replacement (TOOLS-14)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5828
- **Branch:** fix/5828-pipe-database-fraction-sort
- **PR:** #5829
- **Paths:** see #5829
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`62c61158`; collated from changes/5828-pipe-database-fraction-sort.md)
- **Summary:** Sort nominal pipe sizes by true physical diameter instead of string replacement (TOOLS-14)
- **Next step:** Merge the PR.

### DL-#5825 · Ensure Integrate_Ode Includes T_End in Uniform Output Grid (TOOLS-27)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5825
- **Branch:** fix/5825-simulation-core-integrate-ode-endpoint
- **PR:** #5827
- **Paths:** see #5827
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`253e297f`; collated from changes/5825-simulation-core-integrate-ode-endpoint.md)
- **Summary:** Ensure integrate_ode includes t_end in uniform output grid (TOOLS-27)
- **Next step:** Merge the PR.

### DL-#5823 · Include Entropy of Mixing in Ideal Gas Mixture Entropy Calculation (TOOLS-07)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5823
- **Branch:** fix/5823-thermo-mixture-entropy-of-mixing
- **PR:** #5824
- **Paths:** see #5824
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`03870cc9`; collated from changes/5823-thermo-mixture-entropy-of-mixing.md)
- **Summary:** Include entropy of mixing in ideal gas mixture entropy calculation (TOOLS-07)
- **Next step:** Merge the PR.

### DL-#5821 · Compute Syngas Mixture Heat Capacity Ratio From Molar Heat Capacities (TOOLS-04)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5821
- **Branch:** fix/5821-syngas-mixture-heat-capacity-ratio
- **PR:** #5822
- **Paths:** see #5822
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`6d2c9def`; collated from changes/5821-syngas-mixture-heat-capacity-ratio.md)
- **Summary:** Compute syngas mixture heat capacity ratio from molar heat capacities (TOOLS-04)
- **Next step:** Merge the PR.

### DL-#5806 · Glass Interface Fallback Contract: RuntimeError Escapes DEMO/LEGACY Fallback; Sigma Defaults Drift (TOOLS-31, TOOLS-30, TP-04, GLASS-06)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5806
- **Branch:** fix/5806-glass-interface-fallback-contract
- **PR:** #5819
- **Paths:** see #5819
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`559a9846`; collated from changes/5806-glass-interface-fallback-contract.md)
- **Summary:** Glass interface fallback contract: RuntimeError escapes DEMO/LEGACY fallback; sigma defaults drift (TOOLS-31, TOOLS-30, TP-04, GLASS-06)
- **Next step:** Merge the PR.

### DL-#5803 · Fix Box Inertia Tensor Axis Alignment in anthropometric_model.py

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5803
- **Branch:** fix/5803-urdf-box-inertia
- **PR:** #5811
- **Paths:** see #5811
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`356e3765`; collated from changes/5803-align-box-inertia-axes-urdf-builder.md)
- **Summary:** Fix box inertia tensor axis alignment in anthropometric_model.py
- **Next step:** Merge PR

### DL-#5798 · Use Polytropic Exponent (Gamma-1)/(Gamma*Eta) for Discharge Temperature and Work (TOOLS-03)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5798
- **Branch:** fix/5798-polytropic-exponent-irreversibility
- **PR:** #5816
- **Paths:** see #5816
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`273c280b`; collated from changes/5798-polytropic-exponent-irreversibility.md)
- **Summary:** Use polytropic exponent (gamma-1)/(gamma*eta) for discharge temperature and work (TOOLS-03)
- **Next step:** Merge the PR.

### DL-#5797 · Compute Isentropic Efficiency as W_Isentropic / W_Actual <= 1 and Allow Low-Efficiency Warning (TOOLS-02)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5797
- **Branch:** fix/5797-syngas-isentropic-efficiency
- **PR:** #5815
- **Paths:** see #5815
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`5a591908`; collated from changes/5797-syngas-isentropic-efficiency.md)
- **Summary:** Compute isentropic efficiency as W_isentropic / W_actual <= 1 and allow low-efficiency warning (TOOLS-02)
- **Next step:** Merge PR

### DL-#5805 · Use Actual Gas Mass Flux in Scrubber Eckert Flow Parameter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5805
- **Branch:** fix/5805-scrubber-gas-flux
- **PR:** #5809
- **Paths:** see #5809
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`5b144a79`; collated from changes/5805-use-actual-gas-mass-flux-in-scrubber-eck.md)
- **Summary:** Use actual gas mass flux in scrubber Eckert flow parameter
- **Next step:** Shipped in PR #5809.

### DL-#5802 · Mirror Bilateral Limb Joint Origins With Lateral Offsets Using Shoulder/Hip Widths

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5802
- **Branch:** fix/5802-urdf-bilateral-joint-origins
- **PR:** #5812
- **Paths:** see #5812
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`8bd3732b`; collated from changes/5802-urdf-bilateral-joint-origins.md)
- **Summary:** Mirror bilateral limb joint origins with lateral offsets using shoulder/hip widths
- **Next step:** Shipped in PR #5812.

### DL-#5801 · Convert Kmol/H to Nm3/H for GHSV Sizing and Allow Negative Extent for Reverse Shift (TOOLS-10, TOOLS-11)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5801
- **Branch:** fix/5801-wgs-volume-and-reverse-shift
- **PR:** #5813
- **Paths:** see #5813
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`7ee84f18`; collated from changes/5801-wgs-volume-and-reverse-shift.md)
- **Summary:** Convert kmol/h to Nm3/h for GHSV sizing and allow negative extent for reverse shift (TOOLS-10, TOOLS-11)
- **Next step:** Shipped in PR #5813.

### DL-#5800 · Decide Steam Phase From Saturation Temperature at Given Pressure in Simplified Mode (TOOLS-06)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5800
- **Branch:** fix/5800-steam-phase-saturation-temp
- **PR:** #5814
- **Paths:** see #5814
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`e94d1b18`; collated from changes/5800-steam-phase-saturation-temp.md)
- **Summary:** Decide steam phase from saturation temperature at given pressure in simplified mode (TOOLS-06)
- **Next step:** Shipped in PR #5814.

### DL-#5769 · Fetch Complete Quality-Gate History so Multi-Commit Push Diffs Retain the Before SHA.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5769
- **Branch:** merged via #5794
- **PR:** #5794
- **Paths:** see #5794
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`2e722742`; collated from changes/5769-fetch-complete-quality-gate-history-so-m.md)
- **Summary:** Fetch complete quality-gate history so multi-commit push diffs retain the before SHA.
- **Next step:** Shipped in PR #5794.

### DL-#5804 · Regularise Double Pendulum Coulomb Friction With Tanh

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5804
- **Branch:** fix/5804-double-pendulum-friction
- **PR:** #5808
- **Paths:** see #5808
- **Started:** 2026-10-27
- **Last verified:** 2026-10-27 (`545dc19f`; collated from changes/5804-regularise-double-pendulum-coulomb-frict.md)
- **Summary:** Regularise double pendulum Coulomb friction with tanh
- **Next step:** Shipped in PR #5808.

### DL-#5717 · Bound CI Apt Lock and Package Acquisition With Process-Group Cleanup.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5717
- **Branch:** cursor/ci-apt-bounded-script-b9dd
- **PR:** #5787, #5790
- **Paths:** see #5787
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`6ad4516e`; collated from changes/5717-handle-missing-sudo-and-sigterm-cleanup.md)
- **Summary:** Bound CI apt lock and package acquisition with process-group cleanup.
- **Next step:** Merge after CI green.

### DL-#5744 · Route Repo-Owned Agent Auto-Merge Instructions Through automerge_guard.py and Hold Blocked: PRs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5744
- **Branch:** merged via #5751
- **PR:** #5751, #5856
- **Paths:** see #5751
- **Started:** 2026-10-25
- **Last verified:** 2026-10-28 (`0cdb4ff6`; collated from changes/5744-automerge-arm-guard.md)
- **Summary:** Route repo-owned agent auto-merge instructions through automerge_guard.py and hold Blocked: PRs.
- **Next step:** Shipped in PR #5751.

### DL-#5775 · Add Cursor Cloud Environment.Json Fleet Bootstrap via Repository_Management Scripts/Cursor/Bootstrap.Sh.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5775
- **Branch:** merged via #5776
- **PR:** #5776
- **Paths:** see #5776
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`0204fc2f`; collated from changes/5775-add-cursor-cloud-environment-json-fleet.md)
- **Summary:** Add Cursor Cloud environment.json fleet bootstrap via Repository_Management scripts/cursor/bootstrap.sh.
- **Next step:** Shipped in PR #5776.

### DL-#5761 · Gitleaks Allowlist for Impact-Acoustics Evidence JSON; Public Path Sanitation Guard and Scrub in Guarded Tree.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5761
- **Branch:** cursor/security-cleanup-scrub-c049
- **PR:** #5761
- **Paths:** see #5761
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`434ec4b9`; collated from changes/5761-gitleaks-allowlist-for-impact-acoustics.md)
- **Summary:** Gitleaks allowlist for impact-acoustics evidence JSON; public path sanitation guard and scrub in guarded tree.
- **Next step:** Expand SCAN_PREFIXES after broader docs and workflow scrub.

### DL-#5727 · Wire Antigravity (Agy) CLI Adapter to Nonce-Bound Tool Bridge and Add AgenticAppHost Protocol.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5727
- **Branch:** merged via #5785
- **PR:** #5785
- **Paths:** see #5785
- **Started:** 2026-10-25
- **Last verified:** 2026-10-25 (`c5abe04b`; collated from changes/5727-wire-antigravity-agy-cli-adapter-to-nonc.md)
- **Summary:** Wire Antigravity (agy) CLI adapter to nonce-bound tool bridge and add AgenticAppHost protocol.
- **Next step:** Shipped in PR #5785.

### DL-#5767 · Do Not Delete In-Flight Transcriber Audio When Constructing Another VoiceTranscriber

- **State:** shipped
- **Owner:** codex
- **Issue:** #5767
- **Branch:** codex/5767-voice-temp
- **PR:** #5768
- **Paths:** src/shared/python/sidekick/voice.py,tests/unit/sidekick/test_voice_transcriber.py,manuals/tools/manifests/module-inventory/entries-src-shared-python-sidekick.json,manuals/tools/manifests/module-inventory.json
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`ac1a060d`; collated from changes/5767-do-not-delete-in-flight-transcriber-audi.md)
- **Summary:** Do not delete in-flight transcriber audio when constructing another VoiceTranscriber
- **Next step:** Review and merge after canonical voice regression gates pass

### DL-#5723 · Restore the Public KnowledgePack FTS Query Sanitizer

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5723
- **Branch:** codex/5723-fts-query-sanitizer
- **PR:** #5724
- **Paths:** see #5724
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`160c9c78`; collated from changes/5723-restore-the-public-knowledgepack-fts-que.md)
- **Summary:** Restore the public KnowledgePack FTS query sanitizer
- **Next step:** Merge the sanitizer compatibility PR.

### DL-#5737 · Isolate Compatibility Shim Tests From Global Import State

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5737
- **Branch:** merged via #5745
- **PR:** #5745
- **Paths:** see #5745
- **Started:** 2026-10-24
- **Last verified:** 2026-10-24 (`9fb27a4c`; collated from changes/5737-isolate-compatibility-shim-tests-from-gl.md)
- **Summary:** Isolate compatibility shim tests from global import state
- **Next step:** Shipped in PR #5745.

### DL-#5725 · Security-Scan Contract for the Fleet Scanner

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5725
- **Branch:** claude/oss-scanner-contract
- **PR:** #5726
- **Paths:** .oss-scanner/project.yaml,.oss-scanner/Dockerfile,.oss-scanner/threat_model.md
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`37332d91`; collated from changes/5725-add-a-oss-scanner-security-scan-contract.md)
- **Summary:** Add a .oss-scanner security-scan contract (project.yaml, offline audit Dockerfile, threat model) for the fleet scanner pilot.
- **Next step:** Shipped in PR #5726.

### DL-#5720 · Extract KnowledgePack Search Engine

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5720
- **Branch:** codex/5718-knowledgepack-repo-filter
- **PR:** #5719
- **Paths:** src/shared/python/ai/knowledge/pack.py,src/shared/python/ai/knowledge/search.py,tests/api_baselines/knowledge_api_baseline.json,manuals/tools/manifests/module-inventory.json,manuals/tools/manifests/module-inventory/entries-src-shared-python-ai.json
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`c913821b`; collated from changes/5720-extract-knowledgepack-search-engine.md)
- **Summary:** Extract KnowledgePack BM25 and hybrid ranking into a private search module without changing the public API.
- **Next step:** Shipped in PR #5719.

### DL-#5718 · Filter KnowledgePack Search by Repository Before Ranking Limits

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5718
- **Branch:** codex/5718-knowledgepack-repo-filter
- **PR:** #5722
- **Paths:** see #5722
- **Started:** 2026-10-23
- **Last verified:** 2026-10-23 (`6f064e60`; collated from changes/5718-filter-knowledgepack-search-by-repositor.md)
- **Summary:** Filter KnowledgePack search by repository before ranking limits
- **Next step:** Merge the filtered search API.

### DL-#5712 · Retire Unscoped Shared Toolcache Repair

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5712
- **Branch:** merged via #5715
- **PR:** #5715
- **Paths:** see #5715
- **Started:** 2026-10-22
- **Last verified:** 2026-10-22 (`00e29dcc`; collated from changes/5712-retire-unscoped-shared-toolcache-repair.md)
- **Summary:** retire unscoped shared toolcache repair
- **Next step:** Shipped in PR #5715.

### DL-#5708 · Require Patched Pypdf 6.21.1 in PDF Dependency Declarations

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5708
- **Branch:** merged via #5709
- **PR:** #5709
- **Paths:** see #5709
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`a258ac47`; collated from changes/5708-require-patched-pypdf.md)
- **Summary:** Require patched pypdf 6.21.1 in PDF dependency declarations
- **Next step:** Shipped in PR #5709.

### DL-#5695 · Chore(Ci): RM-CI-000 Runner Load Measurement Script Snapshot and Scheduled-Workflow Follow-Ups (#5695)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5695
- **Branch:** cursor/measure-schedule-utility-share-3893
- **PR:** #5706
- **Paths:** see #5706
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`fbef9621`; collated from changes/5695-chore-ci-rm-ci-000-runner-load-measureme.md)
- **Summary:** chore(ci): RM-CI-000 runner load measurement script snapshot and scheduled-workflow follow-ups (#5695)
- **Next step:** Merge PR #5706 after CI green and arm auto-merge.

### DL-#5686 · Accept Valid Change Fragments in the SPEC Freshness Hook

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5686
- **Branch:** fix/native-wheel-matrix-5685
- **PR:** #5689
- **Paths:** see #5689
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`04a8ee88`; collated from changes/5686-accept-valid-change-fragments-in-the-spe.md)
- **Summary:** Accept valid change fragments in the SPEC freshness hook
- **Next step:** Merge the fragment-hook conformance fix with issue 5685.

### DL-#5685 · Deduplicate Native Linux Ai_Backend Wheels With Bounded Fail-Closed Smoke Selection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5685
- **Branch:** fix/native-wheel-matrix-5685
- **PR:** #5689
- **Paths:** see #5689
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`04a8ee88`; collated from changes/5685-remove-duplicate-native-linux-ai-backend.md)
- **Summary:** Deduplicate native Linux ai_backend wheels with bounded fail-closed smoke selection
- **Next step:** Merge PR after required checks pass.

### DL-#5680 · Initialize and Retain the ORT Environment Required by Local Embeddings

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5680
- **Branch:** merged via #5683
- **PR:** #5683
- **Paths:** see #5683
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`2669f2e0`; collated from changes/5680-fix-local-embeddings-environment.md)
- **Summary:** Initialize and retain the ORT environment required by local embeddings
- **Next step:** Shipped in PR #5683.

### DL-#5675 · Apply Prettier to Files Already Drifted on Main; Ignore Generator-Owned Outputs and LaTeX Docs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5675
- **Branch:** merged via #5676
- **PR:** #5676
- **Paths:** see #5676
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`3229d840`; collated from changes/5675-apply-prettier-to-files-already-drifted.md)
- **Summary:** Apply prettier to files already drifted on main; ignore generator-owned outputs and LaTeX docs
- **Next step:** Shipped in PR #5676.

### DL-#5672 · Coalesce Chat Scroll-to-Bottom Into One Named Timer and Stop It When the Dock Closes, so a Pending Scroll Cannot Run Qt Work Mid-Teardown (CI Standard Segfault in Scroll_Now)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5672
- **Branch:** fix/tools-5672-owned-scroll-segv
- **PR:** #5674
- **Paths:** see #5674
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`37a7477e`; collated from changes/5672-chat-scroll-timer-close.md)
- **Summary:** Coalesce chat scroll-to-bottom into one named timer and stop it when the dock closes, so a pending scroll cannot run Qt work mid-teardown (CI Standard segfault in scroll_now)
- **Next step:** Merge the PR.

### DL-#5671 · Promote Active Dl_State to Shipped When Collating Merged Fragments

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5671
- **Branch:** merged via #5673
- **PR:** #5673
- **Paths:** see #5673
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`303dcba7`; collated from changes/5671-promote-active-dl-state-to-shipped-when.md)
- **Summary:** Promote active dl_state to shipped when collating merged fragments
- **Next step:** Shipped in PR #5673.

### DL-#5669 · Parent Leaked QTimer.singleShot UI-Reset Timers to Their Widgets via Shared.Python.Ui.Owned_Timer (Hover Copy, Assistant Copy, Data-Processor Status, Chat Scroll, Unit Converter)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5669
- **Branch:** fix/leaked-singleshot-ui-timers
- **PR:** #5670
- **Paths:** see #5670
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`5ac74834`; collated from changes/5669-owned-single-shot-timers.md)
- **Summary:** Parent leaked QTimer.singleShot UI-reset timers to their widgets via shared.python.ui.owned_timer (hover copy, assistant copy, data-processor status, chat scroll, unit converter)
- **Next step:** Merge the PR.

### DL-#5661 · Full Clean and Jerk: Make_Clean_and_Jerk_Config on a Timed via Schedule at the Standard's Eight Phases and a Hang/Rack/Overhead Phase-Correct Bar Path

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5661
- **Branch:** merged via #5662
- **PR:** #5662
- **Paths:** see #5662
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`0c585b39`; collated from changes/5661-full-clean-and-jerk-make-clean-and-jerk.md)
- **Summary:** Full clean and jerk: make_clean_and_jerk_config on a timed via schedule at the standard's eight phases and a hang/rack/overhead phase-correct bar path
- **Next step:** Shipped in PR #5662.

### DL-#5658 · Stop and Join Realtime-File-Tail Thread When Last Subscriber Unsubscribes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5658
- **Branch:** merged via #5659
- **PR:** #5659
- **Paths:** see #5659
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`53d5e798`; collated from changes/5658-stop-and-join-realtime-file-tail-thread.md)
- **Summary:** Stop and join realtime-file-tail thread when last subscriber unsubscribes
- **Next step:** Shipped in PR #5659.

### DL-#5652 · Snatch Gets a Pull Phase: Ordered Timed via Points (ViaSchedule) and a Phase-Correct Bar Path

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5652
- **Branch:** merged via #5653
- **PR:** #5653
- **Paths:** see #5653
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`a4595051`; collated from changes/5652-snatch-gets-a-pull-phase-ordered-timed-v.md)
- **Summary:** Snatch gets a pull phase: ordered timed via points (ViaSchedule) and a phase-correct bar path
- **Next step:** Shipped in PR #5653.

### DL-#5645 · Import Ezc3d Lazily on First C3D Read and Add Availability Check to Prevent Windows DLL Conflicts With OpenSim

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5645
- **Branch:** merged via #5654
- **PR:** #5654
- **Paths:** see #5654
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`32ff75b8`; collated from changes/5645-import-ezc3d-lazily-on-first-c3d-read-an.md)
- **Summary:** Import ezc3d lazily on first C3D read and add availability check to prevent Windows DLL conflicts with OpenSim
- **Next step:** Shipped in PR #5654.

### DL-#5638 · Rust Golfer Solver Is Now the Same Solver as Python: Closure Equations, INDEPENDENT_CLOSURE_ROWS=[0,1] (Exposed to Python), Zero Multipliers on Rows 2-3, Typed Rank-Loss Error (ValueError), Analytical Jacobian, 2\*Alpha/Beta^2 Baumgarte, Hub/Gravity/Coriolis Parity; Adds Rust-vs-Python Parity Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5638
- **Branch:** merged via #5644
- **PR:** #5644
- **Paths:** see #5644
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`46c78180`; collated from changes/5638-rust-golfer-solver-is-now-the-same-solve.md)
- **Summary:** Rust golfer solver is now the same solver as Python: closure equations, INDEPENDENT_CLOSURE_ROWS=[0,1] (exposed to Python), zero multipliers on rows 2-3, typed rank-loss error (ValueError), analytical Jacobian, 2\*alpha/beta^2 Baumgarte, hub/gravity/Coriolis parity; adds Rust-vs-Python parity tests
- **Next step:** Shipped in PR #5644.

### DL-#5640 · Chore: Remove Committed Clutter (Unused Root Scripts, Empty Issue Template)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5640
- **Branch:** merged via #5639
- **PR:** #5639, #5664
- **Paths:** see #5639
- **Started:** 2026-10-16
- **Last verified:** 2026-10-20 (`91e41ee1`; collated from changes/5640-golfer-coriolis-christoffel.md)
- **Summary:** chore: remove committed clutter (unused root scripts, empty issue template)
- **Next step:** Merge the PR; resync the UpstreamDrift child copy with the next vendor/ud-tools bump.

### DL-#5635 · Golfer Constraint Solver Solves the KKT System on INDEPENDENT_CLOSURE_ROWS=(0,1) Only (Min-Norm Lstsq), Zero Multipliers on Redundant Rows 2-3, Raises ValueError on Rank Loss; GolferParams.G Default Kept at 9.81 Pending Owner Decision and Now Rejects Non-Finite G

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5635
- **Branch:** merged via #5637
- **PR:** #5637, #5668, #5688
- **Paths:** see #5637
- **Started:** 2026-10-16
- **Last verified:** 2026-10-20 (`d520de5a`; collated from changes/5635-force-source-standard-gravity.md)
- **Summary:** Golfer constraint solver solves the KKT system on INDEPENDENT_CLOSURE_ROWS=(0,1) only (min-norm lstsq), zero multipliers on redundant rows 2-3, raises ValueError on rank loss; GolferParams.g default kept at 9.81 pending owner decision and now rejects non-finite g
- **Next step:** Merge the PR; UpstreamDrift needs no change (its default is already 9.80665).

### DL-#5630 · Label Movement_Optimizer.Spine_Loads as a Planar Ergonomic L5/S1 Estimate, Document UD ADR 0055 Sign Conventions, Add MODEL_METADATA and to_Adr0055_Ap_Shear

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5630
- **Branch:** merged via #5634
- **PR:** #5634
- **Paths:** see #5634
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`432c5bcb`; collated from changes/5630-spine-loads-planar-label.md)
- **Summary:** Label movement_optimizer.spine_loads as a planar ergonomic L5/S1 estimate, document UD ADR 0055 sign conventions, add MODEL_METADATA and to_adr0055_ap_shear
- **Next step:** Shipped in PR #5634.

### DL-#5632 · Security Dependency Upgrades: Patch/Minor Lockfile Bumps for Open Dependabot Alerts in Uv.Lock and 12 Npm Lockfiles (Major-Only Fixes Deferred)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5632
- **Branch:** merged via #5632
- **PR:** #5632
- **Paths:** see #5632
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`99d00cbc`; collated from changes/5632-security-dependency-upgrades-patch-minor.md)
- **Summary:** Security dependency upgrades: patch/minor lockfile bumps for open Dependabot alerts in uv.lock and 12 npm lockfiles (major-only fixes deferred)
- **Next step:** Shipped in PR #5632.

### DL-#5626 · Fix Python/JAX Golfer Forward-Kinematics Club Y Sign and Club Jacobians to Match the Rust Kernel; Add Central-Difference and Shaft-Direction Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5626
- **Branch:** merged via #5628
- **PR:** #5628, #5627
- **Paths:** see #5628
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`e3cf5a43`; collated from changes/5626-fix-pendulum-core-golfer-club-jacobians.md)
- **Summary:** Fix Python/JAX golfer forward-kinematics club y sign and club Jacobians to match the Rust kernel; add central-difference and shaft-direction tests
- **Next step:** Shipped in PR #5628.

### DL-#33 · Consume UpstreamDrift Ud.Putt_Scene/1 Export via Mirror Pattern

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #33
- **Branch:** merged via #5622
- **PR:** #5622
- **Paths:** see #5622
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e27f1a5d`; collated from changes/33-consume-upstreamdrift-ud-putt-scene-1-ex.md)
- **Summary:** consume UpstreamDrift ud.putt_scene/1 export via mirror pattern
- **Next step:** Shipped in PR #5622.

### DL-#5617 · Repair the 5609 Fragment and Validate Pending Fragments in Unit Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5617
- **Branch:** merged via #5618
- **PR:** #5618
- **Paths:** see #5618
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`07414813`; collated from changes/5617-repair-the-5609-fragment-and-validate-pe.md)
- **Summary:** Repair the 5609 fragment and validate pending fragments in unit tests
- **Next step:** Shipped in PR #5618.

### DL-#5615 · Test Impact_Interval Time Cap During Restitution and Timestep Refinement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5615
- **Branch:** merged via #5616
- **PR:** #5616
- **Paths:** see #5616
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e51ed398`; collated from changes/5615-test-impact-interval-time-cap-during-res.md)
- **Summary:** Test impact_interval time cap during restitution and timestep refinement
- **Next step:** Shipped in PR #5616.

### DL-#5609 · Refresh Detect-Secrets Baseline for SCADA F-Matrix Test SHAs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5609
- **Branch:** merged via #5611
- **PR:** #5611
- **Paths:** see #5611
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2d680ea2`; collated from changes/5609-refresh-detect-secrets-baseline.md)
- **Summary:** Refresh detect-secrets baseline for SCADA f-matrix test SHAs
- **Next step:** Shipped in PR #5611.

### DL-#5462 · Enforce Muscle Definition Compatibility and Add Synthetic Same-Input Agreement vs Model Divergence Fixtures

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5462
- **Branch:** merged via #5620
- **PR:** #5620
- **Paths:** see #5620
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5d304135`; collated from changes/5462-enforce-muscle-definition-compatibility.md)
- **Summary:** Enforce muscle definition compatibility and add synthetic same-input agreement vs model divergence fixtures
- **Next step:** Shipped in PR #5620.

### DL-#5446 · Fix Mirrored Wedge Sole Bounce Sign in Wedge Geometry and CAD

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5446
- **Branch:** merged via #5613
- **PR:** #5613
- **Paths:** see #5613
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9b605f30`; collated from changes/5446-fix-mirrored-wedge-sole-bounce-sign-in-w.md)
- **Summary:** Fix mirrored wedge sole bounce sign in wedge geometry and CAD
- **Next step:** Shipped in PR #5613.

### DL-#4260 · \ Add Python Calculation-Runtime-Manifest/V1 Emitter,Cross-Manifest Compatibility Verifier,and Specification\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #4260
- **Branch:** merged via #5614
- **PR:** #5614
- **Paths:** see #5614
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8bcaae41`; collated from changes/4260-add-python-calculation-runtime-manifest.md)
- **Summary:** \ Add Python calculation-runtime-manifest/v1 emitter,cross-manifest compatibility verifier,and specification\
- **Next step:** Shipped in PR #5614.

### DL-#5606 · Allowlist Merged Commit SHAs in test_f_matrix.py for Detect-Secrets

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5606
- **Branch:** merged via #5607
- **PR:** #5607
- **Paths:** see #5607
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`f851bbc7`; collated from changes/5606-allowlist-merged-commit-shas-in-test-f-m.md)
- **Summary:** Allowlist merged commit SHAs in test_f_matrix.py for detect-secrets
- **Next step:** Shipped in PR #5607.

### DL-#5508 · Document and Enforce Coupled PyQt6-Sip 13.12.0 Deferral (#5508, Dependabot #5486)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5508
- **Branch:** merged via #5600
- **PR:** #5600
- **Paths:** see #5600
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`dd834170`; collated from changes/5508-document-and-enforce-coupled-pyqt6-sip-1.md)
- **Summary:** Document and enforce coupled PyQt6-sip 13.12.0 deferral (#5508, Dependabot #5486)
- **Next step:** Shipped in PR #5600.

### DL-#4089 · SCADA F-Matrix Rows F01/F03/F04/F08/F10/F16 Cite Their Merged Slices (PRs #5551, #5564, #5557, #5560, #5559, #5563) and Stay Partial With Refreshed Gaps

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #4089
- **Branch:** merged via #5602
- **PR:** #5602
- **Paths:** see #5602
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`840da341`; collated from changes/4089-scada-f-matrix-rows-f01-f03-f04-f08-f10.md)
- **Summary:** SCADA F-matrix rows F01/F03/F04/F08/F10/F16 cite their merged slices (PRs #5551, #5564, #5557, #5560, #5559, #5563) and stay partial with refreshed gaps
- **Next step:** Shipped in PR #5602.

### DL-#5595 · Split C3DDataReader.Force_Plate_Dataframe Into Private Helpers (Pure Refactor, Adds Characterization Tests)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5595
- **Branch:** merged via #5596
- **PR:** #5596
- **Paths:** see #5596
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`f7440f18`; collated from changes/5595-split-c3ddatareader-force-plate-datafram.md)
- **Summary:** Split C3DDataReader.force_plate_dataframe into private helpers (pure refactor, adds characterization tests)
- **Next step:** Shipped in PR #5596.

### DL-#5542 · Historian H3: TimescaleDB Schema Migrations for Hypertable Compression Continuous Aggregates and Retention

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5542
- **Branch:** merged via #5588
- **PR:** #5588
- **Paths:** see #5588
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`7c44b38f`; collated from changes/5542-historian-h3-timescaledb-schema-migratio.md)
- **Summary:** Historian H3: TimescaleDB schema migrations for hypertable compression continuous aggregates and retention
- **Next step:** Shipped in PR #5588.

### DL-#5529 · SCADA F08: /Api/Trends Gains Additive Quality Array and CSV Export Appends Quality Column (Unknown for NULL/Unrecognised)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5529
- **Branch:** merged via #5560
- **PR:** #5560
- **Paths:** see #5560
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`649d7106`; collated from changes/5529-scada-f08-api-trends-gains-additive-qual.md)
- **Summary:** SCADA F08: /api/trends gains additive quality array and CSV export appends Quality column (unknown for NULL/unrecognised)
- **Next step:** Shipped in PR #5560.

### DL-#5533 · SCADA F16: Advisory-Only AdvisoryResult Provenance Record (Model/Inputs Hash, Window, Constraints, Solver Status) on /Api/Mpc/Simulate, With Import-Guard Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5533
- **Branch:** merged via #5563
- **PR:** #5563
- **Paths:** see #5563
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`4bf32727`; collated from changes/5533-scada-f16-advisory-only-advisoryresult-p.md)
- **Summary:** SCADA F16: advisory-only AdvisoryResult provenance record (model/inputs hash, window, constraints, solver status) on /api/mpc/simulate, with import-guard test
- **Next step:** Shipped in PR #5563.

### DL-#5526 · SCADA F03: Per-Tag Alarm Priority Override (Validated WARNING/TRIP Enum on the Interlock Config, Persisted via the Routing Config Store; BadQuality Floor; Fails Closed to Trip)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5526
- **Branch:** merged via #5564
- **PR:** #5564
- **Paths:** see #5564
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`58002d40`; collated from changes/5526-scada-f03-per-tag-alarm-priority-overrid.md)
- **Summary:** SCADA F03: per-tag alarm priority override (validated WARNING/TRIP enum on the interlock config, persisted via the routing config store; BadQuality floor; fails closed to trip)
- **Next step:** Shipped in PR #5564.

### DL-#12065 · Route Sidekick C3d_Reader Ezc3d Import Through Shared Native-DLL Guard (UpstreamDrift #12065 PR3 Step 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12065
- **Branch:** merged via #5581
- **PR:** #5581
- **Paths:** see #5581
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`bdc3eed0`; collated from changes/12065-route-sidekick-c3d-reader-ezc3d-import-t.md)
- **Summary:** Route sidekick c3d_reader ezc3d import through shared native-DLL guard (UpstreamDrift #12065 PR3 step 1)
- **Next step:** Shipped in PR #5581.

### DL-#5524 · SCADA F01: Audit Rows Carry Reason, Redacted Before/After and Config_Revision Columns (Additive Migration); Alarm Ack Records Credential-Derived Actor Instead of the Free-Text User

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5524
- **Branch:** merged via #5551
- **PR:** #5551
- **Paths:** see #5551
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`b6e3e670`; collated from changes/5524-scada-f01-audit-rows-carry-reason-redact.md)
- **Summary:** SCADA F01: audit rows carry reason, redacted before/after and config_revision columns (additive migration); alarm ack records credential-derived actor instead of the free-text user
- **Next step:** Shipped in PR #5551.

### DL-#5527 · SCADA F04: Append-Only ConfigRevision History (SHA-256 Canonical JSON, Credential Fingerprint), Atomic With PersistedConfig Upsert, Admin-Gated List/Get Endpoints, Revision_Id on Config Write

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5527
- **Branch:** merged via #5557
- **PR:** #5557
- **Paths:** see #5557
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`2b528052`; collated from changes/5527-scada-f04-append-only-configrevision-his.md)
- **Summary:** SCADA F04: append-only ConfigRevision history (SHA-256 canonical JSON, credential fingerprint), atomic with PersistedConfig upsert, admin-gated list/get endpoints, revision_id on config write
- **Next step:** Shipped in PR #5557.

### DL-#5522 · Pendulum_Simulator: Base_Force (Python + TS) Is the Shoulder Reaction of Net_Joint_Forces: Pivot Holds Weight up (+Mg), Arm M1 at the Wrist; Archived Base_Force_Y / Control_Vector Outputs Need Re-Run Under #5523

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5522
- **Branch:** merged via #5568
- **PR:** #5568
- **Paths:** see #5568
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`cd6d2cce`; collated from changes/5522-pendulum-simulator-base-force-python-ts.md)
- **Summary:** pendulum_simulator: base_force (Python + TS) is the shoulder reaction of net_joint_forces: pivot holds weight up (+mg), arm m1 at the wrist; archived base_force_y / control_vector outputs need re-run under #5523
- **Next step:** Shipped in PR #5568.

### DL-#5431 · Pendulum_Simulator: Distributed Club Convention (Uniform Shaft M2 + Head Point Mass) Canonical in Dynamics, Forces, Energy, Transfer Diagnostics, Rust and Web; Presets Retuned to the Real Driver Split

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5431
- **Branch:** merged via #5554
- **PR:** #5554
- **Paths:** see #5554
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`db5318ea`; collated from changes/5431-pendulum-simulator-distributed-club-conv.md)
- **Summary:** pendulum_simulator: distributed club convention (uniform shaft m2 + head point mass) canonical in dynamics, forces, energy, transfer diagnostics, Rust and web; presets retuned to the real driver split
- **Next step:** Shipped in PR #5554.

### DL-#5532 · SCADA F11: PLC Driver Registry (Register_Driver/Capabilities) Replaces the PLCFactory if/Elif; Broken Registered Drivers Raise Instead of Falling Back to the Simulator

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5532
- **Branch:** merged via #5558
- **PR:** #5558
- **Paths:** see #5558
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`f3c358ac`; collated from changes/5532-scada-f11-plc-driver-registry-register-d.md)
- **Summary:** SCADA F11: PLC driver registry (register_driver/capabilities) replaces the PLCFactory if/elif; broken registered drivers raise instead of falling back to the simulator
- **Next step:** Shipped in PR #5558.

### DL-#5525 · SCADA F02: HMI DataAgeIndicator Renders the Scan Data_Source (Held/Fault/Simulated/Unknown) Instead of Inferring CONNECTED From Frame Age

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5525
- **Branch:** merged via #5561
- **PR:** #5561
- **Paths:** see #5561
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`0f3bcaad`; collated from changes/5525-scada-f02-hmi-dataageindicator-renders-t.md)
- **Summary:** SCADA F02: HMI DataAgeIndicator renders the scan data_source (held/fault/simulated/unknown) instead of inferring CONNECTED from frame age
- **Next step:** Shipped in PR #5561.

### DL-#5520 · Connect Calculator_State_Mixin Signals to Bound Methods so Calculators Are Freed by Refcount, Not Only Cyclic GC

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5520
- **Branch:** merged via #5552
- **PR:** #5552
- **Paths:** see #5552
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`24bb0c38`; collated from changes/5520-calculator-gc-cycles.md)
- **Summary:** Connect calculator_state_mixin signals to bound methods so calculators are freed by refcount, not only cyclic GC
- **Next step:** Shipped in PR #5552.

### DL-#5515 · Movement_Optimizer: BodyModel Uses Shared Standard Gravity (9.80665), Add DbC Accessors (Torque_Joint_Count, Has_Com_Samples, Max_Abs_Swing_Angle_rad, Frame_Count) and an Injectable Force-Arrow Renderer Hook on Draw_Force_Arrows (UD Parity)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5515
- **Branch:** merged via #5555
- **PR:** #5555
- **Paths:** see #5555
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`55c081d1`; collated from changes/5515-movement-optimizer-bodymodel-uses-shared.md)
- **Summary:** movement_optimizer: BodyModel uses shared standard gravity (9.80665), add DbC accessors (torque_joint_count, has_com_samples, max_abs_swing_angle_rad, frame_count) and an injectable force-arrow renderer hook on draw_force_arrows (UD parity)
- **Next step:** Shipped in PR #5555.

### DL-#5518 · Fix(Ci): Requalify Libfreetype6 2.13.2+Dfsg-1Ubuntu0.2 for the Rate PyQt Font-Stack Guard; All 10 PyQt References Pass Unchanged Under Existing Tolerances (Closes #5518, Closes #5501)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5518
- **Branch:** merged via #5549
- **PR:** #5549
- **Paths:** see #5549
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8d4201fc`; collated from changes/5518-fix-ci-requalify-libfreetype6-2-13-2-dfs.md)
- **Summary:** fix(ci): requalify libfreetype6 2.13.2+dfsg-1ubuntu0.2 for the Rate PyQt font-stack guard; all 10 PyQt references pass unchanged under existing tolerances (Closes #5518, Closes #5501)
- **Next step:** Shipped in PR #5549.

### DL-#1917 · Sync the Agent-Identity Managed Block and Vendored automerge_guard.py: The Owner's Personal Token Is an Approved Agent Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1917
- **Branch:** merged via #5514
- **PR:** #5514
- **Paths:** see #5514
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`9bfe68b4`; collated from changes/1917-sync-the-agent-identity-managed-block-an.md)
- **Summary:** Sync the agent-identity managed block and vendored automerge_guard.py: the owner's personal token is an approved agent identity
- **Next step:** Shipped in PR #5514.

### DL-#5516 · Install Pyserial Alongside Pymodbus so Pymodbus 3.16.0 Imports in CI and the P1am/Test Extras

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5516
- **Branch:** merged via #5517
- **PR:** #5517
- **Paths:** see #5517
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`880ebffb`; collated from changes/5516-install-pyserial-alongside-pymodbus-so-p.md)
- **Summary:** Install pyserial alongside pymodbus so pymodbus 3.16.0 imports in CI and the p1am/test extras
- **Next step:** Shipped in PR #5517.

### DL-#2064 · Bootstrap Rustup With --No-Modify-Path Before Dtolnay/Rust-Toolchain so CI Stops Appending CARGO_HOME Env Lines to Runner Shell Profiles (Repository_Management#2064)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2064
- **Branch:** merged via #5505
- **PR:** #5505
- **Paths:** see #5505
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`7a453259`; collated from changes/2064-bootstrap-rustup-with-no-modify-path-bef.md)
- **Summary:** Bootstrap rustup with --no-modify-path before dtolnay/rust-toolchain so CI stops appending CARGO_HOME env lines to runner shell profiles (Repository_Management#2064)
- **Next step:** Shipped in PR #5505.

### DL-#5506 · Dependabot: Group Cargo and Pip Minor/Patch Updates Into One Weekly PR per Ecosystem

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5506
- **Branch:** merged via #5507
- **PR:** #5507
- **Paths:** see #5507
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`74030f4a`; collated from changes/5506-dependabot-group-cargo-and-pip-minor-pat.md)
- **Summary:** dependabot: group cargo and pip minor/patch updates into one weekly PR per ecosystem
- **Next step:** Shipped in PR #5507.

### DL-#5456 · CI: Isolate RUSTUP_HOME/CARGO_HOME per Workspace for Every Fleet Rust Job (RM#2021)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #5456
- **Branch:** merged via #5457
- **PR:** #5457
- **Paths:** .github/workflows,.github/actions/isolate-rust-homes,tests/ops
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`6e824112`; collated from changes/5456-ci-isolate-rustup-home-cargo-home-per-wo.md)
- **Summary:** CI: isolate RUSTUP_HOME/CARGO_HOME per workspace for every fleet Rust job (RM#2021)
- **Next step:** Shipped in PR #5457.

### DL-#2018 · Vendor Automerge_Guard and Requeue_Stalled_Merges (Stalled Merge-Queue Requeue) From Repository_Management

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2018
- **Branch:** merged via #5449
- **PR:** #5449, #5453
- **Paths:** see #5449
- **Started:** 2026-10-05
- **Last verified:** 2026-10-06 (`863cb3fb`; collated from changes/2018-resync-requeue.md)
- **Summary:** Vendor automerge_guard and requeue_stalled_merges (stalled merge-queue requeue) from Repository_Management
- **Next step:** Shipped in PR #5449.

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Quarantined Historical Records

These terminal records remain unverified. Their exact original bytes and decisions are preserved in `docs/development/provenance/tools-devlog-migration-receipt.json`; quarantine does not qualify their features or change issue status.

- DL-#5360 (original line 320): unresolved historical record.
- DL-#5346 (original line 346): unresolved historical record.
- DL-#5345 (original line 359): unresolved historical record.
- DL-#1755 (original line 372): unresolved historical record.
- DL-#5333 (original line 385): unresolved historical record.
- DL-#5315 (original line 424): unresolved historical record.
- DL-#5299 (original line 437): unresolved historical record.
- DL-#5297 (original line 450): unresolved historical record.
- DL-#5301 (original line 463): unresolved historical record.
- DL-#5303 (original line 489): unresolved historical record.
- DL-#4951 (original line 502): unresolved historical record.
- DL-#4946 (original line 515): unresolved historical record.
- DL-#4253 (original line 528): unresolved historical record.
- DL-#4225 (original line 541): unresolved historical record.
- DL-#4252 (original line 554): unresolved historical record.
- DL-#4162 (original line 567): unresolved historical record.
- DL-#4186 (original line 580): unresolved historical record.
- DL-#4918 (original line 593): unresolved historical record.
- DL-#5072 (original line 606): unresolved historical record.
- DL-#4922 (original line 619): unresolved historical record.
- DL-#4382 (original line 632): unresolved historical record.
- DL-#4300 (original line 645): unresolved historical record.
- DL-#5223 (original line 658): unresolved historical record.
- DL-#5073 (original line 671): unresolved historical record.
- DL-#5074 (original line 684): unresolved historical record.
- DL-#4220 (original line 697): unresolved historical record.
- DL-#5218 (original line 710): unresolved historical record.
- DL-#1614 (original line 723): unresolved historical record.
- DL-#5168 (original line 749): unresolved historical record.
- DL-#5157 (original line 762): unresolved historical record.
- DL-#5155 (original line 775): unresolved historical record.
- DL-#5153 (original line 788): unresolved historical record.
- DL-#5151 (original line 802): unresolved historical record.
- DL-#5145 (original line 816): unresolved historical record.
- DL-#5147 (original line 830): unresolved historical record.
- DL-#5073 (original line 844): unresolved historical record.
- DL-#5138 (original line 860): unresolved historical record.
- DL-#5101 (original line 914): unresolved historical record.
- DL-#5074 (original line 927): unresolved historical record.
- DL-#5095 (original line 940): unresolved historical record.
- DL-#5062 (original line 953): unresolved historical record.
- DL-#8942 (original line 976): unresolved historical record.
- DL-#4130 (original line 995): unresolved historical record.
- DL-0054 (original line 1017): unresolved historical record.
- DL-0054 (original line 1067): unresolved historical record.
- DL-#5223 (original line 2129): unresolved historical record.
