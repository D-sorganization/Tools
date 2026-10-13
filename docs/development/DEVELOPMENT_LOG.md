# Development Log — Tools

State table for every feature in flight in this repository. Update
entries **in place**; never append dated sections. One entry per
feature, from proposal to ship. See the `development-logs` section of
`AGENTS.md` for the binding rules and
`shared_scripts/development_log.py` for the validator.

- **Portfolio:** infra
- **WIP limit:** 4
- **Last audited:** 2026-09-30 by Luna (Tools#5392 longitudinal residual variance)

### DL-#4149 · Exact golf-club head CAD families (iron family slice)

- **State:** in_review
- **Owner:** claude
- **Issue:** #4149
- **Branch:** `claude/issue-4149-iron`
- **PR:** #5425 (draft)
- **Paths:** `src/shared/python/golf_club/iron_*.py`, `src/shared/python/golf_club/_head_cad.py`, `src/shared/python/golf_club/wedge_cad.py`, `src/shared/python/golf_club/wedge_export.py`, `tests/shared/python/golf_club/test_iron_*.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-05 @ 696b70a74 (base; golf_club API baseline realigned to `Any`) — test_iron_parameters + test_iron_cad pass with build123d 0.13.0; wedge CAD/export tests unchanged and green; ruff and mypy clean.
- **Summary:** Iron family following the wedge precedent: versioned `golf_club.iron_parameters/1` schema with boundary DbC, ruled-loft B-Rep with fused hollow hosel, mass properties recovered from the solid and checked against a closed-form prismatoid and the exported-mesh divergence theorem, STEP/BREP/STL export with `golf_club.iron_export/1` manifest. Shared datum-face, hosel-tube and file-export helpers extracted to `_head_cad.py`. Driver, hybrid and putter families and shape optimisation remain open under #4149.
- **Next step:** owner reviews the iron schema ranges and topology in the draft PR body.

### DL-#5434 · Align Workflow Python Matrices with requires-python

- **State:** in_review
- **Owner:** claude
- **Issue:** #5434
- **Branch:** `fix/5434-matrix-python-floor`
- **PR:** #5435
- **Paths:** `.github/workflows/maturin-file-watcher.yml`, `.github/workflows/maturin-swing-core.yml`, `tests/ops/test_workflow_python_floor.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`d61e4015ecc4f52e9713b636e9840e61bf929e96`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Split `maturin-file-watcher` and `maturin-swing-core` into a `wheel-gate` (build + import, Python 3.10-3.12, proves the crates' >=3.10 claim) and the existing wrapper/parity job (root code, now 3.11-3.12 only, since conftest will not collect it below the root floor and a 3.10 leg exited 5). A policy test fails any workflow leg below the floor of the code it exercises, and any maturin crate not built on its declared floor (`rust_core/tools-core` is a listed known gap).
- **Next step:** Merge the PR once the Maturin workflows are green.

### DL-#5432 · Fail fast on stale module inventory test

- **State:** in_review
- **Owner:** claude
- **Issue:** #5432
- **Branch:** `fix/5432-inventory-fast-fail`
- **PR:** #5433
- **Paths:** `tests/architecture/_inventory_diff.py`, `tests/architecture/test_inventory_diff_5432.py`, `tests/architecture/test_tools_module_inventory_contract.py`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`44a1ac0e3e76220177da9af93f53d4ffcbdbbd3a`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** The inventory freshness test compared two huge dicts with `assert ==`, whose quadratic pytest diff stalled a CI shard for 90 minutes. It now fails with a bounded shard-level summary; strictness is unchanged.
- **Next step:** Merge PR #5433 via auto-merge.

### DL-#4464 · Fork PRs never run on the self-hosted fleet

- **State:** in_review
- **Owner:** claude
- **Issue:** #4464
- **Branch:** `fix/4464-fork-pr-self-hosted`
- **PR:** not created
- **Paths:** `.github/workflows/`, `scripts/fork_pr_runner_guard.py`, `tests/ops/test_fork_pr_runner_guard.py`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`696b70a74739813e05fb07d330889fb8efef91e4`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Workflow-level defence in depth for #4464: every job in a PR-triggered workflow that can reach `d-sorg-fleet` skips fork PRs, the required `tests` lane routes fork PRs to a hosted runner, and a static checker enforces both. Admin settings (fork-PR approval, default token permissions, runner-group access) remain owner-only.
- **Next step:** Owner applies the admin-only settings listed in the PR body, then closes #4464.

### DL-#5429 · Fix Clippy Rust 1.99 Errors in pendulum-core

- **State:** in_review
- **Owner:** claude
- **Issue:** #5429
- **Branch:** `fix/pendulum-core-clippy-f64`
- **PR:** see PR body
- **Paths:** `src/pendulum_simulator/pendulum-core/src/cmaes.rs`, `src/pendulum_simulator/pendulum-core/src/dynamics.rs`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-05
- **Last verified:** 2026-10-12 (`eb4b886ad29af93ab7b22493ad9f4e4fb1bc31d8`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Removed the deprecated `use std::f64;` in `cmaes.rs` (it made `f64::INFINITY`/`NEG_INFINITY` resolve to the deprecated module constants) and replaced constant-size `chunks_exact(2)` with `as_chunks` with const parameter 2 in `dynamics.rs`, so the pendulum_core rust quality gate passes on the fleet's Rust 1.99.
- **Next step:** Merge the PR once the pendulum_core rust quality gate is green.

### DL-#5422 · Fix Clippy Deprecated f64 Constants in tools-core

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #5422
- **Branch:** `fix/issue-5422-clippy-f64-deprecated`
- **PR:** not created
- **Paths:** `rust_core/tools-core/src/rrt.rs`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`57e90bce55e475c890a37f7dba8b53184e845497`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Removed deprecated `use std::f64;` in `rust_core/tools-core/src/rrt.rs` that shadowed primitive `f64` associated constants (`f64::NEG_INFINITY` and `f64::MAX`) causing Clippy `-D warnings` failure in `rust-quality-gate` on CI Standard.
- **Next step:** Open ready PR, arm auto-merge via Repository_Management `scripts/automerge_guard.py`.

### DL-#4433 · Visual-first first-viewport geometry beyond the initial state

- **State:** in_review
- **Owner:** claude
- **Issue:** #4433
- **Branch:** `claude/issue-4433`
- **PR:** #5424 (draft), #5447
- **Paths:** `tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py`, `tests/rate_of_closure/pyqt_launch_monitor_state_probe.py`, `src/rate_of_closure/web/e2e/visualization-tab-visibility.spec.ts`, `docs/audits/rate_of_closure_visual_first_epic_4433.v1.json`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-06 (`8c3cc237`; collated from changes/4433-launch-monitor-analytics-no-longer-scrol.md)
- **Summary:** First tab pair (Launch Monitor Analytics) of the epic's state-by-state geometry gap: the primary scatter stays in the first viewport through result, error, loading and empty on React (3 reference viewports) and PyQt (1.0/1.5 DPI). Other pairs, approved baselines and the AT protocol stay open under the epic.
- **Next step:** Merge the PR; the 1440x900 evidence captures are layout-identical and need no refresh.

### DL-#5417 · ADR-008 dependency and license deny-list check

- **State:** in_review
- **Owner:** claude
- **Issue:** #5417
- **Branch:** `claude/issue-5417`
- **PR:** not created
- **Paths:** `scripts/check_license_denylist.py`, `config/license_denylist.json`, `tests/architecture/test_license_denylist.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`d7cef753e54ce89f30eca336ebe037f8a6b9ab86`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Enforces the ADR-008 licensing rule (no FreeMoCap or SkellyCam, AGPL ids) with a static scan of pyproject.toml and requirements files plus an optional installed-closure license walk. Lists contain only what ADR-008 names. CI wiring is out of scope.
- **Next step:** workflow owner wires `python scripts/check_license_denylist.py` into CI as a separate workflow PR.

### DL-#5416 · Python validator for calculation-runtime-manifest/v1

- **State:** in_review
- **Owner:** claude
- **Issue:** #5416
- **Branch:** `claude/issue-5416`
- **PR:** draft (see issue)
- **Paths:** `src/shared/python/swing_sim/runtime_manifest.py`, `tests/shared/python/test_runtime_manifest_parity_fixture.py`, `docs/specs/active/CALCULATION_RUNTIME_MANIFEST.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-12 (`f729eca97dd26fd4955bcabeb73e7083a47609bd`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** One-for-one Python port of the TypeScript `parseRuntimeManifest`, pinned by the shared `runtime_manifest_parity_v1.json` fixture.
- **Next step:** frontier agent reviews the draft PR before it is marked ready.

### DL-#5392 · Longitudinal regression residual variance

- **State:** in_review
- **Owner:** codex
- **Issue:** #5392
- **Branch:** `fix/longitudinal-residual-variance-20260930`
- **PR:** #5394 (draft)
- **Paths:** `src/rate_of_closure/web/src/model/launchMonitorLongitudinal.ts`, `src/rate_of_closure/web/src/model/launchMonitorLongitudinal.test.ts`, `manuals/tools/manifests/module-inventory/`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-30
- **Last verified:** 2026-09-30 — controlled c096 baseline regression RED (0 vs 0.00020716016510533694), candidate focused GREEN (3/3), and one-worker full web Vitest rerun 2361/2361 across 238 files; type-check, lint, build, and required design-manual checks run. Initial two-worker full run retained its single import-boundary timeout; isolated test passed.
- **Summary:** The merged #5389 optimization retained the covariance/sum-of-squares pass but derived residual variance by subtracting nearly equal aggregates, reporting zero standard error for a representable near-perfect regression with nonzero ordered residuals. Restored the intercept and ordered direct-residual sum of squares, with an independent public-analysis regression test. This is a focused follow-up to merged #5389.
- **Next step:** root reviews the final diff and draft PR/CI before any merge decision.

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked`
reachable from any live state and `abandoned` from `parked`.
`shipped` never returns to `in_progress`; open a new entry instead.

## Active

### DL-#5577 · Normalize Compatible Contract-Level Enums

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #5577
- **Branch:** fix/contract-enum-interop-5577-20261012
- **PR:** #5579
- **Paths:** see #5579
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`9b880daf`; collated from changes/5577-normalize-compatible-contract-level-enum.md)
- **Summary:** Normalize compatible contract-level enums
- **Next step:** Complete scoped checks, review the transaction behavior, and publish the fix through a PR.

### DL-#5548 · Historian H9: Historian Operator/Engineer Documentation, Runbook and ADR

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5548
- **Branch:** docs/5548-historian-adr-runbook
- **PR:** #5583
- **Paths:** see #5583
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`f71ee279`; collated from changes/5548-historian-h9-docs-runbook-adr.md)
- **Summary:** Historian H9: Historian operator/engineer documentation, runbook and ADR
- **Next step:** Merge the PR, flip H9 matrix row to landed citing it.

### DL-#5523 · Pendulum_Simulator: Re-Run Swing_Objectives/Impact_Optimality/Model_Adequacy Results on the Distributed Club (51.4 m/s, 8.02 m/s Hands, 14% Braking; Centrifugal Is the Dissenting Objective) With a Reproducible Script

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5523
- **Branch:** fix/5523-rerun-distributed-club
- **PR:** #5576
- **Paths:** see #5576
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`5c1f1657`; collated from changes/5523-pendulum-simulator-re-run-swing-objectiv.md)
- **Summary:** pendulum_simulator: re-run swing_objectives/impact_optimality/model_adequacy results on the distributed club (51.4 m/s, 8.02 m/s hands, 14% braking; centrifugal is the dissenting objective) with a reproducible script
- **Next step:** Merge the PR; exact-provider qualification keeps results exploratory.

### DL-#5530 · SCADA F09: Declarative Procedure Model With Hold/Abort, Gated to the Simulated Driver

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5530
- **Branch:** feat/5530-scada
- **PR:** #5578, #5580
- **Paths:** see #5578
- **Started:** 2026-10-12
- **Last verified:** 2026-10-13 (`98d0ea6c`; collated from changes/5530-scada-f09-matrix-row-flipped-to-landed-c.md)
- **Summary:** SCADA F09: declarative procedure model with hold/abort, gated to the simulated driver
- **Next step:** Merge the PR; flip F09 matrix row to landed citing it.

### DL-#5531 · SCADA F10: Advisory-Only Flatline and Command/Feedback-Mismatch Detectors (asset_health.py)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5531
- **Branch:** feat/5531-asset-health
- **PR:** #5559
- **Paths:** see #5559
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`86819437`; collated from changes/5531-scada-f10-advisory-only-flatline-and-com.md)
- **Summary:** SCADA F10: advisory-only flatline and command/feedback-mismatch detectors (asset_health.py)
- **Next step:** Merge the PR.

### DL-#5572 · Preserve Signal Toolkit Import Identity

- **State:** in_review
- **Owner:** codex
- **Issue:** #5572
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/__init__.py, tests/shared/python/signal_toolkit/test_import_identity.py, manuals/tools/manifests/module-inventory.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5572-signal-toolkit-import-identity.md)
- **Summary:** Bind Signal toolkit import aliases before package children capture provider classes.
- **Next step:** Complete governed log migration, full pre-PR validation, and protected provider publication.

### DL-#5571 · Preserve Signal Fitting Consumer Contracts

- **State:** in_review
- **Owner:** codex
- **Issue:** #5571
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/fitting.py, tests/shared/python/signal_toolkit/test_fitting_consumer_contracts.py, manuals/tools/manifests/module-inventory.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5571-signal-fitting-consumer-contracts.md)
- **Summary:** Preserve Signal fitting empty-input and failed-fit ndarray identity contracts in the canonical Tools provider.
- **Next step:** Complete governed log migration, full pre-PR validation, and protected provider publication with Signal #5569.

### DL-#5570 · Reconcile Development Log Provenance

- **State:** in_review
- **Owner:** codex
- **Issue:** #5570
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** .secrets.baseline, docs/development/DEVELOPMENT_LOG.md, docs/development/development-log-provenance-migration.md, docs/development/provenance/tools-devlog-migration-plan.json, docs/development/provenance/tools-devlog-metadata-review.md, docs/development/provenance/tools-devlog-migration-receipt.json, shared_scripts/development_log_migration.py
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5570-governed-development-log-provenance.md)
- **Summary:** Reconcile inherited Tools development-log findings through the reviewed, lossless RM provenance migration.
- **Next step:** Resolve PR #5573 CI findings, rerun canonical RM-6, and complete protected merge; separately review stale historical feature states.

### DL-#5569 · Keep Signal Structural Shape Validation Active Under Optimized Python

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5569
- **Branch:** codex/signal-optimized-structural-guards
- **PR:** #5573
- **Paths:** src/shared/python/signal_toolkit/core.py; tests/shared/python/signal_toolkit/test_core_structural_guards.py; manuals/tools/manifests/module-inventory.json; manuals/tools/manifests/module-inventory/entries-src-shared-python-signal-toolkit.json
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`8a11691b`; collated from changes/5569-keep-signal-structural-shape-validation.md)
- **Summary:** Keep Signal structural shape validation active under optimized Python
- **Next step:** Review and merge Tools fix, then repin and mirror the provider in UpstreamDrift #7704

### DL-#5528 · SCADA F07: Publish Firmware Trip Latch Tri-State (Plc_Trip_Latch) and First-Out Tag (First_Out_Tag_Id) From One Register Read; InterlocksPanel Shows PLC Trip and an Explicit Unknown State

- **State:** in_review
- **Owner:** claude
- **Issue:** #5528
- **Branch:** feat/5528-first-out-tag
- **PR:** #5550
- **Paths:** see #5550
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`aa48c2e1`; collated from changes/5528-scada-f07-read-firmware-first-out-trip-t.md)
- **Summary:** SCADA F07: publish firmware trip latch tri-state (plc_trip_latch) and first-out tag (first_out_tag_id) from one register read; InterlocksPanel shows PLC trip and an explicit unknown state
- **Next step:** Review and merge; F07 matrix row stays partial until cited.

### DL-#5474 · Require Matching Versioned Actuator-Profile References for Mixed Command Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5474
- **Branch:** feat/5474-actuator-profile-reference-admission
- **PR:** #5475
- **Paths:** see #5475
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`86d0f28b`; collated from changes/5474-compiled-actuator-profile-evidence.md)
- **Summary:** Require matching versioned actuator-profile references for mixed command evidence
- **Next step:** Resolve and verify profile bytes against native execution in UpstreamDrift #11955

### DL-#5454 · Deterministic 1440X900 Visual-Baseline Candidates: Settle Tab-Pill Transitions Before Capture and Pin Determinism With a Two-Session Test

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5454
- **Branch:** fix/5454-deterministic-visual-baselines
- **PR:** #5481
- **Paths:** see #5481
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`24eede77`; collated from changes/5454-deterministic-1440x900-visual-baseline-c.md)
- **Summary:** Deterministic 1440x900 visual-baseline candidates: settle tab-pill transitions before capture and pin determinism with a two-session test
- **Next step:** Merge the PR.

### DL-#5472 · Repair Canonical Manual Source-Count Regression While Preserving Release Blockers

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5472
- **Branch:** fix/canonical-manual-count-5472
- **PR:** #5473
- **Paths:** see #5473
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`25b7c1fb`; collated from changes/5472-repair-canonical-manual-source-count-reg.md)
- **Summary:** Repair canonical manual source-count regression while preserving release blockers
- **Next step:** Complete PR5473 CI; retain artifact integrity and native provider qualification gates

### DL-#5470 · Add Opaque Native Restart Transport and Strict Replay Envelope With Native-Clock Integrity and Dynamic Complete Manual Page Inspection

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5470
- **Branch:** feat/native-state-artifact-5470
- **PR:** #5471
- **Paths:** see #5471
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`89415ee8`; collated from changes/5470-add-opaque-native-restart-transport-and.md)
- **Summary:** Add opaque native restart transport and strict replay envelope with native-clock integrity and dynamic complete manual page inspection
- **Next step:** Review Tools transport; integrate actual R2025b native restart provider in UpstreamDrift11921

### DL-#5463 · Bound Replay Experiment Resources and Privacy-Safe Preview Artifacts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5463
- **Branch:** feat/5463-bounded-preview-resources
- **PR:** #5468
- **Paths:** src/shared/python/sidekick/lab/mocap,schemas/mocap/preview-manifest-v1.schema.json,tests/shared/python/sidekick/lab/mocap,test_sidekick_public_api_stability.py,docs/development/feedback_controls
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`b1480ff6`; collated from changes/5463-add-bounded-replay-resources-and-privacy.md)
- **Summary:** Add bounded replay resources and privacy-aware preview manifest contracts
- **Next step:** Run targeted mocap, API, inventory and pre-PR validation; then publish T03 PR

### DL-#5460 · Plan Shared Controlled-Matching and Muscle Replay Interchange With Private Evidence Boundaries

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5460
- **Branch:** docs/feedback-controls-5460
- **PR:** #5465
- **Paths:** docs/development/feedback_controls
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`789fd312`; collated from changes/5460-controlled-matching-contracts.md)
- **Summary:** Plan shared controlled-matching and muscle replay interchange with private evidence boundaries
- **Next step:** Start T01 contract schema and synthetic conformance fixture

### DL-#5440 · Drop Faulthandler Timeout That Crashed Tests-Shared

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5440
- **Branch:** fix/5440-drop-faulthandler-timeout
- **PR:** #5443, #5442
- **Paths:** pyproject.toml,tests/architecture/test_faulthandler_timeout_contract.py
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`ead6be5f`; collated from changes/5440-variation-gui-lost-signal.md)
- **Summary:** Remove faulthandler_timeout=50 added by #5442: its stack dump fired inside legitimately long numerical tests and crashed the tests-shared shard on main (exit 249). A contract test keeps any future watchdog at or above the longest timeout marker.
- **Next step:** Merge the PR, then root-cause any future hang from its faulthandler stack dump.

### DL-#5436 · Prove Tools-Core Python Floor in CI

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #5436
- **Branch:** fix/5436-tools-core-wheel-gate
- **PR:** #5438
- **Paths:** .github/workflows/maturin-tools-core.yml,tests/ops/test_workflow_python_floor.py
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`625c4e0c`; collated from changes/5436-ci-tools-core-new-maturin-tools-core-whe.md)
- **Summary:** ci(tools-core): new maturin-tools-core wheel-gate proves the crate's >=3.10 floor on Python 3.10-3.12; floor policy test now has no exemptions
- **Next step:** Merge PR #5438 once the wheel-gate is green on the fleet.

### DL-#1976 · Add Collate-Changes Workflow and Wire Spec-Check to Check_Spec_Freshness

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #1976
- **Branch:** ci/1976-tools-collate-changes
- **PR:** #5448, #5451, #5428
- **Paths:** see #5448
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`ad530938`; collated from changes/1976-tools-change-fragments.md)
- **Summary:** Add collate-changes workflow and wire spec-check to check_spec_freshness
- **Next step:** Merge the pull request.

### DL-#5385 · Propagate flight termination states and gate landing metrics

- **State:** in_review
- **Owner:** local (agy executor)
- **Issue:** #5385
- **Branch:** `fix/tools-flight-termination-5385`
- **PR:** #5391
- **Paths:** `src/shared/python/swing_sim/flight/`, `src/rate_of_closure/application/`, `tests/api_baselines/swing_sim_api_baseline.json`
- **Started:** 2026-09-30
- **Last verified:** 2026-10-12 (`6be5b5b164a64bca8a5c6bedad57f25bd11f7b91`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Propagate explicit FlightTermination enum states (LANDED, TIME_LIMIT, SOLVER_FAILED, CANCELLED) before reporting landing metrics. Gate carry_distance, landing_angle, and lateral_deviation to None on non-landed flights. Fail closed across CenteredClubDeliveryAdapter, ground simulation transfer, and flight execution profiles.
- **Next step:** Push branch, open PR #5385, and link UpstreamDrift #11145.

### DL-#5353 · Pre-impact bundle wire, modal-state record and public validators

- **State:** in_review
- **Owner:** local (agy executor)
- **Issue:** #5353
- **Branch:** `feat/5353-pre-impact-bundle-modal-state`
- **PR:** #5368
- **Paths:** `src/shared/python/golf_club/_pre_impact_contracts.py`, `src/shared/python/golf_club/_pre_impact_serde.py`, `src/shared/python/golf_club/modal_state.py`, `src/shared/python/golf_club/pre_impact_bundle.py`, `src/shared/python/golf_club/pre_impact_frames.py`, `tests/shared/python/golf_club/`, `tests/api_baselines/golf_club_api_baseline.json`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-12 (`b61fc5b5021b34d239b738da1390bda6a67497bd`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Complete implementation of #5353: items 1-2 add the versioned pre-impact bundle wire (`PreImpactBundle`, `Provenance`, `TimeBase`, `HeadState`, `BallState`, `HandWrench`, `Pose`, `Quantity`), Plücker frame transforms with $10^{-12}$ wrench power invariance, and modal-state records with M-orthogonal projection ($0.5 \dot{q}^T M_r \dot{q} + 0.5 q^T K_r q$). Follows items 3-4 (validation and field origin) landed in #5366.
- **Next step:** get CI green on the PR and merge through auto-merge squash.

### DL-#5362 · Input contracts surfaced by the UpstreamDrift shadow burn-down

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #5362
- **Branch:** `claude/tools-5362-shadow-contracts`
- **PR:** #5364 (draft)
- **Paths:** `src/shared/python/signal_toolkit/`, `src/shared/python/model_generation/`, `src/shared/python/ai/gui/session_manager.py`, `src/shared/python/humanoid_character_builder/__init__.py`, `tests/shared/python/signal_toolkit/`, `tests/shared/python/model_generation/`, `tests/unit/ai/gui/test_session_manager_2872.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-12 (`3678409fc51024150ab28970b72e3b468935f345`; log metadata only; feature acceptance unverified; reviewed receipt `docs/development/provenance/tools-devlog-metadata-review.md`)
- **Summary:** Preconditions on signal_toolkit smoothing/saturation/rate-limit/deadband/noise and primitive inertia mass, standard gravity from the shared constant, UTC-normalised chat session timestamps, and a corrected humanoid_character_builder docstring, so UpstreamDrift can retire its quarantined expectations by bumping the pin instead of editing its shadow copies.
- **Next step:** get CI green on the PR and arm it through `automerge_guard.py`.

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
