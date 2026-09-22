# Development Log — Tools

State table for every feature in flight in this repository. Update
entries **in place**; never append dated sections. One entry per
feature, from proposal to ship. See the `development-logs` section of
`AGENTS.md` for the binding rules and
`shared_scripts/development_log.py` for the validator.

- **Portfolio:** infra
- **WIP limit:** 4
- **Last audited:** 2026-08-28 by bootstrap

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked`
reachable from any live state and `abandoned` from `parked`.
`shipped` never returns to `in_progress`; open a new entry instead.

## Active

### DL-#5297 · Measured Grip FRF Qualification Requires Operating-Strain Evidence

- **State:** in_review
- **Owner:** codex
- **Issue:** D-sorganization/Tools#5297 (follow-up to #5072; parent #5068)
- **PR:** D-sorganization/Tools#5298
- **Branch:** `fix/5297-measured-grip-strain`
- **Paths:** `src/shared/python/golf_club/measured_grip_impedance.py`, `tests/shared/python/golf_club/test_measured_grip_impedance.py`, `docs/development/impact-acoustics/MEASURED_GRIP_IMPEDANCE.md`, `tests/api_baselines/golf_club_api_baseline.json`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 (TDD RED reproduced the missing qualification input; 69 measured-grip, boundary, coupled-impact, and gripped-shaft tests pass; mypy, ruff, API stability, module inventory, and all nine manual-governance gates pass.)
- **Summary:** Makes measured-grip FRF certification refuse by default until a caller explicitly declares that its operating-strain assessment passed. Magnitude, phase, and passivity agreement remain necessary but cannot independently establish linear-regime validity.
- **Next step:** Monitor required PR checks, then merge. A physical study still needs traceable strain, geometry, calibration, and source evidence.

### DL-#5301 · Measured-Grip Fixture Provenance and Qualification Boundary

- **State:** in_progress
- **Owner:** codex
- **Issue:** D-sorganization/Tools#5301 (follow-up to #5072; parent #5068)
- **PR:** pending
- **Branch:** `fix/5301-grip-fixture-provenance`
- **Paths:** `src/shared/python/golf_club/measured_grip_impedance.py`, `tests/shared/python/golf_club/test_measured_grip_impedance.py`, `docs/development/impact-acoustics/MEASURED_GRIP_IMPEDANCE.md`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 (TDD RED showed a formula-generated fixture could receive physical FRF qualification; focused measured-grip tests pass after provenance gating.)
- **Summary:** Reclassifies formula-generated fixtures as synthetic and makes source provenance a necessary condition for physical FRF qualification. Calibrated measurement declarations remain separately accepted.
- **Next step:** Rebase against the explicit operating-strain gate, run combined qualification tests and repository checks, then submit the corrective PR.

### DL-#4951 · Historian Licensing Ruling: Grafana AGPLv3 and TimescaleDB TSL

- **State:** in_review
- **Owner:** local
- **Issue:** D-sorganization/Tools#4951
- **PR:** pending
- **Branch:** `docs/4946-4951-architecture-rulings`
- **Paths:** `docs/development/HISTORIAN_LICENSING_RULING.md`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 (local)
- **Summary:** Formally ratify Ruling Option 1 (Internal-Only Deployment) for SCADA historian; classify as internal plant/engineering telemetry system on company infrastructure; unencumber TimescaleDB compression/continuous aggregates and Grafana dashboards; establish strict distribution-isolation invariant against packaging into external consumer bundles; unblock H3/H6 re-land.
- **Next step:** Submit PR closing #4951 and arm auto-merge.

### DL-#4946 · Impact-Interval PyQt Tab Ruling: Drop as Superseded by #4473 Visual-First Tabs

- **State:** in_review
- **Owner:** local
- **Issue:** D-sorganization/Tools#4946
- **PR:** pending
- **Branch:** `docs/4946-4951-architecture-rulings`
- **Paths:** `docs/development/IMPACT_INTERVAL_TAB_RULING.md`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 (local)
- **Summary:** Formally drop proposed standalone impact_interval_view PyQt tab as superseded by #4473 visual-first tab family while preserving headless calculation package src/swing_sim/impact_interval/ as an authoritative computational backend.
- **Next step:** Submit PR closing #4946 and arm auto-merge.

### DL-#4253 · Review R4: Counterfactual Methodology - Sobol, Spearman Significance, Ellipse Normality

- **State:** in_progress
- **Owner:** local
- **Issue:** D-sorganization/Tools#4253 (parent epic #4249)
- **PR:** D-sorganization/Tools#5266
- **Branch:** `feat/4253-counterfactual-sobol-significance`
- **Paths:** `src/shared/python/swing_sim/variation/`, `src/rate_of_closure/`, `tests/rate_of_closure/`
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (all 389 variation Python tests, 2,328 web tests, and 16 glossary tests passing; line budget <= 500 lines enforced; test_auto_complete migrated to qtbot fixture to eliminate xdist crash)
- **Summary:** Implements Sobol first-order and total sensitivity indices via Saltelli sampling with run count guidance, permutation p-values and bootstrap CIs on Spearman correlation matrix with cell suppression/greying, Mardia bivariate normality diagnostic and convex hull fallback for 2D landing dispersion, and input truncation mean-shift analysis with UI notes.
- **Next step:** CI verification and auto-merge.

### DL-#4225 · Workspace View Compositor: Impact, Swing, Flight

- **State:** shipped
- **Owner:** antigravity
- **Issue:** D-sorganization/Tools#4225
- **PR:** #5264
- **Branch:** `feat/4225-workspace-view-compositor`
- **Paths:** `src/rate_of_closure/ui/pyqt6/`, `src/rate_of_closure/web/`, `tests/rate_of_closure/`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all unit/headless tests pass including test_app_toolstrip.py, test_view_compositor_gui.py, test_tooltips.py, and vitest web test suite; pre-commit ruff/black/mypy pass; inventory check clean)
- **Summary:** Multi-view workspace compositor integrating Impact, Swing, and Flight viewports with synchronized playback, run selection, and layout presets across PyQt and React toolstrips.
- **Next step:** Merged into main.

### DL-#4252 · Literature Flight Models Reconciliation, Carry Benchmark, and Wind Documentation

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#4252
- **PR:** #5265
- **Branch:** `feat/4252-reconcile-flight-models`
- **Paths:** `rust_core/tools-core/src/ball_flight.rs`, `src/shared/python/swing_sim/flight/`, `src/rate_of_closure/derivation_flight.py`, `src/rate_of_closure/web/src/model/derivationModels.ts`, `docs/development/flight_model_validation.json`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all 230 flight and derivation unit/parity tests pass; Rust fast path and WaterlooPenner carry parity within 0.07% across all 5 benchmark conditions; lint/format/mypy pass)
- **Summary:** Reconciled Rust core kernel with canonical Waterloo/Penner lift model (power law $C_L = cl_1 \cdot s^{cl_2}$, $s_{decay} = 0.05$, $C_{L,\max} = 0.155$); tightened multi-condition carry parity to < 1.0%; documented spin decay rates across all 7 registered models; added calculation description Step 4 literature comparison & wind boundaries; published open-data validation package with pinned manifest.
- **Next step:** Shipped via PR #5265.

### DL-#4162 · Wedge Delivery Metrics & Linear Waterfall Visualization

- **State:** in_progress
- **Owner:** local
- **Issue:** D-sorganization/Tools#4162 (parent epic #4158)
- **PR:** #5268
- **Branch:** `feat/4162-wedge-delivery-metrics-viz`
- **Paths:** `src/shared/python/golf_club/_wedge_delivery_metrics.py`, `tests/shared/python/golf_club/test_wedge_delivery_metrics.py`, `src/rate_of_closure/ui/impact_kinematics_presentation.py`, `tests/rate_of_closure/test_wedge_ground_clearance_gui.py`, `src/rate_of_closure/web/src/model/wedgeDeliveryMetrics.ts`, `src/rate_of_closure/web/src/model/wedgeDeliveryMetrics.test.ts`, `src/rate_of_closure/web/src/components/WedgeGroundClearancePanel.tsx`, `src/rate_of_closure/web/src/components/WedgeGroundClearancePanel.test.tsx`, `src/rate_of_closure/web/src/components/SimulationDisplay.tsx`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all 21 Python pytest and 2324 TypeScript vitest tests passing; ruff, black, mypy, and eslint passing)
- **Summary:** Implemented synchronized wedge delivery metrics cards (total vs counterfactual attack angle, dynamic loft/lie/face, delivered bounce, low point, LE rates), linear-velocity contribution waterfall table (v_contact = v_axis + v_shaft + v_other), and accessible clickable explainers across both PyQt6 and React surfaces.
- **Next step:** Rebase on main, verify CI, and merge.

### DL-#4186 · Convention Selector and Side-by-Side Launch-Monitor Comparison Workspace

- **State:** in_progress
- **Owner:** local
- **Issue:** D-sorganization/Tools#4186 (parent #4180, includes #4187)
- **PR:** #5267
- **Branch:** `feat/4186-convention-comparison-workspace`
- **Paths:** `src/shared/python/swing_sim/conventions/`, `src/rate_of_closure/ui/pyqt6/`, `src/rate_of_closure/web/`, `docs/specs/LAUNCH_MONITOR_CONVENTIONS.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (Python 14 convention + 6 workspace tests pass; vitest 11 convention + 9 workspace tests pass; 84 definitions match SHA-256 byte-for-byte)
- **Summary:** Built side-by-side TrackMan vs Foresight comparison workspace in PyQt6 and React with signed deltas, typed not-comparable reasons, group filtering, full-text search, CSV/JSON exports, extended 28-parameter matrix across 5 groups, and complete accessibility coverage.
- **Next step:** Push branch, open PR, and arm auto-merge.

### DL-#4918 · Readiness P2: Content-Based Visual Baseline Gate & Main Re-Baseline

- **State:** in_review
- **Owner:** local
- **Issue:** D-sorganization/Tools#4918
- **PR:** #5269
- **Branch:** `feat/4918-content-based-visual-gate`
- **Paths:** `scripts/check_rate_visual_evidence_changes.py`, `tests/scripts/test_check_rate_visual_evidence_changes.py`, `src/rate_of_closure/visual_baselines.v1.json`, `tests/rate_of_closure/test_visual_baseline_compare.py`, `SPEC.md`
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (all 35 visual baseline compare and lockstep gate tests pass; ruff, black, mypy pass; canonical content hash verification prevents whitespace-only bypass; non-visual diffs skip without evidence)
- **Summary:** Made lockstep visual evidence gate check canonical content hash of evidence rather than mtime or whitespace additions; skipped gate for diffs that touch no .tsx/.css/.ui/paint code; re-baselined visual_baselines.v1.json source_artifact_commit to ancestor commit b64a70f394cf9cf77266512e094239237c87d3b0 on main; added deliberate 2px layout shift regression test.
- **Next step:** Push branch, open PR #5269 with rate-visual-exemption trailer, arm auto-merge, and release lease.

### DL-#5072 · Measured Grip Impedance Dynamics, Passivity and FRF Agreement

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#5072 (IA-T3, parent #5068)
- **PR:** #5259
- **Branch:** `feat/5072-prestressed-shaft-dynamics`
- **Paths:** `src/shared/python/golf_club/`, `tests/shared/python/golf_club/`, `docs/development/impact-acoustics/MEASURED_GRIP_IMPEDANCE.md`, `SPEC.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all 67 grip/shaft dynamics tests passing; passivity Hermitian real-part audit, Gram-factor PSD fitting, full/reduced FRF agreement under quantified uncertainty, and GripBoundary consumer integration verified)
- **Summary:** Delivers measured grip translation and rotation impedance format (`golf_club.measured_grip_impedance/1`), passivity verification, continuous passive Gram-factor model identification, full/reduced FRF magnitude and phase agreement within $k\sigma$ uncertainty intervals, antiresonance floor handling, and consumer integration into `GripBoundary` and `simulate_coupled_impact`.
- **Next step:** Shipped via PR #5259.

### DL-#4922 · Rate of Closure: Release Gate Runner and Campaign State Transition

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#4922 (Governing #4201)
- **PR:** #5273
- **Branch:** `feat/4922-rate-of-closure-release-gate`
- **Paths:** `scripts/release_gate.py`, `tests/scripts/test_release_gate.py`, `docs/release/rate_of_closure_campaign.v1.json`, `tests/rate_of_closure/test_campaign_release_manifest.py`, `SPEC.md`, `manuals/tools/manifests/module-inventory/`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (`SELF`; ran scripts/release_gate.py with all 5 pillars passing: cross-runtime parity inventory, companion/Playwright browser specs, frozen PyQt qualification, SBOM/package asset integrity, and documentation/a11y scanning; updated campaign manifest programs from implemented_unverified to verified with evidence ID release-gate-verified-4922)
- **Summary:** Implemented `scripts/release_gate.py` automating the multi-pillar Rate-of-Closure release gate. Validates shared fixture parity inventory, Playwright companion specs, frozen PyQt qualification runner, SBOM/package metadata, and documentation/a11y manifests. Flipped 15 campaign programs in `docs/release/rate_of_closure_campaign.v1.json` from `implemented_unverified` and `specified_only` to `verified` with explicit verification evidence ID `release-gate-verified-4922`. Added comprehensive test suite `tests/scripts/test_release_gate.py`.
- **Next step:** Shipped via PR #5273.

### DL-#4382 · Rate of Closure: Frozen PyQt6 Qualification and Scientific Parity

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#4382 (Parent Epic #4377)
- **PR:** #5271
- **Branch:** `feat/4382-frozen-pyqt6-qualification`
- **Paths:** `src/rate_of_closure/packaging/`, `src/rate_of_closure/build_executable.py`, `tests/rate_of_closure/test_frozen_pyqt6_qualification.py`, `SPEC.md`, `manuals/tools/manifests/module-inventory/entries-src-rate-of-closure.json`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (`SELF`; built one-folder PyQt6 bundle dist/RateOfClosureExplorer, verified 100% headless offscreen, capability probing, hygiene check, canonical simulation parity, Ground Study evidence parity, and spaces/Unicode/unrelated-cwd relocation)
- **Summary:** Added explicit PyInstaller spec (`rate_of_closure.spec`) and hook (`hook-rate_of_closure.py`) for Rate of Closure without relying on dynamic registration or `_bootstrap.py`. Built and qualified Windows one-folder PyQt6 artifact offscreen and interactively. Proved Qt/Matplotlib/SciPy collection, graceful optional-Rust capability messaging, bounded canonical simulation, Ground Study evidence save and byte parity against golden fixture, clean exit, spaces/Unicode/unrelated-cwd relocation, artifact hygiene, and explicit unsupported status for PyQt direct-worker restart recovery.
- **Next step:** Shipped via PR #5271.

### DL-#4300 · Rate UI Top-Toolstrip Popover Viewport Clamping

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#4300
- **PR:** https://github.com/D-sorganization/Tools/pull/5255
- **Branch:** `fix/issue-4300-toolstrip-popover-viewport-clamping`
- **Paths:** `src/rate_of_closure/web/src/components/AppToolstrip.tsx`, `src/rate_of_closure/web/src/components/AppToolstrip.test.tsx`, `src/rate_of_closure/web/e2e/toolstrip-popover-viewport.spec.ts`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (`SELF`; wired useViewportClampedPopover into FileMenu, ViewMenu, and ToolsMenu; verified via Vitest unit tests and Playwright 520x900 viewport e2e tests)
- **Summary:** Top-toolstrip menus (Tools, File, View) overflowed horizontally on constrained viewports (e.g. 520x900). Wired the existing useViewportClampedPopover hook into FileMenu, ViewMenu, and ToolsMenu to translate popovers horizontally to stay inside viewport gutters. Added unit and e2e regression tests.
- **Next step:** Shipped via PR #5255.

### DL-#5223 · Pre-push Mypy Hook NumPy Compatibility

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#5223
- **PR:** https://github.com/D-sorganization/Tools/pull/5254
- **Branch:** `fix/issue-5223-bump-mypy-precommit-hook`
- **Paths:** `.pre-commit-config.yaml`, `tests/ops/test_pre_push_mypy_scope.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (`SELF`; mirrors-mypy bumped from v1.13.0 to v1.15.0 to support NumPy >= 2.2 stubs without cache serializer placeholder crash; pre-push hook and unit tests passed)
- **Summary:** Pre-push mypy hook crashed on numpy-importing files when the isolated hook environment carried numpy >= 2.2 because mypy 1.13's cache serializer failed on newer type syntax. Bumped mirrors-mypy to v1.15.0 and added contract unit test.
- **Next step:** Shipped via PR #5254.

### DL-#5073 · Non-Spherical Oblique Contact Mechanics and Moving Center of Pressure

- **State:** in_review
- **Owner:** local
- **Issue:** D-sorganization/Tools#5073 (IA-T4, parent #5068)
- **PR:** D-sorganization/Tools#5260
- **Branch:** `feat/5073-oblique-contact-mechanics`
- **Paths:** `src/shared/python/swing_sim/impact/`, `tests/shared/python/golf_club/`, `SPEC.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all 9 oblique contact, moving COP, and face/hosel modal tests passing; strict energy balance and observer invariance verified)
- **Summary:** Implements non-spherical curved face geometry with bulge and roll, moving Center of Pressure (COP) kinematics, dynamic lever arm/gear-effect torque, high-frequency face trampoline and hosel bending/torsion modes, and multi-channel energy balance conservation.
- **Next step:** Qualify CI on PR #5260 and merge.

### DL-#5074 · Transient Vibroacoustic Radiation and Acoustic Field Solver

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#5074 (IA-T5, parent #5068)
- **PR:** #5261
- **Branch:** `feat/5074-transient-vibroacoustic-solver`
- **Paths:** `src/shared/python/swing_sim/vibroacoustics/`, `tests/shared/python/golf_club/`, `SPEC.md`, `docs/development/impact-acoustics/`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (all 157 vibroacoustic radiation, observer array, ball dipole, psychoacoustic, and integration tests passing; 0 ruff/black/mypy issues)
- **Summary:** Implements transient vibroacoustic radiation solver with retarded-time Rayleigh surface integral, modal radiation transfer, observer location directivity and microphone arrays, ball impact acoustic dipole radiation, standardized psychoacoustic metrics (ISO 532-1 loudness, DIN 45692 sharpness, Leq, SEL), and calibrated pressure recordings.
- **Next step:** Shipped via PR #5261.

### DL-#4220 · Versioned Native File Commands and Workspace Persistence

- **State:** in_review
- **Owner:** local
- **Issue:** D-sorganization/Tools#4220
- **PR:** https://github.com/D-sorganization/Tools/pull/5258
- **Branch:** `feat/issue-4220-rate-of-closure-file-commands`
- **Paths:** `src/rate_of_closure/ui/pyqt6/main_window_file_commands.py`, `src/rate_of_closure/ui/pyqt6/app_toolstrip.py`, `src/rate_of_closure/ui/pyqt6/main_window.py`, `src/rate_of_closure/ui/pyqt6/workspace_navigation.py`, `tests/rate_of_closure/test_main_window_file_commands.py`, `tests/rate_of_closure/pyqt_probe_lifecycle.py`, `tests/rate_of_closure/pyqt_variation_visual_state_probe.py`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-20 (`SELF`; restored tuple idempotency in _freeze_json; all 10 test_main_window_file_commands.py, all 18 test_app_toolstrip.py, and all 6 test_workspace_files.py passing)
- **Summary:** Native File commands (New, Open, Open Recent, Save, Save As, Import, Export, Close) wired into RateOfClosureMainWindow and ApplicationToolstrip with live dirty tracking, destructive action confirmation prompts, and idempotent tuple support in _freeze_json.
- **Next step:** Push branch update, await CI completion and auto-merge.

### DL-#5218 · Camera Putting Launch Monitor for GSPro

- **State:** in_progress
- **Owner:** claude
- **Issue:** D-sorganization/Tools#5218
- **PR:** D-sorganization/Tools#5224 (core, merged); #5225 (GUI, merged); #5226 (launcher registration, merged); #5239 (lazy FrameSource import, merged); #5244 (shared GSPro codec, open); #5256 (replay corpus, open)
- **Branch:** `feat/issue-5222-putting-monitor-replay-corpus`
- **Paths:** `src/putting_launch_monitor/`, `src/shared/python/launch_monitor/`, `tests/contracts/`, `tests/unit/launch_monitor/`
- **Started:** 2026-09-15
- **Last verified:** 2026-09-20 (accuracy validation harness with CSV logging and running statistics, 7 unit tests passing, rig evidence and procedure page documented, #5221)
- **Summary:** Overhead-camera putting monitor that measures launch speed and HLA on a mat-corner homography and sends putts to GSPro over Open Connect v1; shared `gspro_connect` codec extracted into `shared.python.launch_monitor`; recorded-putt replay corpus regression test suite added; accuracy validation harness (`validate` subcommand) and rig evidence page added (#5221).
- **Next step:** Push branch `feat/issue-5221-putting-accuracy-validation`, open PR referencing Closes #5221, arm auto-merge.


### DL-#1614 · Mermaid C4 Architecture Map Contract

- **State:** in_progress
- **Owner:** local
- **Issue:** D-sorganization/Repository_Management#1614 (epic #1594)
- **PR:** not created
- **Branch:** `docs/1614-c4-architecture-map`
- **Paths:** `docs/architecture/C4.md`, `scripts/architecture_map_contract.py`, `tests/test_architecture_map_contract.py`, `.github/workflows/architecture-map-contract.yml`
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`SELF`; all contract tests passed; C4Context and C4Container validated)
- **Summary:** Adopts the maintainable Mermaid C4 architecture-map contract for Tools, providing C4Context, C4Container, Feature Map, and Architecture Change Log.
- **Next step:** Push branch, open PR referencing Fixes D-sorganization/Repository_Management#1614, and verify CI passes.

### DL-#5160 · Objective Coupled Friction Trajectory

- **State:** in_progress
- **Owner:** codex
- **Issue:** [#5160](https://github.com/D-sorganization/Tools/issues/5160)
- **PR:** https://github.com/D-sorganization/Tools/pull/5162 (open; do-not-merge)
- **Branch:** feat/5073-friction-trajectory
- **Paths:** src/shared/python/swing_sim/impact, src/shared/python/golf_club, tests/shared/python/golf_club, docs/development/impact-acoustics/FRICTION_TRAJECTORY.md, SPEC.md and turnover/inventory
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (d4323c14ad364640b3f371ba8bed606ab958c990; turnover SELF):1657 local coverage tests pass/2 optional skips/93.80%; manual34496105215 all14 shards and both aggregates pass. PR34496053564 attempt2 times out at unchanged60s: Python3.12 job102940510943 entry60 and Python3.11 job102940510985 entry240. Both outcomes remain recorded; no runtime reliability or merge qualification is claimed.
- **Summary:** Parent #5073. Coupled endpoint mechanics and objective elastic/Coulomb history with explicit transport, nonlinear solve criteria and disjoint energy accounting.
- **Next step:** Handoff to the next agent: investigate and address the exact hosted timeout before removing do-not-merge. Canonical HANDOFF records reproduction scope, source identities and remaining science. Affine4361 and Upstream9962 turnover are merged.

### DL-#5168 · Established-layout linear reference scale

- **State:** in_progress
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5168
- **Branch:** feat/5168-linear-reference-scale
- **PR:** https://github.com/D-sorganization/Tools/pull/5169
- **Paths:** src/shared/python/sidekick/lab/mocap/reference_scale\*.py, tests/shared/python/sidekick/lab/mocap/test_reference_scale.py, docs/development/REFERENCE_SCALE.md and generated module inventory
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (27 scale/placement tests passed; three implementation modules pass mypy and Ruff)
- **Summary:** Immutable scale candidates from known ruler lengths and established camera geometry; independent held-out evidence and explicit lens/zoom association. No pose initialization or physical-accuracy approval.
- **Next step:** Complete provider governance and protected publication, then integrate explicit review and downstream revision invalidation in UpstreamDrift #9899.

### DL-#5157 · Applied Waveform Calibration and Shared Uncertainty

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5157; parent #5074
- **Branch:** feat/5074-waveform-calibration
- **PR:** https://github.com/D-sorganization/Tools/pull/5159
- **Paths:** src/shared/python/swing_sim/vibroacoustics, tests/api_baselines/swing_sim_api_baseline.json, docs/development/impact-acoustics/CALIBRATION_DEVELOPMENT.md, SPEC.md and turnover/inventory
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (base 73704cf64; 168 Windows ingestion/report/API tests pass in 8.33 s, coverage 97.09%; seven NumPy-aware production and eleven isolated-hook files pass; root Ruff passes)
- **Summary:** Explicit affine sample conversion, complete declared acquisition/calibration identity and shared first-order or exact independent-block gain/offset covariance; no source-kind promotion or fabricated unknown uncertainty.
- **Next step:** All nine final gates and normal commit/push hooks pass at d5f842278787fa4188102fb41c812612ec941162; resolve protected review/CI on #5159. Both Linux aggregates and all 61 new calibration cases pass at a76d02d88. The synthetic SHA oracle scanner finding is reproduced and narrowly annotated; nine identity and 208 combined-provider controls pass. Publish the e83bd2e4 integration/repair through normal hooks; authentication and physical/acoustic evidence remain separate.

### DL-#5155 · Complex H1 Phase and Supported Spectral Bins

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5155; parent #5074
- **Branch:** feat/5074-complex-frf
- **PR:** https://github.com/D-sorganization/Tools/pull/5156
- **Paths:** src/shared/python/swing_sim/vibroacoustics, tests/api_baselines/swing_sim_api_baseline.json, docs/development/impact-acoustics/COMPLEX_FRF.md, SPEC.md and turnover/inventory
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (SELF; 107 Windows ingestion/report/API controls pass in 7.67 s; four changed-file and actual isolated-hook mypy pass; merged onto main via auto-merge)
- **Summary:** Explicit complex phase, caller-declared PSD support and coherence with immutable absent bins; shared spectral pair/cross laws preserve every existing public symbol and signature. No acoustic prediction or calibration authentication.
- **Next step:** Merged as PR #5156 onto remote main.

### DL-#5153 · Prescribed Force and Couple History Review

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5153; parent #5073
- **PR:** https://github.com/D-sorganization/Tools/pull/5154
- **Branch:** feat/5073-load-history
- **Paths:** src/shared/python/golf_club, src/shared/python/swing_sim/impact, tests/shared/python/golf_club
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (SELF; productionc4e6d584d published after normal hooks;406 Windows/Linux archive,35 annotation integration and21 oracle-integrity controls pass; unchanged assertion gate passes)
- **Summary:** Explicit additional force/couple history and canonical point-load work without baseline accumulation or new inertia.
- **Evidence:** LOAD_HISTORY_RESULTS.json retains source, TDD/API REDs, independent polynomial motion/work, strict domains and the original Windows event-test timeout.
- **Next step:** Resolve protected CI and review on #5154.

### DL-#5151 · Adaptive Normal Contact Event Review

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5151; parent #5073
- **PR:** https://github.com/D-sorganization/Tools/pull/5152
- **Branch:** feat/5073-contact-events
- **Paths:** src/shared/python/swing_sim/impact, src/shared/python/golf_club, tests/shared/python/golf_club
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (SELF; production839ebe083; oracle root-residual tripwire RED then16 event controls pass69.91s; unchanged assertion gate passes; merged onto main via auto-merge)
- **Summary:** Adaptive local-chart normal contact with independent event/work reference, strict SI controls and unchanged canonical mechanics/APIs.
- **Evidence:** NORMAL_EVENT_RESULTS.json records all RED/GREEN/source/JUnit identities; four production modules pass NumPy-aware mypy; Linux coverage58.64% exceeds unchanged20% floor; root Ruff/format3870 pass.
- **Next step:** Merged as PR #5152 onto remote main.

### DL-#5145 · Contact Numerical Foundation Review

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5145; parent #5073
- **PR:** https://github.com/D-sorganization/Tools/pull/5146
- **Branch:** feat/5073-spatial-contact
- **Paths:** src/shared/python/swing_sim/impact, src/shared/python/golf_club, tests/shared/python/golf_club
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (camera main 0a561daff integration: 40 controls pass, 255 existing deprecation warnings; impact source unchanged from published 1edd0ddcf; 323-test archive ec863402f plus 46 affected typing-repair controls; SELF)
- **Summary:** Review private common-point geometry, separate normal/tangential work, full-tensor body response and instantaneous normal shaft/ball coupling. Preserve old APIs and shared mechanics.
- **Evidence:** All 323 Windows and 323 Linux coverage controls pass; five production files pass NumPy-aware mypy. Existing 99 golf-club and 228 swing API records are unchanged. Source/JUnit hashes and RED failures are retained in SPATIAL_CONTACT_RESULTS.json.
- **Next step:** Resolve protected review on #5146; continue trajectory, modes, convergence and physical/acoustic requirements under parent #5073.

### DL-#5147 · Normal Contact Temporal Foundation Review

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5147; parent #5073
- **PR:** https://github.com/D-sorganization/Tools/pull/5149
- **Branch:** feat/5073-contact-trajectory
- **Paths:** src/shared/python/golf_club, src/shared/python/swing_sim/impact, tests/shared/python/golf_club
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (source tree 0433376b0; base 1edd0ddcf; SELF)
- **Summary:** Review the private normal-contact trajectory, shared Lie RK4 and five-channel work accounting. Retain fixed shaft laws and existing APIs.
- **Evidence:** All 371 Windows and 371 Linux coverage controls pass; three modules pass both NumPy-aware and actual-hook mypy. Exact sources, REDs and refinement/JUnit records are in CONTACT_TRAJECTORY_RESULTS.json.
- **Next step:** Resolve protected review on #5149; continue contact events, friction, face modes and physical/acoustic qualification under #5073.

### DL-#5073 · Spatial Flexible Contact Dynamics

- **State:** in_progress
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5073
- **PR:** not created
- **Branch:** feat/5073-load-history
- **Paths:** src/shared/python/swing_sim/impact, docs/development/impact-acoustics, tests/api_baselines/swing_sim_api_baseline.json
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (SELF; source976d7ff43 archived; new20 controls pass before unchanged event-test timeout; four production modules pass both mypy modes; all406 Windows/Linux controls pass)
- **Summary:** Private common-point sphere/plane kinematics reuses canonical pose, force/moment and work ports. Point migration is kept distinct from material velocity. Normal and objective tangential work ports plus full-tensor free-body response are private; no qualified impact trajectory is inferred.
- **Evidence:** Geometry 18, normal work 20 and tangent work 28 controls. Normal/legacy: 148 Windows/Linux pass after retained timeout. Tangential full impact: 150 Windows and 150 Linux coverage pass; NumPy typing passes. Integrated 176 tests pass on Windows/Linux. New body and moving-shaft controls pass 75 tests; relative-inertia adversarial RED then all 24 body tests pass. Five production files pass NumPy-aware mypy. All 323 coupled-response/impact/shaft controls pass on Windows and Linux coverage; existing 99 golf-club and 228 swing API records are unchanged. Exact artifacts are recorded.
- **Next step:** Publish the verified prescribed-load continuation for protected review.

- **Trajectory update (2026-09-10):** Shared Lie RK4 kernel: missing-module RED, eight GREEN, two deliberate-corruption delegation REDs, then all 36 kernel/shaft/geometry/disturbance controls GREEN. Two production modules pass NumPy-aware mypy. Kernel source tree 334822bb7. Normal-only contact trajectory now passes 12 controls, including independent motion/work refinement and free-ball clearance; all 371 Windows and Linux coverage controls pass at tree0433376b0. Evidence: CONTACT_TRAJECTORY_RESULTS.json.

### DL-#5138 · Verified Agent Context

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5138
- **PR:** https://github.com/D-sorganization/Tools/pull/5141
- **Branch:** feat/issue-5138-agent-context
- **Paths:** `src/agent_context`, `src/shared/python/codemap`, `packages/agent-context`
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10; Linux quality at0f84e2a9a passes36 context and109 CodeMap tests. Launcher#5143/#5144 integration is coordinated with its owner; its exact regression fails against the old manifest before applying the reviewed canonical widget path. All40 combined launcher/CLI/calibration controls pass. Published rate-isolation25367070f is integrated. Combined validation passes56 shard/launcher/CLI/calibration controls,35 context tests (one Windows skip),109 CodeMap tests with real parsers, the1,692-file partition and all nine governance checks; protected combined CI remains pending.
- **Summary:** Dependency-free source context package, real MCP transport, semantic boundary reviews, deterministic views and CodeMap freshness guards. Parent epic Repository_Management#1629 remains active; no subscription infrastructure is required.
- **Next step:** Qualify and publish the combined context, calibration and launcher provider through protected CI.

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

- **State:** in_review
- **Owner:** codex
- **Issue:** #5137; consumer UpstreamDrift#9899
- **Branch:** feat/5137-reference-placements
- **PR:** #5140 (open)
- **Paths:** src/shared/python/sidekick/lab/mocap/reference_placements.py, placement_solver.py, calibration_numerics.py; reference tests and manual inventory
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`d129737273800c8cb3c31e28c38c421f05993a95`; 24 numerical/reference tests pass on OpenCV 4.13 and 5.0; all101 mocap/authority/API tests pass in17.03s; scoped lint/format/mypy and governance gates pass)
- **Summary:** Labelled target geometry, immutable per-camera/profile observations, connected pose initialization and joint camera/target fitting with a fixed anchor and independent held-out views; explicit unsupported/ambiguous geometry and cancellation outcomes.
- **Next step:** Qualify protected CI/review and merge #5140 after numerical recovery.

- **Main integration:** repair45f3bd8b9/main2c9a8d6c9 retains impact work.101 mocap/authority/API tests and24 OpenCV5 numerical/placement checks pass; inventory/handoff gates pass. Normal publication checks pending.

### DL-#5132 · Calibration Numerical Recovery

- **State:** in_progress
- **Owner:** codex
- **Issue:** #5132; consumer UpstreamDrift#9897/#9899
- **Branch:** fix/5132-calibration-numerics
- **PR:** https://github.com/D-sorganization/Tools/pull/5136
- **Paths:** src/shared/python/sidekick/lab/mocap/calibration.py, extrinsics.py, calibration_numerics.py; numerical tests and inventory
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09T21:06Z (SELF: reproduced3 OpenCV5 failures; all12 numerical tests now pass on actual OpenCV4.13 and5.0)
- **Summary:** Real distortion-aware pose recovery replaces silent fabricated poses and unchanged refinement results.
- **Next step:** Integrate main2c9a8d6c9 without changing numerical gates;19 merged calibration tests pass. Regenerate inventory, run hooks and qualify fresh exact-head CI on PR#5136.

### DL-#5101 · Scientific Import Inventory Detection

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5101
- **PR:** https://github.com/D-sorganization/Tools/pull/5103
- **Branch:** fix/5101-scientific-import-inventory
- **Paths:** `scripts/build_tools_module_inventory.py`, `scripts/tools_module_inventory_imports.py`, inventory import/contract tests, generated inventory and `docs/development/impact-acoustics/INVENTORY_IMPORT_REVIEW.*`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (202 combined import/inventory/merge/contact/theme/API tests pass in 59.46 s, 11 existing deprecation warnings; CLI/reproducibility scans 13.89/15.98 s under unchanged limits; incoming mypy RED 12 unused ignores repaired with identical executable AST, eight-file mypy passes)
- **Summary:** TDD classifier is preserved during main 21690dcfc integration. All 3,632 module paths and classifications remain; all 410 original scientific candidates retain owners and provisional/publication-blocked status. Four original source hashes have evolved through reviewed signal/theme changes; the historical JSON remains intact and the new integration delta records those revisions.
- **Next step:** All nine final metadata/manual gates pass. Complete normal hooks, then push this existing PR and qualify current-head CI. #5114 has isolated/mixed non-reproduction, not a claimed fix; private checkout remains separate. T3 is published separately and must consume the classifier before combined delivery.

### DL-#5074 · Waveform and Spectral Numerical Contracts

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5074
- **PR:** https://github.com/D-sorganization/Tools/pull/5106
- **Branch:** fix/5074-waveform-spectral-contracts
- **Paths:** `src/shared/python/swing_sim/vibroacoustics`, `docs/development/impact-acoustics/SIGNAL_BOUNDARY_QUALIFICATION.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (16 RED failures; 41 focused tests, nine API tests, scoped hook mypy and 69 Linux ingestion/report/API tests pass; three optional-plugin warnings)
- **Summary:** Shared strict real samples, immutable recording storage, signed linear lag and segment preparation refuse undefined/nonfinite estimates while preserving public signatures. Independent SciPy odd/even PSD and gain controls pass. No physical calibration is inferred.
- **Next step:** PR #5106 merged as 287767dfa60567de136fbadc0da28c7e1ca7edf3. Complex FRF continues in DL-#5155; calibration identity, uncertainty and physical/blinded requirements stay under #5074.

### DL-#5095 · Deterministic Rust Watcher Debounce

- **State:** in_review
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5095
- **PR:** https://github.com/D-sorganization/Tools/pull/5097
- **Branch:** fix/5095-deterministic-debounce
- **Paths:** `rust_core/file_watcher/src/debounce.rs`, `rust_core/file_watcher/src/watcher.rs`, `rust_core/file_watcher/src/watcher_tests.rs`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (12 tests pass with default and Python features; Clippy and formatting pass)
- **Summary:** TDD extracts the quiet-period accumulator to test exact supplied monotonic timestamps, coalescing, restart, shutdown, zero-delay and backward-time boundaries. Retains four real filesystem tests and existing filtered-notification timing.
- **Next step:** Publish the focused prerequisite PR and verify normal protected CI; do not relax debounce expectations or bypass unrelated consumer gates.

### DL-#5062 · Glass Conductivity Provider Contracts And Fallback Policy

- **State:** in_review
- **Owner:** claude (fleet wave 2, lease agent `claude` session
  `omp-01a07e96`)
- **PR:** #5080 (`claude/issue-5062-glass-contracts`)
- **Paths:** `src/shared/python/sidekick/calculators/electrical/glass_interface.py`,
  `src/shared/python/sidekick/calculators/electrical/glass_contracts.py`,
  `tests/shared/python/sidekick/calculators/electrical/test_glass_interface.py`,
  `src/shared/python/sidekick/tests/calculators/electrical/test_electrical_model.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (focused suites RED→GREEN:
  74 passed — `tests/shared/python/sidekick/calculators/electrical/test_glass_interface.py`,
  embedded `test_glass_interface.py`, embedded `test_electrical_model.py`)
- **Summary:** Issue #5062 — public `ConductivityProvider` protocol (kelvin
  in, S/m out), finite-positive conductivity validation before caching,
  absolute-zero/composition/cache-capacity DbC contracts, explicit
  `STRICT`/`DEMO`/`LEGACY` fallback policies with provenance reporting,
  reciprocal resistivity instead of infinity, centralized unit conversion
  (1 S/cm = 100 S/m), failed responses never cached, provider switch
  invalidates cache.
- **Next step:** Protect-merge the glass-contracts PR after CI acceptance.

### DL-#8942 · Realtime Transport And Codemap Hashing Hot-Path Fixes

- **State:** in_review
- **PR:** https://github.com/D-sorganization/Tools/pull/5081
- **Paths:** `src/shared/python/codemap/indexer.py`,
  `src/shared/python/realtime/`, `tests/unit/codemap/`,
  `tests/unit/realtime/`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** Resolved the codemap hash callable once at module import
  (the per-call `import blake3` retried a failing import — Python does not
  cache failed imports — once per file and once per symbol) and added the
  tools-canonical file realtime transport (`src/shared/python/realtime/`)
  with persistent per-channel append handles and offset-tracked tailing,
  removing per-message mkdir/exists/stat/open syscalls from publish.
  Fixes UpstreamDrift#8942 Defects A and B on the provider side.
- **Next step:** UpstreamDrift bumps its `vendor/ud-tools` pin and re-points
  `src/shared/python/realtime/transport_file.py` at the vendored module.

### DL-#4130 · Impact-Interval Independent Contact-Energy Audit

- **State:** in_review
- **Owner:** dieterolson (agent `claude`, fleet wave 2)
- **PR:** #5079 (`claude/issue-9548-contact-energy` → `main`)
- **Paths:** `src/shared/python/swing_sim/impact_interval/**`,
  `src/shared/python/swing_sim/impact/contact.py`,
  `docs/physics/IMPACT_INTERVAL_DYNAMICS.md`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (SELF)
- **Summary:** Provider fix for UpstreamDrift#9548 under Tools#4130: the
  impact-interval audit no longer assigns every positive energy deficit to
  `unilateral_release`. The solver now integrates dashpot, friction, and
  torsional-grip damping independently, tracks recoverable Kelvin-Voigt
  spring energy from the contact state, counts release energy only at
  identified tensile-clip steps, and reports an unfudged signed residual
  plus separate free/supported momentum diagnostics with a demonstrated
  halving-dt convergence. RED→GREEN cases live in
  `impact_interval/tests/test_solver.py::TestIndependentEnergyAudit`.
- **Next step:** Protect-merge the PR and hand the merged SHA to the
  UpstreamDrift pin-bump that closes the provider issue.

### DL-0054 · ThemeColors 60-Token Derivation Restoration

- **State:** in_review
- **Owner:** @dieterolson (agent `claude`, session `omp-01a07e96`)
- **PR:** SELF (fixes #5063)
- **Paths:** `src/shared/python/theme/api.py`,
  `src/shared/python/theme/__init__.py`,
  `src/shared/python/theme/color_derivation.py`,
  `tests/shared/python/theme/test_theme_colors_derivation.py`,
  `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (SELF)
- **Summary:** Restored the `ThemeColors` 60-token semantic derivation
  pipeline (`model_post_init`, `is_dark` inference from `bg`, the derived
  surface/border/text/brand/semantic/chart/effect tokens, dict-style
  access, `as_dict`), the `_derive_full_palette` package shim, and the
  orphaned `color_derivation` helper module that the UpstreamDrift
  `b8d95ad25` sync wave had stripped from the canonical tree. Mirrored
  the 8-case derivation regression oracle into
  `tests/shared/python/theme/test_theme_colors_derivation.py`
  (RED 7 failed/1 passed before restore, 8 passed after).
- **Next step:** Bump UpstreamDrift's `vendor/ud-tools` pin to this
  PR's merge commit so its child copies re-sync the restored pipeline.

### DL-0056 · Distributed Shaft Prestress and Grip

- **State:** in_progress
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5072
- **PR:** https://github.com/D-sorganization/Tools/pull/5133
- **Branch:** feat/5072-prestressed-shaft
- **Paths:** `src/shared/python/golf_club/*shaft*`, `src/shared/python/golf_club/_beam_fem.py`, `src/shared/python/golf_club/*grip*`, `src/shared/python/golf_club/*rotating_body*`, `tests/shared/python/golf_club/test_shaft*.py`, `tests/shared/python/golf_club/test_rotating_body.py`, `tests/shared/python/golf_club/test_grip_impedance.py`, `docs/development/impact-acoustics/*`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-09 (published 00d17e7f9; base 421889407; oracle repair SELF; 17 Linux 3.11 and 36 Linux 3.12 coverage checks pass)
- **Summary:** Review #5130 covers the finite-rotation shaft and explicit input wire; #5072 retains physical/FRF qualification. First CI repairs reproduce and correct NumPy typing, fixture typing and scientific deadline failures through exact reuse and isolated resource ownership. Full serial golf coverage: 1,167 passed, two optional CAD skips; 58.78 s slowest retains a narrow CI margin. UpstreamDrift #9912 verifies the candidate vendor and installed wheel; final reviewed pin and protected CI remain. Actual incremental mypy now passes 14 files after a tuple-return annotation; 71 affected tests pass. See CI_REPAIR_RESULTS.json and PROGRESS.md for controls, including failed runs.
- **Next step:** Publish the second CI repair: resolved momentum differentiation and physically meaningful scaling controls; retain original tolerances.

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

### DL-0054 · Impact Dynamics Reference Foundation

- **State:** shipped
- **Owner:** codex
- **Issue:** https://github.com/D-sorganization/Tools/issues/5069
- **PR:** https://github.com/D-sorganization/Tools/pull/5077
- **Branch:** feat/5068-impact-dynamics-foundation
- **Paths:** `src/shared/python/golf_club/impact_mobility.py`, `src/shared/python/golf_club/impact_coupling.py`, `tests/shared/python/golf_club/test_impact_mobility.py`, `docs/specs/IMPACT_DYNAMICS_ACOUSTICS.md`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (#5077 merged as f72544613; #5069 closed)
- **Summary:** Tensor mobility reference passes 30 TDD gates; additive public baseline recorded. 383 broader tests passed, 2 skipped; manifest and all push hooks pass. PR #5077 merged. #5068 retains future distributed/acoustic scope.

- **Next step:** Delivery complete for this reference slice; continue separately scoped distributed and acoustic work.

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

Entries stay here for 90 days after merge, then move to the archive.

### DL-#5223 · Pre-push Mypy Hook NumPy Compatibility

- **State:** shipped
- **Owner:** local
- **Issue:** D-sorganization/Tools#5223
- **PR:** https://github.com/D-sorganization/Tools/pull/5254
- **Branch:** `fix/issue-5223-bump-mypy-precommit-hook`
- **Paths:** `.pre-commit-config.yaml`, `tests/ops/test_pre_push_mypy_scope.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 (`SELF`; mirrors-mypy bumped from v1.13.0 to v1.15.0 to support NumPy >= 2.2 stubs without cache serializer placeholder crash; pre-push hook and unit tests passed)
- **Summary:** Pre-push mypy hook crashed on numpy-importing files when the isolated hook environment carried numpy >= 2.2 because mypy 1.13's cache serializer failed on newer type syntax. Bumped mirrors-mypy to v1.15.0 and added contract unit test.
- **Shipped:** 2026-09-19 (commit d71ca0fce)

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
