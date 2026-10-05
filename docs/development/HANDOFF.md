# Current handoff — Re-vendor fork PR runner guard (RM#1996)

- Repository: D-sorganization/Tools
- Working directory: `/home/user/wt/tools-guard-sync`
- Branch: `chore/sync-fork-guard-1996`; commit: SELF; PR: #5439
- Governing issue: Repository_Management#1996 (development log entry DL-#4464, paths refreshed)

## Objective and status

Re-vendor the updated guard (`scripts/fork_pr_runner_guard.py` plus new `scripts/fork_pr_guard_analysis.py`) byte-identical from Repository_Management, re-vendor its tests, regenerate the module inventory. Done.

## Validation

- RED: RM canonical tests vs old guard 10 failed; GREEN: `python -m pytest -q -o addopts="" tests/ops/test_fork_pr_runner_guard.py` 69 passed.
- `python scripts/fork_pr_runner_guard.py` on Tools workflows: 0 violations before and after.
- ruff, mypy 1.13.0 and `python -m scripts.build_tools_module_inventory --check` clean (`mypy.ini` gained a no-redef override for the vendored dual import).

## Next step

- Merge once CI is green; nothing else outstanding.

---

# Past handoff — Align workflow Python matrices with requires-python (Tools#5434)

- Repository: D-sorganization/Tools
- Working directory: `/home/user/wt/tools-5434`
- Branch: `fix/5434-matrix-python-floor`; commit: SELF; PR: #5435
- Governing issue: Tools#5434 (development log entry DL-#5434)

## Objective and status

Stop the 3.10 workflow legs that collect 0 tests (pytest exit 5) because root conftest refuses code below `requires-python >=3.11`, without losing the proof of the crates' own `>=3.10` claim. Done: `maturin-file-watcher` and `maturin-swing-core` each split into `wheel-gate` (build + import, 3.10-3.12) and the old wrapper/parity job (ids `rust-backend-gate`, `parity-gate` unchanged, now 3.11-3.12). `tests/ops/test_workflow_python_floor.py` enforces both directions. Known gap: `rust_core/tools-core` declares `>=3.10` but is built only on 3.11/3.12 in `ci-standard.yml` (listed in `KNOWN_UNPROVEN`).

## Validation

- RED then GREEN: `python -m pytest -q -o addopts="" tests/ops/test_workflow_python_floor.py tests/test_python_version_contract.py` (16 passed incl. `tests/ops/test_maturin_swing_core_workflow.py`).
- `python -m scripts.build_tools_module_inventory --check` clean after regeneration; ruff check/format clean; actionlint shows only pre-existing SC2012 info notes.
- After merging main (fork guard #4464 landed): both new `wheel-gate` jobs carry the same fork guard as the wrapper jobs; `python scripts/fork_pr_runner_guard.py` reports no violations; `tests/ops/test_fork_pr_runner_guard.py` + the floor tests: 50 passed.

## Next step

- Merge once CI is green; nothing else outstanding.



---

# Past handoff — Fast-fail stale inventory test (Tools#5432)

- Repository: D-sorganization/Tools
- Working directory: `/home/user/wt/tools-5432`
- Branch: `fix/5432-inventory-fast-fail`; commit: SELF; PR: #5433 (ready, armed)
- Governing issue: Tools#5432. Development-log entry: DL-#5432.

## Objective and status

`test_inventory_is_deterministic_and_fresh` no longer asserts on two huge dicts; it fails with a bounded summary naming stale shards (`tests/architecture/_inventory_diff.py`). Strictness unchanged. Module inventory regenerated for the new test files.

## Validation

- `python -m pytest -q -o addopts="" tests/architecture/test_inventory_diff_5432.py`: 7 passed (RED first: helper missing).
- `python -m scripts.build_tools_module_inventory --check`: passes after regeneration.

## Next step

- None; auto-merge armed. If the inventory goes stale on rebase, regenerate it.


---

# Past handoff — Fork PRs never run on the self-hosted fleet (Tools#4464)

- Repository: D-sorganization/Tools
- Working directory: `/home/user/wt/tools-4464`
- Branch: `fix/4464-fork-pr-self-hosted`; commit: SELF; PR: see PR body
- Governing issue: Tools#4464 (development log entry DL-#4464)

## Objective and status

Make it impossible, through the workflows themselves, for fork PR code to run on
`d-sorg-fleet`. Done in this branch: canonical job-level fork guard on 47 jobs,
fork routing to `ubuntu-latest` for `ci-standard` `pick-runner`/`tests`/`tests-gate`,
and `scripts/fork_pr_runner_guard.py` with `tests/ops/test_fork_pr_runner_guard.py`.
On `pull_request` a fork can edit the workflow file, so this is defence in depth;
the repository settings in the PR body are still required to close the issue.

## Validation

- `python -m pytest tests/ops/test_fork_pr_runner_guard.py`: RED on main (50 jobs), GREEN after.
- `python -m pytest tests/ops tests/scripts tests/test_python_version_contract.py`: passed.
- `actionlint -shellcheck= .github/workflows/*.yml`: no findings before or after.
- Review fix: base-context head checkouts aliased through workflow- or job-level `env`
  (`PR_SHA: ${{ github.event.pull_request.head.sha }}` then `ref: ${{ env.PR_SHA }}` /
  `$PR_SHA`) are now detected; 8 new cases RED before, 34 tests GREEN after.
- Re-vendored byte-identical from Repository_Management#1990 (canonical copy; adds
  bracket-form head refs and privileged `workflow_call` callers); 34 tests GREEN.

## Next step

- Owner applies the admin-only settings in the PR body; new PR-triggered fleet jobs must carry the guard (the test enforces it).
# Current handoff — Fix Rust 1.99 clippy errors in pendulum-core (Tools#5429)

- Repository: D-sorganization/Tools
- Working directory: `/home/user/wt/tools-clippy`
- Branch: `fix/pendulum-core-clippy-f64`; commit: SELF; PR: see PR body
- Governing issue: Tools#5429 (development log entry DL-#5429)

## Objective and status

Make `pendulum_core rust quality gate (fmt + clippy + test)` pass on the fleet's Rust 1.99. Done: removed `use std::f64;`
from `cmaes.rs` and switched `dynamics.rs` to `as_chunks::<2>()`. Behaviour is unchanged.

## Validation

- `cargo +1.99 clippy --all-targets -- -D warnings` (in `src/pendulum_simulator/pendulum-core`): 5 errors before, clean after.
- `cargo +1.99 test`: 42 passed. `cargo fmt --check`: clean.

## Next step

- Merge once CI is green; nothing else outstanding.

---

# Past handoff — Fix clippy deprecated f64 constants in tools-core (Tools#5422)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/_worktrees/Tools-5422`
- Branch: `fix/issue-5422-clippy-f64-deprecated`; commit: SELF; PR: see PR body
- Governing issue: Tools#5422

## Objective and status

Fix clippy failures in `rust-quality-gate` where `use std::f64;` shadowed primitive `f64` associated constants (`f64::NEG_INFINITY` and `f64::MAX`). Removed shadowing import.

## Validation

- `cargo clippy --package tools-core --all-targets -- -D warnings`: passed cleanly (0 errors, 0 warnings).
- `cargo test --package tools-core`: 184 passed, 0 failed.

## Next step

- Open ready PR, arm auto-merge via Repository_Management `scripts/automerge_guard.py`.

---

# Past handoff — ADR-008 license deny-list check (Tools#5417)

# Current handoff — Launch Monitor state geometry, first tab pair (Tools#4433 sub-task)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Tools-worktrees/claude-4433`
- Branch: `claude/issue-4433`; commit: SELF; PR: #5424 (ready, armed)
- CI fix: the new probe is allowlisted in `scripts/test_assertion_allowlist.txt` (subprocess entrypoint, like its siblings); module inventory regenerated (the new probe is listed as a related test in three shards); `origin/main` merged.
- Governing issue: Tools#4433 (sub-task; epic stays open). Development-log entry: DL-#4433.

## Objective and status

Add first-viewport geometry assertions beyond the initial state for one reciprocal tab pair. Chosen pair: React `launch-monitor-analytics` / PyQt `launch_monitor_analytics`. Implemented; tests only, no production change.

## Files and decisions

- `tests/rate_of_closure/pyqt_launch_monitor_state_probe.py`: new subprocess probe (QT_SCALE_FACTOR is fixed per process). Drives result, error, loading (measured inside the synchronous read), and empty through the tab's own handlers; the file dialog and message box are replaced by recorders so no modal opens. Reuses the helpers of `pyqt_visualization_tab_probe.py`.
- `tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py`: `_probe` takes a `probe_script` argument; new test parametrized at 1.0 and 1.5 DPI.
- `src/rate_of_closure/web/e2e/visualization-tab-visibility.spec.ts`: new state test at all three reference viewports; read hold via a `Blob.arrayBuffer` gate; the threshold rule moved into `requiredVisibleSize`, shared with the initial-state test.
- Ledger: V1.4, V4.2, V4.3 rationale, evidence and gaps updated; statuses unchanged (partial). The R14.6 blocking-gap text is mirrored in the #4142 ledger and left unchanged.
- Found, not fixed: 58 px horizontal document overflow at 390x844 once results render (single-column grid track widened by the result tables).

## Validation

- `QT_QPA_PLATFORM=offscreen python -m pytest tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py tests/rate_of_closure/test_visual_first_epic_4433_evidence.py tests/scripts/test_check_rate_visual_evidence_changes.py tests/rate_of_closure/test_visualization_tab_manifest.py -q`: 63 passed.
- RED: a result-only 1200 px (PyQt) / 2000 px (React) spacer above the scatter fails both new tests; not committed.
- `npx playwright test e2e/visualization-tab-visibility.spec.ts --project chromium-desktop`: 4 passed.

## Next step

- Frontier review of the draft PR; then the next tab pair as a separate CLI-tier sub-task.

---

# Current handoff — ADR-008 license deny-list check (Tools#5417)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Tools-worktrees/claude-5417`
- Branch: `claude/issue-5417`; commit: SELF; PR: draft, see PR body
- Governing issue: Tools#5417 (refs #4719)

## Objective and status

Add `scripts/check_license_denylist.py` and `config/license_denylist.json` enforcing ADR-008 (no FreeMoCap/SkellyCam, no AGPL ids) statically and with `--installed`. Implemented and tested.

## Files and decisions

- `config/license_denylist.json`: only what ADR-008 names.
- `tests/architecture/test_license_denylist.py`: 11 tests, incl. real-repo static pass.
- Not wired into CI (out of scope; workflow PR for the workflow owner).

## Validation

- `python -m pytest -q tests/architecture/test_license_denylist.py`: 11 passed; ruff check/format and mypy clean.

## Next step

- Frontier review of the draft PR; separate workflow PR for CI wiring.

# Current handoff — Python runtime-manifest validator (Tools#5416)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Tools-worktrees/claude-5416`
- Branch: `claude/issue-5416`; commit: SELF; PR: draft, see PR body
- Governing issue: Tools#5416 (parent #4260)

## Objective and status

Port the TypeScript `parseRuntimeManifest` (`calculation-runtime-manifest/v1`) to Python with shared-fixture parity. Implemented; awaiting review.

## Files and decisions

- `src/shared/python/swing_sim/runtime_manifest.py`: new validator (`parse_runtime_manifest`, `runtime_manifest_from_json`, `stable_runtime_manifest_json`); returns fresh plain dicts. TS `TypeError`/`RangeError` map to `TypeError`/`ValueError` with the same key phrases. Nonempty-text check uses the ECMAScript `trim()` whitespace set, not `str.strip()`.
- `tests/shared/python/test_runtime_manifest_parity_fixture.py`: parser parity cases driven from the unchanged shared fixture.
- `docs/specs/active/CALCULATION_RUNTIME_MANIFEST.md`, `SPEC.md`, module-inventory shard: bookkeeping.
- Producer, fixture, TypeScript validator and `canonical_numeric_json` untouched (out of scope).

## Validation

- `python -m pytest -q tests/shared/python/test_runtime_manifest_parity_fixture.py`: 52 passed.
- ruff check, ruff format --check, mypy clean on the new module and test.

## Next step

- Frontier review of the draft PR, then mark ready and arm auto-merge via Repository_Management `scripts/automerge_guard.py`.

---


# Past handoff — Restore Reverted PBKDF2 Hardening and Swing-Objectives Fix (Tools#5404)

# Current handoff — Restore Reverted PBKDF2 Hardening and Swing-Objectives Fix (Tools#5404)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Tools`
- Branch: `fix/restore-reverted-fixes-5404`; commit: SELF; PR: not created
- Governing issue: Tools#5404

## Objective and status

Restore the PBKDF2 hardening from PR #5399 (600,000 iterations default with transparent legacy 100,000 fallback on InvalidToken) and the swing-objectives inertia equivalence and evidence limits corrections from PR #5401 (closing #5393) which were inadvertently reverted during the stale-branch squash merge of PR #5400.

## Files and decisions

- `src/folder_packer_pro/encryption.py`: Restored 600,000 PBKDF2 iterations with legacy fallback.
- `tests/test_folder_packer_pro.py`: Added explicit regression test verifying 600,000 iterations default and legacy 100,000 archive decryption fallback.
- `docs/specs/SWING_ACTUATION_AND_REALISM.md`: Restored corrected heuristic reference interval wording and wrist-pivot measurement origin caveats.
- `src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/`: Restored actuation, club_equivalence, impact_optimality, model_adequacy, objective_realism, reference_kinematics.
- `src/pendulum_simulator/tests/test_club_equivalence.py`: Restored regression test `test_inertia_arithmetic_distinguishes_masses`.
- `manuals/tools/manifests/module-inventory/`: Regenerated module inventory shards via `python -m scripts.build_tools_module_inventory`.
- `SPEC.md`: Restored change log rows for #5400, #5393, #5399, and added row for #5404.

## Validation

- Pytest: all 18 tests in `test_club_equivalence.py`, all 9 tests in `test_folder_packer_pro.py`, and all 16 tests in `test_spec_version_freshness.py` pass.
- Ruff: `ruff check` passes (0 errors on changed files).
- Formatter: `ruff format --check` passes.
- Mypy: `mypy` passes cleanly (0 errors on changed files).
- Module inventory: `python -m scripts.build_tools_module_inventory --check` passes cleanly.

## Next step

- Commit, push branch, open PR referencing Closes #5404, and arm squash auto-merge.

---


# Past handoff — v1.23.1 release preparation (Tools#5376)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Worktrees/luna-pr5380-20260929`
- Branch: `bot/luna-release5376-20260929`; commit: SELF; PR: #5376 (open, targets `main`)
- Governing issue: none (release preparation)

## Objective and status

Prepare the v1.23.1 release branch. The latest `main` is `95800a5647603bf1211a98b0899674540cc16151` and still declares v1.23.0; the release branch declares v1.23.1. Merged current `main` into this branch without rebasing. The release notes include the merged PRs through #5384.

## Files and decisions

- Refreshed the v1.23.1 changelog date and added merged PRs #5377, #5378, #5379, #5380, #5381, and #5384.
- Added the keyed #5376 release-preparation row to SPEC.md; retained the current main rows #5379 and #5384.
- Refreshed the development-log audit date; no feature continuation state changed.
- The merged main changes remain intact, including their workflow, documentation, manifest, source, and test updates.

## Validation

- `python -m pytest -n 1 -o addopts='' tests/scripts/test_release_changelog.py tests/ops/test_release_workflow.py tests/shared/python/sidekick/calculators/electrical/test_norm_einsum_equivalence.py` — 39 passed.
- Ruff check and format check on the changed electrical model and test — passed; targeted mypy — passed.
- Design-manual governance, module inventory, textbook lint, exemplars, calculation freshness, manual QA, publication projection, handoff, and render checks — passed. Manual QA and publication projection report the existing unapproved state with two release blockers.
- P1AM frontend Vitest was unavailable because `vitest` is not installed in the checkout; no dependency installation was attempted.

## Blockers and risks

- Root review and CI are pending. This worker will not merge or trigger workflows.

## Next step

- Root reviews the final diff and CI before deciding whether to merge PR #5376.

## Change log

- `SELF` — 2026-09-29: merge current `main` and refresh v1.23.1 release metadata and handoff.
- `SELF` — Regenerate the module inventory for `scripts/fork_pr_runner_guard.py`: the stale inventory failed `test_inventory_is_deterministic_and_fresh`, whose assertion diff stalled the `tests-unit` xdist shard until the 90-minute timeout.

---

# Past handoff - self-hosted npm cache (2026-09-29)

- Repository: D-sorganization/Tools; branch `claude/tools-npm-cache-per-job`; commit: SELF; PR: #5384.
- Problem: on self-hosted runners `~/.npm` is shared by every job on the host, so `setup-node` `cache: npm` saved 2.4 GB. The HMI gate (`p1am-frontend.yml`) spent its 20-minute budget restoring it at ~1 MB/s. The signed URL expired after 10 minutes and the job was cancelled (PR #5380, two attempts on d-sorg-local-Oglaptop-3).
- Change: job-level `NPM_CONFIG_CACHE: ${{ github.workspace }}/.npm-cache` in `p1am-frontend.yml` and `rate-of-closure-visual-evidence.yml`, the two self-hosted `cache: npm` jobs. This follows `tauri-build.yml`. Hosted `ubuntu-24.04` jobs already start clean and are unchanged.
- The cache key is only the lockfile hash, so the four >1 GB cache entries were deleted to stop them being restored. They regenerate at about 35 MB.
- Next: after merge, confirm the next HMI gate run saves a cache of tens of MB.

---

# Current handoff — Project Steward status pass 2026-09-26

- Repository: D-sorganization/Tools
- Working directory: `/home/dieterolson/staff-worktrees/Tools-run-3ab98d2c47ee`
- Branch: `staff/project-steward-task-c496cc`; commit: SELF
- Pull request: #5356
- Governing issue/epic: Project Steward scheduled pass (no governing issue)

## Objective and status

Scheduled Project Steward pass. Audited all changes since 2026-09-23 from
GitHub and git; refreshed `docs/project/STATUS.md`, `docs/project/CHARTER.md`,
and reconciled 5 stale `DEVELOPMENT_LOG.md` entries to `shipped`.

Key findings this pass:
- Three epics closed: #4103 (Swing-Impact-Ball-Flight), #4707 (Design Manual), #5218 (Putting LM)
- Three knowledge-pack features shipped: #5345/#5348, #5346/#5350, #5347/#5351
- **`tests (3.11)` is red on `main`** as of 2026-09-26 (sha 1a8f012); `tests (3.12)` passes
- Two duplicate v1.22.0 bot release PRs open (#5349, #5352)
- P0 security #4464 still unaddressed (16 days; approaching Board proposal threshold)

## Files and decisions

- `docs/project/CHARTER.md`: marked TOOLS-4103, TOOLS-5218, TOOLS-4707 as `shipped`; added TOOLS-5345 row for knowledge-pack features
- `docs/project/STATUS.md`: full refresh — what moved, stuck items, CI health, open PRs, decisions needed
- `docs/development/DEVELOPMENT_LOG.md`: DL-#5347, DL-#5346, DL-#5345, DL-#1755, DL-#5333 all moved to `shipped`; last-audited updated to 2026-09-26
- `docs/development/HANDOFF.md`: this file

No source code changed; docs-only PR.

## Validation

- `git diff --stat`: only `docs/` files changed
- No SPEC.md §12 entry required (no `src/**` changes)
- spec-check gate passes (SOURCE_CHANGED=false)

## Blockers and risks

- `tests (3.11)` red on main since 2026-09-26: known failures include
  `test_wgs_engine_imports_without_pyqt6` and `test_python_310_fallback_exports_timezone_utc_and_str_enum`.
  This is not a regression introduced by this steward pass (docs-only). A separate
  triage issue should be filed.
- P0 security #4464: 16 days unaddressed — nearing the 14-day Board proposal
  threshold. If still unresolved at next pass, the steward should submit a Board proposal.

## Next steps

- Open the draft PR, confirm CI passes for docs-only changes.
- File a triage issue for `tests (3.11)` red on main.
- Close one of the duplicate v1.22.0 release PRs (#5349 or #5352).
- At next pass: if #4464 is still unresolved, submit a Board proposal per the playbook.

## Change log

- `SELF` — Project Steward 2026-09-26: refresh STATUS.md/CHARTER.md for 3 closed epics, 3 knowledge-pack ships, CI red flag; reconcile 5 dev-log entries to shipped.

---

# Past handoff — Pre-Impact Bundle Wire and Modal-State Record (Tools#5353)

- Repository: D-sorganization/Tools
- Branch: `feat/5353-pre-impact-bundle-modal-state`; commit SELF; PR: #5368
- Issue: #5353 — all items (1-4) now implemented. Items 3-4 (validation and field origin) landed in #5366; items 1-2 (pre-impact bundle wire and modal-state record) implemented here.
- Built:
  - `golf_club._pre_impact_contracts`: `PreImpactBundleError`, `AbsentFieldError`, `Quantity`, explicit-raise validators.
  - `golf_club.pre_impact_frames`: `Pose`, `twist_to_parent`, `wrench_to_parent`, `shift_wrench_origin`, `shift_twist_reference`, Plücker spatial motion/force transforms with $10^{-12}$ wrench power invariance.
  - `golf_club.modal_state`: `ModalBasis`, `ShaftState`, `ModalProjection`, `project_onto_basis` (M-orthogonal with residual reporting and $0.5 \dot{q}^T M_r \dot{q} + 0.5 q^T K_r q$ energy ledger).
  - `golf_club.pre_impact_bundle`: `PreImpactBundle`, `Provenance`, `TimeBase`, `HeadState`, `BallState`, `HandWrench`, `grip_pose_from_delivery_sample`.
  - `golf_club._pre_impact_serde`: dict/json serialization helpers keeping all files strictly $\le 500$ LOC.
- Validation: `pytest tests/shared/python/golf_club/test_pre_impact_bundle.py tests/shared/python/golf_club/test_pre_impact_frames_modal.py tests/shared/python/golf_club/test_field_origin.py tests/shared/python/golf_club/test_public_validation.py tests/test_shared_package_api_stability.py` -> all 92 tests pass; `ruff check`, `ruff format --check`, module inventory check clean.
- Next: open PR, verify CI passes, auto-merge squash; UpstreamDrift can consume canonical Tools bundle via thin adapter.

---

# Past handoff — Shadow-surfaced input contracts (Tools#5362)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5362` (agy Gemini 3.8 Flash slice from `agy-tools-5362-shadow-contracts`, reviewed and re-applied on a fresh `origin/main`)
- Branch: `claude/tools-5362-shadow-contracts`; commit SELF; PR: #5364 (draft)
- Issue: #5362 — UpstreamDrift's unit-gate quarantine burn-down (UD#9411) found expectations that only the canonical Tools packages can satisfy; UD may not grow its shadow copies, so they land here.
- Built: signal_toolkit preconditions (`apply_exponential_smoothing` alpha in (0, 1]; saturation `lower <= upper` checked once in `_apply_saturation_values`; `apply_rate_limiter` `max_rate >= 0`; `apply_deadband` `threshold >= 0`; `NoiseGenerator` `amplitude >= 0`); `InertiaCalculator.compute_from_primitive` rejects `mass <= 0`; `PhysicsValidator` default gravity uses `GRAVITY_M_S2` (9.80665) instead of a literal 9.81; `ChatSessionManager` normalises session timestamps to UTC so naive and aware ISO strings sort together; humanoid_character_builder docstring states it is layered on model_generation. Item 7 (sidekick data_processing `@value`) was already resolved on main.
- Review changes over the agy patch: dropped a duplicated `lower > upper` guard in `apply_saturation` (the helper it calls enforces it); replaced the docstring bullet that still claimed "no dependencies" with one naming model_generation.
- Validation: `python -m pytest -o addopts="" tests/shared/python/signal_toolkit tests/shared/python/model_generation/inertia/test_calculator.py src/shared/python/model_generation/tests/test_physics_validation.py src/shared/python/humanoid_character_builder/tests/test_api.py tests/unit/ai/gui/test_session_manager_2872.py` -> 327 passed; module inventory regenerated and `--check` clean.
- Next: open the draft PR, CI green, arm through `automerge_guard.py`; then one UD vendor-pin bump (covers #5361 and this) and retire the 11 re-quarantined IDs plus the `test_safe_eval` IDs (UD#9411).

---

# Past handoff — Launch-monitor covariation sums (Tools#5355)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-pr-5355`
- Branch: `bolt-covariation-reduce-15618261669777904218`; commit SELF; PR: #5355 (Bolt, reworked by Claude onto main after #5354)
- Issue: none (Bolt performance PR); no DL entry — No material development-log change: a behaviour-preserving refactor of one model file.
- Built: `launchMonitorCovariation.ts` gains private `sum` and `pairMean` helpers. `centeredPairs` and `meanPairs` share `pairMean` instead of the Bolt commit's two pasted loops; `metaAnalyze` sums the random weights once instead of inside the per-player `forEach` (O(N^2) -> O(N)).
- Validation: `npx vitest run src/model/launchMonitorCovariation.test.ts` -> 5 passed; `npx tsc --noEmit` and `eslint` on the changed file clean; `python -m scripts.build_tools_module_inventory --check` passes after regeneration.
- Next: CI green, then arm through `automerge_guard.py`.

---

# Past handoff — safe_eval power-result bound (Tools#5360)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5360`
- Branch: `claude/tools-5360-safe-eval-pow-bound`; commit SELF; PR: not created at commit time
- Issue: #5360 (security): nested and runtime integer powers bypassed the static exponent and chain-depth guards. The chain-depth walk only follows `.right`, so `((a**b)**c)**d` passed at any depth; `2 ** x` with a large namespace `x` was never checked.
- Built: `MAX_POW_RESULT_BITS = 10_000`; `_bounded_pow` (rejects integer results whose `exponent * log2(|base|)` exceeds the bound; floats and numpy values pass through); `_PowToBoundedCall` rewrites every `**` to call it, and `safe_eval` injects it as a global the validated expression cannot name. Two-argument scalar `pow()` uses the same helper; modular `pow(a, b, m)` is unchanged.
- Validation: `PYTHONPATH=src python -m pytest tests/shared/python/test_safe_eval.py tests/shared/python/test_safe_pandas_eval.py tests/test_shared_package_api_stability.py -o addopts=""` -> 84 passed (6 new cases); ruff and mypy clean; module inventory `--check` passes.
- Next: open the draft PR, get CI green, arm through `automerge_guard.py`; then bump the UD vendor pin and retire the five `tests/unit/test_safe_eval.py` quarantine IDs whose remaining expectations Tools satisfies (UD#9411).

---

# Past handoff — Launch-monitor column helper (Tools#5354)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-pr-5354`
- Branch: `bolt-optimize-column-extraction-13047720`; commit SELF; PR: #5354 (Bolt, reviewed by Claude)
- Issue: none (Bolt performance PR); no DL entry — No material development-log change: a three-call-site helper extraction.
- Built: `launchMonitorColumns(rows)` in `src/rate_of_closure/web/src/model/launchMonitorAnalysis.ts` (sorted union of row keys, one Set pass). The Bolt commit had pasted the same loop into `LaunchMonitorPerformanceWorkspace`, `LaunchMonitorPlayerWorkspace` and `NeuralModelLabPanel`; all three now call the helper.
- Validation: `npx vitest run src/model/launchMonitorAnalysis.test.ts src/model/launchMonitorPerformanceWorkspace.test.ts` -> 15 passed (2 new helper tests); `npx tsc --noEmit` and `eslint` on the changed files clean; `python -m scripts.build_tools_module_inventory --check` passes after regeneration.
- Next: CI green, then arm through `automerge_guard.py`.

---

# Past handoff — Knowledge-pack golden Q&A evaluation + optional MiniLM hybrid ranking (Tools#5347)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/agy-5347`
- Branch: `agy/issue-5347`; commit SELF; PR: #5351 (draft)
- Issue: #5347 (K4 of Repository_Management#1772: Q&A evaluation and optional MiniLM hybrid ranking)
- Built: `src/shared/python/ai/knowledge/eval.py` (GoldenQACase, EvalSummary, evaluate_pack, load_golden_set, CLI) and optional dense MiniLM embeddings with hybrid BM25 + cosine Reciprocal Rank Fusion ranking in `src/shared/python/ai/knowledge/pack.py`. Off by default and gated by manifest's `embeddings: true`. Core remains stdlib + PyYAML only.
- Rebase (2026-09-25): rebased onto `main` after #5350 (Sidekick Wizards) merged; preserved all public API symbols in `pack.py`; compacted `pack.py` to 496 LOC to satisfy the $\le 500$ LOC budget; fixed embedder mypy `no-any-return` typing; regenerated module inventory.
- Validation: `py -3.12 -m pytest tests/shared/python/ai/knowledge` -> 67 passed, 1 skipped; `tests/test_shared_package_api_stability.py` -> 10 passed; `scripts/check_file_size_budget.py` -> 0 violations; `ruff check` clean on changed files; `py -3.12 -m mypy src/shared/python/ai/knowledge` clean (9 files checked); `scripts/build_tools_module_inventory.py --check` passes.
- Next: push commits to `agy/issue-5347`, mark PR #5351 ready, monitor CI and squash-merge.

---

# Past handoff — Sidekick Wizards (Tools#5346)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5346`
- Branch: `feat/5346-sidekick-wizards`, stacked on `feat/5345-knowledge-pack` (PR #5348); commit SELF; PR not created until #5348 merges.
- Issue: #5346 (K3a of Repository_Management#1772)
- Built:
  - `ai/knowledge/wizard.py`: `WizardConfig` + `load_wizard_config`, `KnowledgeContext.render()`, `WizardKnowledge` with TTL-cached freshness. It needs only the stdlib and PyYAML.
  - `ai/wizards.py`: the Sidekick glue. It holds a per-root cache, calls `register_app_context` and provides `knowledge_for_context`.
  - `BaseAgentAdapter.build_context_instruction_section` appends Wizard knowledge. Every adapter already uses this path.
  - `search_knowledge_base` prefers the pack.
  - `RAGContextProvider` emits a DeprecationWarning.
- Host contract: `knowledge/wizard.yml` (key, name, description, capabilities, manifest=`knowledge/pack.yml`, pack=`.knowledge/pack.sqlite`, k, roots). Build with `python -m shared.python.ai.knowledge build knowledge/pack.yml --root <Repo>=. --out .knowledge/pack.sqlite`.
- Validation: `py -3.12 -m pytest tests/shared/python/ai src/shared/python/ai/tests tests/test_shared_package_api_stability.py -o addopts="" -n 8` -> 570 passed; ruff and mypy clean. The knowledge API baseline gains wizard (the theme baseline rewrite was reverted).
- Next: open the PR after #5348 merges; hosts UD#10943 and GM#5089 (tier:cli); refresh job Tools#5347.

---

# Past handoff — Knowledge-pack engine (Tools#5345)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5345`
- Branch: `feat/5345-knowledge-pack`; commit SELF; PR: see DL-#5345
- Issue: #5345 (K0 of Repository_Management#1772: Vision Quest, Disciple, Sidekick Wizards)
- Built: `src/shared/python/ai/knowledge/` (manifest, chunking, sources, pack, cli). It uses only the stdlib and PyYAML and imports no other Tools module, so Runner_Dashboard vendors it (RD#1479). The pack format is gated by `PRAGMA user_version = 1`.
- Contract: `build_pack(manifest, roots, out) -> PackInfo`; `KnowledgePack.open(p).search(q, k=8, include_superseded=False) -> list[Passage]`, `.info()`, `.is_stale(roots)`. Status precedence: manifest override > front-matter `status:` > current. Ties break by authority (published > findings > reviews > product > reference > notes).
- Baseline: `knowledge` is added to `VENDORED_PACKAGES`; new file `tests/api_baselines/knowledge_api_baseline.json`. Regeneration also rewrote the theme baseline, which was reverted by hand.
- Validation: `py -3.12 -m pytest tests/shared/python/ai/knowledge tests/test_shared_package_api_stability.py -o addopts=""` -> 42 passed; ruff and mypy clean; smoke build of RM `staff/knowledge/findings.yml` over local UD + AffineDrift -> 10,288 passages in 2.8 s.
- CI follow-up: the module inventory was regenerated with `py -3.12 -m scripts.build_tools_module_inventory`, then LF-normalized because the Windows write is CRLF. The divergence-ledger gate needs `UD-PAIR:` in the PR body (paired with UD#10943; `ai/knowledge` has no UD copy).
- Next: K3a Tools#5346 (per-product Wizard packs for Sidekick); K4 Tools#5347 (refresh job).

---

# Past handoff — Retire the review-comment-to-issue converter (RM#1755)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-retire-converter`
- Branch: `chore/retire-comment-converter`
- Issue: Repository_Management#1755
- What was removed: `.github/workflows/Comment-to-Issue-Converter.yml` (already disabled 2026-09-25; no processor script, tests, or lingering references were present).
- Validation: `py -3.12 <RM>/scripts/campaigns/review_comment_converter_retirement/retire_converter.py --repo . --check` -> exit 0 after `--apply`.
- Next step: open the draft PR for review.

---

# Night Watch Development Log Maintenance — 2026-09-23

## Identity

- Repository: D-sorganization/Tools
- Working directory: `/home/dieterolson/staff-worktrees/Tools-run-350067a6bf64`
- Branch: `staff/night-watch-task-239d87`
- Baseline commit: `96d5681328ac62e53d94371f07e09c95f0a3b2cd`
- Implementation commit: SELF
- Pull request: SELF
- Governing issue/epic: Night Watch scheduled pass (no governing issue)

## Objective and status

Docs compliance sweep: reconciled the development log state table with actual
merged PR and closed-issue records on GitHub. Found 29+ entries still marked
`in_review` or `in_progress` whose governing PRs had already merged (some as
recently as 2026-09-23, others as old as 2026-09-08). Marked 30 entries
`shipped` and 1 entry `parked`. The log is now current.

## Files and decisions

- `docs/development/DEVELOPMENT_LOG.md`: 31 entries reconciled. 30 marked
  `shipped` (each with the correct PR reference); 1 (DL-#5132, calibration
  numerical recovery) marked `parked` because PR #5136 was closed without
  merge and the issue is closed. Last-audited timestamp updated to
  2026-09-23 by night-watch.
- `docs/development/HANDOFF.md`: this file, replacing the stale codex session
  HANDOFF from #5322.

## Validation

- No source code changed; only `docs/development/` files touched.
- All formerly-active entries now have accurate `shipped` or `parked` states.
- Zero `in_review` / `in_progress` entries remain.

## Blockers and risks

- DL-#5132 (calibration numerical recovery, PR #5136): PR closed without
  merge and issue closed. Marked `parked`. Retry would require a new issue
  and a new entry.
- DL-#0001 (backup-pyo3-split): pre-existing parked orphan with no governing
  issue; not changed in this pass.

## Next steps

- Open the draft PR, let CI pass, and merge.
- A future Night Watch pass should audit the older DL-00NN entries for
  candidates to move to `abandoned`.

## Change log

- `SELF` — Night Watch 2026-09-23: reconcile 31 stale development log entries
  to `shipped` or `parked` based on verified GitHub merge records.

---

# Handoff — Tools PR #5387 (2026-09-30)

- Repository: D-sorganization/Tools; branch `bot/luna-tools5387-20260930`; PR #5387 remains a draft.
- Scope: dispersion aggregation loops and regression coverage, SPEC/changelog metadata, generated module inventory, and four compatible web lockfile security patches.
- Evidence: 238 web test files / 2,360 tests passed; lint, type-check, build, 27 governance checks, and npm audit at `--audit-level=high` passed (0 high/critical). Two moderate entries remain for Vitest 3.2.7 and `@vitest/mocker`; Vitest 4 is a major update and was not included.
- Remaining: root review and PR CI. No performance benchmark was run; the journal makes no GC or speed claim.
