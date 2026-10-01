# Current handoff — Flight Termination States & Metric Gating (Tools#5385)

- Repository: D-sorganization/Tools
- Working directory: `C:/Users/diete/Repositories/Tools`
- Branch: `fix/tools-flight-termination-5385`; commit: SELF; PR: #5391 (open, targets `main`)
- Governing issue: Tools#5385 (paired consumer issue: UpstreamDrift#11145)

## Objective and status

Propagate explicit termination reasons (`FlightTermination`: `LANDED`, `TIME_LIMIT`, `SOLVER_FAILED`, `CANCELLED`) from flight integration before reporting landing metrics (`carry_distance`, `landing_angle`, `lateral_deviation`). Non-landed flights have landing metrics gated to `None` and fail closed across downstream consumers: `CenteredClubDeliveryAdapter` (returns `ForwardStatus.FAILED`), `build_ground_simulation_request` (raises `FlightGroundTransferError`), and `_recompute_registered` in `flight_execution_profiles` (returns `RECOMPUTATION_FAILED`).

## Files and decisions

- `src/shared/python/swing_sim/flight/types.py`: Added `FlightTermination` enum, `IncompleteFlightError(RuntimeError)`. Updated `FlightResult` dataclass to validate landing metrics are `None` when `termination != FlightTermination.LANDED`. Added `require_landing()`, `flight_completed`, `landed` properties.
- `src/shared/python/swing_sim/flight/models.py`: Updated `BallFlightModel._run_ode_simulation` to inspect `sol.status`, `sol.success`, and `sol.t_events` safely and propagate `termination`, `terminal_event`, and `actual_horizon`. Attached partial `FlightResult` to `FlightSimulationCancelled`.
- `src/shared/python/swing_sim/flight/impact_solution_adapter.py`: Added landing gate returning `ForwardStatus.FAILED` with `"flight_incomplete:<termination>"`.
- `src/shared/python/swing_sim/flight/ground_transfer.py`: Added landing gate raising `FlightGroundTransferError` if `not result.landed`.
- `src/rate_of_closure/application/flight_execution_profiles.py`: Added landing gate returning `RECOMPUTATION_FAILED` if `not result.landed`.
- `src/shared/python/swing_sim/flight/tests/test_flight_termination.py`: Added 10 tests verifying the 5 canonical termination fixtures and downstream fail-closed gating.
- `tests/api_baselines/swing_sim_api_baseline.json`: Regenerated to record breaking API changes (`FlightResult` methods, `compute_flight_metrics` kwargs, `FlightSimulationCancelled` optional result).

## Validation

- Full test suite: 231 tests in `flight` and `test_shared_package_api_stability.py` pass.
- Linters: `ruff check` (0 errors), `ruff format --check` (clean), `black --check` (clean), `mypy` (clean).

## Next step

- Push branch and create PR #5385 with breaking API notice citing UpstreamDrift #11145.

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
