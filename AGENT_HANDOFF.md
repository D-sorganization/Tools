# AGENT_HANDOFF — Tools (Monorepo Root)

## Impact Dynamics and Acoustics: #5068

- Full program remains active: distributed/flexible impact, calibrated acoustics, consumer studies and physical/blinded evidence are required.
- T1 #5069/#5077 merged at f72544613 and T2 #5071/#5082 merged at 80d580d57; both are ancestors of current main. Theory AffineDrift #4258 and the UpstreamDrift #9706 integration plan are merged.
- T3 #5072 remains on feat/5072-prestressed-shaft, published at 6bf223941 through normal hooks; it is not yet merged. There are 920 Linux scientific/API passes, two optional CAD skips, with conditional frequency-interval and independent bending/mesh/modal checks. These do not qualify physical acoustic bandwidth.
- T5 #5106 boundary repair is on main at 287767dfa; calibration identity, complex FRF, radiation and physical/blinded validation remain open. T6 #5083 is merged; strict evidence/consumer qualification remains.
- #5114: deferred Qt deletion has an ownership RED and two passing Windows regressions (17.10 s), with optional-Qt collection skips verified separately. Full Linux patch failed: 2,880 passed, 29 skipped, three worker losses and a 900-second cap. Fourteen serial profiled GUI controls pass but do not qualify the full suite; #5114 stays open.
- Continue at docs/development/impact-acoustics/RATE_QT_ISOLATION.md. Classifier #5103 merged at 33144678cb; its Python 3.12 Rate shard was cancelled after another worker loss. Preserve failed controls and exact native sources; keep #5114 open.
- GitHub CLI access recovered at 08:56 UTC Sept 9. #5114 evidence is published in issuecomment-5599216206; local checkpoint f788ca612 passed normal commit hooks and awaits push hooks. #5072 ownership was renewed through 10:57:31 UTC; scientific implementation resumes separately.

> **Update this file with every PR and every push to main.**
> Last updated: 2026-09-09
> **Current state only**, capped at 150 lines by `CLAUDE.md`; history lives in git and `docs/agent_handoff_archive/`.
> Do not append dated entries here again.

## Merge Governance

- Subepic #4728 protected-squash-merged as `682c1402b4bdb1b387877cbdaaf4999fa04a074a`; verify current `origin/main` before acting.
- Live `main` rules require zero approving reviews. Do not require or request a named maintainer's approval; `@dieterolson` is not a standing release gate.
- Never use admin bypass, force-push, check bypass, or protection changes to merge a failing or stale head.
- `cross-repo-python-integration.yml` initializes UpstreamDrift's `vendor/ud-tools` pin in the \_downstream checkout (#5085): UD retired tools-canonical child copies (UD #9569), so the consumer contract lane fails on every PR without it.

## Where This Repo Is Headed

Tools is the fleet's shared engineering-tools monorepo (45+ tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). Delivery follows scoped consolidations directly on `main`.

| Epic  | Status (one line)                                                                                                                                                                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| #4103 | Swing-Impact-Ball-Flight platform. Open. Remaining: camera cluster (#4571) and Phase 7 (WASM web parity, Pages CI).                                                                                          |
| #4120 | Investigation & Variation Suite. Open. PR #4124 merged.                                                                                                                                                      |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 merged. H5 (release-management repo) pending.                                                                                                |
| #4130 | Impact-interval club dynamics. F1–F4 landed (#4577, #4945); independent contact-energy audit (provider fix for UpstreamDrift#9548) in review — UI/report surfacing awaits the Tools#4946 model-type ruling.  |
| #4142 | Variation and sensitivity. R13.3, R13.5, R14.3 merged; R14.6/calibrated-renderer PRs #4835/#4837 on main.                                                                                                    |
| #4146 | Shared Club Builder. Open. Assembly physics contracts landed in #4157.                                                                                                                                       |
| #4433 | Visual-first tab visibility and visualization-led UX. 8 verified / 23 partial; #4832 adds fifth acceptance manifest.                                                                                         |
| #4430 | Rotating-base companion. **COMPLETED** via #4618/#4619; UpstreamDrift consumed provider through #8954.                                                                                                       |
| #4549 | Club Fitting Tester. **COMPLETED** (#4557, #4577) — C1–C7 delivered (mesh inertia, shaft delivery, OEM doc).                                                                                                 |
| #4562 | Heavy Hit. **COMPLETED** (#4568, #4577) — H1–H4 delivered (coupled mechanics, MJCF/URDF/.osim import).                                                                                                       |
| #4583 | Launch monitor analytics. Release A merged; Release B open (vendor emulation requires real paired data).                                                                                                     |
| #4584 | Strokes gained v2. **COMPLETED** (#4599, #4600, #4602, #4608, #4610, #4613) — shared-first analytics.                                                                                                        |
| #4706 | Markerless mocap. Open. TOOLS-M0 (#4708) / TOOLS-M1 (#4710) in review under PR #4734.                                                                                                                        |
| #4707 | Engineering design manual authority. TOOLS-D1 (#4711), TOOLS-D2 (#4714), TOOLS-D3 (#4717) through TOOLS-D8 completed. TOOLS-D9 (#4730) active final subepic.                                                 |
| #5068 | Impact and vibroacoustics program open. T1/T2 and initial T5/T6 merged; distributed T3 remains partial on feat/5072-prestressed-shaft; T4 flexible contact and calibrated acoustic/physical evidence remain. |

## Active Delivery Boundaries

- #4844 renderer prerequisite: `fix/4844-consistent-pyqt-renderer`, worktree `C:/Users/diete/Repositories/Tools-impact-render`, PR #5090. Published `df4101f28` passes 73 browser and 23 PyQt tests in two Linux captures; all ten PyQt images repeat byte-identically. A reviewed 20-image reference set is proposed with unchanged tolerances; 60 local contracts and both candidate comparisons pass. Hooks/fresh CI remain. See `docs/development/rate-pyqt-renderer-4844-reference-review.md`; #5087 is closed unmerged.
- Rust watcher #5095 / PR #5097: deterministic quiet-period batching replaces the scheduler-dependent flush-count test; four filesystem tests remain. Validation and CI prerequisite evidence: `docs/development/file-watcher-debounce-5095.md`.
- TOOLS-D8 (#4728 / PR #5054) merged: enforces immutable public publication projection (`tools-publication-projection/1.0.0`).
- TOOLS-D9 (#4730) enforces governed completion-audit handoff and maintenance contract (`tools-handoff-maintenance/1.0.0`) across root and per-tool handoffs with diff-aware CI gating, line budgets (<= 150 lines), and machine-checked evidence.
- Completing and merging #4730 closes subepic #4730 and closes the entire parent epic [DOC-TOOLS] (#4707).
- Issue #5062 (glass conductivity provider contracts and explicit fallback policy) in review on branch `claude/issue-5062-glass-contracts`: public `ConductivityProvider` protocol in `glass_contracts.py`, validated finite-positive outputs, STRICT/DEMO/LEGACY policies, reciprocal resistivity.
- `src/shared/python/theme/` derivation restore (fix #5063) is in review on branch `claude/issue-5063-theme-restore`: the `ThemeColors` 60-token semantic derivation pipeline, `_derive_full_palette`, and `color_derivation` helper deleted by the UpstreamDrift `b8d95ad25` sync wave are restored with a mirrored 8-case regression oracle. After merge, UpstreamDrift's `vendor/ud-tools` pin must be bumped so its child copies re-sync the restored pipeline.
- UpstreamDrift#8942 provider perf fix landed: codemap hashes resolve once at
  module import, and `src/shared/python/realtime/transport_file.py` now ships
  tools-canonical (persistent per-channel append handles, offset-tracked
  tailing). Pending downstream wave: UpstreamDrift bumps its `vendor/ud-tools`
  pin and re-points its transport copy at the vendored module.
- Issue #3992 (single-source Catppuccin palette) in review on branch
  `bot/issue-3992-theme-single-source`: the six hand-copied stylesheet/palette
  sites named by the issue (function_generator, pressure_drop_calculator,
  steam_engine_calculator, financial_calculator, asteroid_jumper, help_system)
  now source colors from `shared.python.theme.catppuccin` (`get_stylesheet()` /
  `CATPPUCCIN_MOCHA`). Guarded by
  `tests/architecture/test_issue3992_no_pasted_palette.py`.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — conventions, CI gates, and cross-repo dependency rules; Tools is a leaf consumed by UpstreamDrift and Gasification_Model.
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — §12 requires a dated row for every PR touching `src/`.
4. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for a new tool's handoff doc.
5. `manuals/tools/manifests/module-inventory.json` — strict tracked-module inventory.
6. `manuals/tools/schemas/handoff-maintenance.schema.json` — machine-checked completion-audit handoff schema.
7. `docs/shared/divergence_ledger.v1.json` — machine-readable seam-divergence ledger.

## Gate Commands (Repo-Wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check (Ruff, NOT Black)
python3 -m pytest -n auto --timeout=60            # full test suite (tests/ only)
python3 scripts/ci_test_shards.py --check        # every test file claimed by exactly one CI shard (#4913)
python3 -m scripts.check_design_manual_governance
python3 -m scripts.build_tools_module_inventory --check
python3 -m scripts.lint_tools_textbook_chapters
python3 -m scripts.check_tools_exemplars
python3 -m scripts.check_tools_calculation_freshness --check
python3 -m scripts.check_tools_manual_qa --check
python3 -m scripts.check_tools_publication_projection --check
python3 -m scripts.check_tools_handoff --check
python3 -m scripts.render_tools_design_manual --check
```

## Do-Not List

- **Do not exceed 150 lines in any AGENT_HANDOFF.md file.**
- **Do not append dated entries to handoff files.** Put history in git commits and SPEC.md §12.
- **Do not use `git commit --no-verify` or bypass CI/CD checks.**
- Do not regenerate `tests/sidekick_api_baseline.json` with `--regenerate-api-baseline` without a coordinated breaking change.
- Do not import across package boundaries — Law of Demeter is enforced.
- Do not edit shared Python surface from inside a consumer's vendored copy; Tools is source of truth.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web` — Phase 7 of #4103 owns this.

## Short-Term Roadmap (Ordered)

1. Protect-merge #4730 (TOOLS-D9) and close parent epic DOC-TOOLS (#4707).
2. Advance TOOLS-M0 (#4708) / TOOLS-M1 (#4710) markerless mocap authority under PR #4734.
3. Validate and protect-merge #4792/R14.3 from the R13.5 protected mainline.
4. Implement Sidekick S1–S5 plan in `docs/development/epic_sidekick_unified_impact_model_and_launcher_integration.md`.
5. Land camera-cluster epic #4571 to close #4466.

## PR Disposition — 2026-09-07/08 Fleet Backlog Sweep (Tools, 22 open PRs)

| PR                             | Disposition                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #5076                          | MERGED (1e6f21d61): dirty conflict resolved via local merge with the module-inventory-regen merge driver + sidekick shard regen.                                                                                                                                                                                                                                                       |
| #5081                          | MERGED (squash 69aeeed83): minimum-test-contract + divergence-ledger UD-PAIR fixes; main-merge conflict resolved in-branch first.                                                                                                                                                                                                                                                      |
| #5078                          | Theme-restore CI repaired (mypy duplicate module, inventory, theme API baseline) + main merged; merging when clean.                                                                                                                                                                                                                                                                    |
| #5079                          | API baseline (additive KelvinVoigt method), handoff-manifest re-pin, inventory regen; pushed, merging when clean.                                                                                                                                                                                                                                                                      |
| #5080                          | Sidekick API baseline re-pinned (glass contracts) + main merged; merging when clean.                                                                                                                                                                                                                                                                                                   |
| #5082                          | No PR-caused failure: E2E = hosted-runner font-stack drift (#5087), cancelled shards rerunning; merge when required checks green.                                                                                                                                                                                                                                                      |
| #5077                          | Awaiting CI (full-tensor impact reference; upstream defect UpstreamDrift#8942).                                                                                                                                                                                                                                                                                                        |
| 15 release bumps (#5024-#5070) | Stale mutually-conflicting v1.16.2 duplicates (newest = #5070 v1.17.0); branch-updated twice, required checks green; merge #5070 first, then close the rest as superseded duplicates under an exempt label.                                                                                                                                                                            |
| #5086                          | bot/ infra PR: initialize UD vendor/ud-tools pin in downstream lane (fixes #5085).                                                                                                                                                                                                                                                                                                     |
| #5087                          | bot/ infra PR: run PR-lane Worker E2E on the fleet-calibrated renderer image (hosted font-stack drift).                                                                                                                                                                                                                                                                                |
| 8 release bumps (#5089-#5104)  | CLOSED as redundant: main already v1.17.0 via #5070 (four stale v1.16.2 downgrades, four duplicate v1.17.0); superseded-by-#5070 comments posted.                                                                                                                                                                                                                                      |
| #5093                          | MERGED via auto-merge after branch update chain (incl. ToolsPRs squash d5a85a23e + main 287767dfa): added missing §12 row and refreshed module-inventory components shard after Spec Check (source changed without SPEC update) and Docs Governance (stale inventory shard) failures; Rate Web Playwright PR visual-evidence policy gate red (non-required; documented in PR comment). |
| #5106                          | MERGED (squash 287767dfa): vibroacoustics ingestion/spectral contracts; SPEC/AGENT_HANDOFF/module-inventory unions + manifest re-pin during main merges.                                                                                                                                                                                                                               |
| #5103                          | MERGED at 33144678cb from be38a827a on Sept 9. Scientific import inventory is on main. Its cancelled Python 3.12 Rate shard contains another Club Tester worker loss; #5114 remains open.                                                                                                                                                                                              |
| #5085 (infra)                  | spec_rows_merge_driver.py Py3.14 fix (argparse %-format help strings) applied in Repository_Management shared clone; not a Tools PR.                                                                                                                                                                                                                                                   |
