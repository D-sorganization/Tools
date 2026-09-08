# AGENT_HANDOFF — Tools (Monorepo Root)

## Impact Dynamics and Acoustics: #5068

- Full epic execution is active; completion requires all model and empirical gates.
- T1 foundation: Tools PR #5077 remains open with downstream CI failure.
  Theory PR AffineDrift #4258 and integration plan UpstreamDrift #9706 are merged.
- T2 #5071 / draft PR #5082: `fix/5071-qualified-impact-coupling`, based on `5932146f1`; 396 broader tests pass, 2 skip, 15 final audit tests and 6 browser parser tests pass.
  Event-resolved solver and additive preload/energy audit; v1 wire retained.
- RED: timeout/step refusal and energy-record validation failed before fixes.
  GREEN: 1,614 unit tests also pass in isolated Python 3.12 + pytest-qt; use serial offscreen hooks. Protected Tools CI remains open.
- Continuation matrix: `docs/development/impact-acoustics/PROGRESS.md`.
  T3–T6, consumer adapters/studies and physical/perceptual validation remain open.
- T2 lease posting failed; #9735 compatibility lease succeeded. Fresh downstream import regression reproduced; see continuation matrix.

> **Update this file with every PR and every push to main.**
> Last updated: 2026-09-08
> **Current state only**, capped at 150 lines by `CLAUDE.md`; history lives in git and `docs/agent_handoff_archive/`.
> Do not append dated entries here again.

## Merge Governance

- Subepic #4728 protected-squash-merged as `682c1402b4bdb1b387877cbdaaf4999fa04a074a`; verify current `origin/main` before acting.
- Live `main` rules require zero approving reviews. Do not require or request a named maintainer's approval; `@dieterolson` is not a standing release gate.
- Never use admin bypass, force-push, check bypass, or protection changes to merge a failing or stale head.

## Where This Repo Is Headed

Tools is the fleet's shared engineering-tools monorepo (45+ tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). Delivery follows scoped consolidations directly on `main`.

| Epic  | Status (one line)                                                                                                                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| #4103 | Swing-Impact-Ball-Flight platform. Open. Remaining: camera cluster (#4571) and Phase 7 (WASM web parity, Pages CI).                                          |
| #4120 | Investigation & Variation Suite. Open. PR #4124 merged.                                                                                                      |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 merged. H5 (release-management repo) pending.                                                |
| #4130 | Impact-interval club dynamics. **COMPLETED** (F1–F4 in PR #4577) — 6-DOF transient package, tests, impact wire.                                              |
| #4142 | Variation and sensitivity. R13.3, R13.5, R14.3 merged; R14.6/calibrated-renderer PRs #4835/#4837 on main.                                                    |
| #4146 | Shared Club Builder. Open. Assembly physics contracts landed in #4157.                                                                                       |
| #4433 | Visual-first tab visibility and visualization-led UX. 8 verified / 23 partial; #4832 adds fifth acceptance manifest.                                         |
| #4430 | Rotating-base companion. **COMPLETED** via #4618/#4619; UpstreamDrift consumed provider through #8954.                                                       |
| #4549 | Club Fitting Tester. **COMPLETED** (#4557, #4577) — C1–C7 delivered (mesh inertia, shaft delivery, OEM doc).                                                 |
| #4562 | Heavy Hit. **COMPLETED** (#4568, #4577) — H1–H4 delivered (coupled mechanics, MJCF/URDF/.osim import).                                                       |
| #4583 | Launch monitor analytics. Release A merged; Release B open (vendor emulation requires real paired data).                                                     |
| #4584 | Strokes gained v2. **COMPLETED** (#4599, #4600, #4602, #4608, #4610, #4613) — shared-first analytics.                                                        |
| #4706 | Markerless mocap. Open. TOOLS-M0 (#4708) / TOOLS-M1 (#4710) in review under PR #4734.                                                                        |
| #4707 | Engineering design manual authority. TOOLS-D1 (#4711), TOOLS-D2 (#4714), TOOLS-D3 (#4717) through TOOLS-D8 completed. TOOLS-D9 (#4730) active final subepic. |

## Active Delivery Boundaries

- TOOLS-D8 (#4728 / PR #5054) merged: enforces immutable public publication projection (`tools-publication-projection/1.0.0`).
- TOOLS-D9 (#4730) enforces governed completion-audit handoff and maintenance contract (`tools-handoff-maintenance/1.0.0`) across root and per-tool handoffs with diff-aware CI gating, line budgets (<= 150 lines), and machine-checked evidence.
- Completing and merging #4730 closes subepic #4730 and closes the entire parent epic [DOC-TOOLS] (#4707).
- `src/shared/python/theme/` derivation restore (fix #5063) is in review on branch `claude/issue-5063-theme-restore`: the `ThemeColors` 60-token semantic derivation pipeline, `_derive_full_palette`, and `color_derivation` helper deleted by the UpstreamDrift `b8d95ad25` sync wave are restored with a mirrored 8-case regression oracle. After merge, UpstreamDrift's `vendor/ud-tools` pin must be bumped so its child copies re-sync the restored pipeline.
- UpstreamDrift#8942 provider perf fix landed: codemap hashes resolve once at
  module import, and `src/shared/python/realtime/transport_file.py` now ships
  tools-canonical (persistent per-channel append handles, offset-tracked
  tailing). Pending downstream wave: UpstreamDrift bumps its `vendor/ud-tools`
  pin and re-points its transport copy at the vendored module.

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
