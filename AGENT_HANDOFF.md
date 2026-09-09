# AGENT_HANDOFF — Tools (Monorepo Root)

## Impact Dynamics and Acoustics: #5068

- Full program active. AffineDrift theory #4258/#4282/#4298 and UpstreamDrift integration plan #9706 are merged. #4298's auxiliary benchmark made no measurements; no performance result is claimed.
- T1 #5069/#5077 is closed/merged at f72544613. T2 #5071/#5082 is closed/merged at 80d580d57; merged golf source/tests match both reviewed e47fde4e and prior validated 476eaa98.
- T3 #5072 publishes autonomous decay 58f33e403 and turnover 2d290079b through all normal hooks. There are 39 decay controls and 731 Linux golf/API passes, two optional CAD skips and three plugin warnings. Separate G/C spectra, all-node finite-grip response and shared plant validation are retained; the envelope only qualifies the stated constant homogeneous ODE. See AUTONOMOUS_DECAY.md.
- T3 has not yet integrated current main or the corrected inventory classifier. PR #5103 has advanced to 02b53e2d8 with CI pending at the last observation. The old classifier falsely marks the new decay calculation non-calculation; integrate the AST correction before combined delivery without granting scientific approval.
- T5 #5106 has advanced to c8f3b4d1: the private-consumer lane fails and some checks remain pending. Historical local qualification at 3a9362530 is preserved; do not extend it to differing remote bytes without review. Calibration identity, complex FRF, radiation and physical/blinded gates remain open.
- T6 #5083 is merged; #5075 retains strict-wire/evidence gaps. Driven/nonlinear stability, transient/bandwidth convergence, flexible impact, final synthesis and exact-pin consumer studies remain required.
- UpstreamDrift #9826 publishes fced8c0d6 after the reconciled PDF correction and current-main integration. All 26 incoming/publication tests and normal hooks pass; remote SHA is verified. Current-head CI is pending after the shallow-diff false deletion correction. Preserve historical 6235789dc; follow claim_preservation_9825_turnover.md in that worktree.
- UpstreamDrift #9830 publishes three synthetic shooting/reference refinements in turnover 1887ac59f. Tiny own-grid residuals coexist with substantial ODE mismatch; reference accuracy and component budgets need a TDD implementation. No solver threshold or physical claim has been changed. #8920/#8556 retain physical-data and parameter-identification gates.
- T3 uses codex lease session impact-acoustics-01a07d8a-t3; check live ownership before edits. Detailed requirements and historical numerical evidence: docs/development/impact-acoustics/PROGRESS.md and its linked derivation/turnover files.

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
