# AGENT_HANDOFF — Tools (Monorepo Root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-25

> **Current state only**, capped at 150 lines by `CLAUDE.md`; history lives in
> git and in [`docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md`](docs/agent_handoff_archive/2026-08_tools_root_handoff_log.md).
> Do not append dated entries here again.

## Merge Governance

- Use feature-branch pull requests and ordinary protected merges; all configured status checks must pass on the exact head.
- The live `main` rules require zero approving reviews. Do not require or request a named maintainer's approval; `@dieterolson` is not a standing release gate. Optional review remains available for risk, expertise, or unresolved feedback.
- Never use admin bypass, force-push, check bypass, or protection changes to merge a failing or stale head.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, grown from a closure-rate
calculator into a swing → impact → ball-flight simulation platform. Since
early August the delivery pattern has shifted from long stacked PRs to
**scoped consolidations rebuilt directly onto current `main`**.

| Epic  | Status (one line)                                                                                                                                                                                                                |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 | Swing-Impact-Ball-Flight platform. Open. Stack PR #4119 closed; content landed in slices. Remaining: camera cluster (#4571) and Phase 7 (WASM web parity, Pages CI).                                                             |
| #4120 | Investigation & Variation Suite. Open. PR #4124 **merged**.                                                                                                                                                                      |
| #4125 | Realistic clubs / kinetics / putting / showcase. Open. PR #4129 **merged**. H5 (public release-management repo) still not started.                                                                                               |
| #4130 | Impact-interval club dynamics. **COMPLETED** (F1–F4 landed in PR #4577) — 6-DOF transient package, tests, and impact wire.                                                                                                       |
| #4142 | Ensemble variation, quiet zones, sensitivity attribution. Open. Merged PR #4703 advances R15.4 and the ledger to 22 verified / 9 partial; R14.6 and eight other explicit gaps remain.                                            |
| #4146 | Shared Club Builder. Open. Assembly physics contracts landed in #4157.                                                                                                                                                           |
| #4433 | Visual-first tab visibility and visualization-led UX. Open. The V0.1 candidate advances the 31-item fail-closed audit to 7 verified / 24 partial obligations; R14.6 retains seven exact gaps and two human actions.              |
| #4430 | Qualified rotating-base companion. **COMPLETED** via #4618/#4619; UpstreamDrift consumed the immutable provider through #8954.                                                                                                   |
| #4549 | Club Fitting Tester (OEM-grade). **COMPLETED** (#4557, #4577) — C1–C7 delivered (mesh inertia, shaft delivery, OEM doc, counterfactuals, PyQt6/React GUI tabs).                                                                  |
| #4562 | Heavy Hit - hand/body coupling at impact. **COMPLETED** (#4568, #4577) — H1–H4 delivered (coupled mechanics, MJCF/URDF/.osim import, GUI readout).                                                                               |
| #4583 | Professional launch-monitor program. Release A and source-backed SG are merged; #4603 adds canonical dataset/covariation consumers through the ordinary protected flow. Release B physical collection remains external and open. |
| #4698 | Coordinate-explicit pendulum force attribution and impulse optimization. Active on `feat/4698-force-attribution`; schema `force-attribution/v1` is the planned Upstream boundary.                                                |

Per-tool detail: `src/rate_of_closure/AGENT_HANDOFF.md`, `src/pendulum_simulator/AGENT_HANDOFF.md`, and `src/rotation_converter/AGENT_HANDOFF.md`.

## Open PR Situation — Read Before Filing Anything

No PR is a draft. Query `gh pr list` before acting; the current queue below is
an exact 2026-08-25 snapshot and can change independently of this handoff.

| PR    | Scope                                                            |
| ----- | ---------------------------------------------------------------- |
| #4705 | Fail-closed #4433 acceptance audit and R14.6 gap correction      |
| #4687 | RotationConverter fixed-array JSX optimization                   |
| #4686 | CSV-export allocation optimization                               |
| #4685 | Immutable Upstream R15 evidence and #4142 handoff reconciliation |
| #4682 | P1AM FastAPI route introspection and dependency pins             |
| #4677 | Automated v1.9.0 release bump                                    |
| #4676 | Automated v1.8.0 release bump                                    |

**Current #4142 state.** PRs #4669 and #4674 are merged. Tools PR #4703 merged
as `57b443201b402fc110ec5623885c7e310d6ad6d3` from exact contribution head
`e1d8d098d038ae2cf6bc5ace7c4864ef1df05ed1`; the merge commit was verified as
remote `main`. Protected UpstreamDrift
PR #9039 consumes exact Tools revision
`17474249b9267d0e73a779c1d72f231e7b8de39c` through a thin gateway with typed
outcomes and deterministic parity gates. PR #4703 advanced
the ledger to 22 verified, 9 partial, and zero unverified items by consolidating
theory, schemas, method assumptions, performance evidence, quick start,
reproduction commands, falsifiers, and scientific limits. #4142 remains open
because R14.6 is gated by #4433 and eight other requirements retain explicit
gaps. This is model-data and provenance evidence, not human or scientific
validation.
PR #4705 maps all 31 V0--V5 obligations and the completion condition. Trusted
run `32689177846` proves the initial React/PyQt tier, not all states or human
approval. PR #4731 restores detect-secrets with an unchanged baseline.
Branch `feat/4433-tab-authority` implements V0.1 with bounded purpose,
prerequisites, and reciprocal counterparts for all 20 workspaces.
PR #4683 subsequently fixed the Python 3.12 tolerance, Morris readiness, and
error-taxonomy regressions exposed by protected execution. Do not weaken the
provider contract or infer human validity from the corrected model-data gates.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — conventions, CI gates, and cross-repo dependency rules; Tools is a leaf consumed by UpstreamDrift and Gasification_Model.
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — §12 requires a dated row for every PR touching `src/`.
4. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for a new tool's handoff doc.

## Gate Commands (Repo-Wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check (Ruff, NOT Black)
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m contract                     # API contract tests (downstream-facing)
python3 -m pytest -m integration --timeout=60     # cross-repo integration
```

SPEC freshness (CI job `spec-freshness` in `.github/workflows/spec-check.yml`):
any PR touching `src/**`, `tests/**`, `config/**`, `pyproject.toml`,
`Cargo.toml`, `package.json`, or `requirements.txt` must also modify
`SPEC.md` in the same PR, or carry the `spec-exempt` label. This includes
`src/<tool>/AGENT_HANDOFF.md` edits, since they live under `src/`.

Note: `ruff format --check` reports four pre-existing failures under
`src/data_processing/data_processor/python/tests/` — on `main`, not yours.

## Do-Not List

- **Do not reopen-and-rebase the closed `codex/4142-*` / `codex/4433-*`
  branches.** They are superseded by merged #4473 and have diverged far enough
  that merging one would remove current `main` content.
- **Do not add per-file Python version guards.** `conftest.py` reads each
  package's declared `requires-python` and skips below it (`CLAUDE.md`).
- **Do not append a dated entry to this file.** It hit 2,708 lines that way,
  18x the limit. Put history in the commit message.
- Do not regenerate `tests/sidekick_api_baseline.json` with
  `--regenerate-api-baseline` — it blesses existing drift, not just your change.
  Hand-edit it and coordinate a downstream migration.
- Do not change public signatures in `src/shared/python/**` without coordinated
  migration issues in UpstreamDrift and Gasification_Model (`CLAUDE.md`).
- Do not edit the shared Python surface from inside a consumer's vendored copy;
  Tools is the source of truth (#4495 exists because three fixes were orphaned).
- Do not import across package boundaries — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths. The repo-wide protected budget is 1,200 lines.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not assume delta CI covers you. Changed-file selection hides whole-tree
  debt on the lint and type gates until a PR touches enough shared files.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web` —
  Phase 7 of #4103 owns this.
- **Do not `git add -A` after a stash pop or merge.** Raw conflict markers were
  staged into `SPEC.md` and pushed twice this way. `grep '<<<<<<<'` and confirm
  `git diff origin/main -- SPEC.md` is exactly one insertion before pushing.
- Do not remove a stdlib-XML `# nosec` without replacing the justification —
  `quality-gate` runs bandit (B405/B314) and the repo's convention is
  nosec-with-a-written-reason, not swapping in `defusedxml`.

## Short-Term Roadmap (Ordered)

1. **Sidekick Unified Integration**: implement the S1–S5 plan in
   `docs/development/epic_sidekick_unified_impact_model_and_launcher_integration.md`.
2. Restore the isolated advisory benchmark lane through #4582.
3. **Land the camera-cluster epic #4571** so #4466 can finally close.
4. Advance #4433's 24 partial audit items without promoting initial-state evidence; manual AT and rendered-review approval remain human actions.
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy.
6. #4125 H5: stand up the public release-management repo (cross-repo).
7. Approve #4600's inspected post-merge PyQt launch-monitor visual reference.
