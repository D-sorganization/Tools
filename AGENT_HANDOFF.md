# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-09.

## LOCAL CONTINUATION (2026-08-09): capability wire/input hardening

Branch `feat/4201-capability-contract-hardening` is based on exact carrier
head `18fe89201d657116bbca99922297c14968356c44`. This local-only continuation
removes the remaining Python workflow parser coercions, requires exact JSON
primitive types throughout the v1 capability document, and uses one shared
golden accept/reject fixture in Python and TypeScript. Mathematically integral
JSON numbers remain valid for integer fields; numeric strings, booleans,
fractional integers, and numbers in text fields fail closed.

The React Shot Optimizer numeric editor now keeps a decimal text draft so a
normal keyboard sequence can enter `-3.5` without the browser number input
discarding the leading minus sign. A `userEvent` regression test exercises the
real clear/type/tab/run interaction and verifies the dispatched numeric value.

TDD evidence before implementation: Python `8 failed, 16 passed`; React
`2 failed, 26 passed`. Final focused capability evidence: Python `43 passed`;
React `9 files / 69 tests passed`; Ruff check and format, focused mypy,
TypeScript type-check, zero-warning ESLint, structural budgets, and
`git diff --check` pass. The Vite production build passes with 187 modules and
no size warning. This branch has not been pushed and no GitHub state was
changed.

## STABILIZATION RECORD (2026-08-09): interpretable capability results

The isolated `feat/4201-capability-results-diagnostics` continuation is based
on exact carrier head `18fe89201`. It makes ranked optimizer output fully
inspectable in both PyQt6 and React: score, miss CVaR, downside carry,
success/no-impact/failure counts, failure fraction, evidence-envelope status,
Pareto status, limiting constraints, and parameter units are now visible.
React scalar scatters add numeric grid/ticks, explicit axis ranges, and a
cohort legend. Both clients add versioned result JSON and spreadsheet-safe
ranked-alternative CSV exports; raw observation exports remain separate.

RED/GREEN coverage spans the export contract, React results/scatter, and PyQt
workflow. Local verification passed 813 Rate Python/PyQt tests and 104 React
files / 628 tests, plus Ruff, targeted mypy for the new export and tab modules,
TypeScript, zero-warning ESLint, and the 188-module production build. The
pre-existing mypy 1.19.1/Python 3.13 internal serialization assertion remains
reproducible on the unchanged `capability_results.py` base and is not a source
diagnostic. This work is not released merely because it is committed: integrate it into
`feat/4199-wind-workflow`, rerun exact-head hosted CI, and preserve #4282's
existing base and release order.

## RELEASE STABILITY RECORD (2026-08-09): Rate web entry and packaging truth

The isolated branch `feat/4201-web-release-stability`, based exactly on carrier
head `18fe89201`, repairs three bounded release blockers without changing
physics or application UI behavior. The documented direct command
`python src/rate_of_closure/launch_web.py` now imports `GUI_INFO` through the
package-qualified path and is protected by a subprocess delegation smoke test.
Rate now declares only release surfaces that exist: PyQt6/PyInstaller desktop
and a static Vite web bundle. The nonexistent Tauri wrapper, scripts, CLI
dependency, and corresponding claims were removed from the manifest, lockfile,
README, executable-builder documentation, and current specification.

`FlightExplorerPanel.test.tsx` now settles the code-split Wind Strategy import
inside React `act` before making its synchronous assertion. This preserves real
lazy integration coverage while eliminating dependence on the testing
library's default polling timeout; no timeout was increased. Keep this branch
local until its focused and broad gates below are reviewed and it is integrated
through the existing campaign carrier. No PR was created or pushed here.

Local evidence on this branch: 813 Rate Python/PyQt tests passed with 15
pre-existing warnings; the focused registration/launcher file passed 9 tests;
all 102 React files / 624 tests passed; and the Flight Explorer file passed
five consecutive focused runs (25 tests total). TypeScript type-check,
zero-warning ESLint, the 187-module Vite build, focused Ruff check/format,
targeted mypy, `npm ci`, zero-vulnerability audit, and `git diff --check` pass.
The broader `tests/test_dry_compliance.py` remains red only for the unrelated
pre-existing Movement Optimizer and Optimizer GUI PyQt launchers.

## CI REMEDIATION RECORD (2026-08-09): hermetic swing-core parity lane

The exact PR #4282 head `3186a265b1` built and loaded the `swing_core` wheel,
but its Python 3.11 parity job failed before collection because the self-hosted
runner auto-loaded a cached `pytest-qt` plugin without PyQt6 installed. The
Rust-only parity step now sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; a focused
workflow regression test prevents the unrelated GUI-plugin dependency from
returning. This is a CI-environment isolation fix, not a physics change.

At this handoff, PR #4282 remains draft and unmergeable by policy: its new
exact-head CI has not run yet, several governance jobs remain queued, and no
review approval is recorded. Preserve its existing base and stack order.

## COMPLETION RECORD (2026-08-08): interruption recovered

The uncommitted capability-optimization-ui slice was reviewed against
`docs/specs/CAPABILITY_OPTIMIZATION.md`, re-verified, repaired, and
committed. The dying agent's "gates green" claim was partially false;
the recovery fixed: Ruff formatting/import-sort in three files, a
mypy-1.13 `call-arg` failure from positional-after-star bounds
unpacking in `capability_controls.py` (replaced with typed spec
factories), TypeScript errors in `CapabilityOptimizationPanel.test.tsx`
(untyped `vi.fn` mock), and an eager import that pushed the main Vite
chunk to 511 kB (the panel is now lazy-loaded like WindStrategyPanel;
main chunk back to 474.32 kB, no size warning).

Verified gates on the committed head: 1423 Python tests passed (808
`tests/rate_of_closure` + 615 swing_sim in-package, 0 skipped); 102
React files / 619 tests passed; Ruff check/format clean on all changed
files; CI-equivalent mypy 1.13 (`--ignore-missing-imports
--follow-imports=skip`) clean on all 10 changed src files; `tsc
--noEmit`, zero-warning ESLint, and the 187-module Vite build pass;
changed-only 500-LOC budget and `git diff --check` pass. The slice is
published as PR #4294 on `feat/4197-capability-flight-evaluator`, and
the capability stack was flipped ready-for-review in merge order:
observer #4283 → evaluator #4289 → this #4294.

## Current Rate of Closure continuation

The active checkout is
`C:\Users\diete\Repositories\Tools-worktrees\capability-optimization-ui` on
branch `feat/4197-capability-optimization-ui`. It is based exactly on evaluator
commit `c280407d432c153639bb266c9c721a014a129723`, published as draft PR
#4289 on `feat/4197-capability-flight-evaluator`. Preserve that parent
relationship: do not retarget, rebase, force-push, or merge this child ahead of
its protected stack.

This continuation supplies the matched end-user optimizer for issue #4197.
PyQt6 and React now author the same versioned profile, club, target, objective,
search budget, fixed-spin source, and deterministic seed; strictly save/load
`capability-optimization-workflow/v1`; execute the qualified Waterloo/Penner
evaluator off the UI thread; expose truthful progress and cancellation; and
retain every attempted sample in `scalar-ensemble/v1`. Results include ranked
alternatives, complete/no-impact/failed counts, selectable scalar axes,
autofit/zoom, an accessible paged raw table, lossless spreadsheet-safe CSV,
and stable JSON. The UI states that v1 is still-air carry to first ground
crossing and does not model wind, bounce, roll, or total distance.

Rendered browser and desktop review corrected three issues before publication:
duplicate diagnostic labels are stage-qualified, saved v1 layouts reveal newly
registered modules without undoing prior hide/show choices, and the PyQt
control/results split keeps both panes readable. Every new interactive PyQt
control has a frame-aware tooltip. Current verified local evidence is 808 Rate
Python/PyQt tests plus 615 swing_sim tests and 102 React files / 619 tests
passed. Ruff and formatting, CI-equivalent mypy 1.13, TypeScript type
checking, zero-warning ESLint, diff checks, and the 187-module Vite production
build (lazy-loaded Shot Optimizer chunk, no size warning) pass. New production
modules are below 400 lines and functions below 50 lines.

Publish this branch only as the next protected stacked draft PR. Issue #4197
must remain open until protected CI, independent review, merge order, and
downstream UpstreamDrift parity are proven.

## Durable monorepo guidance

Tools is the D-sorganization fleet's shared engineering-tools monorepo. It
contains PyQt6 applications, FastAPI/React mirrors, and Rust kernels consumed
by downstream repositories. Rate of Closure is only one package; preserve
unrelated tool boundaries and user changes.

Before changing public shared APIs, read:

1. `CLAUDE.md` for repo-wide CI and downstream dependency rules;
2. `docs/architecture/CANONICAL_TOPOLOGY.md` for repository topology;
3. `docs/AGENT_HANDOFF_TEMPLATE.md` before adding another tool handoff;
4. the target tool's own `AGENT_HANDOFF.md`.

Any source/config/dependency change must update the canonical `SPEC.md`
change log in the same PR unless an authorized `spec-exempt` path applies.
Do not modify a public signature under `src/shared/python` without a
coordinated migration for UpstreamDrift and other consumers. Do not import
across unrelated package boundaries, regenerate API baselines to hide a
breaking change, bypass hooks/checks, or create an ad hoc Pages workflow.

Fleet handoff policy is tracked by Repository_Management #1393 and enforcement
issue #1397: every implementation commit must update the repository-specific
handoff in the same commit, while no-material-change commits state that fact.

## Protected stack and critical cautions

- #4119 is still the outer platform PR and has unresolved integration/conflict
  risk; none of the Rate campaign is released merely because a nested PR is
  merged into another feature branch.
- #4280 adds complete selected-scatter CSV/raw-table parity for #4144.
- #4281 adds the shared wind scalar adapter; #4282 adds the PyQt/React wind
  workflow; #4283 adds capability observation/cancellation and scalar adapters.
- #4285 and #4288 are later ground-contract/flight-transfer descendants. Keep
  their publication blockers separate from this evaluator slice.
- Impact-interval PR #4133 is not present in the current #4119 head. Do not
  repeat the stale claim that it is already integrated; reconcile its files and
  tests explicitly before closing #4130.

Use the verified GitHub App CLI route in the same PowerShell process:

```powershell
. C:\Users\diete\codex-tools\setup-github-for-codex.ps1
```

Never bypass protected checks, rewrite parent branches, use an administrator
merge, or treat queued/skipped checks as passing evidence.

## Required reading

1. `AGENTS.md` for TDD, DbC, DRY, LoD, size, and GitHub rules.
2. `CLAUDE.md` for repo-wide CI and downstream dependency rules.
3. `docs/specs/CAPABILITY_OPTIMIZATION.md` for the evaluator contract.
4. `src/rate_of_closure/AGENT_HANDOFF.md` for the detailed Rate stack.
5. `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` for the campaign
   history and remaining cross-surface work.
6. `SPEC.md` section 12 for the required source-change freshness entry.

## Current validation commands

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest tests/rate_of_closure -q
python -m ruff check <changed-python-files>
python -m ruff format --check <changed-python-files>
cd src/rate_of_closure/web
npm test -- --run
npm run type-check
npm run lint
npm run build
```
