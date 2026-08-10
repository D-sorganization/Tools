# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-10.

## PR #4288 corrected-ground propagation

Draft PR #4288 remains on `feat/4269-flight-ground-transfer` with unchanged
base `feat/4268-ground-contract`. Exact corrected #4285 parent
`788aa547651a3685a363ea401824a5d81477bafb` is incorporated by the normal merge
containing this handoff. The child preserves its signed terminal state,
physical sphere/terrain contact brackets, strict provenance, and qualified
Python/TypeScript/Rust/PyO3/WASM transfer while inheriting deterministic
workspace timestamps and canonical `swing_sim` import identity. No branch was
rebased, retargeted, rewritten, or force-pushed.

This propagation adds no bounce, skid, roll, terrain response, total distance,
or UI execution. Protected CI, independent review, and dependency-ordered
collapse remain separate release gates.

Merged-tree validation is `951` Python tests, `106` React files / `661` tests,
and `26` focused Rust transfer/wire tests. The affected cross-version suite is
also `110` tests on real CPython 3.10.20. The 189-module Vite production build,
TypeScript, zero-warning ESLint, Ruff check/format across 82 files, pinned mypy
1.13 across the six-file transfer and 13-file parent production namespaces,
Rust workspace format plus warning-denied `tools-core` clippy, campaign-manifest
validator plus eight contracts, documentation governance, 13-file 500-LOC
budget, marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.

## PR #4285 workspace timestamp propagation

Draft PR #4285 remains on `feat/4268-ground-contract` with unchanged base
`feat/4197-capability-observer`. Exact corrected #4282 carrier
`5f77af4add23547a21cc3fabce98ae9ad4260427` is incorporated through the normal
merge containing this handoff. No branch was rebased, retargeted, rewritten,
or force-pushed.

The child retains its strict, UI-neutral flight-to-ground schemas, canonical
fixture, migrations, legacy-result adapter, explicit dependency, and pinned
typing repairs while inheriting the deterministic Python 3.10-3.12 UTC parser
and the corrected variation, scalar, wind, capability, and campaign-release
ancestry. This merge does not claim bounce, skid, roll, terrain profiles,
total distance, UI execution, or Rust/WASM delivery. Protected CI, review, and
normal descendant propagation remain separate release gates.

Exact-head CI on the preceding child head exposed an actionable collection
defect after installing the package: embedded `src.shared.python.swing_sim`
tests and canonical `shared.python.swing_sim` imports could load distinct
package trees, making the ground and impact subpackages unavailable by import
order. The shared alias registry now coalesces `swing_sim`; an isolated RED/GREEN
identity contract and both affected public-API suites pin the correction. The
separate file-size job was cancelled during checkout and did not execute its
budget check, so it remains infrastructure evidence rather than a source
failure.

Merged-tree evidence is `915` Rate/ground/impact Python tests, the `12`-test
shared-alias architecture suite, `28` focused ground/impact/alias tests on
CPython 3.12, `28` direct compatibility tests on real CPython 3.10.20, and
`104` React files / `642` tests. The 188-module Vite build, TypeScript,
zero-warning ESLint, Ruff, format, pinned mypy 1.13, campaign-manifest
validation plus eight contracts, documentation governance, the 44-file
500-LOC budget, conflict-marker scan, and diff checks are clean.

## 2026-08-09 Flight-transfer corrected-parent propagation

Draft PR #4288 remains on `feat/4269-flight-ground-transfer` with unchanged
base `feat/4268-ground-contract`. Exact carrier-reconciled #4285 parent
`6a2bc9d06f6f9a28a0d615b19d2ed4fc13871059` is incorporated through the
normal local merge containing this handoff; no branch was rebased, retargeted,
force-pushed, or published. The result retains the qualified signed terminal
state and physical contact transfer across Python, TypeScript, Rust, PyO3, and
WASM while inheriting the corrected wind/scalar/variation, capability,
Python-3.10, campaign-authority, and strict-ground ancestry.

The public flight-facade conflict was resolved semantically: the child keeps
its structural frozen-dataclass protocol and transfer API inventory, while the
parent's package-relative import preserves Linux/editable collection. No
bounce, skid, roll, terrain response, total distance, or UI delivery is added.
Protected CI, independent review, and exact child-first merge remain required.

Focused evidence is 113 strict-ground, flight-transfer/facade, compatibility,
scalar-adapter, and responsive-wind tests on Python 3.11 and the same 113 on
real CPython 3.10.20. Ruff check/format passes 36 focused Python files. Pinned
mypy 1.13 passes the 13-file transfer delta and 12-file ground production set
in their established separate namespace invocations. The type gate required
binding each terminal trajectory sample before `FlightStatePoint` narrowing;
runtime assertions and physics are unchanged. The inherited campaign manifest
validates and its nine manifest/parity contracts pass. Transfer modules remain
within 400/50-line structural budgets; the sole placeholder scan hit is the
intentional fail-closed base-model `NotImplementedError` extension boundary.

## 2026-08-09 Flight-to-ground transfer parent propagation

Draft PR #4288 remains on `feat/4269-flight-ground-transfer` with unchanged
base `feat/4268-ground-contract`. This checkout incorporates exact published
parent head `8e8df7b9c633affb986326137338313faf46d2db` through a normal merge;
neither branch was rebased, retargeted,
force-pushed, or merged on GitHub. The child retains its extracted
`flightIntegrator.ts` rather than restoring the parent's superseded inline RK4
implementation, and the Python public-contract inventory includes the parent's
two capability-evaluator dataclasses alongside the child's transfer types.
Focused merge testing also caught a circular package-facade dependency:
`ground.__init__ -> result_adapter -> flight.__init__ -> ground_transfer ->
ground.__init__`. The transfer adapter now imports the exact ground record/type
modules it consumes, preserving the package facades while satisfying LoD.
The reconciled branch passes the complete affected Python gate: `1483 passed,
7 skipped` across `tests/rate_of_closure` and `src/shared/python/swing_sim`;
all skips are optional local Rust-wheel paths. Focused transfer/contract Python
tests are `82 passed`, focused React transfer/capability tests are `38 passed`,
and focused Rust transfer/wire tests are `26 passed`.
The complete React suite is `104 files / 643 tests passed`; type-check, lint,
and the production Vite build pass with the main bundle at 476.51 kB. Full
`tools-core` Rust validation is `137 passed` (111 unit, 20 transfer, 6 wire).
Changed Python Ruff check/format and CI-pinned mypy 1.13 pass, as do docs
governance and staged/unstaged diff checks.

The propagated parent is the pinned-mypy schema repair documented below. Its
wire-neutral `str(...)` boundary and explicit adversarial-test casts introduce
no transfer conflict. Re-run the focused transfer/ground suites and the exact
changed-file mypy 1.13 profile on this merged child before publishing it.
That post-merge verification is now complete: `70` focused ground/transfer/API
tests pass, and the stronger mypy 1.13 run is clean across all `13` changed
Python files, including tests. The frozen-dataclass inventory uses an explicit
structural protocol for its introspection boundary, preserving the assertion
while avoiding a skipped-import union ambiguity. Ruff check/format, test
assertion policy, docs governance, and diff checks also pass.

Before propagation, remote PR #4288 was cleanly stacked at
`d2d3d0f53a78aa863574afe43290a29c48318d94`, had no reviews or unresolved
threads, and remained draft/unstable because hosted checks failed. The Python
3.12 log's only numerical assertion is in the separate shared wind fixture:
`test_python_matches_the_shared_cross_client_wind_fixture` differed by
`3.494e-12` against a `1e-12` absolute tolerance. No flight-to-ground transfer
tolerance failed, and this branch does not modify the wind workflow. The Rust
`-lpython3.11` linker failure is runner/toolchain infrastructure.

## 2026-08-09 Strict ground-contract base propagation

The first protected run on published head
`2d9a06fae46e0601a05896b71934ca0c6b8dc59a` reached the exact pinned mypy
1.13 gate and failed in `ground/json_schema.py`: with unchanged imports skipped,
the Python 3.10-compatible string-enum boundary was represented as `str`, so
the checker could not prove enum iteration or `.value` access. The scoped
follow-up builds wire enum values through `str(item)` and uses `str(...)` for
the fixed target frame. Deliberately invalid test inputs now use explicit
casts instead of stale `type: ignore` comments; runtime validation semantics
are unchanged. The exact CI profile
`--ignore-missing-imports --follow-imports=skip` passes under mypy 1.13 for all
19 changed Python files, Ruff check/format passes, and the focused ground suite
is `46 passed`. Run `31341468033` is evidence for the diagnosed old head, not
green evidence for this follow-up. Publish normally, then merge the resulting
parent head into PR #4288 and re-verify that child; do not retry the obsolete
failed run.

Draft PR #4285 remains on `feat/4268-ground-contract` with the unchanged base
`feat/4197-capability-observer`. This checkout incorporates the current parent
head `9bbb98e16e435a0d4c74153b909f2ebfefbbce7a` through a normal merge commit;
the branch was not rebased, retargeted, force-pushed, or merged on GitHub. The
only textual merge conflict was this root handoff. The ground schemas,
canonical fixture, migration, and legacy result adapter did not conflict with
the capability evaluator/workspace implementation.

The pre-propagation PR head `3235af71150a774954e7673fc81d7179330fbe76`
still had no reviews or unresolved review threads. Its hosted Python 3.11/3.12
lanes exposed an undeclared `jsonschema` test dependency, while the Rust gate
failed because the runner could not link `-lpython3.11`. Treat the latter as
runner/toolchain infrastructure, not ground-model evidence. Re-run focused
contract and affected Rate gates on the merged ancestry before publishing any
follow-up, and keep issue #4269 / PR #4288 stacked behind this contract PR.

The bounded follow-up `2025b504fb3e308a4141b1c20df6a88e05a59d1f` declares `jsonschema>=4.23.0` in the repository's
test/quality dependency set, pins the verified 4.24.0 build in the lock file,
and routes all three new ground-contract enums
through the existing shared `StrEnum` compatibility boundary. A package-wide
AST regression test was red against `contract_types.py`, `contract_wire.py`,
and `unavailable_types.py`, then green after the imports were corrected. The
complete ground package is `46 passed`; the affected combined Rate+swing_sim
suite is `1463 passed, 5 skipped` (optional local Rust wheels), with Ruff
check/format, targeted mypy, documentation governance, and diff checks clean.
Python 3.10 failures originating in older capability/Rate modules
remain outside this ground-only repair and must not be hidden by broad edits.

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

The historical capability optimization checkout is
`C:\Users\diete\Repositories\Tools-worktrees\capability-optimization-ui` on
branch `feat/4197-capability-optimization-ui`. Its functionality is now part of
the corrected #4282 ancestry incorporated above. Preserve the protected stack;
do not retarget, rebase, force-push, or merge descendants out of order.

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

The #4285 ground-contract carrier incorporates #4282 exact head
`5f77af4add23547a21cc3fabce98ae9ad4260427` by normal ancestry. Keep downstream
ground PRs stacked behind the resulting exact #4285 head.

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
