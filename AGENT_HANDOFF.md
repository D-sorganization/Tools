# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-10.

## PR #4282 workspace timestamp propagation

The current carrier remains `feat/4199-wind-workflow` with base
`feat/4199-wind-scalar-adapter`. Exact corrected parent
`cf52529b1e68479321bb93b1be3d59c77f782008` is incorporated through the normal
merge containing this handoff. No branch was rebased, retargeted, rewritten,
or force-pushed. The consolidated wind/capability workflow retains its release
authority and inherits the deterministic Python 3.10-3.12 workspace timestamp
parser plus the complete variation and scalar-ensemble history.

The monotonic SPEC assigns 1.14.10 to compatibility, 1.14.11 to variation,
1.14.12 to the wind scalar adapter, and 1.14.13 to this workflow carrier.
The reconciled tree passes `924` combined Python tests, `28` direct
compatibility tests on real CPython 3.10.20, and `104` React files / `642`
tests. The Vite production build, TypeScript, zero-warning ESLint, Ruff,
format, pinned mypy 1.13, campaign-manifest validator and its eight contracts,
documentation governance, source-size, conflict-marker, and diff gates are
clean. Protected CI, review, and downstream propagation remain separate
release gates.

## Active Rate of Closure campaign

The active checkout is
`C:\Users\diete\Repositories\Tools-worktrees\toolstrip-workspace` on the
existing PR #4282 carrier `feat/4199-wind-workflow`. The local continuation
starts from exact published head
`de49580a3c0888b44f66dcc09bba2ab2fa33914a` and normally incorporates exact
corrected #4281 parent `cf52529b1e68479321bb93b1be3d59c77f782008`
without changing the branch base `feat/4199-wind-scalar-adapter`. It composes
four reviewed slices:
strict capability parsing and signed decimal entry; complete capability
diagnostics/result exports and quantitative React scatter scales; package-safe
static-web release entrypoints; and the strict `rate-of-closure-campaign/v1`
release-evidence authority. The normal merge also carries the parent's
Python 3.10 compatibility, variation-export, and scalar-ensemble corrections.
No branch was rebased, retargeted, rewritten, or force-pushed. Protected CI
and review remain due after normal publication.

Canonical files:

- `docs/release/rate_of_closure_campaign.v1.json` — normalized campaign state;
- `scripts/rate_campaign_manifest.py` — schema and contradiction validator;
- `docs/release/RATE_OF_CLOSURE_CAMPAIGN_MANIFEST.md` — maintenance procedure;
- `src/rate_of_closure/AGENT_HANDOFF.md` — current Rate-specific handoff;
- `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` — historical detail.

The manifest deliberately reports the campaign as **not released**. It
distinguishes specified scope, feature-stack implementation, protected parent
merge, and exact `main` release. A local pass, a feature-branch merge, and a
default-branch release are never interchangeable.

## Current carrier and release state

Capability PRs #4294, #4289, and #4283 were merged top-down into
`feat/4199-wind-workflow`; their feature parents were not protected. PR #4282
is the current open carrier on base `feat/4199-wind-scalar-adapter`. Exact
corrected parent `cf52529b1e68479321bb93b1be3d59c77f782008` is incorporated
through the normal local merge containing this handoff.

The previous exact #4282 head `3186a265b1` built and loaded `swing_core`, but
its Python 3.11 parity job failed before collection because a cached
`pytest-qt` plugin loaded without PyQt6. Commit `18fe89201` disables third-party
pytest plugin auto-loading only in that Rust parity step. Its focused workflow
contract test passes; a fresh hosted parity run is still required.

Combined local evidence on the composed continuation is 828 Rate Python/PyQt
tests and 104 React files / 642 tests, plus TypeScript, zero-warning ESLint,
and the 188-module Vite production build. The deterministic manifest,
generated-schema JSON, Ruff, targeted mypy, and nine manifest/parity contracts
also pass on implementation head `2c1a77baa`.

Hosted quality-gate run `31340032608` reached mypy 1.13 and exposed
CI-context-only `no-any-return` findings in the Pydantic manifest loader and
Qt elapsed-timer adapter because the delta lane uses `--follow-imports=skip`.
Both boundaries now narrow their return values explicitly. The exact Python
3.12/mypy 1.13 delta is clean across 54 files; Ruff passes and 62 focused
regression tests plus eight campaign-manifest tests pass (only pre-existing
optional-plugin config warnings).

Corrected-parent propagation evidence is 62 focused wind/scalar/variation and
compatibility tests on both Python 3.11 and real CPython 3.10.20, plus 8 React
files / 35 tests, TypeScript, and focused zero-warning ESLint. The Python 3.10
run exposed one child-owned direct `enum.StrEnum` import in capability
observations; it now uses the shared runtime compatibility contract and is
included in the source-level regression. Ruff check/format passes 15 focused
files, pinned mypy 1.13 passes 10 production modules, and all nine campaign
manifest/parity contracts pass.

The direct web launcher dynamically loads the root bootstrap through
`importlib` instead of mutating `sys.path` in the changed entrypoint. Its real
child-process delegation test and the changed-Python policy guard cover that
release path.

Only `main` is the release boundary. It requires `quality-gate` and
`tests (3.11)`. Outer PR #4119 remains the main-targeting platform carrier and
requires current-main reconciliation. Impact-interval PR #4133 merged after
its parent stack had already propagated and is not proven in #4119.

## Critical open programs

- #4142/#4144: variation graphics are substantial, but global sensitivity,
  bounded performance, downstream pinning, and release remain.
- #4146: club-builder core exists; manufacturing export, image fitting, full
  client workflow, and downstream qualification remain.
- #4158: wedge kinematics, clearance, turf, visualization, and forgiveness are
  feature-stack implementations without a protected release.
- #4191/#4201: wind, inverse flight, playback, and capability work exist on the
  stack; scientific, installed-package, and cross-surface release gates remain.
- #4218/#4234: toolstrip and selected layout fixes exist; camera issue #4284,
  complete persistence, high-DPI, keyboard, and visual baselines remain.
- #4260/#4267: four-surface parity is specified; ground contracts/transfer are
  partial and bounce, skid, roll, total distance, UI, and parity remain open.

## Required validation

```powershell
python scripts/rate_campaign_manifest.py
python -m pytest tests/rate_of_closure/test_campaign_release_manifest.py -q
python -m pytest tests/ops/test_maturin_swing_core_workflow.py -q
python -m ruff check scripts/rate_campaign_manifest.py `
  tests/rate_of_closure/test_campaign_release_manifest.py
python -m ruff format --check scripts/rate_campaign_manifest.py `
  tests/rate_of_closure/test_campaign_release_manifest.py
python -m mypy --ignore-missing-imports scripts/rate_campaign_manifest.py
```

For a feature change, also run the complete affected Rate Python/PyQt/shared
swing and React suites recorded in the per-tool handoff.

## Do not

- Do not mark a program released without a 40-character `main` merge SHA.
- Do not convert queued, cancelled, skipped, or failed checks into passing
  evidence.
- Do not retarget or rewrite the protected stack, force-push, or admin-merge.
- Do not duplicate carrier or test metadata in individual program records.
- Do not treat an UpstreamDrift launcher tile as four-surface parity.

Fleet handoff policy is tracked by Repository_Management #1393/#1397. Any
`src/**` handoff change also requires the same-commit `SPEC.md` update.
