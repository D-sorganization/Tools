# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-08.

## Current Rate of Closure continuation

The active checkout is
`C:\Users\diete\Repositories\Tools-worktrees\capability-flight-evaluator` on
branch `feat/4197-capability-flight-evaluator`. It is based exactly on the
published capability-observation branch at `49612946138b1021f80c9f8d2a4d06f1610825db`
(draft PR #4283). Preserve that parent relationship: do not retarget, rebase,
force-push, or merge this child ahead of its protected stack.

This continuation supplies the missing real forward evaluator for issue #4197.
Python and TypeScript now bind an existing `player-capability-profile/v1` and
`capability-optimization-request/v1`, validate requested clubs, parameter IDs,
units, finite values, declared safe bounds, and physical flight domains, then
run the actual Waterloo/Penner flight model. The adapters convert trajectory
and spin into the canonical target frame, include the configured target in
metric derivation, and return all available scalar `ball-flight-result/v1`
metrics to the optimizer. Three-variable profiles require a sourced spin
default for each requested club; there is no global driver fallback. Profiles
may instead vary `total_spin` and `spin_axis_tilt` together. Positive tilt is
the app-native fade/right convention, and each result records the spin source.

Typed behavior is deliberate:

- a qualified ground crossing returns `complete`, including carry/offline and
  target residuals;
- a horizon without ground crossing returns `nonconverged` with no partial
  metrics;
- expected Python floating-point overflow returns `failed` with a stable
  non-leaking reason, while programming/contract errors surface;
- schema, unit, physical-domain, or safe-bound violations raise before physics;
- this post-impact launch evaluator never fabricates `no_impact`.

Current local evidence after the independent review corrections:

- all shared flight tests: 138 passed / 4 optional-Rust skips;
- complete React suite: 97 files / 597 tests passed;
- shared 16-scalar parity fixture, both tilt signs, per-club default
  provenance, physical domains, and supported coarse sampling are covered;
- one shared gyro-projected tilt function now backs result, impact, and
  variation producers in both runtimes;
- Ruff, formatting, targeted mypy, TypeScript, and zero-warning ESLint pass;
- Vite production build passes with 176 modules transformed;
- new production modules are below 400 lines and functions below 50 lines.

The next implementation slice is the user-facing capability optimization
workflow in PyQt6 and React: profile/club/target/environment editing, worker
progress and cancellation, observation-to-`scalar-ensemble/v1` scatter/table/
CSV, versioned persistence, and rendered interaction review. Issue #4197 must
remain open until that UI, protected CI, review, merge, and downstream parity
are proven.

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
python -m pytest src/shared/python/swing_sim/flight/tests -q
python -m ruff check src/shared/python/swing_sim/flight
python -m ruff format --check src/shared/python/swing_sim/flight
python -m mypy src/shared/python/swing_sim/flight/capability_flight_evaluator.py
cd src/rate_of_closure/web
npm test -- --run
npm run type-check
npm run lint -- --quiet
npm run build
```
