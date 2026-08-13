# Launch Monitor Analytics Handoff

## Scope and repository state

- Repository: `D-sorganization/Tools`
- Draft pull request: `#4212`, `feat(rate-of-closure): add launch monitor analytics tabs`
- Head branch: `feat/4205-launch-monitor-analytics`
- Base branch: `feat/4181-launch-monitor-registry`
- Last remote head before this CI recovery: `4b22e79cf829bac12217e60634ffbfbea5c40d6b`
- Published CI-recovery implementation: `ee917d80f`
- Related integration pull request: `#4217`
- Consumer pull request: `D-sorganization/UpstreamDrift#8369`

PR #4203's next current-head repair also removes seven inherited Python
3.11-only `enum.StrEnum` runtime imports from the Rate/shared swing dependency
surface. The repository compatibility helper supplies identical string-enum
wire behavior on Python 3.10, while `TYPE_CHECKING` preserves native enum types
for pinned mypy 1.13. This is a compatibility-only parent correction: launch
monitor schema, registry values, analysis logic, and UI behavior are unchanged.
Focused evidence is 64 tests plus a real CPython 3.10.20 probe; propagate the
new exact parent through the stack after guarded publication.

The parent follow-up also routes the torque-profile controller's UTC constant
through the shared compatibility module. It does not change launch-monitor or
torque-profile data; it removes one additional Python 3.11-only import from the
Rate package collection surface.

PR #4203 current-head CI recovery is intentionally limited to Linux test
collection: the in-package flight and solver facade-contract tests now use
relative package imports so pytest's `src.shared...` collection namespace does
not cross into the editable `shared...` alias. Run `31199764932` is diagnostic
evidence for the old head; its Rust missing-`libpython3.11` failure is runner
infrastructure. No launch-monitor analytics contract or runtime value changes.
The repaired modules pass `12` tests on both Windows and WSL Python 3.11 under
importlib collection; Ruff/format and pinned mypy 1.13 pass.

The analytics implementation supplies UI-neutral Python and TypeScript contracts,
statistics, parsing, and dataset fingerprinting behind stable facade modules. The
facades are intentionally retained so the PyQt6 and React clients do not depend on
private module layout.

## Completed hardening

The production modules were split by responsibility without changing their public
facade APIs. The resulting focused Python and React validation completed before
publication:

- 583 Rate of Closure Python tests passed.
- 374 React tests passed.
- TypeScript type-check, ESLint, and the production build passed.
- Python 3.12 mypy, Ruff, Black, and the repository module-size gate passed.
- Python 3.13 mypy encountered an upstream internal cache assertion; this is not
  claimed as a successful lane.

## Exact-head CI recovery

The first protected run on remote head `4b22e79c` exposed two deterministic issues:

1. CI-pinned Ruff 0.14.10 required one formatter-only line break in
   `src/rate_of_closure/launch_monitor_analysis.py`.
2. `detect-secrets` classified the published SHA-256 test vector for `"abc"` as a
   high-entropy credential. The exact test-vector line now carries the scanner's
   narrow `pragma: allowlist secret` annotation; no baseline entry or broad
   exclusion was added.
3. The next protected run reached pinned mypy 1.13 and rejected raw Qt combo-box
   strings at three Literal-typed request boundaries, then inferred one reused
   loop variable as incompatible correlation and coefficient types. The adapter
   now narrows the three values at the UI boundary and uses type-specific loop
   names; the analysis contract itself was not weakened.

Local recovery evidence:

- `ruff check` and `ruff format --check`: passed.
- Focused `detect-secrets` scan: zero findings.
- `launchMonitorAnalysis.test.ts`: 5 passed.
- TypeScript type-check and ESLint with zero allowed warnings: passed.
- Python fail-closed/missingness boundary test: 1 passed.
- PyQt analytics tab tests: 2 passed.
- Exact mypy 1.13 check of the corrected PyQt adapter: passed.
- `git diff --check`: passed.

The full Python analytics file is comparatively expensive in this Windows test
environment and exceeded a three-minute wrapper during the recovery rerun. The
earlier 583-test hardening run remains the full-suite evidence; the recovery changes
are formatter-only Python and a scanner-only TypeScript comment.

## Remaining release gates

- Verify that the current branch head contains `ee917d80f`, then require a new
  protected run on that exact head; do not reuse results from `4b22e79c`.
- Resolve any new actionable failures and wait for queued repository-runner jobs.
- Obtain required review and resolve all review threads before undrafting or
  merging.
- Preserve the declared PR stack and dependency order. Do not force-push,
  retarget, admin-merge, or bypass protected checks.
- Reconcile the exact released Tools dependency in UpstreamDrift before claiming
  consumer release completion.

## Next epic

Tools epic `#4218` and children `#4219` through `#4225` track the modern top
toolstrip, persistent File operations, Glossary/Theme/hotkeys, module visibility,
Impact/Swing/Flight multi-view compositor, and distinct plot/legend controls. Per
the epic sequencing contract, implementation starts only after the current
ball-flight/variation/wedge campaign reaches its declared completion gate.

## 2026-08-07 ground-distance authority note

Branch `feat/4268-ground-contract`, stacked from PR #4283 head
`60ac5b46c78988225862d9b89a33ddc3656a3413`, adds the strict
`flight-to-ground-request/v1` and `flight-to-ground-result/v1` authority for
issue #4268. Launch-monitor total distance and roll remain unavailable unless a
complete qualified ground result is explicitly projected through
`to_ground_model_result`; carry is never substituted. The implementation and
both handoff updates are committed together as `SELF`. Issue #4269 must supply
terminal 3D angular velocity and physical sphere/terrain contact bracketing
before analytics can consume the new contract.

The v1 boundary now preserves signed pre/post angular state on every ground
event and carries typed field/reason/provenance evidence for unavailable input.
Its strict JSON entry points reject duplicate keys, unsafe cross-runtime
integers, surrogate text, and raw range violations before canonical rounding.

Visualization child issue #4284 now extends epic #4218 with bounded clubhead
camera tracking and canonical Face On, Down the Line, and Overhead snap views
for matched PyQt/React 3D animations.
