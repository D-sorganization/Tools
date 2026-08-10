# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-10

## 2026-08-10 PR #4304 corrected-impact propagation

Exact corrected #4302 parent
`846653c21bd61a40aab99ab838c29915d0728e70` is incorporated into
`feat/4271-ground-skid-roll` by the normal merge containing this handoff. PR
#4304 keeps base `feat/4270-ground-impact-bounce`; neither branch was rebased,
retargeted, rewritten, or force-pushed. The child retains arbitrary-plane skid,
pure roll, resistance, qualified rest, finite-edge handling, strict result
composition, and passive ledgers while inheriting corrected flight transfer,
deterministic workspace timestamps, and canonical `swing_sim` identity.

This remains a partial `not_released` slice. Regional/changing-normal surfaces,
terrain deformation, torsional spin damping, roll-to-skid transitions, UI,
TypeScript/Rust/PyO3/WASM physics, and downstream parity remain excluded.
Protected CI, independent review, and normal stack collapse remain open.

Merged-tree validation is `115` focused ground tests on both the current
runtime and real CPython 3.10.20, `1020` broad Python tests, `106` React files /
`661` tests, and the complete `tools-core` Rust suite at `137` tests (`111`
unit, `20` transfer, `6` wire). The combined compatibility/ground/flight/alias
suite is `179` tests on real CPython 3.10.20. The 189-module Vite production
build, TypeScript, zero-warning ESLint, Ruff check/format across 41 files,
pinned mypy 1.13 across all 25 ground and nine transfer production modules,
Rust workspace format plus warning-denied `tools-core` clippy, campaign-manifest
validator plus eight contracts, documentation governance, 18-file 500-LOC
budget, marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.

## 2026-08-10 PR #4304 deterministic-digest secret-scan repair

Protected detect-secrets run `31360998491` classified the version-locked
ground skid/roll golden-fixture SHA-256 digest as a high-entropy string on
exact head `d09f3129a68322bfc5dd30763556ac356ef2e55c`. The test now explicitly
allowlists that immutable non-secret digest without changing its value or the
fixture bytes. There is no physics, numerical, schema, or API change. SPEC
1.14.20 records the CI correction. All `115` ground tests, Ruff, formatting,
a finding-free local scan of the affected file, documentation governance, the
`370`-line source-size check, and diff gates pass; fresh protected CI/review
remain required after a normal guarded push.

## 2026-08-09 PR #4304 corrected implementation evidence

The campaign registry now advances PR #4304 and its immutable local evidence
to qualified implementation commit `f475ae85feea1b2c628f756699b2aba6ea9334fb`.
That commit is the narrow scalar-boundary correction for hosted run
`31354071845`; its 115-test CPython 3.11.9 and 3.10.20 suites, pinned MyPy
1.13, pinned Ruff 0.14.10, manifest contracts, and documentation governance
all pass locally. This registry/handoff commit is documentation-only and makes
no material runtime, physics, numerical, schema, or API change. Fresh
protected CI and review remain required after an ordinary guarded push.

## 2026-08-09 PR #4304 isolated-MyPy correction

Hosted quality-gate run `31354071845` (job `93350276996`) failed exact
published carrier `aaff1bc536653e90b1e629b91365f55b171bf689` with ten
`no-any-return` findings. Its changed-file MyPy 1.13 invocation deliberately
uses `--follow-imports=skip`, so imported NumPy-backed scalar helpers were
represented as `Any` at nine public and internal return boundaries.

The correction explicitly normalizes those already validated boundaries to
`float` or `bool`. It changes no physics equation, scalar value, integration
order, schema, serialized output, API, issue scope, or stack base. The exact
hosted invocation now passes locally, all 115 ground tests pass on CPython
3.11.9, and pinned Ruff check/format is clean for the changed production
files. A normal guarded fast-forward publication must trigger fresh protected
CI; the failed run is not retried or treated as evidence.

## 2026-08-09 draft PR #4304 publication

Draft PR #4304 now publishes `feat/4271-ground-skid-roll` at exact reviewed head `dcc801395538bdc7b9a46835f5555abdd72677a4` with unchanged base `feat/4270-ground-impact-bounce` at parent `920c46dee688815691e251777142126bf1489b1a`. The branch was pushed normally after verifying the GitHub App identity, clean worktree, absent remote child, exact parent head, and fast-forward ancestry. No retarget, rebase, force-push, parent rewrite, merge, or check bypass occurred.

The immutable implementation evidence remains the two-commit child ending at `dcc801395538bdc7b9a46835f5555abdd72677a4`: 115 ground tests pass on CPython 3.11.9 and real 3.10.20, pinned MyPy 1.13 is clean across 25 production modules, pinned Ruff 0.14.10 is clean across 18 changed Python files, and manifest, documentation, assertion, structural, file-size, and diff gates pass. Issue #4271 stays open for changing normals and regional surfaces; protected CI, review, dependency integration, UI, compiled runtimes, and downstream parity remain release gates.

This publication-only registry update makes no material physics, numerical, schema, or API change beyond the already committed exact head; it records the carrier and evidence in the campaign manifest and canonical handoffs.


## 2026-08-09 issue #4271 independent-review hardening

Independent review blocked publication of local commit `730b58bba8d9c281e6cdcc1e7e2c6340caa1c3f9`
and produced adversarial regressions before any GitHub write. The follow-up
binds every bounce prefix to the SHA-256 of the complete canonical request;
rejects mismatched surface, ball, limit, and provenance inputs; composes both
phase model identities; preserves typed impact-prefix limitations; and
requires suffix terminal state, trajectory, frame, events, and termination
evidence to agree.

The skid integrator retains exact collinear capture while adaptively bounding
closing oblique Coulomb substeps to one quarter of the slip characteristic
time. This prevents zero-slip overshoot and step-size resonance on inclined
oblique motion. Strictly positive vector roots eliminate zero-duration
downhill-start failures, and zero-speed outward acceleration at a finite edge
is immediate `LEFT_SURFACE`. The manifest now distinguishes #4302's corrected
current carrier head `920c46dee688815691e251777142126bf1489b1a` from immutable
physics evidence `63a6f4bec63c58d28bceed2e8cf348a618c8e366`.

The hardened exact tree passes all 115 ground tests on CPython 3.11.9 and real
CPython 3.10.20. Pinned MyPy 1.13 is clean across 25 ground production modules;
pinned Ruff 0.14.10 check/format is clean across 18 changed Python files. The
manifest validator, eight manifest tests, documentation governance, file-size,
changed-test assertion, structural, and diff gates are publication requirements.
Issue #4271 remains open because changing normals and regional surfaces are
still outside this bounded plane slice.

## 2026-08-09 issue #4271 static-plane skid/roll local slice

Local branch `feat/4271-ground-skid-roll` is based on exact corrected #4270
head `920c46dee688815691e251777142126bf1489b1a` and is intended to target
`feat/4270-ground-impact-bounce`. Nothing from this child has been pushed or
opened on GitHub, and no protected/release evidence is claimed.

The shared Python ground package now consumes the exact #4270 capture handoff
and provides bounded arbitrary-plane skid, pure roll, rolling resistance,
finite-axis edge exit, retained axial spin, qualified rest, distinct relative
skid/roll path, and a passive energy/work ledger. Its strict composer removes
the immediate-capture duplicate, preserves event sequencing, and emits only
representable v1 rest, edge, time-limit, or event-limit results. Censored
endpoint metrics carry explicit warnings; legacy rest-only projection rejects
complete `LEFT_SURFACE` output.

The authority is `docs/specs/GROUND_SKID_ROLL.md`; golden fixture SHA-256 is
`74e23ebe86c8b476a3414b0ff11e561e126810b5358337cb87bc1e35e3a1d73d`.
Local qualification: all 108 ground tests pass on CPython 3.11.9 and real
3.10.20, pinned mypy 1.13 passes 24 production modules, pinned Ruff 0.14.10
passes 15 changed Python files, and manifest validation, eight manifest tests,
and documentation governance pass.

This remains a Python model slice, not product-surface completion. Material
regions, changing normals, deformation/grass response, torsional spin damping,
roll-to-skid, PyQt6/React UI, compiled runtimes, UpstreamDrift consumers,
protected review, parent integration, and main release remain open.

## 2026-08-09 PR #4302 pinned-MyPy current-head correction

Hosted quality-gate run `31350134551` exposed four deterministic MyPy 1.13
findings on published head `ceaed9e548c5b6d147dbbeb17ee3ff2a509436c5`.
The wire serializer now binds its validated lazy-import boundary to the declared type,
and airborne grid sampling advances one guarded local `float` before storing
the next grid time. No material physics, schema, numerical, UI, scope, or base
change occurred; focused pinned MyPy and ground tests remain the publication
gate.

## 2026-08-09 Ground impact and repeated-bounce local slice

Draft PR #4302 publishes issue #4270 on `feat/4270-ground-impact-bounce` at
immutable evidence commit `63a6f4bec63c58d28bceed2e8cf348a618c8e366`.
It targets exact #4288 head `4972e55e0bb6e5b6bf7da0f899eed5d4f54e7d9d`
on `feat/4269-flight-ground-transfer`; no existing stack base was changed.
The reusable Python ground facade now provides a passive restitution/Coulomb
impact with sphere inertia and full spin, exact first contact, deterministic
repeated hops, bounded cancellation/failure semantics, and typed airborne
segments. `max_time_s` is elapsed from interpolated first contact while public
times remain absolute; first contact counts toward `max_events`.

Capture-to-surface output is one exact-contact `SKID` point and handoff state,
with no duplicate event/grid sample. Horizontal bounce-air distance is the sum
of each segment's x-z displacement, exposed as prefix evidence for #4271.
This is not a final ground-run result: #4271 retains skid/roll/rest, total
distance, and `GroundSimulationResult`. Firmness, grass, compression, moisture,
rolling resistance, UI, TS/Rust/PyO3/WASM kernels, and downstream consumers are
explicit non-deliveries. The campaign manifest remains `not_released`; see
`docs/specs/GROUND_IMPACT_BOUNCE.md` and the campaign handoff for qualification.

Final local validation is `82 passed` for the complete ground package on both
CPython 3.11.9 and real CPython 3.10.20. Pinned mypy 1.13 reports no issues
across all 17 ground production modules. Pinned Ruff 0.14.10 check and format
pass the changed Python set. The campaign manifest validates, its eight
contract tests pass, documentation governance and focused changed-test
assertion gates pass, and all changed production modules/functions/signatures
remain within 400-line/50-line/four-parameter budgets.

Independent pre-publication review made no material physics, schema, or scope
change: vector primitives now return explicit `Vector3` tuples without typing
suppressions, and internal initialization invariants raise deterministic
runtime errors instead of relying on optimizable assertions. The complete
82-test ground suite, pinned mypy, Ruff, and diff gates remain green.

## 2026-08-09 Flight-transfer corrected-parent propagation

Draft PR #4288 keeps base `feat/4268-ground-contract` and normally
incorporates exact carrier-reconciled #4285 parent
`6a2bc9d06f6f9a28a0d615b19d2ed4fc13871059`. No branch was rebased,
retargeted, force-pushed, or published. The transfer's signed terminal state,
physical sphere/terrain contact brackets, strict provenance, and cross-runtime
contracts remain intact while the corrected wind-to-ground ancestry becomes
complete.

The flight facade retains both the child's transfer inventory/structural
frozen-dataclass typing and the parent's package-relative collection fix.
This propagation adds no bounce, skid, roll, terrain response, total-distance,
or UI claim. Protected CI and child-first merge of #4288 into #4285 remain
separate release gates.

Focused evidence is 113 ground/transfer/facade/carrier tests on Python 3.11 and
113 on real CPython 3.10.20. Ruff check/format passes 36 focused files; pinned
mypy 1.13 passes the 13 transfer-delta files and 12 ground production modules
through their established separate namespace invocations. The terminal-spin
test now binds each trajectory sample before exact `FlightStatePoint`
narrowing, preserving runtime assertions while satisfying the pinned checker.
The campaign manifest and all nine manifest/parity tests pass.

## 2026-08-09 Flight-transfer parent propagation

Draft PR #4288 keeps base `feat/4268-ground-contract`. The exact local parent
head `8e8df7b9c633affb986326137338313faf46d2db` is now incorporated through a
normal merge; no branch was
rebased, retargeted, force-pushed, or merged on GitHub. Semantic resolution
keeps the child's focused `flightIntegrator.ts` extraction, carries the
parent's capability-evaluator value types into the frozen public-contract
inventory, and advances the transfer specification to 1.14.9 after the
parent's schema-gate repair at 1.14.8. Focused Python, React, Rust, and full affected
gates now pass: `82` focused Python tests, `38` focused React tests, `26`
focused Rust tests, and `1483 passed, 7 skipped` for the complete affected
Rate+swing_sim Python suite. The skips are optional local Rust-wheel paths.
Complete React validation is `104 files / 643 tests passed`, followed by clean
type-check, lint, and production build; full `tools-core` validation is `137
passed`. Changed Python Ruff check/format and the CI-pinned mypy 1.13 gate are
clean, as are documentation governance and diff checks.
The first focused run was correctly RED for a circular import between the
ground and flight package facades; `ground_transfer.py` now imports its direct
ground record/type dependencies instead of reaching through `ground.__init__`.
The hosted 3.12 numerical failure is a separate wind-fixture drift of
`3.494e-12` against a `1e-12` absolute tolerance; no transfer assertion failed,
so that out-of-scope wind contract is left unchanged.
After propagating exact parent `8e8df7b9c`, the focused ground/transfer/API
suite is `70 passed`. Pinned mypy 1.13 is clean across all `13` child-delta
Python files, including the adversarial and facade tests; the frozen-value
inventory now exposes its dataclass metadata through a test-only structural
protocol instead of relying on ambiguous checker narrowing. Ruff, formatting,
assertion policy, docs governance, and diff checks pass.

## 2026-08-09 Ground-contract ancestry and CI compatibility repair

Protected quality-gate run `31341468033` on exact head
`2d9a06fae46e0601a05896b71934ca0c6b8dc59a` exposed a second, narrower
compatibility boundary: pinned mypy 1.13 with skipped imports sees the shared
string-enum shim as `str`, so schema generation could not type-check iteration
or `.value`. The follow-up derives schema strings with `str(item)` and replaces
test-only suppressions with explicit casts around intentional invalid inputs.
No schema value or runtime rejection rule changes. Exact mypy 1.13 now passes
all 19 changed Python files; Ruff check/format and all 46 ground tests pass.
This evidence applies locally to the follow-up until its new exact head runs in
protected CI. Do not retry or cite the obsolete failed head as passing, and
propagate the published repair into #4288 through a normal merge.

Draft PR #4285 keeps its original base `feat/4197-capability-observer`. The
current parent head `9bbb98e16e435a0d4c74153b909f2ebfefbbce7a` was propagated
through normal merge commit `25f181450`; no rebase, retarget, force-push, or
GitHub merge occurred. The merge had no ground-source conflict and the only
manual resolution retained the newer root handoff plus this ground status.

Hosted checks on the preceding exact head found an undeclared `jsonschema`
dependency during ground-schema test collection. The dependency is now
declared and locked at the locally verified 4.24.0 build, and the three ground
enum modules import the repository's canonical
`shared.python.compatibility.StrEnum` rather than Python 3.11-only
`enum.StrEnum`. A new package-wide source contract failed first on exactly
those three modules and then passed. Current focused evidence is `46 passed`
for `swing_sim/ground/tests`; the combined Rate+swing_sim gate is `1463 passed,
5 skipped`, where every skip is an optional local Rust-wheel path. Ruff
check/format, targeted mypy, documentation governance, and diff checks are clean.
The Rust gate's missing `-lpython3.11` is runner/toolchain infrastructure.
Do not widen this repair to older non-ground Python 3.10 imports, and keep
#4288 stacked behind #4285 until the parent follow-up is reviewed and published.

## 2026-08-08 COMPLETION RECORD — capability-optimization-ui recovered

The interrupted optimization-UI slice was reviewed, re-verified,
repaired, and committed on `feat/4197-capability-optimization-ui`.
Recovery fixes beyond the dying agent's state: Ruff formatting and
import-sort in three files; a mypy-1.13 `call-arg` failure in
`capability_controls.py` (positional-after-star bounds unpacking
replaced with typed `_numeric_spec`/`_integer_spec` factories);
TypeScript errors in `CapabilityOptimizationPanel.test.tsx` (the mock
runner is now `vi.fn<CapabilityRunner>`); and eager panel import
bloating the main Vite chunk past 500 kB — `PrimaryWorkspacePanel` now
lazy-loads the Shot Optimizer behind `Suspense`, matching the
WindStrategyPanel precedent (main chunk 474.32 kB, no warning).

Verified gates on the committed head: 808 `tests/rate_of_closure` plus
615 swing_sim in-package tests passed with zero skips; 102 React files
/ 619 tests passed; Ruff check/format clean; CI-equivalent mypy 1.13
clean on all 10 changed src files; `tsc --noEmit`, zero-warning ESLint,
and the 187-module Vite build pass; changed-only 500-LOC budget and
`git diff --check` pass. Published as PR #4294 on
`feat/4197-capability-flight-evaluator`.

**Stack collapse order.** Only `main` carries branch protection
(`quality-gate` + `tests (3.11)`); none of `feat/4197-*` or
`feat/4199-wind-workflow` is protected, so these merges are not gated by
required checks. Each PR's base is its own parent branch, so the stack
collapses **top-down**: fold #4294 into `feat/4197-capability-flight-evaluator`,
then #4289 into `feat/4197-capability-observer`, then #4283 into
`feat/4199-wind-workflow`. Folding a child into its parent never releases
it ahead of that parent — the parent carries it forward. Merging
bottom-up instead strands each parent one slice behind and needs extra
reconciliation PRs.

The first hosted quality-gate run on #4294 rejected the dict-splat
construction of `CapabilityWorkflowInputs` (`**dict[str, float]` cannot
be proven against integer fields under the CI mypy context); commit
`101020b5b` builds the snapshot with explicit typed keyword arguments.
Do not "fix" the `tests (3.10)` `StrEnum` ImportError seen on PR #4280
by converting bare `from enum import StrEnum` to the compatibility
shim. `pyproject.toml` has declared `requires-python = ">=3.11"` since
2026-05-14, so those imports are correct; `shared.python.compatibility`
serves older subsystems only. Branch protection requires exactly
`quality-gate` and `tests (3.11)`, and `ci-standard.yml` states that
only the 3.11 lane is required (`fail-fast: false` protects it). The
3.10 lane is a stale-matrix artifact, not a merge blocker.

## 2026-08-08 Matched Capability Optimization Workspace

Active branch `feat/4197-capability-optimization-ui` is a normal child of
evaluator commit `c280407d432c153639bb266c9c721a014a129723` (draft PR #4289).
Do not retarget, rewrite, or merge it ahead of the evaluator and observation
parents.

PyQt6 and React now expose a discoverable Shot Optimizer module backed by the
same strict `capability-optimization-workflow/v1` document. It captures profile
and club IDs, capability center/spread, canonical positive-right launch and
spin-tilt frames, sourced fixed spin, target, objective, budgets, alternatives,
and seed. Both clients execute outside the UI thread, publish progress, cancel
without partial-result publication, rank alternatives, retain all observation
cohorts in `scalar-ensemble/v1`, and provide stage-qualified axis selection,
zoom/autofit, paged raw rows, lossless CSV, and stable JSON. Generic Python
scalar CSV serialization now lives in the UI-neutral variation layer.

Rendered QA in the live Vite app and a uniquely titled standalone PyQt window
verified execution, progress, large-result paging, plot controls, and layout.
It also drove fixes for hidden newly registered modules, ambiguous duplicate
axis labels, narrow split-pane geometry, and missing hover guidance. Current
verified gates: 808 Python/PyQt plus 615 swing_sim tests and 102 React files /
619 tests pass; Ruff, formatting, CI-equivalent mypy 1.13, TypeScript, ESLint,
the 187-module Vite production build, structural scans, and diff checks pass. V1 remains explicitly still-air, carry-only, and excludes
wind, bounce, roll, and total distance. Keep #4197 open for protected CI,
review, ordered merge, and downstream parity.

## 2026-08-08 Evaluator CI repair and descending-launch parity fix

Two defects were found and fixed on this branch after its first protected run,
both reported from the stacked child #4294:

1. **Descending-launch exception.** `flight.ts` starts the ball at `z = 0`, so a
   negative launch angle puts it below ground on step 1, which the `t > dt`
   guard skips. By step 2 the interpolation ratio went negative and
   extrapolated the ground crossing to a negative sample time, raising
   `RangeError: timeS must be nonnegative`. The observation layer absorbed it as
   an untyped `evaluator_exception` — 12 of 96 samples on a default driver
   search, where Python reported zero. The ratio is now clamped to `[0, 1]`, so
   these samples report `nonconverged`, matching Python exactly. A pinned
   regression test covers both reproducing samples; it fails without the clamp.
2. **Delta-mypy quality gate.** `result_derivation.py` had an unused
   `type: ignore` and an `Any` return. `_lerp_vector` now builds the 3-tuple
   explicitly (genuinely typed, no ignore needed in either import mode) and
   `_curve` coerces its result to `float`.

The other two red checks on #4289 were **not** code failures: `file-size-budget`
and `detect-secrets` both had their setup steps *cancelled* by runner
infrastructure. Re-run them rather than chasing a nonexistent violation.

Verified after the fixes: 1406 Python tests and 599 React tests pass; Ruff,
format, CI-equivalent mypy 1.13, TypeScript, zero-warning ESLint, and the
176-module Vite build pass.

## 2026-08-08 Model-Backed Capability Flight Evaluator

Active branch `feat/4197-capability-flight-evaluator` is a normal child of
`feat/4197-capability-observer` at exact parent
`49612946138b1021f80c9f8d2a4d06f1610825db` (draft PR #4283). Do not retarget,
rewrite, or merge it ahead of that parent.

The new shared Python and React-model adapters bind the real capability profile
and optimization request to the actual Waterloo/Penner flight model. They use
the established `ball_speed`, `launch_angle`, and `launch_direction` IDs,
explicit sourced per-club spin defaults for older profiles, paired optional
variable `total_spin` and `spin_axis_tilt`, positive-fade/right convention,
canonical target-frame trajectory/spin conversion, target residuals, all
available scalar metrics, typed nonconvergence, and fail-fast invariants.
Profile units, safe bounds, and physical domains are enforced before an
integration call. The post-impact evaluator never invents `no_impact`.

Independent-review corrections removed the global 2,686-rpm fallback, aligned
the canonical metric catalog with the existing app convention, unified
cross-runtime sampling validation, narrowed exception handling, and added a
shared 16-scalar parity fixture. Result, impact, and variation producers now
share the same gyro-projected tilt function. Current complete-suite evidence is
138 Python passes / four optional-Rust skips and 97 React files / 597 tests. Ruff,
formatting, targeted mypy, TypeScript, zero-warning ESLint, and the 176-module
Vite build also pass. This completes the qualified
evaluator prerequisite, not issue #4197: matched PyQt6/React authoring, worker,
progress/cancel, scatter/table/export, persistence, and rendered QA remain.

## 2026-08-07 Universal Variation Visualization Continuation

Branch `feat/4144-variation-export-continuation` is stacked on the published
toolstrip head `c36ca36e9` and continues issue #4144 without adding unrelated
toolstrip changes to PR #4279. The core PyQt6/React variation workspace already
has selectable input/contact/impact/shot scatter, a scatter matrix with
marginals, landing dispersion, and an all-trial 3D swing-arc overlay with
reference trace, principal spread, RMS variability, quiet zones, filtering,
and linked trial selection.

This continuation closes an export/accessibility parity gap: both clients now
export the complete selected scatter axes as CSV, including every trial,
typed outcome, and explicit unavailable values. PyQt now also exposes the raw
selected-axis rows in a bounded read-only table. Shared table population is
factored into `variation_trial_table.py`; the scalar scatter view is isolated
in `variation_scatter_view.py`. Changed production modules remain below 400
lines and changed functions remain at or below 50 lines.

The continuation is published as draft PR #4280. Current exact local evidence
on implementation commit `b3b09215e` is 890
Python/PyQt/shared swing tests passed with one expected optional-Rust skip and
15 existing warnings, and 89 React test files / 545 tests passed. Ruff,
formatting, Black, targeted mypy, TypeScript, zero-warning ESLint, the 166-module
Vite build, and `git diff --check` pass. This is local evidence only; no PR,
protected CI, review, merge, or default-branch release has yet been established.

Literal audit of "every many-trial simulation" found two dependent gaps that
must remain explicit:

- wind-strategy uncertainty retains paired per-strategy outcomes in both
  runtimes but has no PyQt/React workflow and does not feed the universal
  scalar plot facade;
- the capability optimizer stores aggregate statistics but discards the
  individual evaluation rows needed for honest scatter plots.

Issue #4199 already owns the wind UI/scatter requirement. Its adapter must use
composite row identity `(strategy_id, trial_index)`, retain completed,
nonconverged, and invalid cohorts, and report that impact variables are
unavailable because this runner begins at prescribed launch. Do not coerce it
into the impact-specific cohort enum or fabricate impact/landing values.

Branch `feat/4199-wind-scalar-adapter` is published as
[draft PR #4281](https://github.com/D-sorganization/Tools/pull/4281), stacked
on exact draft PR #4280 head `d71b0ea01b5659d3049ff05627c41f06481207e4`.
Implementation commit `4a28114aa` supplies that first UI-neutral model slice.
Python and React share the exact
snake-case `scalar-ensemble/v1` wire structure: structured provenance,
labeled stages/categories/cohorts, unit-bearing variables, RFC3986 composite
row identity, immutable nullable raw rows, and overall/per-cohort x/y/paired
availability. The wind adapter validates deterministic request/analysis trial
agreement, retains every actual and perfect-information status, and exposes
true/estimated wind, launch/aim, target, landing, miss, cost, and information
delta without invoking the flight solver. Impact variables remain honestly
absent because this analysis begins at launch.

Exact local gates for this branch are green: 906 Python/PyQt/shared-swing
tests passed with one expected optional-Rust skip and 15 existing warnings;
91 React files / 555 tests passed. Ruff, Ruff formatting, Black, focused mypy,
TypeScript, zero-warning ESLint, the 166-module Vite production build, and
`git diff --check` pass. Production Python modules remain below 400 lines and
no changed Python function exceeds 50 lines. This contract/adapter does not
complete #4199: the next slice still needs background execution, progress and
cancellation, PyQt/React scatter/strategy UI, persistence, and export wiring.

## 2026-08-07 Ground Model and Four-Surface Parity Revalidation

The user's requested rolling-ground and site-wide parity programs already
exist as [epic #4267](https://github.com/D-sorganization/Tools/issues/4267)
with children #4268-#4276 and
[epic #4260](https://github.com/D-sorganization/Tools/issues/4260) with
children #4261-#4266. Do not create duplicate epics. A fresh repository and
live-GitHub audit was recorded in
[the ground comment](https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5222725556)
and [the parity comment](https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5222726010).

There is no qualified landing/bounce/roll implementation yet. The current
airborne solvers stop at the relative launch plane and the shared
`TrajectoryPoint` omits terminal angular velocity. For a teed shot, translating
that relative trajectory into the course frame can leave the terminal ball
center at tee elevation. Complete #4268 and #4269 before implementing or
presenting bounce, roll, or total distance.

Reuse the Tools `GroundModelResult` fail-closed boundary, putting skid/roll
limiting cases, and turf provenance/variation/cancellation patterns. Adapt
UpstreamDrift's split terrain material, elevation, normal, and region contracts
through a one-way versioned DTO. Do not qualify Upstream's scalar landing
helper, heuristic putting spin relaxation, legacy duplicate terrain model, or
Rust `(1-friction)` contact law as production high-speed landing physics.

The parity matrix must keep Rate and Upstream's separate products distinct:
standalone Rate PyQt6/React, Upstream's Rate PyQt provider/React route, Shot
Tracer PyQt6/React, and the legacy ball-flight GUI. Current Upstream `main`
has no native Rate React route, and its Tools source resolvers disagree about
vendor-first versus sibling-first precedence. A launcher tile cannot satisfy a
calculation-parity row; #4261, #4262, and #4264 must make the runtime source,
exact Tools pin, and support state machine-verifiable.

## 2026-08-05 Advanced Wedge Impact Visualization

Branch `feat/4162-wedge-impact-visualization` extends issue #4162 on top of the
validated turf stack. It corrects the impact adapter to evaluate pose, twist,
and articulated wrist geometry at the exact event time rather than silently
using the nearest retained sample. The new versioned
`rate-of-closure.impact-scene/v1` contract carries complete scene geometry,
velocity components, metric equations, frames, assumptions, availability, and
screw-axis data without placing physics in either UI.

PyQt6 adds an exact-event Impact Inspector layer, locked physical axis scaling,
isometric/face-on/down-the-line cameras, and 300-DPI PNG, vector SVG, and strict
JSON export. React adds an orbitable and keyboard-controllable impact still,
the same named cameras and velocity toggles, visibly expandable engineering
metric definitions, and device-resolution PNG, true-primitive SVG, and JSON
exports. The web mirror now retains and shortest-arc interpolates the canonical
head rotation; the older limitation note saying it lacked full head pose is no
longer accurate for this branch.

Scientific boundaries remain explicit: articulated sources do not yet have an
independent torsional head state; the scene is rigid-body instantaneous
kinematics; illustrative turf profiles cannot support optimal-bounce or
forgiveness claims; and turf force is not replayed into the retained swing.

Current-head verification: all 576 Rate Python/PyQt tests passed (one existing
polynomial-generator legend warning); all 347 React/model tests passed; the
production Vite build, TypeScript, ESLint, Ruff, formatting, changed-module
strict mypy, and protected module-size budget passed. Headless Chrome visual QA
at 1600×1400 exercised named views, a vector toggle, keyboard orbit, and an
expanded metric definition with zero console exceptions/log errors. The new
web branch is running at `http://localhost:5260/`; the current PyQt process was
also launched successfully and remained responsive.

## 2026-08-05 Wedge Impact Inspector Integration

Draft PR #4173 (`feat/4163-impact-inspector`) integrates the draft variation
branch with the shared golf-club stack through wedge kinematics PR #4172. It
adds the first bounded implementation slice for wedge epic #4158 /
impact-inspector issue #4163:

- Canonical inspection time and event label on every `SimulationRun`: impact
  for hits, closest approach for misses.
- Exact `Jump to Impact` / `Jump to Closest Approach` controls in PyQt6 and
  React, with playback paused before the jump.
- `simulation/impact_kinematics.py`, which adapts retained pose/twist/contact
  data to `shared.python.golf_club.WedgeKinematicState` and preserves geometry
  provenance and model limitations.
- Engineering readouts in both clients for contact/reference AoA, remove-shaft
  counterfactual, shaft rotation and vertical velocity, face-normal rate,
  leading-edge/arc rate where available, and screw-axis distance.
- Restored manual angular velocity in the React simulation path; the previous
  hard-coded zero made all closure and shaft metrics false zeros.
- Deterministic midpoint tie-breaking for a flat maximum-speed plateau, so the
  manual source's automatic event is its documented square pose at 30 ms.

Physics boundary: articulated pendulum runs expose the measured wrist-to-head
shaft line but have no shaft-twist degree of freedom. The inspector reports
that absence rather than inventing torsional motion. The web mirror still does
not retain full head pose; its readout declares that limitation until WASM or a
canonical backend replaces the temporary TypeScript mirror.

Current-head release evidence: 1,006 Python/PyQt/shared-swing tests passed with
five optional Rust-wheel parity skips; 334 React/model tests and 12 swing-core
Rust tests passed; Vite production build, TypeScript, ESLint, Ruff, formatting,
and mypy across 165 source files passed. A focused post-refactor PyQt run passed
46 tests. Browser QA verified the 1,307 deg/s manual fixture at 30 ms and a
1.430 s closest-approach miss with zero console warnings/errors. Native-window
QA confirmed the control and readout are visible in the standalone PyQt6 app.

## Status Note

`src/rate_of_closure` and `src/shared/python/swing_sim` do **not exist on
`main` yet** — they land with PR #4119. This doc describes the tool as it
exists on the in-flight branch stack (`feat/impact-simulation-platform` →
`feat/investigation-suite` → `feat/course-showcase`) so the next agent has
full context the moment #4119 merges. If you're reading this on a fresh
`main` checkout and don't see `src/rate_of_closure/`, check out one of those
branches or wait for #4119 to land.

## Where This Tool Is Headed

Rate of Closure started as a single-page "closure rate" calculator (twist
model: GC-path vs impact-point-path gap, °/ft). Epics #4103 → #4120 → #4125 →
#4130 are growing it into a full swing → impact → ball-flight simulator with
PyQt6 + web parity and eventual public GitHub Pages distribution. Read
`src/rate_of_closure/README.md` (frame conventions, Cheetham dossier sourcing,
run instructions) before touching physics code.

## The #4119 → #4124 → #4129 PR Stack

Each PR consolidates a whole epic's stacked feature branches into one PR to
keep self-hosted CI load down (see `CLAUDE.md` — these are big diffs).

- **#4119** `feat/impact-simulation-platform` (epic #4103, open, auto-merge
  armed). Base tool (twist model, PyQt6 + React/Vite web, Cheetham dossier
  data) + STL clubheads/club library/inertial model + `swing_sim` shared
  package (`src/shared/python/swing_sim/`: swing sources, impact model ported
  from UpstreamDrift with 3 physics fixes, gear effect, 7 literature
  ball-flight models, goal-driven multi-start solver) + `swing-core` Rust
  crate (pyo3 + wasm) + app integration (simulation session, impact-time
  scrubber, screw-axis overlay via the rotation_converter adapter, video
  controls, CSV/JSON export, solver panel). 404 pytest + 72 vitest + 111
  cargo tests, all local gates green. Supersedes #4092/#4097/#4098/#4112-4118.

- **#4124** `feat/investigation-suite` (epic #4120, open, **draft state —
  do not merge yet**, stacked on #4119). Adds `rate_of_closure/plotting/`
  (40-variable data catalog, `PlotSpec`, Plots tab + custom-plot wizard),
  scale-separated Strike/Swing/Flight viewers + standalone Flight Explorer,
  `swing_sim/variation/` (seeded Monte Carlo/NoiseSpec engine, dispersion +
  Spearman sensitivity, Variation tab), and V4: glossary (60 terms),
  cold-user help system, "Derivation & Traceability" → "Calculation
  Description" rename, full-model derivations, hover-hint completeness sweep.
  Supersedes draft PRs #4121/#4122/#4123. 566 pytest + 125 vitest passing.

- **#4129** `feat/course-showcase` (epic #4125, open, **draft state — do
  not merge yet**, stacked on #4124). Merges `feat/realistic-heads` (H1:
  parametric club-type geometry, volumetric COG via divergence theorem),
  `feat/swing-kinetics` (H2: joint torque/force plots + 3D overlays from
  pendulum inverse dynamics), `feat/putting-vertical` (H3: `swing_sim`
  putting module + app Putting tab), then adds on top: H7 course scene
  (`ui/course.py` / `course_scene.py`) + target optimization
  (`swing_sim/solver/targets.py`, `TargetRegion`, `ImpactGoal.target_region`),
  and H6 showcase styling (`ui/pyqt6/app_style.py`, UpstreamDrift launcher
  visual language) + yards-default distance units. 413 pytest + 309 swing_sim
  - 174 vitest passing. H4 (AffineDrift putting research content) and H5
    (public release-management repo) are cross-repo and tracked in #4125
    directly, not in this PR.

**Do not merge #4124 or #4129 before their base merges** — SPEC.md sections
were unioned assuming sequential merge order; merging out of order will
produce conflicting/duplicate changelog rows.

## swing_sim Packages

`src/shared/python/swing_sim/` (introduced by #4119, home for physics shared
with UpstreamDrift via the established shared-module arrow):

- `swing_sim/flight/` — 7 literature ball-flight models (drag/lift/spin
  decay), citations in registry metadata.
- `swing_sim/impact/` — impact model ported from UpstreamDrift (offset-drop
  fix, opt-in 3×3 MOI tensor, inverted friction spin axis fix), gear effect,
  `SpringDamperImpactModel` (Kelvin-Voigt contact force history, 1e-7s steps)
  — this is the contact-force law epic #4130 will extend for the full
  contact-interval integration rather than duplicate.
- `swing_sim/solver/` — goal-driven multi-start least-squares solver;
  `targets.py` (added in #4129) adds `TargetRegion` for green/fairway
  optimization goals.
- `swing_sim/variation/` (added in #4124) — namespaced variable registry,
  `NoiseSpec`/`VariationPlan`, seeded parallel N-run Monte Carlo engine,
  dispersion/OAT sensitivity/Spearman/landing-ellipse stats.

Epic #4130 (Impact-Interval Club Dynamics) will add `impact_interval/` to
this same package — its home is explicitly `swing_sim` so UpstreamDrift
reaches it via vendor, per that epic's F2 phase description. Not started yet
(foundation-only epic, no PR).

## How #4103/#4120/#4125/#4130 Relate to This Tool

All four are rate_of_closure epics specifically (unlike the wider-monorepo
epics tracked in the root `AGENT_HANDOFF.md`, e.g. SCADA #4085-#4089).
#4103 is the foundation platform; #4120, #4125 are sequential feature waves
stacked directly on its PR; #4130 is a physics-depth epic that extends the
impact model #4103 introduced (contact-interval integration replacing the
instantaneous-impulse approximation) — foundation phase only so far.

## Web Mirror + GitHub Pages

The web mirror (`src/rate_of_closure/web/`, React/Vite/TS) is pinned
test-for-test against the PyQt6 model today (TS mirrors hand-written, not yet
WASM — that swap is explicitly deferred to Phase 7 of #4103). It builds to a
static bundle (`npm run build`) and carries Tauri scripts for desktop
packaging, same as other web tools in the repo.

**There is no automated GitHub Pages CI deploy for this tool yet.** No
`.github/workflows/*.yml` references `rate_of_closure` or Pages deploy
actions as of this writing. The only Pages-publishing precedent in the repo
is `src/web_applications/unit_converter/unit-converter-app/DEPLOYMENT.md`'s
manual branch-folder publish (Settings → Pages → select branch/folder).
Phase 7 of #4103 ("GitHub Pages mirror updated (public share link), parity
tests as deploy gates") owns building a real workflow — do not improvise one
in an unrelated PR.

## Must-Read Architecture Pointers

1. `src/rate_of_closure/README.md` — frame conventions, unit conventions,
   dossier sourcing, run/build instructions.
2. `src/rate_of_closure/model.py` (base twist physics, no Qt) once #4119
   lands.
3. `src/shared/python/swing_sim/impact/` — the contact-force law shared
   with (and about to be extended by) epic #4130.
4. `rust_core/swing-core/` — pendulum EOM + plane projection, pyo3 + wasm
   targets, follows the `tools-core` feature-contract pattern.
5. `.github/workflows/maturin-swing-core.yml` (added by #4119) — Rust wheel
   build for this crate.

## Gate Commands (this tool)

```bash
python3 -m pytest tests/rate_of_closure src/shared/python/swing_sim -n auto --timeout=60
cd src/rate_of_closure/web && npm run test && npm run build && npx tsc --noEmit && npx eslint .
cargo test -p swing-core
python3 -m ruff check src/rate_of_closure src/shared/python/swing_sim
python3 -m mypy src/rate_of_closure src/shared/python/swing_sim
```

## Do-Not List

- Do not duplicate the Kelvin-Voigt contact-force law — #4130 requires
  `SpringDamperImpactModel` and the new `impact_interval/` package to share
  one implementation (DRY, explicit in the epic's binding standards).
- Do not exceed 500 LOC per file in `rate_of_closure`, `swing_sim`, or
  `swing-core` — sub-package instead.
- Do not hand-mirror physics into the TS web layer once WASM lands (Phase 7)
  — that's the whole point of the wasm-pack build; today's hand-written TS
  mirrors are a stopgap, not the target architecture.
- Do not merge the stacked PRs out of order (see stack section above).
- Do not invent citations in derivation docstrings or the AffineDrift
  putting research content (H4) — sourced/verifiable only, dossier
  discipline per epic #4125.

## Roadmap (ordered)

1. Merge #4119, then #4124, then #4129 in order.
2. Start epic #4130 Phase F1 (formulation document) — six-DOF rigid-club
   contact-interval derivation, validation program design.
3. Phase 7 of #4103: WASM swap + real Pages CI deploy workflow.
4. #4125 H4/H5: AffineDrift putting research content and the public
   release-management repo (both cross-repo, not started).

## 2026-08-07 Wind-Strategy Workflow Continuation

The active child branch is `feat/4199-wind-workflow`, published as
[draft PR #4282](https://github.com/D-sorganization/Tools/pull/4282) at exact
implementation head `fdcc25008`, stacked on exact PR #4281 head
`8b8690e8760d82ba814e8d95588d2540d28a6759`. Do not fold this work into,
retarget, or merge it ahead of PR #4281.

This branch turns the shared `scalar-ensemble/v1` wind adapter into matched
end-user workflows. Python runs the immutable request in a `QThread`; React
uses a real, lazy-loaded Vite module Worker. Both expose exact `0..N`
progress, cancellation, current launch plus canonical landing target, trial
and wind-estimate controls, summaries, every scalar axis, explicit
completed/nonconverged/invalid availability, cohort-colored scatter, generic
all-row CSV, and fail-closed result invalidation. Scatter controls include
pan/zoom, Auto Fit, toolbar-history reset, and movable/hidden legends in
PyQt; React includes zoom, Auto Fit, clipped marks, numeric ticks/gridlines,
and movable/hidden legend. Captured calculation-basis regions make model,
seed, target, integration, risk, and aim-policy settings visible.

Final native-window QA at 1280 x 768 added matched ball-flight Loop controls
to PyQt and React and verified that Play/Pause, replay from landing, granular
speed, and continuous wrap all use the single owned animation clock. The
PyQt wind workspace now separates a compact two-column Setup view from a
plot-first Results view, automatically selects Results after completion, and
keeps run/cancel/export plus progress/status visible in both views. A live
five-trial run completed 5/5 and rendered its basis, summary, scatter, native
pan/zoom toolbar, Auto Fit, and legend-position control without overlap.

Lifecycle and safety details are contractual: the PyQt worker never reads
widgets, window shutdown cancels and joins it, queued stale signals are
ignored, and the main window explicitly stops Flight Explorer. React
terminates its Worker on completion, error, cancellation, unmount, or consumed
input change. Both clients preserve unavailable values as null/empty cells;
CSV strings and headers that could become spreadsheet formulas are
neutralized without altering numeric negatives. PyQt accepts the complete
shared uint32 seed range.

Current local evidence on this working tree:

- `1350 passed, 5 skipped, 15 warnings` for the complete
  `tests/rate_of_closure` plus `src/shared/python/swing_sim` suite. Skips are
  optional local Rust-wheel paths; warnings are the existing Hypothesis
  `norecursedirs` and empty polynomial-preview legend warnings.
- `94` React test files / `566` tests passed; focused playback and wind suites
  also pass.
- Ruff, Black, targeted mypy, TypeScript, zero-warning ESLint,
  `cargo test -p swing-core` (`12 passed`), and `git diff --check` pass.
- The 175-module production build emits separate wind-worker and lazy
  wind-workspace chunks; the main chunk is 472.34 kB and has no size warning.
- Every changed production source is at most 400 lines and every changed wind
  function is at most 50 lines.

This completes the #4199 current-launch workflow slice, not epic #4199 or the
universal many-run objective. Capability optimization still discards its
individual evaluator rows. The next child must add the optional streaming
`CapabilitySampleObservationV1` sink and cancellation hook described in
[issue #4197](https://github.com/D-sorganization/Tools/issues/4197#issuecomment-5223170071),
then adapt those rows to `scalar-ensemble/v1` without bloating the compact
optimization result.

Ground and four-surface parity remain open epics. The latest executable
acceptance refinements are in
[ground #4267](https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5223106106)
and
[parity #4260](https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5223106465).
Do not treat a launcher tile as a fourth UI implementation, and do not equate
launch-monitor total displacement with accumulated ground path length.

### 2026-08-07 CI repair and current ground/parity findings

The first hosted run for PR #4282 found one actionable delta-mypy defect:
`WindStrategyLifecycleMixin.closeEvent` conflicted with Qt's nullable close
event signature under Python 3.12. Commit
`424b4c395370aea26069386c070a65f7abe885bc` introduces a concrete
`WindStrategyGroupBox`, keeps worker teardown in the mixin, and gives the Qt
override the correct `QCloseEvent | None` contract. Exact Python 3.12 mypy
now passes for 11 changed production files, as do Ruff, formatting,
`git diff --check`, and 19 focused wind/playback tests. Do not merge until the
new protected checks and the entire parent stack are green and approved.

Read-only audits against current UpstreamDrift remote `main`
`0782853295e005af68818617e4725eb980890f43` found useful but unqualified
contact, terrain, turf, and putting-roll code. Preserve the direction
`UpstreamDrift adapter -> Tools ground-run/v1 authority`; Tools must not import
UpstreamDrift. Do not reuse the terrain serialization without fixing its lost
material fields, and do not start bounce/roll physics until first physical
sphere contact, arbitrary surface normal, target-frame conversion, and full
terminal angular velocity are available through a strict transfer contract.

The four-surface parity baseline is not complete: UpstreamDrift PyQt launches
the Tools native window, UpstreamDrift React has no Rate route, and Tools React
still has narrower impact/flight authority than Tools PyQt. Close parity
through shared versioned calculation contracts and golden fixtures, not by
counting launchers, separate simulators, or copied model implementations.

## 2026-08-07 Capability-Observation Continuation

Active branch `feat/4197-capability-observer` is an unretargeted child of
`feat/4199-wind-workflow` at exact parent head
`6e3c1029f1f3a80ae09020ef7d0afacb3c0d5484`.
It is published as [draft PR #4283](https://github.com/D-sorganization/Tools/pull/4283);
the validated implementation/hardening commit is
`5c6073bd68ed4c8f23b343d4d11c2dc4277ea246`.

The Python and TypeScript capability optimizers now stream one immutable,
versioned observation for every attempted ensemble sample and support typed
cooperative cancellation before an evaluator call. Existing callers and the
compact optimization result are unchanged. Status normalization is fail
closed, evaluator exception text is not leaked, all valid metrics retain their
declared order and provenance, and malformed or incomplete landing results do
not become fabricated successes.

The Rate app adapters turn the stream into bounded `scalar-ensemble/v1`
datasets with a complete scalar flight catalog, null unavailable outputs,
nominal/perturbed parameters, target residuals, and source lineage. They
require contiguous zero-based attempts, reject overflow before retention,
deep-copy/freeze TypeScript inputs, and serialize ASCII and Unicode fixtures
byte-identically across runtimes.

Current local gates: 120 Python tests passed with four optional Rust-wheel
skips; 96 React files / 580 tests passed; Python 3.12 mypy, Ruff, Black,
TypeScript, ESLint, Vite build, structural budgets, and diff checks pass.
Publish only as the next protected stacked draft PR, then keep #4197 open for
its remaining user-facing capability-optimization workflow.

An independent review blocked PR creation after the first branch push and the
four findings are now fixed locally. Stable JSON uses a shared raw-number
policy rather than native runtime spelling; derived parameter labels uppercase
only an initial ASCII letter; public observations enforce complete landing and
empty incomplete-status metric invariants; and TypeScript compares parameter
declarations structurally instead of with delimiter-concatenated signatures.
Adversarial IEEE rounding/exponent/negative-zero, Unicode, control-delimiter,
non-finite, and status-matrix tests are green. Current totals are 135 Python
flight/adapter tests passed with four optional Rust-wheel skips and 96 React
files / 584 tests passed, with the previously recorded lint, type, build, and
budget gates still green. The corrections are committed and published in draft
PR #4283. Monitor its protected checks and reviews together with PRs
#4279–#4282; do not retarget, rewrite, bypass, or merge this child ahead of
`feat/4199-wind-workflow`.

The first hosted CI Standard run found one actionable delta-mypy boundary in
the new `_capability_observation_runtime.py`: unchanged imported request fields
resolve as `Any` under `--follow-imports=skip`, so `total_count` now casts their
already contract-validated product to `int`. No runtime behavior changes. The
exact seven-file Python 3.12 mypy command, Ruff/format, diff check, and full
135-test flight/adapter suite pass with four optional Rust-wheel skips. This
fix and handoff update are
`60ac5b46c78988225862d9b89a33ddc3656a3413`; that commit is already present
in the propagated capability ancestry.

## 2026-08-07 Strict ground-contract and transfer continuation

Draft PR #4285 on `feat/4268-ground-contract` is the protected child of PR
#4283 and owns only the strict shared flight-to-ground v1 contract, schemas,
migrations, canonical cross-runtime fixture, and one-way legacy adapter. Its
first hosted CI Standard run found no behavioral defect: the changed-test
assertion gate classified `ground/tests/__init__.py` and the deterministic
`ground/tests/_support.py` record builder as assertion-free tests. Exact-path
fixture exemptions are now recorded in `scripts/test_assertion_allowlist.txt`;
all `test_*.py` modules remain checked. Reproduce the gate against
`feat/4197-capability-observer`, commit this repair with both durable handoffs,
push normally, and re-verify protected checks.

Issue #4269 continues independently in worktree
`C:\Users\diete\Repositories\Tools-worktrees\flight-ground-transfer` on
`feat/4269-flight-ground-transfer`, based on the published #4285 contract
commit. It must deliver the terminal angular state and physical sphere/terrain
contact bracket in Python, TypeScript, Rust, and WASM before bounce/roll is
wired to flight. Do not infer terminal spin from launch spin or substitute a
launch-plane crossing for physical contact.
The subsequent protected run's `detect-secrets` job classified the two pinned
Python/TypeScript SHA-256 parity digests as high-entropy strings. These are
expected deterministic test outputs, not secrets. Use only exact inline
`pragma: allowlist secret` annotations on the four digest constants; preserve
the repository baseline and scanner scope. Commit the annotations and this
handoff together on the parent capability-observer branch, then re-run the
protected stack without force-pushing or bypassing the gate. Parent commit
`49612946138b1021f80c9f8d2a4d06f1610825db` is now propagated into this child
by a normal merge commit.

The active #4269 branch now merges PR #4285 head
`3235af71150a774954e7673fc81d7179330fbe76` normally. Do not publish its
uncommitted transfer adapters until the repaired terrain geometry, strict wire
contract, origin proof, bounded web integrator, and native-wheel parity have all
passed a second independent integrated review.

## 2026-08-07 Physical flight-contact transfer

Issue #4269 is active on `feat/4269-flight-ground-transfer` after alignment
merge `13184096e`. Its local implementation adds full signed terminal angular
state and strict physical sphere/terrain contact transfer in Python,
TypeScript, Rust, PyO3, and WASM. Web origin proof and the 50,000-step
synchronous RK4 budget fail closed, and a partial final step stops at the exact
requested horizon; native Python integrates to arbitrary configured planes;
Rust entry points preserve the complete v1 request evidence. Bounce, skid, roll,
and UI wiring remain later child issues. At this pre-publication checkpoint,
the pending implementation/specification/handoff commit became
`d2d3d0f53a78aa863574afe43290a29c48318d94`; the later sections record its
completed independent review and current parent propagation.

The second integrated review's Python chronology, Rust canonical-wire, and
Rust/web exact-horizon blockers are repaired. Final local counts are 208 Python
tests with the rebuilt CPython 3.12 wheel and no skips, 603 React tests, and 160
Rust tests. Exact PyO3/Python token parity, production web build, PyO3/wasm32
checks, and `wasm-pack build` pass. The final independent closure audit found no
P0-P2 issue and declared #4269 locally publication-ready.

Full-crate Rust Clippy remains blocked only by pre-existing unrelated warnings;
the changed flight-ground scope is clean. New code respects the source and
function budgets. Existing oversized append-only registries and preserved
multi-parameter public compatibility signatures are baseline constraints, not
new structures introduced by this issue.

## 2026-08-09 PR #4302 deterministic-digest scanner repair

At exact head `920c46dee688815691e251777142126bf1489b1a`, protected
`detect-secrets` flagged the public SHA-256 golden-fixture assertion in
`ground/tests/test_impact_impulse.py`. The exact constant now uses the narrow
inline `pragma: allowlist secret` convention. No baseline, scanner scope,
physics, fixture bytes, or expected value changed.

Publish this as a normal follow-up commit on
`feat/4270-ground-impact-bounce`, with the three handoffs in the same commit.
Keep #4270 and #4267 open; do not bypass protected CI or propagate into child
branches until the parent push is verified.
