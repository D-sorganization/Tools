# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-10.

## 2026-08-10 issue #4274 matched regional surface editor local child

Local branch `feat/4274-regional-surface-ui` started from published PR #4335
head `d382ca9928628a16fec7ddd4fa1b1cc144b4c490`. When that parent
advanced, normal merge `6051d89a685ef009cfeef7c77bb3591cd124574a` preserved both
histories and exact corrected parent
`74a053d2d544da9f44a88007660ad28c0127f285`. No GitHub write,
protected evidence, review, merge-to-parent, or release claim has been made for
this child.

The PyQt6 and React shells now register matched `Ground Surfaces` primary
modules. Each exposes one complete SI base material/domain and one to eight
finite overlay rows with editable material evidence, region/surface/request
identities, precedence, bounds, and source revision. Both load explicitly
illustrative/unvalidated discovery data, hash the actual draft into provenance,
and delegate to the strict regional-plan v1 parser. Errors preserve input and
publish accessible state; successful validation exposes canonical schema, SI,
source, digest, and request readback. Navigation migration reveals the new
module without discarding a user's saved order or visibility.

RED-first evidence captured the missing Python and React adapter/component
boundaries. Before the parent advance, all five Python/PyQt editor tests and
seven focused React/wire tests passed. After the normal parent merge, the three
non-GUI Python adapter tests and six PyQt shell/help integration tests pass;
five final React editor tests and four regional wire tests pass,
including the shared illustrative-draft provenance digest
`2b3bf1b705bf86f5bf3cbe17970ddff63887410ad9f255200e5cfa31e5717db3`.
TypeScript, zero-warning ESLint, the 198-module production build, Ruff, and
MyPy pass. The campaign manifest and its eight tests, documentation governance,
changed-test assertions, and diff checks pass. The isolated regional PyQt
rerun and an ad hoc full-shell exit check hit bounded workstation
startup/lifecycle timeouts without an assertion failure; the pre-merge editor
GUI tests and post-merge shell/help GUI tests remain green, while the post-merge
adapter tests exercise the exact incoming validation/type-guard changes.

This is a session-only request editor/readback slice, not completion of #4274.
The regional v1 schema has no calibration record, so `unvalidated` remains
visible source qualification included in the draft digest rather than a
fabricated wire field. Request import/export, measured calibration workflows,
workspace model-input persistence, physics execution, result playback,
terrain/interval visualization, TypeScript or compiled regional physics,
UpstreamDrift parity, protected CI/review, parent integration, and main release
remain open.



## 2026-08-10 PR #4335 isolated-MyPy return typing

Protected CI at exact head `d382ca9928628a16fec7ddd4fa1b1cc144b4c490`
found two `no-any-return` errors under its changed-file
`--follow-imports=skip` profile. The strict text and JSON validators still
perform the same runtime checks; their already validated return values now
carry explicit local casts so the isolated CI type boundary remains precise.
This correction changes no schema, digest, physics, numerical result, or API.
Fresh exact-head CI and review remain required after the ordinary push.

## 2026-08-10 issue #4271 regional-plan wire-contract local child

Local branch `feat/4271-regional-wire-contract` starts from exact regional
physics parent `1a48d749af508843fac2a5102f4dd56294429bda`. No GitHub write,
protected evidence, review, merge, or release claim has been made for this
child.

Python and TypeScript now share separate strict
`ground-regional-material-plan-request/v1` and result/v1 records without
silently widening either frozen flight-to-ground v1 contract. The request
requires a finite base interval and one or more finite in-domain overlays,
exact SI/schema/geometry/limitation values, explicit provenance, stationary
coplanar geometry, and unique region, precedence, and surface IDs. Counts are
bounded at 4,096 regions and documents at 1 MiB. Duplicate JSON keys, unknown
fields, nonfinite values, changed geometry/velocity, invalid intervals, and
unsupported qualifications fail closed.

The result embeds the exact request, its canonical SHA-256, the same regions
in deterministic precedence/ID order, and producer provenance bound to the
request digest. Both runtimes reject reordered or changed surface evidence.
The Python adapter constructs the existing qualified `SurfaceResolver`;
TypeScript validates and serializes only and does not claim regional physics.
The shared golden request/result digests are
`a890b6fd544d73114ec5d0cd042f87aa2358d01ca85543a8c4d71ef2cb18cab1`
and
`8d9bc2f53897da241580f7b5fdaff7c6614077bed8a486cc6d7619d02b0e3e55`.

Local qualification is green: all 132 Python ground tests and all 107 React
files / 666 tests pass; TypeScript, zero-warning ESLint, and the 190-module
production build pass. Pinned Ruff 0.14.10 is clean over 45 ground files and
pinned MyPy 1.13 is clean over 31 production modules. The campaign manifest
and its eight contracts plus documentation governance pass. The changed
production modules remain below 400 lines.

This child remains `not_released`. Protected CI, independent review, normal
stack integration, UI, TypeScript/Rust/PyO3/WASM regional physics, changing
normals/heights/velocities, internal transition-ledger wire export,
UpstreamDrift parity, and main release remain open.


## 2026-08-10 issue #4271 coplanar regional-material local child

Local branch `feat/4271-regional-surface-transitions` starts from exact current
draft PR #4304 head `ee77b059bd83f7dafac7e0d411665231cdb7435c`.
No GitHub write, PR, protected evidence, review, merge, or release claim has
been made for this child.

The Python reference now supports finite coplanar material overlays on the
request-bound skid/roll plane. Region IDs and nonnegative precedence values are
unique; higher precedence wins overlaps; quadratic boundary roots split motion
exactly; and a coincident base-domain exit wins over a material change. Every
overlay must retain the base frame, height, normal, axis, and surface velocity.
A transition preserves time, position, velocity, spin, phase, and energy,
emits the strict Python/TypeScript `surface_transition` event, and records exact
from/to region and surface IDs in the internal suffix ledger. Request event
limits, a positive `max_surface_transitions` bound, and the existing step
limit prevent unbounded transition sequences. Model version `1.1.0` and the
`REGIONAL_PLANAR_V1` warning make the new qualification visible.

RED-first analytic/property evidence is green: all `121` ground tests pass,
including 24 randomized piecewise-analytic examples; the React contract suite
and full web suite pass at `106` files / `662` tests; TypeScript, zero-warning
ESLint, and the 189-module production build pass. Pinned MyPy 1.13 passes all
`28` ground production modules and the isolated `12` changed-module CI
profile. Ruff check/format, the campaign manifest and its eight contracts,
documentation governance, changed-test assertions, the 400-LOC changed-file
budget, and diff checks are clean.

This remains local, partial, and `not_released`. Arbitrary changing normals,
height or surface-velocity discontinuities, deformation/grass response,
torsional-spin damping, roll-to-skid transitions, regional PyQt6/React UI,
a versioned regional wire request/result schema, TypeScript/Rust/PyO3/WASM
regional physics, UpstreamDrift parity, protected CI, review, normal stack
integration, and main release remain open. Region plans and from/to identity
records are execution-scoped non-wire data in this child.

## 2026-08-10 PR #4304 corrected-impact propagation

Draft PR #4304 remains on `feat/4271-ground-skid-roll` with unchanged base
`feat/4270-ground-impact-bounce`. Exact corrected #4302 parent
`846653c21bd61a40aab99ab838c29915d0728e70` is incorporated by the normal merge
containing this handoff. The child preserves arbitrary-plane kinetic skid,
static-feasible pure roll, rolling resistance, qualified rest, finite-axis
edge localization, strict prefix/suffix result composition, and passive energy
ledgers while inheriting corrected flight-transfer ancestry, deterministic
workspace timestamps, and canonical `swing_sim` import identity. No branch was
rebased, retargeted, rewritten, or force-pushed.

The campaign remains partial and `not_released`. Material regions, changing
normals, terrain deformation, torsional spin damping, roll-to-skid transitions,
UI, TypeScript/Rust/PyO3/WASM physics, and downstream parity remain excluded.
Protected CI, independent review, normal dependency collapse, and consumer
delivery remain separate release gates.

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

Protected detect-secrets run `31360998491` correctly failed exact head
`d09f3129a68322bfc5dd30763556ac356ef2e55c` because the immutable SHA-256
golden-fixture digest looked like a high-entropy hexadecimal credential. The
test now carries the scanner's explicit inline allowlist annotation. The
digest and fixture bytes are unchanged, and this correction changes no
physics, numerical result, schema, or API. SPEC 1.14.20 records the repair.
All `115` ground tests, Ruff, formatting, a finding-free local scan of the
affected file, documentation governance, the `370`-line source-size check,
and diff gates pass before an ordinary guarded fast-forward publication.
Fresh protected CI and review remain required.

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

The local `feat/4271-ground-skid-roll` worktree continues exact corrected
#4270 parent `920c46dee688815691e251777142126bf1489b1a` without rewriting or
publishing any branch. Its intended normal base is
`feat/4270-ground-impact-bounce`. No GitHub write, PR, protected check, review,
or release claim exists for this child yet.

The Python ground facade now continues an exact `SETTLED_TO_SKID` handoff over
one immutable arbitrary-orientation plane through kinetic Coulomb skid,
static-feasible pure roll, rolling resistance, retained axial spin, and
qualified rest. A finite tangent-axis domain localizes `LEFT_SURFACE` exactly.
Typed cancellation, step, time, event, and unsupported-surface outcomes fail
closed; invalid numerical states raise without a result. A passive ledger
retains translation, rotation, gravity work, moving-plane work, and
dissipation; skid and roll paths remain distinct.

The result composer joins #4270 and #4271 evidence without duplicate or
epsilon-time points, reconstructs immediate-capture `IMPACT` from the signed
first event, and constructs strict v1 summaries only for representable rest,
left-surface, time-limit, or event-limit outcomes. Partial/edge endpoint totals
are explicitly censored, and the legacy result adapter now refuses non-rest
complete output.

`docs/specs/GROUND_SKID_ROLL.md` is the scientific authority. The shared
analytic fixture is locked at SHA-256
`74e23ebe86c8b476a3414b0ff11e561e126810b5358337cb87bc1e35e3a1d73d`.
The complete ground suite is `108 passed` on CPython 3.11.9 and real CPython
3.10.20. Pinned mypy 1.13 passes all 24 ground production modules; pinned Ruff
0.14.10 check/format passes the 15 changed Python files. The manifest validates
with all eight contract tests, and documentation governance passes.

The campaign remains partial and `not_released`. Material regions, changing
normals, terrain deformation, torsional spin damping, roll-to-skid transitions,
UI, TypeScript/Rust/PyO3/WASM physics, and downstream parity remain open.
Protected CI, independent review, exact-head publication, parent integration,
and consumer delivery are still required.

## 2026-08-09 PR #4302 pinned-MyPy current-head correction

Hosted quality-gate run `31350134551` exposed four deterministic MyPy 1.13
findings on published head `ceaed9e548c5b6d147dbbeb17ee3ff2a509436c5`:
the lazy wire-serializer import was inferred as `Any`, and repeated-bounce
sampling repeatedly accessed an optional mutable grid-time attribute after its
runtime guard. The correction binds the already validated serializer boundary
to its declared mapping type and narrows the initialized grid time into one local
`float` before advancing it and writing it back. No physics, schema, numerical
ordering, result content, issue scope, or stack base changes. Focused pinned
MyPy and ground tests must be green before the normal fast-forward push.

## 2026-08-09 Ground impact and repeated-bounce local slice

Draft PR #4302 publishes issue #4270 on `feat/4270-ground-impact-bounce` at
immutable evidence commit `63a6f4bec63c58d28bceed2e8cf348a618c8e366`.
It targets exact published #4288 head
`4972e55e0bb6e5b6bf7da0f899eed5d4f54e7d9d` on
`feat/4269-flight-ground-transfer`; no existing stack base was changed.

The self-facaded ground package now exposes a typed passive restitution plus
Coulomb sphere-plane impulse, full angular coupling, moving-boundary energy
ledger, exact bracket contact, and deterministic repeated ballistic hops.
Absolute event/sample times are retained while `max_time_s` starts at first
contact; `max_events` includes first contact. Capture emits one exact-contact
`SKID` point and `handoff_state` without a duplicate timestamp. Typed airborne
segments make `bounce_air_distance_m` reproducible as accumulated x-z arc
length. Cancellation and time/event/no-recontact/numerical limits return only
a validated prefix.

`docs/specs/GROUND_IMPACT_BOUNCE.md` is the scientific authority and the shared
golden fixture is locked by SHA-256. The campaign remains `not_released`.
Issue #4271 still owns skid/roll/rest, total distance, and the final
`GroundSimulationResult`; terrain deformation/material response, UI,
TypeScript physics, Rust/PyO3/WASM, and downstream adapters remain excluded.
Final local validation is `82 passed` for the complete ground package on both
CPython 3.11.9 and real CPython 3.10.20. Pinned mypy 1.13 reports no issues
across all 17 ground production modules. Pinned Ruff 0.14.10 check and format
pass the changed Python set. The campaign manifest validates, its eight
contract tests pass, documentation governance and focused changed-test
assertion gates pass, and all changed production modules/functions/signatures
remain within 400-line/50-line/four-parameter budgets. Protected CI, review,
and ordinary parent integration remain required.

Independent pre-publication review made no material physics, schema, or scope
change: vector primitives now return explicit `Vector3` tuples without typing
suppressions, and internal sampling/contact initialization invariants raise
deterministic runtime errors instead of relying on optimizable assertions. The
complete 82-test ground suite, pinned mypy, Ruff, and diff gates remain green.

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

## 2026-08-09 PR #4302 deterministic-digest scanner repair

Protected CI correctly remained blocking at head
`920c46dee688815691e251777142126bf1489b1a`, but `detect-secrets` classified
the SHA-256 assertion for the committed cross-runtime impact golden fixture as
a high-entropy credential. The value is deterministic public test evidence,
not a secret. Its exact assertion now carries the repository-standard inline
`pragma: allowlist secret`; scanner scope and the shared baseline are unchanged.

Commit this narrow repair with all three canonical handoffs and push normally
to `feat/4270-ground-impact-bounce`. Do not retry the unchanged failed run,
amend history, or force-push. Descendant PRs #4304 and #4305 inherit this file
and must later receive the parent by ordinary merge commits.
