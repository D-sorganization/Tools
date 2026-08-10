# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-10.

## 2026-08-10 PR #4316 exact-head CI type repair

Current-head quality-gate run `31389230948` identified four redundant `str`
casts in the PyQt persistence export accessors under the repository's pinned
MyPy 1.13 delta configuration. The serializers already declare exact `str`
returns; explicit typed local bindings preserve that contract across differing
import-following environments without runtime coercion. The casts are removed
without changing runtime behavior. SPEC 1.14.38 records the repair. Fresh
isolated MyPy 1.13 and 1.15 checks pass on all five production modules; 15
focused Python/PyQt tests, Ruff, Python 3.10 compilation, formatting, and
governance also pass. Protected current-head CI remains required; #4274 and
#4267 remain open.

## 2026-08-10 issue #4274 playback workspace persistence and exports

A reviewed continuation on `feat/4274-ground-playback-persistence` adds matched
PyQt6/React persistence and deterministic evidence export from exact reviewed
implementation head `80f0f3ebdb0835c300f9f1e60e7ef2f8703e6cc8`. The strict
`rate-of-closure-ground-playback-workspace/v1` document embeds the validated
`flight-to-ground-result/v1` plus paused absolute time, supported speed, loop
state, and a UI-neutral orbit camera. Imports reject duplicate/unknown fields,
unsupported versions, nonfinite or out-of-range state, oversized documents,
and oversized trajectories before replacing the last-good workspace. Active
timers are never persisted and importing always restores paused.

Both surfaces export lossless canonical result JSON and deterministic
LF-terminated trajectory/event CSV with every raw position, linear velocity,
angular velocity, frame, phase/event, time, and sequence field. PyQt uses
atomic `QSaveFile` replacement. The existing PyQt tab was reduced from 422 to
367 lines by separating contract, persistence orchestration, and file controls.
This slice executes no physics and does not add surface editors, terrain
meshes, comparison overlays, ensembles, solvers, compiled runtimes, or
UpstreamDrift integration. Keep issue #4274 and epic #4267 open.

Evidence passes 17 focused Python/PyQt tests (including tooltip governance),
the complete Rate suite (891 tests), 12 focused React/model tests, the complete
React suite (110 files / 678 tests), and the 198-module Vite production build.
Ruff check/format, pinned-style MyPy on five production modules, TypeScript,
zero-warning ESLint, documentation, scoped secrets, module size, minimum-test,
conflict-marker, and final diff gates pass. A normal two-parent merge now has
original feature commit `abb55c177af19a3cc08dd6bd5d258ea5ce3a61b9`
first and exact ready-for-review parent PR #4315 head
`2618ab025622bf1a4fa21e771b30f808f783648b` second. The base remains
`feat/4274-ground-playback`; no branch was rewritten or retargeted. SPEC
1.14.36 records the propagation. Independent exact-head review is READY at
merge `0ef91e84b6d49551723ba0fbfb8eb1bf7b1ebfa2`, and ready-for-review
PR #4316 is published against unchanged `feat/4274-ground-playback`. SPEC
1.14.37 records publication. Protected CI, approval, parent landing, issue
acceptance, and epic closure remain open.

## 2026-08-10 issue #4274 exact-parent propagation

The locally reviewed Ground Playback continuation now incorporates exact
current parent `f4ca3f801f60c1c3042d4ed1a6100fdd7cfebd4b` from draft PR #4309 by a
normal two-parent merge, with original playback head
`9045f8f3684fcf87bbe0ef3f5c1e1afba0ed5708` first and the corrected ground
reference executor second. The base remains
`feat/4275-ground-reference-execution`; no parent branch was rewritten or
retargeted. SPEC 1.14.33 records the propagation. Fresh current-diff evidence
passes `77` focused Python tests, `1,105` Rate/ground Python tests, the complete
React suite (`109` files / `673` tests), TypeScript, zero-warning ESLint, and a
`197`-module production build. Ruff/format, pinned MyPy 1.13 on six production
files, real CPython 3.10 compilation, documentation and minimum-test governance,
a `10`-file secrets scan, conflict-marker scans, and diff checks also pass. The
exact parent supplies current green Rust/fmt/clippy evidence because this child
adds no Rust delta. Independent exact-diff review is READY at implementation
merge `80f0f3ebdb0835c300f9f1e60e7ef2f8703e6cc8`. Ready-for-review PR #4315 is
now published with unchanged base `feat/4275-ground-reference-execution`; SPEC
1.14.34 records that live publication state. Protected CI, approval, parent
landing, dependency integration, issue acceptance, and epic closure remain
open.

## 2026-08-10 issue #4274 playback clock and evidence parity repair

Independent release review found that the PyQt player advanced one nominal
timer interval per callback and discarded loop overshoot, and that both client
evidence tables omitted scientifically material state and provenance fields.
PyQt now anchors playback to an injected monotonic clock, re-anchors speed and
loop-mode changes without discontinuity, uses modulo loop wrap, and has
deterministic delayed-tick, speed, toggle, and overshoot coverage. PyQt and React now expose full
trajectory linear/angular velocity, event pre/post linear/angular velocity,
result identity/status/termination, input SHA-256, calibration ID/confidence,
and warnings from the same shared golden result. The full
Rate suite passes 873 tests and the full React suite passes 109 files / 673
tests; pinned MyPy 1.13, Ruff, TypeScript, zero-warning ESLint, and production
build gates pass. Independent re-review, exact-head publication, protected CI,
required approval, dependency integration, and epic closure remain open.

## 2026-08-10 issue #4274 strict browser-import repair

The React Ground Playback importer now routes raw text through the existing
duplicate-key-aware `flightToGroundResultFromJson` facade before semantic
validation. This closes a review finding in which `JSON.parse` discarded
duplicate fields before the strict result parser could reject them. A
regression proves duplicate `request_id` fields fail atomically, leave the
last valid trajectory and summary intact, and tell the user that the valid
result remains loaded. The complete React suite passes 109 files / 673 tests;
zero-warning ESLint, TypeScript type-checking, and the production Vite build
also pass. This is local evidence only: independent re-review, exact-head
ready-for-review publication, protected CI, review approval, dependency integration, and epic
closure remain open.

## 2026-08-10 PR #4309 corrected-scalar-study propagation

Draft PR #4309 remains on `feat/4275-ground-reference-execution` with
unchanged base `feat/4273-ground-study-scalar-adapter`. Exact corrected #4308
parent `edd898089d017e36b814bfea408a7845734c7706` is incorporated by the normal
two-parent merge containing this handoff. The child preserves its bounded
one-shot Python reference executor: immutable request/settings are validated
before execution, the same cooperative-cancellation callback spans existing
repeated bounce, exact settled-to-skid handoff, skid/roll, and canonical result
composition, and only representable suffixes compose. Typed cancellation and
execution failures retain phase, native reason, and request fingerprint, while
the deterministic full-pipeline fixture pins the integrated result. No branch
was rebased, retargeted, rewritten, or force-pushed.

This remains a partial `not_released` slice. Changing normals or material
regions, terrain deformation, torsional damping, roll-to-skid transitions,
production material profiles, ensembles, inverse solving, UI, compiled
runtimes, and downstream consumer parity remain excluded. Keep issues #4273
and #4275 and epic #4267 open. Protected CI, independent review, normal stack
collapse, and consumer delivery remain separate release gates. Downstream PRs
#4274 and #4312 still descend from the prior #4309 head and require their own
ordinary propagation after this push.

Merged-tree validation is `238` focused ground/scalar tests on both CPython
3.11.9 and real CPython 3.10.20, plus `1,404` passed and seven documented
optional-Rust-wheel skips across the broader Rate of Closure/swing/flight/
ground/import-alias selection. React passes `106` files / `661` tests, the
189-module Vite production build, and zero-warning ESLint. The complete
`tools-core` Rust suite passes `137` tests (`111` unit, `20` transfer, `6`
wire), workspace formatting, and warning-denied Clippy. Pinned Ruff 0.14.10
check/format passes all six net changed Python files; pinned MyPy 1.13 passes
51 ground/flight/scalar production modules. The campaign manifest validator
and eight contract tests, documentation governance, ground-module budget,
six-file protected changed-only 500-LOC budget, 13-file scoped marker scan,
and diff checks are clean. A separate exploratory whole-repository size scan
still reports four unchanged legacy modules outside this PR diff; it is not
the protected changed-file gate. Hosted checks and review apply to the new
exact merge head only.

## 2026-08-10 PR #4308 corrected-result-adapter propagation

Draft PR #4308 remains on `feat/4273-ground-study-scalar-adapter` with
unchanged base `feat/4273-ground-study-result-adapter`. Exact corrected #4307
parent `76292d7a97e891aa88b06b3ea85f9e7e5b506e9e` is incorporated by the normal
two-parent merge containing this handoff. The child preserves its bounded,
deterministic `ground-study/scalar-ensemble/v1` adapter: callers supply explicit
series/trial identity, overflow and duplicate rows fail closed, complete and
censored studies retain observed values, and failed/unavailable studies retain
evidence with null scalars. Rows keep study/request/result/profile digests,
calibration and producer provenance, surface/frame and target geometry,
qualification and operating conditions, eligibility reasons, and typed target
availability while inheriting corrected qualified-result, material-profile,
impact/roll, timestamp, and canonical `swing_sim` ancestry. No branch was
rebased, retargeted, rewritten, or force-pushed.

This remains a partial `not_released` slice. Rendered variation/dispersion
plots, ensemble runners, optimizers, UI, compiled runtimes,
regional/changing-normal terrain, and four-surface consumer parity remain
excluded. Keep issue #4273 and epic #4267 open; protected CI, independent
review, normal dependency collapse, and consumer delivery remain separate
release gates.

Merged-tree validation is `217` focused ground/scalar tests on both CPython
3.11.9 and real CPython 3.10.20, plus `1,383` passed and six expected skips
across the broader 1,389-case Rate of Closure/swing/flight/ground/import-alias
selection. React passes `106` files / `661` tests, the 189-module Vite
production build, and zero-warning ESLint. The complete `tools-core` Rust suite
passes `137` tests (`111` unit, `20` transfer, `6` wire), workspace formatting,
and warning-denied Clippy. Pinned Ruff 0.14.10 check/format passes both net
changed Python files; pinned MyPy 1.13 passes 49 ground/flight/scalar production
modules. The campaign manifest validator and eight contract tests,
documentation governance, module budget, changed-only 500-LOC budget, eight-file
scoped marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.

## 2026-08-10 PR #4307 corrected-study propagation

Draft PR #4307 remains on `feat/4273-ground-study-result-adapter` with
unchanged base `feat/4273-ground-study-projection`. Exact corrected #4306
parent `99f7fefbd61a7eb9285c4a9297618bf52344055e` is incorporated by the normal
two-parent merge containing this handoff. The child preserves its fail-closed
`qualified_study_to_ground_model_result` bridge, which exposes total, roll,
bounce count, and final offline values only from a solver-eligible study and
keeps the study as the provenance authority, while inheriting corrected
material-profile, impact/roll, deterministic timestamp, and canonical
`swing_sim` ancestry. No branch was rebased, retargeted, rewritten, or
force-pushed.

This remains a partial `not_released` slice. Production presets/calibration
claims, profile UI, regional/changing-normal terrain, compiled runtimes, and
four-surface consumer parity remain excluded. Keep issue #4273 and epic #4267
open; protected CI, independent review, normal dependency collapse, and
consumer delivery remain separate release gates.

Merged-tree validation is `198` focused ground tests on both CPython 3.11.9
and real CPython 3.10.20, plus `1,371` passed and six expected skips across the
broader 1,377-case Rate of Closure/swing/flight/ground/import-alias selection.
React passes `106` files / `661` tests, the 189-module Vite production build,
and zero-warning ESLint. The complete `tools-core` Rust suite passes `137`
tests (`111` unit, `20` transfer, `6` wire), workspace formatting, and
warning-denied Clippy. Pinned Ruff 0.14.10 check/format passes all four net
changed Python files; pinned MyPy 1.13 passes 47 ground/flight production
modules. The campaign manifest validator and eight contract tests,
documentation governance, module budget, four-file changed-only 500-LOC
budget, ten-file scoped marker scan, and diff checks are clean. Hosted checks
and review apply to the new exact merge head only.

## 2026-08-10 PR #4306 corrected-material-profile propagation

Draft PR #4306 remains on `feat/4273-ground-study-projection` with unchanged
base `feat/4272-ground-material-profiles`. Exact corrected #4305 parent
`dcfc8ef9fe522b817e64e72e964264d1770a916d` is incorporated by the normal
two-parent merge containing this handoff. The child preserves the strict
`ground-study-projection/v1` record, arbitrary-plane contact-target geometry,
qualified/calibrated solver-eligibility gates, canonical revalidation, typed
unavailable evidence, and removal of the unqualified direct metric adapter
while inheriting corrected impact/roll ancestry, deterministic workspace
timestamps, and canonical `swing_sim` import identity. No branch was rebased,
retargeted, rewritten, or force-pushed.

This campaign slice remains partial and `not_released`. Production presets and
calibration claims, profile UI, regional/changing-normal terrain, compiled
runtimes, and four-surface consumer parity remain excluded. Issue #4273 and
epic #4267 remain open; protected CI, independent review, normal dependency
collapse, and consumer delivery remain separate release gates.

Merged-tree validation is `194` focused ground tests on both CPython 3.11.9
and real CPython 3.10.20, plus `1,367` passed and six expected skips across the
broader 1,373-case Rate of Closure/swing/flight/ground/import-alias selection.
React passes `106` files / `661` tests, the 189-module Vite production build,
and zero-warning ESLint. The complete `tools-core` Rust suite passes `137`
tests (`111` unit, `20` transfer, `6` wire), workspace formatting, and
warning-denied Clippy. Pinned Ruff 0.14.10 check/format passes all 18 net
changed Python files; pinned MyPy 1.13 passes 47 ground/flight production
modules. The campaign manifest validator and eight contract tests,
documentation governance, module budget, 18-file changed-only 500-LOC budget,
scoped marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.

## 2026-08-10 PR #4305 corrected-skid-roll propagation

Draft PR #4305 remains on `feat/4272-ground-material-profiles` with unchanged
base `feat/4271-ground-skid-roll`. Exact corrected #4304 parent
`ee77b059bd83f7dafac7e0d411665231cdb7435c` is incorporated by the normal merge
containing this handoff. The child preserves strict qualified SI material
profiles/libraries, fail-closed write-through atomic CAS persistence, exact
operating-condition solver binding, and provenance-complete neutral terrain
snapshot adaptation while inheriting corrected impact/roll ancestry,
deterministic workspace timestamps, and canonical `swing_sim` import identity.
No branch was rebased, retargeted, rewritten, or force-pushed.

The campaign remains partial and `not_released`. Production presets, profile
UI, regional/changing terrain physics, compiled runtimes, and downstream
consumer parity remain excluded. Protected CI, independent review, normal
dependency collapse, and consumer delivery remain separate release gates.

Merged-tree validation is `168` focused ground tests on both the current
runtime and real CPython 3.10.20, `1073` broad Python tests, `106` React files /
`661` tests, and the complete `tools-core` Rust suite at `137` tests (`111`
unit, `20` transfer, `6` wire). The combined compatibility/ground/flight/alias
suite is `232` tests on real CPython 3.10.20. The 189-module Vite production
build, TypeScript, zero-warning ESLint, Ruff check/format across 59 files,
pinned mypy 1.13 across all 38 ground and nine transfer production modules,
Rust workspace format plus warning-denied `tools-core` clippy, campaign-manifest
validator plus eight contracts, documentation governance, 20-file 500-LOC
budget, marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.

## 2026-08-10 PR #4305 deterministic-digest secret-scan repair

Exact parent repair `1a65d638cc0787c4e32f28bb37862205d5068671` is
incorporated by the normal merge containing this handoff. Protected
detect-secrets run `31361053024` also classified this child profile and
library's two immutable canonical SHA-256 digests as high-entropy strings.
Those non-secret scientific integrity values now carry explicit inline
allowlist annotations. All three digest values and fixture bytes remain
unchanged; physics, numerics, schemas, APIs, and persistence behavior are
unchanged. SPEC 1.14.22 records the child repair above relabeled child feature
1.14.21 and parent repair 1.14.20. All `168` ground tests, eight manifest
contracts, Ruff, formatting, finding-free scans of both affected test files,
documentation governance, `370`/`389`-line source-size checks, conflict-marker,
and diff gates pass. Protected CI, review, and downstream propagation remain
open after an ordinary guarded push.

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

## 2026-08-09 issue #4272 evidence SHA correction

The authoritative implementation commit is
`3645fc4d28e332785eb23cd2198ed0be931614d0`. Earlier documentation expanded
the valid short SHA `3645fc4d2` to a nonexistent full object ID. This
documentation-only correction updates every local evidence reference before
the branch is used as a child base; it introduces no runtime or scientific
change.

## 2026-08-09 PR #4305 protected quality-gate portability repair

Protected CI run `31358547585`, job `93362698271`, failed at exact head
`c242fdacd5c9e9a59e5ffb8934542eaa67114452` because Linux-hosted MyPy 1.13
correctly does not expose Windows-only `ctypes.WinDLL` and
`ctypes.get_last_error` attributes. The repair isolates those members behind
the already guarded Windows-only helper, uses the module namespace for
platform-conditional lookup, and adds an adversarial unit test that proves the
wide-character `MoveFileExW` call retains `REPLACE_EXISTING | WRITE_THROUGH`
flags and ctypes signatures. The focused store suite (12 tests), full ground
suite (168 tests on CPython 3.11.9 and real CPython 3.10.20), pinned Ruff
0.14.10 across 55 files, CI-equivalent MyPy 1.13 across 14 changed production
files, manifest plus 8 tests, documentation governance, and diff checks pass
locally. This is a portability/type-check repair only; persistence semantics
and numerical contracts are unchanged. Protected rerun evidence remains
pending.

## 2026-08-09 issue #4272 draft publication

Draft [PR #4305](https://github.com/D-sorganization/Tools/pull/4305) was opened
from `feat/4272-ground-material-profiles` at exact documentation carrier
`e90b3b36a1b1eb2f051fae1dd549bd9da77a6a8b`, based on unchanged
`feat/4271-ground-skid-roll` at
`482cdf272b04c78b50da91a6d2ddd4d15e063c7b`. The independently audited
implementation evidence remains
`3645fc4d28e332785eb23cd2198ed0be931614d0`. Protected CI, review approval,
parent integration, production presets, UI, compiled runtimes, and consumer
parity remain open. No parent branch was rewritten or retargeted.

## 2026-08-09 issue #4272 immutable implementation evidence

Implementation commit `3645fc4d28e332785eb23cd2198ed0be931614d0`
is the independently audited ground-profile, persistence, and adapter evidence.
Its exact tree passes 167 ground tests on CPython 3.11.9 and real CPython
3.10.20, 75 focused adversarial/API tests, pinned Ruff 0.14.10 across 55 ground
files, and CI-isolated MyPy 1.13 across 38 production modules. Manifest, eight
manifest tests, documentation governance, changed-test assertions,
structural/file-size budgets, and diff checks pass. This documentation-only
carrier records immutable local evidence; no runtime, schema, API, or numerical
change is introduced. Protected CI, review, parent integration, production
presets, UI, compiled runtimes, and consumer parity remain open.

## 2026-08-09 issue #4272 ground material profile contract slice

Draft PR #4305 continues exact PR #4304 carrier
`482cdf272b04c78b50da91a6d2ddd4d15e063c7b` with unchanged base
`feat/4271-ground-skid-roll`. No protected-check, review, integration, or
release claim exists yet for this child.

The bounded Python slice adds strict versioned SI profile/library documents,
uncertainty, evidence-linked validity bounds, seven-gate qualification and a
separate calibrated/illustrative scientific status, structural schemas plus
authoritative semantic validation, canonical identities, and explicit
applicability-aware solver binding. Fail-closed atomic CAS persistence includes
typed recovery/indeterminate outcomes, Windows write-through replacement,
POSIX directory sync, and reparse/root-identity checks under a documented
cooperative single-principal boundary. A one-way neutral Upstream terrain
snapshot adapter retains separate terrain/material identities and revisions,
source, frame, velocity, transform, adapter version, interpretation, individual
digests, combined input digest, and complete field dispositions without
importing UpstreamDrift classes. Exported binding, persistence, and adapter
results enforce their own exact-type, provenance, hash, and coherence DbC.

`docs/specs/GROUND_MATERIAL_PROFILES.md` is the scientific authority. The full
ground suite passes 167 tests on CPython 3.11.9 and isolated real CPython
3.10.20; the latter has only five expected missing-plugin configuration
warnings. Pinned Ruff 0.14.10 is clean across the complete 55-file ground tree,
and CI-isolated MyPy 1.13 is clean across all 38 production modules. The
campaign validator, eight manifest tests, documentation governance, 500-line
file-size gate, focused 400/50-line structural budget, and diff check pass.
Exact-head and protected publication evidence must still be captured before
this slice is considered published.

The final read-only release-gate audit found no remaining code or contract
blockers after adversarial fixes for raw numeric identity, referenced
calibration coverage, exact nested records, schema/semantic separation,
operating-condition and solver-value binding, output hash coherence, Windows
write-through replacement, typed probe failures, reparse points, safe Win32
filenames, and collision-resistant terrain/material solver identities.
Exact-head publication, protected CI, review, dependency integration, and
consumer delivery remain external release gates.

Issue #4272 and epic #4267 remain open. Production presets or calibration
claims, editing UI, regional/changing-normal terrain, TypeScript/Rust/PyO3/WASM
delivery, UpstreamDrift consumers, and four-surface parity are not delivered by
this slice.

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

## 2026-08-09 Ground-study projection continuation

Issue #4273 now has a narrow local foundation on
`feat/4273-ground-study-projection`, based exactly on PR #4305 head
`a35fc8aac0cbc2aeeef757fd1d1c518987f2355c`. The new
`ground-study-projection/v1` contract preserves the canonical result summary,
observed endpoints, caller-request context digest, surface/model/profile identity, warnings,
and typed unavailable evidence. The self-contained record embeds the exact
source-result digest, complete solver surface, ball radius, full material
profile/condition binding, exact result calibration/provenance, typed result warnings, and separate profile warning
codes. Construction and parsing re-derive summary/endpoints, sphere/plane
contact, intrinsic target miss, and profile/surface coherence. Only complete
rest results with a qualified calibrated bound profile and measured/literature
result calibration with positive confidence enter objectives. Estimated,
unvalidated, or zero-confidence result calibration fails closed. A
valid partial airborne endpoint remains censored and carries typed
`endpoint_airborne` target unavailability without an inferred landing.
`ground-result/v1` does not carry its producing request fingerprint, so the
request and result digests are not an attested pair; only shared identity,
surface/frame, calibration, and provenance compatibility are checked.

This is a local implementation checkpoint, not issue completion or release
evidence. Ensemble, variation, wind, optimizer, PyQt6/React, compiled-runtime,
and UpstreamDrift adapters remain open. Publish only as the next stacked draft
PR targeting `feat/4272-ground-material-profiles`, after full ground tests,
structural budgets, manifest validation, and an independent review pass.

The independently reviewed implementation commit is
`0de714842cf4cd1207944044c883c2d8dc83a7ba`; after normal parent propagation,
192 ground tests and 47 focused projection/state/wire/API tests pass.

Draft PR #4306 publishes the stack child at branch
`feat/4273-ground-study-projection` against
`feat/4272-ground-material-profiles`. Its creation head was
`6a1b2f76160de0535fca2126958934c53ad5ed25`. Keep #4273 and #4267 open and
require normal protected checks/review before any merge.

The next local child `feat/4273-ground-study-result-adapter` begins at exact PR
#4306 creation head `17473948f1ce5837bd5b55618d5393b0d8575188` and normally includes current
#4306 head `d44edeb4119048fe7a3f8ccfdcae81c8771561e8`. Its narrow continuation adds
`qualified_study_to_ground_model_result`, which populates the existing total,
roll, bounce-count, and final-offline DTO only from a solver-eligible study.
Target misses remain valid physics observations. The DTO is lossy and never
replaces the study's provenance. This is not yet published or issue completion.
Exact reviewed adapter evidence is commit
`6c296ab35471fc8d2070d229f2921d200f7defdb`: 198 ground tests, 27 flight-first
import/result/transfer tests, and 44 focused adapter/compatibility/API tests
pass. The independent re-review reports no remaining blocker.
Draft PR #4307 publishes this continuation against preserved parent branch
`feat/4273-ground-study-projection`; its creation head is
`dac35e3fd61ee8af80dc8c2262da31ea274dbb1d`. Keep #4273/#4267 open and require
normal protected checks and review.

Post-publication import-order regression: importing the flight package first
exposed a cycle through the expanded ground package facade and solver package.
`flight.models` and `flight.surface_simulation` now import their neutral ground
record types directly from `ground.contract_types`, while the ground facade
loads solver-dependent study exports through `__getattr__` only when requested.
This preserves the public API while restoring flight-first and ground-first
import order. Keep this fix on PR #4306 and propagate it to later children only
by normal merge.

The study now persists exact result calibration/provenance and re-derives
eligibility from model calibration as well as the material profile. Provenance
is retained for audit but is not presented as producer certification.
The legacy direct result-to-metric adapter is now module-only and emits a
deprecation warning; it is deliberately absent from the package facade because
it cannot prove material-profile qualification.

Exact qualification/import repair commit
`940563f222065c4f343b587699c52062c6e1db59` passes 194 ground tests, 27
flight-first import/result/transfer tests, and an independent 75-test adversarial
review with no remaining blocker.
No material handoff change beyond clarifying the deprecated adapter's rejection
text: it no longer calls its unqualified compatibility input "qualified."

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

## 2026-08-09 Ground-study scalar ensemble continuation

Local branch `feat/4273-ground-study-scalar-adapter` starts exactly from PR
#4307 head `de6ea15290f6b3c5c49bd436b846baa8f6cb752b`. It adds a bounded,
deterministic adapter from explicit `(series_id, trial_index,
GroundStudyProjection)` samples into `scalar-ensemble/v1`. No identity is
inferred. Complete and censored rows keep observed metrics; failed/unavailable
rows keep their cohort and evidence with all scalars null. Partial airborne
rows keep first-contact evidence and report null final-target values with typed
unavailability. Target misses and ineligible-but-numeric studies remain visible
rather than being discarded.

Rows retain a whole-study digest, request/result digests, exact target geometry,
calibration and producer provenance, surface/frame, profile qualification and
operating condition, eligibility
reasons, and target availability. Input collection rejects before retaining a
row beyond the configured bound and never truncates. This is a bounded #4273
continuation, not ensemble-runner, optimizer, UI, compiled-runtime, downstream
parity, or protected-release completion. Before publication, run focused and
shared scalar contracts, full ground tests, Ruff, MyPy, structural and manifest
gates, and independent review. Publish only as a stacked draft PR targeting
`feat/4273-ground-study-result-adapter`; keep #4273 and #4267 open.

Independent re-review declares exact implementation commit
`b71bf88b6ed888248ad152f69a2bd2de3892e256` READY after 198 ground and 19
adapter/shared-scalar tests, Ruff, Black, MyPy, manifest, documentation,
assertion, file-size, structural, and diff gates. Draft PR #4308 publishes that
commit against unchanged parent `feat/4273-ground-study-result-adapter` at
`de6ea15290f6b3c5c49bd436b846baa8f6cb752b`. Protected checks and review remain
open; this local evidence does not close #4273 or #4267.

## 2026-08-10 PR #4306 pinned-Ruff formatting repair

At exact PR head `d44edeb4119048fe7a3f8ccfdcae81c8771561e8`, the protected
`quality-gate` found a format-only failure: repository-pinned Ruff 0.14.10
would reformat `ground/__init__.py` and `ground/study_derivation.py`. Both
files are now formatted with that exact tool version; no behavior, public
contract, eligibility rule, or import boundary changed. Keep this repair on
PR #4306, preserve its base at exact #4305 head
`a35fc8aac0cbc2aeeef757fd1d1c518987f2355c`, and require ordinary protected
checks and review before merge. Issue #4273 and epic #4267 remain open.

## 2026-08-10 PR #4307 parent propagation and formatter repair

PR #4307 normally merges exact parent #4306 head
`1e1b576c36cc01e27542dd88747f54f918ff16bf` through merge commit
`6f4009e8e3a1b3cf226b84e761d6d60a9f450d7d`; no rebase, retarget, parent
rewrite, or force-push occurred. Hosted `quality-gate` run `31365680155`
identified one additional Ruff 0.14.10 formatting residual in
`ground/tests/test_study_result_adapter.py`. The helper signature is now in
the pinned canonical form with no behavioral or contract change.

Local repair evidence is 198 ground tests, 26 focused flight API/result/
transfer tests plus clean flight-first and ground-first import smoke checks,
and a 53-test adapter/compatibility/API superset. Ruff 0.14.10 check/format,
Black 26.1.0, MyPy 1.13 for the two changed production modules, the campaign
manifest and its eight contract tests, documentation governance, changed-file
size, and `git diff --check` all pass. Require fresh ordinary protected CI and
review at the pushed exact head; keep #4273 and #4267 open.

## 2026-08-10 Bounded ground-reference executor continuation

Branch `feat/4275-ground-reference-execution` starts from exact PR #4308 head
`c8ebf422669992c4a33db661b0c37dfe72b580ae`. The new
`ground-reference-execution/v1` boundary runs the existing repeated-bounce and
skid/roll solvers once and delegates successful result construction to the
existing composer. It composes only rest, left-surface, time-limit, and
event-limit suffixes after an exact settled-to-skid prefix. Cancellation raises
a distinct typed operational signal; every other non-representable native
terminal state fails closed with phase, native reason, and request fingerprint.
The current skid/roll solver does not emit its reserved `numerical_failure`
enum; native numerical exceptions continue to propagate without broad
reclassification.

The same optional cancellation callback reaches both phases. A committed
cross-runtime fixture pins a full bounce/skid/roll/rest request, result, and
canonical digests, while focused tests cover deterministic replay, censored
time-limit, finite-domain exit, cancellation, bounce failure, suffix step
exhaustion, exact nested controls, and public exports. This bounded #4273/#4275
slice does not add UI, ensembles, changing terrain, compiled runtimes,
production calibration, or UpstreamDrift parity and cannot close #4267. Require
the full ground suite, flight-first and ground-first imports, pinned Ruff,
Black, MyPy, manifest, documentation, assertion, structural, file-size, diff,
and independent-review gates before publishing a draft child of PR #4308.

Independent exact-tree re-review is READY after both identified blockers were
closed: the golden case now carries and reconstructs every concrete phase
setting, and resolver/request compatibility is proven to fail before bounce or
callback side effects. Exact local evidence is 219 ground tests, 44 focused
executor/API tests on CPython 3.12 and isolated CPython 3.10, and 26 flight
contract/result/transfer tests. Pinned Ruff 0.14.10, changed-file Black, pinned
MyPy 1.13, manifest plus eight tests, docs governance, assertion, 400/50/4
structural budgets, both import orders, fixture bytes, and diff checks pass.
Repository-wide Black still reports only the two unchanged inherited files
`ground/study_wire.py` and `ground/tests/test_profile_contract.py`; do not
misstate that baseline as a failure of this slice.

Draft PR #4309 publishes exact reviewed implementation commit
`c93c6f36d361f4c129d702565a9330149e175557` against unchanged parent branch
`feat/4273-ground-study-scalar-adapter` at
`c8ebf422669992c4a33db661b0c37dfe72b580ae`. The campaign manifest records the
carrier and immutable local evidence. Protected CI and repository review remain
open, so #4273, #4275, and #4267 stay open.

## 2026-08-10 Issue #4274 strict ground-result playback slice

Local branch `feat/4274-ground-playback` starts exactly from draft PR #4309
head `51492c3ddc8b15b1358434da9b29f600261c918a`. It adds first-class Ground
Playback modules to the standalone PyQt6 and React navigation surfaces. Both
clients import only a strict `flight-to-ground-result/v1` record, reject files
above 5 MiB or 100,000 samples, apply candidate imports atomically, and retain
the last valid result after an error. The React view explicitly states that it
does not execute ground physics; the PyQt disclosure states the same boundary.

One shared golden result pins matching absolute-time and phase-transition
semantics. Both clients provide play/pause/replay, exact sample stepping, phase
jumps, scrubbing, 0.25x through 4x speed, looping, carry versus total or
observed-end markers, locked-physical-scale 3D orbit/zoom/reset, and complete
trajectory, event, warning, calibration, and provenance evidence. Cross-phase
interpolation holds the preceding exact sample so the display does not invent
motion. Result v1 has no surface geometry, so both views use neutral axes and
do not claim a terrain plane.

Local evidence is 872 full Rate Python tests and 672 full React tests, plus the
focused 9-test adapter/Qt suite; Ruff 0.14.10 check/format, changed-file Black,
CI-pinned MyPy 1.13, React lint/type/build, changed-file policy, assertion,
file-size, documentation, and diff gates pass. Standalone Playwright Chromium
verified import, phase jumps, replay/pause, wheel zoom, disclosures, locked
aspect containment, and zero page-width overflow at 1440x900 and 520x900;
offscreen PyQt rendering verified a usable plot/control split at the supported
1024x700 minimum.

This is the smallest production #4274 slice, not issue or epic completion.
Surface editors, terrain meshes, comparison overlays, workspace persistence or
export, ensembles, inverse optimization, Rust/WASM execution, and UpstreamDrift
consumer parity remain open. Do not publish this local commit without an
independent exact-tree review and ordinary protected CI; keep #4274 and #4267
open.
