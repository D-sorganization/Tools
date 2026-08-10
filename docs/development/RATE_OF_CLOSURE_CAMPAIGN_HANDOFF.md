# Rate of Closure Ball-Flight Campaign Handoff

Status verified 2026-08-08. This isolated integration is published as draft
[PR #4217](https://github.com/D-sorganization/Tools/pull/4217). No source PR
branch was rewritten.

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

The reviewed Ground Playback continuation now descends from exact current
draft PR #4309 head `f4ca3f801f60c1c3042d4ed1a6100fdd7cfebd4b`
through a normal two-parent merge. Original playback head
`9045f8f3684fcf87bbe0ef3f5c1e1afba0ed5708` is first and the corrected ground
reference executor is second. The child base remains
`feat/4275-ground-reference-execution`; no parent was rewritten or retargeted.
SPEC 1.14.33 records this propagation. Fresh current-diff evidence passes `77`
focused and `1,105` broad Python tests, the complete React suite (`109` files /
`673` tests), TypeScript, zero-warning ESLint, and a `197`-module production
build. Ruff/format, pinned MyPy 1.13 on six production files, real CPython 3.10
compilation, documentation/minimum-test governance, a `10`-file secrets scan,
marker scans, and diff checks also pass. The exact parent supplies current green
Rust/fmt/clippy evidence because this child adds no Rust delta. Independent
exact-diff review is READY at implementation merge
`80f0f3ebdb0835c300f9f1e60e7ef2f8703e6cc8`. Ready-for-review PR #4315 is now
published with unchanged base `feat/4275-ground-reference-execution`; SPEC
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

Draft PR #4309 remains on `feat/4275-ground-reference-execution`, targeting
the unchanged `feat/4273-ground-study-scalar-adapter` base. Exact corrected
#4308 parent `edd898089d017e36b814bfea408a7845734c7706` is incorporated by the
normal two-parent merge containing this handoff. The child retains a bounded
one-shot reference executor over the existing repeated-bounce, exact
settled-to-skid, skid/roll, and canonical-composition phases. It validates the
immutable request, execution settings, and optional resolver before side
effects; preserves one cancellation callback across phases; composes only
representable suffixes; and emits typed phase/reason/request-fingerprint
evidence for cancellation or failure. Its full deterministic golden fixture
pins the integrated pipeline while acquiring corrected scalar-study,
qualified-result, material, impact/roll, timestamp, and shared-package
ancestry. No branch was rebased, retargeted, rewritten, or force-pushed.

This does not complete the issue or epic. Changing normals/material regions,
terrain deformation, torsional damping, roll-to-skid, production profiles,
ensembles, inverse solving, UI, compiled runtimes, and four-surface consumer
parity remain excluded. Keep #4273, #4275, and #4267 open; protected CI,
independent review, dependency-order collapse, and consumer delivery remain
required. Downstream PRs #4274 and #4312 still descend from the old #4309 head
and require normal propagation after this update.

Merged-tree validation is `238` focused ground/scalar tests on CPython 3.11.9
and real CPython 3.10.20; the broader Rate of Closure/swing/flight/ground/
import-alias selection reports `1,404` passed and seven documented optional-
Rust-wheel skips. React passes `106` files / `661` tests, its 189-module
production build, and zero-warning ESLint. `tools-core` passes all `137` tests
(`111` unit, `20` transfer, `6` wire), workspace formatting, and warning-
denied Clippy. Pinned Ruff 0.14.10 passes six net changed Python files and
pinned MyPy 1.13 passes 51 ground/flight/scalar production modules. Manifest
validation plus eight contracts, documentation governance, ground-module and
protected changed-only file budgets, 13-file scoped marker scan, and diff
checks pass. The optional whole-repository size scan separately reports four
unchanged legacy modules outside this diff. Hosted evidence must be
re-established on the new exact merge head.

## 2026-08-10 PR #4308 corrected-result-adapter propagation

Draft PR #4308 remains on `feat/4273-ground-study-scalar-adapter`, targeting
the unchanged `feat/4273-ground-study-result-adapter` base. Exact corrected
#4307 parent `76292d7a97e891aa88b06b3ea85f9e7e5b506e9e` is incorporated by the
normal two-parent merge containing this handoff. The child retains explicit
series/trial identity, bounded non-truncating collection, deterministic
`scalar-ensemble/v1` rows, observed complete/censored metrics, null-valued
failed/unavailable cohorts, exact target/qualification evidence, and complete
study/request/result/profile provenance while acquiring corrected qualified
result, material, impact/roll, timestamp, and shared-package ancestry. No
branch was rebased, retargeted, rewritten, or force-pushed.

This does not complete the issue or epic. Rendered variation/dispersion plots,
ensemble runners, optimizers, UI, compiled runtimes, regional/changing-normal
terrain, and four-surface consumer parity remain excluded. Keep #4273 and
#4267 open; protected CI, independent review, dependency-order collapse, and
consumer delivery remain required.

Merged-tree validation is `217` focused ground/scalar tests on CPython 3.11.9
and real CPython 3.10.20; the broader 1,389-case Rate of Closure/swing/flight/
ground/import-alias selection reports `1,383` passed and six expected skips.
React passes `106` files / `661` tests, its 189-module production build, and
zero-warning ESLint. `tools-core` passes all `137` tests (`111` unit, `20`
transfer, `6` wire), workspace formatting, and warning-denied Clippy. Pinned
Ruff 0.14.10 passes both net changed Python files and pinned MyPy 1.13 passes
49 ground/flight/scalar production modules. Manifest validation plus eight
contracts, documentation governance, module and changed-only file budgets,
scoped marker scan, and diff checks pass. Hosted evidence must be
re-established on the new exact merge head.

## 2026-08-10 PR #4307 corrected-study propagation

Draft PR #4307 remains on `feat/4273-ground-study-result-adapter`, targeting
the unchanged `feat/4273-ground-study-projection` base. Exact corrected #4306
parent `99f7fefbd61a7eb9285c4a9297618bf52344055e` is incorporated by the normal
two-parent merge containing this handoff. The child retains the narrow
qualified-study bridge into the existing total/roll/bounce/final-offline DTO,
continues to reject non-solver-eligible studies, and preserves the study as
the provenance authority while acquiring corrected material, impact/roll,
timestamp, and shared-package ancestry. No branch was rebased, retargeted,
rewritten, or force-pushed.

This is not issue or epic completion. Production presets/calibration claims,
profile UI, regional/changing-normal terrain, compiled runtimes, and
four-surface consumer parity remain excluded. Keep #4273 and #4267 open;
protected CI, independent review, dependency-order collapse, and consumer
delivery remain required.

Merged-tree validation is `198` focused ground tests on CPython 3.11.9 and
real CPython 3.10.20; the broader 1,377-case Rate of Closure/swing/flight/
ground/import-alias selection reports `1,371` passed and six expected skips.
React passes `106` files / `661` tests, its 189-module production build, and
zero-warning ESLint. `tools-core` passes all `137` tests (`111` unit, `20`
transfer, `6` wire), workspace formatting, and warning-denied Clippy. Pinned
Ruff 0.14.10 passes four net changed Python files and pinned MyPy 1.13 passes
47 ground/flight production modules. Manifest validation plus eight contracts,
documentation governance, module and changed-only file budgets, scoped marker
scan, and diff checks pass. Hosted evidence must be re-established on the new
exact merge head.

## 2026-08-10 PR #4306 corrected-material-profile propagation

Draft PR #4306 remains on `feat/4273-ground-study-projection`, targeting the
unchanged `feat/4272-ground-material-profiles` base. Exact corrected #4305
parent `dcfc8ef9fe522b817e64e72e964264d1770a916d` is incorporated by the normal
two-parent merge containing this handoff. The child retains its strict study
record, intrinsic arbitrary-plane target geometry, calibrated qualification
and solver-eligibility gates, canonical semantic revalidation, provenance and
typed unavailable evidence while acquiring corrected impact/roll ancestry,
deterministic workspace timestamps, and canonical `swing_sim` import identity.
No branch was rebased, retargeted, rewritten, or force-pushed.

This is still a bounded foundation, not issue or epic completion. Production
presets/calibration claims, profile UI, regional/changing-normal terrain,
compiled runtimes, and four-surface consumer parity remain excluded. Keep
#4273 and #4267 open. Protected CI, independent review, dependency-order
collapse, and consumer delivery remain required.

Merged-tree validation is `194` focused ground tests on CPython 3.11.9 and
real CPython 3.10.20; the broader 1,373-case Rate of Closure/swing/flight/
ground/import-alias selection reports `1,367` passed and six expected skips.
React passes `106` files / `661` tests, its 189-module production build, and
zero-warning ESLint. `tools-core` passes all `137` tests (`111` unit, `20`
transfer, `6` wire), workspace formatting, and warning-denied Clippy. Pinned
Ruff 0.14.10 passes 18 net changed Python files and pinned MyPy 1.13 passes 47
ground/flight production modules. Manifest validation plus eight contracts,
documentation governance, module and changed-only file budgets, scoped marker
scan, and diff checks pass. Hosted evidence must be re-established on the new
exact merge head.

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
detect-secrets run `31361053024` identified the parent skid/roll digest plus
this child profile/library's two immutable canonical SHA-256 digests as
high-entropy strings. Explicit inline allowlist annotations now identify all
three as non-secret scientific integrity evidence. Digest values, fixtures,
physics, numerics, schemas, APIs, and persistence behavior are unchanged.
SPEC 1.14.22 records the child correction. All `168` ground tests, eight
manifest contracts, Ruff, formatting, finding-free scans of both affected
test files, documentation governance, `370`/`389`-line source-size checks,
conflict-marker, and diff gates pass. Protected CI, review, and downstream
propagation remain open after a normal guarded push.

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

## 2026-08-10 PR #4304 deterministic-digest secret-scan repair

Protected detect-secrets run `31360998491` failed exact #4304 head
`d09f3129a68322bfc5dd30763556ac356ef2e55c` after identifying the skid/roll
golden fixture's pinned SHA-256 digest as a high-entropy hexadecimal string.
The test now uses the scanner's explicit inline allowlist annotation for this
non-secret scientific integrity value. The digest, fixture, physics,
numerical results, schema, and API are unchanged. SPEC 1.14.20 records the
repair. All `115` ground tests, Ruff, formatting, a finding-free local scan of
the affected file, documentation governance, the `370`-line source-size
check, and diff gates pass before an ordinary guarded fast-forward
publication; protected CI and review remain open.

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

## 2026-08-09 issue #4271 local static-plane skid/roll continuation

`feat/4271-ground-skid-roll` continues exact corrected #4270 parent
`920c46dee688815691e251777142126bf1489b1a`; its intended normal base is
`feat/4270-ground-impact-bounce`. No GitHub write, carrier, protected evidence,
or release advancement has been made for this child.

The UI-independent Python reference solver validates an exact
`SETTLED_TO_SKID` handoff and advances one immutable arbitrary-orientation
plane through kinetic skid, static-feasible pure roll, rolling resistance, and
qualified rest. It retains normal-axis spin, exact finite tangent-axis edge
exit, global first-contact sampling, global event counts, separate relative
skid/roll paths, and a mechanical ledger with gravity and moving-surface work.
Cancellation and bounded limit/unsupported-surface outcomes remain typed;
invalid numerical states raise without fabricating wire results.

Composition is explicitly cross-slice: immediate #4270 capture becomes one
signed `IMPACT` point, suffix samples begin strictly later, and no duplicate or
epsilon timestamp is introduced. Rest/left-surface map to complete v1 results;
time/event limits map to partial censored results; unrepresentable internal
outcomes fail closed. Carry, bounce-air, skid, roll, surface path, total
displacement, final coordinates, and post-first-contact bounce count retain
their strict definitions. The rest-only legacy adapter rejects complete edge
exit.

`docs/specs/GROUND_SKID_ROLL.md` and the SHA-locked analytic fixture
(`74e23ebe86c8b476a3414b0ff11e561e126810b5358337cb87bc1e35e3a1d73d`)
are the local authorities. The full ground package reports 108 passing tests
on CPython 3.11.9 and isolated real CPython 3.10.20. Pinned mypy 1.13 is clean
across 24 production modules; pinned Ruff 0.14.10 check/format passes all 15
changed Python files. The campaign manifest validates, all eight manifest
contracts pass, and documentation governance passes.

Program #4267 remains `partial_implementation` and `not_released`. This slice
does not qualify material regions, changing normals, deformation, grass or
moisture response, torsional spin damping, roll-to-skid, UI, TypeScript or
compiled physics, or downstream parity. Independent review, exact-head
publication, protected checks, ordinary parent integration, and consumers are
open release gates.

## 2026-08-09 PR #4302 pinned-MyPy current-head correction

Hosted quality-gate run `31350134551` failed published #4302 head
`ceaed9e548c5b6d147dbbeb17ee3ff2a509436c5` on four actionable MyPy 1.13
findings. The lazy wire serializer is now bound to its declared mapping type,
and the guarded optional output-grid time is copied to a local
`float`, advanced deterministically, and stored back once. This is a static
typing correction only: physics, schema, numerical order, outputs, scope, and
the normal #4288 base are unchanged. Pinned MyPy, the focused ground suite,
Ruff, and diff checks are required before publication.

## 2026-08-09 issue #4270 local ground-impact/bounce slice

Draft PR #4302 publishes `feat/4270-ground-impact-bounce` at immutable
evidence commit `63a6f4bec63c58d28bceed2e8cf348a618c8e366`. It targets exact
published #4288 head `4972e55e0bb6e5b6bf7da0f899eed5d4f54e7d9d`
on `feat/4269-flight-ground-transfer`; no existing stack base was changed.
Protected checks, review, parent integration, and release remain open.

The strict ground facade now includes the Python reference impulse/bounce
prefix specified by `docs/specs/GROUND_IMPACT_BOUNCE.md`. It resolves passive
normal restitution plus static/kinetic Coulomb impulse with arbitrary unit
normals, moving tangential surfaces, sphere inertia, and full angular coupling.
The repeat-hop state machine interpolates exact physical contact, uses pinned
standard gravity and analytic recontact roots, retains absolute public times,
interprets `max_time_s` from first contact, counts first contact against
`max_events`, checks cancellation at event boundaries, and never duplicates an
event/sample time.

Capture uses effective restitution zero and emits one exact-contact terminal
`SKID` point plus handoff state. Each completed or time-limited airborne segment
records exact endpoints and x-z arc length; `bounce_air_distance_m` sums that
evidence for #4271. A SHA-locked shared golden fixture and analytic, passivity,
property, bracket/output-convergence, ordering, cancellation, and failure tests
qualify the local slice.

Final local validation is `82 passed` for the complete ground package on both
CPython 3.11.9 and real CPython 3.10.20. Pinned mypy 1.13 reports no issues
across all 17 ground production modules. Pinned Ruff 0.14.10 check and format
pass the changed Python set. The campaign manifest validates, its eight
contract tests pass, documentation governance and focused changed-test
assertion gates pass, and all changed production modules/functions/signatures
remain within 400-line/50-line/four-parameter budgets.

Independent pre-publication review made no material physics, schema, or scope
change: vector helpers now return explicit fixed-length tuples without typing
suppressions, and internal initialization invariants raise deterministic
runtime errors rather than relying on optimizable assertions. The complete
82-test ground suite, pinned mypy, Ruff, and diff gates remain green.

The scope remains partial and `not_released`. Issue #4271 owns skid, roll,
rest, total distance, and final `GroundSimulationResult`. The #4270 law does
not consume firmness, grass, compressibility, moisture, or rolling resistance;
terrain deformation, UI, TypeScript physics, Rust/PyO3/WASM, and UpstreamDrift
adapters remain excluded. Protected CI, required review, ordinary parent
integration, and downstream delivery are still required.

## 2026-08-09 PR #4288 corrected-ground-parent propagation

Draft #4288 remains on `feat/4269-flight-ground-transfer` with unchanged base
`feat/4268-ground-contract`. Exact carrier-reconciled #4285 parent
`6a2bc9d06f6f9a28a0d615b19d2ed4fc13871059` is incorporated through the
normal local merge containing this handoff; no branch was rebased, retargeted,
force-pushed, or published. The descendant retains its qualified cross-runtime
terminal-state/contact transfer and now carries the complete corrected
wind/scalar/variation, capability, Python-3.10, campaign-manifest, and strict
ground-contract ancestry.

The only source conflict was the public flight facade test. Resolution keeps
the child's structural protocol and transfer value inventory plus the parent's
package-relative import required for Linux/editable collection. This is
ancestry propagation, not bounce/roll implementation. Protected CI,
independent review, exact-head publication, and child-first merge of #4288
into #4285 remain required before the ground parent can collapse toward the
wind carrier.

Focused evidence is 113 strict-ground, transfer/facade, compatibility,
scalar-adapter, and responsive-wind tests on Python 3.11 plus the same 113 on
real CPython 3.10.20. Ruff check/format passes 36 focused files. Pinned mypy
1.13 passes the 13-file transfer delta and 12-file ground production set in
separate established namespace invocations; the transfer test binds terminal
samples before exact `FlightStatePoint` narrowing without weakening runtime
assertions. The campaign manifest validates and all nine manifest/parity
contracts pass. Documentation governance, ancestry, SPEC order, and final diff
assertions remain required in the same merge.

## 2026-08-09 Flight-transfer stack propagation

The #4288 worktree now carries exact published #4285 head
`8e8df7b9c633affb986326137338313faf46d2db` through a normal merge while
retaining the declared base `feat/4268-ground-contract`.
The only code overlap was the flight integrator: the child keeps its bounded,
testable `flightIntegrator.ts` extraction rather than restoring the parent's
superseded inline RK4 loop. The Python API contract now inventories both the
parent capability evaluator records and child transfer records. No GitHub
write occurred. Focused validation passes with `82` Python tests, `38` React
tests, and `26` Rust tests; the complete affected Rate+swing_sim Python gate is
`1483 passed, 7 skipped`, with only optional local Rust-wheel skips.
The complete React gate is `104 files / 643 tests passed`, followed by clean
type-check, lint, and production build. Full `tools-core` Rust validation is
`137 passed`. Changed Python Ruff check/format and CI-pinned mypy 1.13, docs
governance, and staged/unstaged diff checks also pass.
The initial focused Python run also exposed a real circular import across the
ground and flight facades. The transfer adapter now talks directly to the
ground record/type modules it consumes; no public facade was widened or
removed.

This latest parent propagation is limited to the schema generator's pinned
mypy compatibility boundary and explicit casts in adversarial contract tests.
It has no wire or runtime transfer behavior change. Re-verify the merged child
before publication and cite only the new exact child head's protected checks.
Local post-merge evidence is now `70 passed` for the ground, transfer, and
flight-facade contract suites. The pinned mypy 1.13 profile passes all `13`
child-delta Python files, including tests, after representing frozen-dataclass
metadata with a test-only structural protocol. Ruff check/format, the changed
test assertion ratchet, docs governance, and diff checks are also clean.

Hosted Python 3.12 logs contain no flight-to-ground transfer tolerance failure.
The only numerical assertion is the separate shared wind fixture, whose
`9.786440272809793` result differs from `9.7864402728063` by `3.494e-12`
against a `1e-12` absolute tolerance. This branch does not change the wind
workflow. The hosted Rust `-lpython3.11` linker failure remains runner/toolchain
infrastructure.

## 2026-08-09 Ground-contract stack recovery

Protected quality-gate run `31341468033` on PR #4285 exact head
`2d9a06fae46e0601a05896b71934ca0c6b8dc59a` then reached pinned mypy 1.13
and found that skipped-import analysis models the Python 3.10 string-enum shim
as `str`. The scoped correction generates all schema enum values and target
frame constants through `str(...)`; deliberate invalid-input tests use typed
casts instead of stale suppressions. Wire values and fail-closed runtime
behavior are unchanged. The full 19-file changed Python delta passes the exact
mypy 1.13 flags, Ruff check/format passes, and the focused ground suite remains
`46 passed`. Treat the failed run only as old-head diagnostic evidence. Push a
new commit normally and propagate it into #4288 by normal merge before using
any child CI result as release evidence.

Draft PR #4285 remains based on `feat/4197-capability-observer`. A normal local
merge now carries exact parent head `9bbb98e16e435a0d4c74153b909f2ebfefbbce7a`
into `feat/4268-ground-contract` without retargeting or rewriting either
branch. The previous PR head had no reviews or unresolved threads and was
reported dirty only because the parent had advanced beyond its 2026-08-07
merge base.

The current-head test logs also proved a bounded ground defect: schema tests
imported `jsonschema` without declaring it, and the new enum modules bypassed
the repository's Python 3.10 compatibility boundary. The follow-up declares
`jsonschema>=4.23.0`, pins the locally verified 4.24.0 build, imports the shared
`StrEnum`, and adds a package-wide
regression test. RED named the three offending ground modules; GREEN is
`46 passed`, and the affected Rate+swing_sim suite is `1463 passed, 5 skipped`
with optional local Rust-wheel skips only. Focused Ruff check/format, targeted
mypy, documentation governance, and diff checks pass. The separate Rust
`-lpython3.11` linker failure is infrastructure. No GitHub write was made; PR
#4288 must receive this parent ancestry through a normal merge before further
flight-transfer publication.

## 2026-08-08 Capability workspace continuation

The active stacked child is `feat/4197-capability-optimization-ui`, based
exactly on evaluator commit `c280407d432c153639bb266c9c721a014a129723`
(draft PR #4289). It adds matched PyQt6/React Shot Optimizer modules with the
strict cross-runtime `capability-optimization-workflow/v1` document, qualified
Waterloo/Penner worker execution, progress/cancellation, complete retained
observation cohorts, ranked alternatives, selectable stage-qualified scalar
axes, managed zoom/autofit, accessible 25-row paging, spreadsheet-safe CSV,
and stable JSON. The captured basis includes profile/club IDs, delivery
center/spread, sourced fixed spin, positive-right target frames, objective,
budgets, alternatives count, and deterministic seed.

Live browser and standalone PyQt rendered review verified the workflows and
found three repaired integration defects: duplicated target-axis labels, old
saved layouts hiding newly registered modules, and a cramped PyQt results
split. All optimizer controls now have substantive hover guidance. Verified
local evidence is 808 Rate Python/PyQt tests plus 615 swing_sim tests and 102
React files / 619 tests; Ruff, formatting, CI-equivalent mypy 1.13,
TypeScript, zero-warning ESLint, the 187-module production build with a
lazy-loaded Shot Optimizer chunk, structural limits, and diff checks pass. The model boundary is visible: still-air carry to
first ground crossing only, with wind, bounce, roll, and total distance outside
v1. Publish as a protected child of #4289 and keep #4197 open through CI,
review, ordered merge, and downstream parity.

## 2026-08-08 Capability evaluator continuation

The active child branch is `feat/4197-capability-flight-evaluator`, based
exactly on capability-observation PR #4283 head
`49612946138b1021f80c9f8d2a4d06f1610825db`. It adds the first qualified
full-flight evaluator for #4197 in shared Python and the React model layer.
The factory binds `player-capability-profile/v1` plus
`capability-optimization-request/v1`; validates requested clubs, exact sample
fields, units, finite values, declared safe bounds, and physical domains; runs
the real Waterloo/Penner model; converts trajectory and spin into the canonical
target frame; binds the request target; and emits every available scalar
canonical metric. Existing three-variable profiles require a sourced spin
default for every requested club, while profiles may opt into paired variable
`total_spin` and `spin_axis_tilt`. Positive tilt is fade/right, matching the
existing Flight Explorer, glossary, D-plane, variation, and solver convention.

No-ground-crossing horizons are typed `nonconverged`; expected Python
floating-point overflow is typed `failed` without leaking exception text;
contract and programming errors surface; and this post-impact adapter cannot
report `no_impact`. Python uses SciPy RK45 and React uses fixed-step RK4, so
logical model/version and metric-set parity are exact while numeric parity is
banded through `capability_flight_evaluator_parity_v1.json` and integrator
provenance remains runtime-specific. Canonical result, impact-diagnostic, and
variation producers share one gyro-projected spin-axis tilt calculation.

Post-review full-suite evidence is `138 passed, 4 skipped` in Python and
`97` files / `597` React tests. Ruff, formatting, targeted mypy, TypeScript,
zero-warning ESLint, and the 176-module Vite build pass. The next required
slice is the end-user PyQt6/React capability workspace with
off-main-thread execution, progress/cancel, profile/target/environment editing,
observation scatter/table/CSV, persistence, and rendered QA. Keep #4197 open.

## Integration checkout

- Worktree: `C:\Users\diete\Repositories\Tools-worktrees\ballflight-campaign-integration`
- Branch: `codex/ballflight-campaign-integration`
- Draft PR: [#4217](https://github.com/D-sorganization/Tools/pull/4217)
- PR base ref: `feat/4181-launch-monitor-registry`
- Integration base: `626cfb64b0eddaa598a2a24dc2a050a420be25be`
- Synchronized base head: `4b659acc1f7fc183dff60daea2553009e82dbab9`
- Published PR head before the current continuation:
  `3f79eb8d15d8558ccf53b441e3842c50ce36e16e`
- Latest implementation commit before this documentation-only handoff update:
  `26fe5a7176eba51988a6a4cc4553f423c5c190ed`
- Pinned-mypy CI compatibility follow-up after exact-head log diagnosis:
  `8d54212e85f251ac812a4edb8f50bf6bff31cb61`
- Final target-frame literal correction from the subsequent exact-head CI run:
  `51bad9009ce929fe89d3a527ca0e6858795dbbb7`
- Launcher-themed wrapped-form correction reproduced from the user's live window:
  `d813d652fc76d90582a20928820d1aa306ab8a91`
- Published documentation continuation before the current audit:
  `280b58622bbfedb686777173fb3b22397d3495ee`
- Paired landing-row integrity fix in both clients:
  `d78d2b0ea3b5662f62c24c36d675371a6ef57704`
- Pinned-mypy variation typing correction exposed by exact-head CI:
  `ec70087e645fee4385e41d065582011fe47739ed`
- React manual-delivery inputs, pose, geometry, and schema-v5 persistence:
  `3eed7c4f6290dbd55f936636d6eb4bd043214e48`
- Python/PyQt manual-delivery inputs, pose, geometry, and schema-v5 persistence:
  `fb6f80d7d0f064a6ca9e7b54318aa138fb5af568`
- Cross-client machine-readable reference-impact boundary:
  `785a988662a8ca13410dfacd6802271ddbd27276`
- React v5 self-import and delivered-loft validation:
  `960bc158b247e5a815cd874bee8a6a23f6f78399`
- Native six-decimal manual-delivery persistence:
  `a11cea81a1b2beef1567dc92d01c914834fcbdca`
- Native source-specific plane-orientation gating:
  `8c0f5999d3ccad4aabb3cd1b2aa3a1785d23a702`
- Cross-client source gating, native/web v5 support, and required settings blocks:
  `b4737c60fcafef44d067a02bd03e67ae1b5135cb`
- React field-level v5 manual-delivery validation and settings-only import wording:
  `7e445ed52f27b4f694a3e74b320eee5e60a36268`
- Native/web v5 fail-closed persistence and atomic native import:
  `3255c01d29a9921361fadefab47649268c77c0a7`
- React field-level v5 ball-setup validation:
  `d12782393f9cacc495df9206c8956e13692adb7c`
- Visible PyQt factor gating and canonical workbench-club synchronization:
  `47d77156d15aba9f69179edebb7e35ec3b99416f`
- Native schema contract correction (accepted native versions 1, 2, and 5):
  `7ae1d2a076737ba03f30c5c97ddbed78fff21c6c`
- Optional-Rust backend documentation correction:
  `ed73e80b244fd4e3bf8d5921912bf3ff5474c14b`
- Compact PyQt manual-delivery and contact-policy labels:
  `fef649a898bbd458232290f2105d2c3e2e0879a4`
- Compact PyQt shaft-datum row label:
  `26fe5a7176eba51988a6a4cc4553f423c5c190ed`

## Included PR stack

The source heads were merged in dependency order. A later source head includes
the earlier commits from that PR.

| PR    | Capability                                                                       | Exact included source head                                                                        |
| ----- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| #4203 | Launch-monitor convention registry and fail-closed unknown signs                 | `3d899c8e95bc6808b07a1b230a21021d845c14ad`                                                        |
| #4209 | Launch Direction convention integration and visible unavailable Foresight option | `98589174273e90e6690a08201c369004c3f568b4` (merged by `4b659acc1f7fc183dff60daea2553009e82dbab9`) |
| #4210 | Canonical flight-result metric catalog                                           | `e6524dbb852e9356ae666dda5307cf0fd7e36960`                                                        |
| #4211 | Desired-flight inverse solver                                                    | `24d891cf78f5de125bb1fda602a7a9136b91f138`                                                        |
| #4215 | Impact solution families                                                         | `8e3af21672b105bcbc6f821644e013896d8293ba`                                                        |
| #4216 | Capability optimizer, including variability and downside/CVaR objectives         | `4e11182d7d72abe66fd1066ca2086c2a87df5323`                                                        |
| #4207 | Paired wind physics and responsive locked-aspect canvases                        | `d668de1f1f808f7d5c8a4c5314a3ca940d71a4b9`                                                        |
| #4213 | Wind-estimate uncertainty analysis and v2 risk metrics                           | `15cc7ac5b32924f69175d85ee0bc71b736f6e856`                                                        |
| #4214 | Interactive 3D playback, correct Launch/Apex/Landing events, responsive canvas   | `a7d337155cbd74c8198d9ef7f21add1b5d52b013`                                                        |
| #4208 | Versioned 3D spatial-target contract                                             | `9aec34d89f91c08bf0882c556b66242d00cf3ba6`                                                        |
| #4212 | PyQt/React Launch Monitor Analytics and split statistics modules                 | `a4dcddde6122bb298c7c20d3353d45e74481ba2a` (merged by `8526f7e0ea7b08f7bd48423bf2416b2a822daf56`) |

Integration-only reconciliation commits are
`16395378ec81c6b4c623804fc65ed886ea1bde7a` (formatting),
`107d8e43246d1ca545be1cb8980622f7a208a895` (Flight Explorer split),
`91a0bba09f5fba560744d9be840787dad500b2cf` (strict typing), and
`18fe8768fe27cc21d2d987a426e1a01fda3f5303` (spec reconciliation).

The `wind-strategy-analysis/v2` result distinguishes actual estimate-driven
outcomes, the same declared policy evaluated with true-wind information, and
the hindsight best result among only the declared presets. Its summaries add
failure-inclusive target-circle hold probability, empirical miss-distance
CVaR at a declared alpha, and short/long/left/right probabilities with
unconditional and conditional mean excess. Legacy regret/best aliases remain,
but the precise names are preset-oracle regret/probability; the signed
information-cost delta is not presented as EVPI.

## Launch and registration

Run both commands from the worktree root in separate PowerShell terminals:

```powershell
python src/rate_of_closure/launch_pyqt6.py
cd src/rate_of_closure/web
npm run dev -- --host 127.0.0.1 --port 5270 --strictPort
```

The web app is then at `http://127.0.0.1:5270/`. Its authoritative Vite
package is `src/rate_of_closure/web`. The React navigation ID
`launch-monitor-analytics` is declared in
`web/src/model/viewPreferences.ts`, rendered by `web/src/App.tsx`, and backed
by `web/src/components/LaunchMonitorAnalyticsPanel.tsx`. The PyQt stable tab ID
`launch_monitor_analytics` is registered in `ui/pyqt6/main_window.py` and
backed by `ui/pyqt6/launch_monitor_analytics_tab.py`.

## Verification evidence

### Spatial-target and compact-layout continuation

The current continuation closes the user-visible spatial-target workflow and
the concrete 1280 x 768 PyQt Simulation defects captured in issue #4235.

- PyQt6 and React now share one canonical target across Flight Explorer and
  integrated Simulation, including app/flight-frame editing, landing/aerial
  kinds, circle/corridor/sphere/box tolerances, visible validation, and
  side/top/3D rendering before and after a run.
- Versioned run/project JSON, CSV metadata, solver manifests, and variation
  manifests carry the exact target. Imports migrate legacy documents, reject
  incomplete version-4 documents atomically, and neutralize spreadsheet
  formula prefixes in CSV text fields.
- Aerial target passage is evaluated continuously between retained trajectory
  samples with an interpolated event time. Landing assessment projects the
  ball center onto the course surface. Ground-only solver/variation requests
  explicitly reject aerial targets and stale solver results cannot be applied.
- The PyQt Swing view keeps key impact metrics visible while placing layer and
  engineering-detail controls in collapsible panels. Legends default beside
  the data and can be moved inside or hidden. Shared height-for-width group
  boxes reserve the real height of wrapped forms, so Ball Setup, Spatial
  Target, and global scenario fields do not collapse in narrow scroll rails.
- The optional `swing_core` accelerator no longer prints a crash-like warning
  during a normal auto-backend launch. Auto mode visibly remains operational
  through the Python integrator; explicit Rust requests continue to fail
  closed with actionable installation guidance.

Current exact local evidence after these changes:

- Complete Rate of Closure Python/PyQt suite after the responsive-group and
  quiet optional-accelerator fixes: `630 passed`, with two known non-failing
  warnings (Hypothesis collection configuration and an empty preview legend).
- The complete `630`-test suite was repeated after correcting themed group-box
  chrome accounting. At 1296 x 759 and 125% scaling, Ball Setup reserves its
  full 227 px height-for-width and clears Contact Policy by 7 px; every nested
  row remains contained.
- Complete React suite: `78` files and `475` tests passed.
- React TypeScript type-check, zero-warning ESLint, and the 153-module Vite
  production build passed.
- Ruff check/format passed across the affected Python domain; clean-cache
  pinned mypy 1.13 passed on `64` changed production files and local mypy
  passed on the corrected target editor. The final focused target GUI suite
  passed all `25` tests.
- Changed-only 500-LOC and module-size budgets passed; `git diff --check`
  passed. New production modules remain below 400 lines.
- Compact/full-window tests passed at 1269 x 731 and 1280 x 768, plus the
  1024 x 700 window floor and an explicit 125% Qt scale factor.
- Live screenshots:
  `C:\Users\diete\AppData\Local\Temp\rate-of-closure-themed-layout-fixed.png`
  and the browser-controlled React app at `http://127.0.0.1:5270/`.

- Full pre-v2 Python campaign suite: `740 passed, 4 skipped, 15 warnings`.
- Post-v2 wind-uncertainty plus flight/solver contract tests: `25 passed`.
- React/Vitest suite: `70` files and `439` tests passed.
- Post-v2 targeted React wind-uncertainty suite: `11 passed`.
- React production build: `tsc && vite build` passed (147 modules).
- React `type-check` and ESLint passed.
- Production Python mypy: no issues in 60 changed source files.
- Ruff and Black: 79 changed Python files passed.
- Module-size budget and `git diff --check` passed. The Flight Explorer and
  launch-monitor analytics production modules are each below 400 lines.
- The four skips are Rust parity cases because a compatible `tools_core` wheel
  is not installed. Other warnings are the existing Hypothesis pytest-plugin,
  Matplotlib legend, and Node local-storage-path warnings.
- A repository-root `npm run build` is not a valid campaign gate in this
  checkout: unrelated workspaces lack `turbo`, `next`, and other dependencies.
  The authoritative Rate of Closure package build above passes.

### Variation ensemble continuation

Issue [#4144](https://github.com/D-sorganization/Tools/issues/4144) and draft
PR [#4167](https://github.com/D-sorganization/Tools/pull/4167) own the universal
multi-trial visualization contract. The integration branch includes that work
through the investigation-suite ancestry.

- Focused Python variation suite: `120 passed` across the shared engine,
  simulation adapter, PyQt controls, complete results workspace, plots,
  linked selection, exports, and cross-runtime fixture.
- Focused React variation suite: `21 passed` across six files, including the
  every-trial arc inspector and geometry performance contract.
- Live integrated React QA at `http://127.0.0.1:5270/` ran a 200-trial
  Delivery/Impact/Flight study and a 24-trial Pendulum/Impact/Flight study.
- The pendulum run rendered `24/24` swing arcs, `36,024/36,024` vertices,
  `33/1501` quiet samples at the declared 5 mm RMS threshold, linked trial
  selection, impact/flight scatter variables, a four-variable matrix with
  marginals, sensitivity results, and `24` honest landing coordinates.
- The arc inspector exposes modeled point, outcome cohort, perturbation source,
  source quantile, phase, linked highlighted trial, reset, PNG, variability SVG,
  and versioned plot-definition export controls. Frame and alignment are shown
  as `app_frame:x_target,y_up,z_right` and common simulation time.
- The default scalar delivery study correctly reports that no geometric
  no-impact cohort exists; the pendulum result carries typed hit/no-impact/
  numerical-failure cohorts without fabricated impact or landing coordinates.
- The continuation audit found and corrected one cross-client missing-data
  defect: carry and lateral values were previously filtered independently, so
  complementary missing values in different trials could be combined into a
  fictitious landing. The shared Python dataset now exposes paired finite-row
  selection, the Python and TypeScript ellipse fits consume those exact rows,
  and both canvases report the exact number of points they draw.
- Post-fix focused verification passed `21` Python engine/PyQt/registration
  tests and `16` React analysis/component tests. Python Ruff check/format and
  mypy passed; React TypeScript, zero-warning ESLint, and the 153-module
  production build passed. The complete React suite independently passed
  `79` files and `477` tests.
- The complete Rate/PyQt suite plus shared variation and wedge-kinematics
  contracts passed `743` tests after the paired-row and generated-head
  cross-check additions; only the existing Hypothesis configuration and empty
  polynomial-preview legend warnings remain.

### Wedge AoA worked example continuation

Commit `cfcc99681` expands
`docs/specs/GOLF_CLUB_WEDGE_KINEMATICS.md` and pins its numeric claims in tests.
The declared 64-degree lie, 15-degree lean, **synthetic** 20 mm offset,
1,307 deg/s shaft rate, and 30 mph state decomposes as follows:

- shaft-datum translation vertical speed: `-2.135647 m/s` (`91.7047%`);
- shaft-axis rotation vertical speed: `-0.193183 m/s` (`8.2953%`);
- total AoA: `-10.0000 deg`;
- no-shaft counterfactual AoA: `-9.18117 deg`;
- direct shaft contribution: `-0.81882 deg`.

That fixture proves the kernel; it is not the generated head geometry. A
separate pinned cross-check uses the Rate `Pitching Wedge` face center and
hosel. With the same lie, lean, rate, total 30 mph contact speed, and -10-degree
AoA, it gives shaft-induced velocity
`(+0.497660, -0.164057, -0.060817) m/s`, 7.0446% of downward speed, and a
`-0.33406 deg` counterfactual AoA contribution.

The manual Simulation in both clients now accepts signed reference AoA/path,
targetward-positive forward shaft lean, and tracked-reference versus registered
generated-hosel shaft datum. The authored hosel is correctly registered through
the authored face center and scenario face-distance datum. With the Pitching
Wedge, 30 mph reference speed, -10-degree reference AoA, zero path, 15-degree
lean, 64-degree lie, an explicit 20 mm reference-to-face override, zero
swing-plane angular rate, 1,307 deg/s about the shaft, centered offsets,
450 microseconds contact, Ground support, Delivery Inspection at `t = 0.030 s`,
and `waterloo_penner` flight, the configured app reports -10.847087-degree
contact AoA, -0.298815-degree shaft contribution, 6.5050% downward-speed share,
and 22.45855 m (24.56 yd) carry. The club-library Pitching Wedge default is
11 mm, so the 20 mm value is a declared sensitivity-case override. Entering
-9.153512-degree reference AoA targets exactly -10-degree contact AoA and gives
-0.333108-degree shaft contribution and 23.024061 m (25.18 yd) carry.

Native and web run schemas emit version 5 with canonical nested
`manual_delivery` fields, explicit legacy migration, atomic import, and
machine-readable contact/impact limitations. Native import accepts only the
versions it historically emitted (`1`, `2`, and `5`); versions `3` and `4` are
rejected because they were web-only and never defined a native document. Web
import accepts its historically emitted versions `1` through `5`. Current
native/web v5 imports fail closed when the canonical spatial-target,
ball-setup, or manual-delivery blocks or required fields are missing. The
import command is deliberately labeled
**Import Settings JSON**: it restores only ball setup, spatial target, and
manual delivery, not the source, club/scenario, contact mode, flight model, or
every other exported run input. It is therefore not yet a full deterministic
run replay surface. Current contact detection tracks the reference point and
rigid impact/flight uses its translation; shaft-induced contact velocity is not
yet fed into ballflight. Articulated sources still lack torsional shaft motion.

Both clients disable and explain swing-plane orientation while Manual is
active, because manual attack angle and path own the reference direction. PyQt
also synchronizes the Simulation club with the canonical workbench club spec,
so the visible club, loft/curvature overrides, lie, and reference-to-face datum
are the values consumed by the run.

Final local executable-head evidence at `fef649a898bbd458232290f2105d2c3e2e0879a4`:
the complete scoped Python/PyQt/shared suite passed `972` tests with `3`
expected skips and `15` warnings. The skips
are the Rust parity case when `swing_core` is absent and the wedge CAD/export
cases when `build123d` is absent; the warnings are `14` existing Hypothesis
collection notices and one Matplotlib empty-legend notice. Ruff check and
format passed across all `18` changed production Python files, and pinned mypy
reported no issues. The complete React suite passed `83` files and `521`
tests; TypeScript, zero-warning ESLint, and the Vite production build all
passed (`157` modules transformed). Three non-failing Vitest-worker
`--localstorage-file` warnings are environmental: no matching option exists in
the Rate web package or repository workflow configuration, and the live browser
reported no warnings or errors. The later Rust-fallback docstring and compact
PyQt label changes do not alter computation. After the final row-label change
at `26fe5a7176eba51988a6a4cc4553f423c5c190ed`, the label-focused PyQt suite
passed all `4` tests with Ruff, formatting, and `git diff --check` clean.

The source boundary is explicit: 1,307 deg/s is Cheetham's mean for 94 tour
**driver** swings, not a claimed wedge norm. The documented sensitivity study
pins 0, 652, 1,003, 1,307, 1,611, and 2,432 deg/s. The current impact and calm
Waterloo-Penner flight chain predicts only `17.566 m` (`19.211 yd`) carry for a
30 mph, -10-degree AoA, 37-degree dynamic-loft case; the same model needs
approximately `37.887 mph` club speed to reach 30 yd. Focused wedge/flight
verification: `31 passed`; the broader post-format regression: `59 passed`.

### Current CI diagnosis

Exact-head run `31180951147` on commit `ef7c5f45e` passed Ruff and format, then
failed pinned mypy 1.13 in `variation/analysis.py`: NumPy percentile tuple
unpacking and an unannotated rank buffer were not inferable under the pinned
stubs. Commit `ec70087e6` normalizes the percentile result to a typed array and
annotates the rank buffer without changing runtime behavior. Mypy 1.13 on
Python 3.12 now passes the corrected module. A new exact-head CI run is required
after the manual-delivery continuation is published.

At the previous published head, PR-triggered CI run `31134083167` failed its
quality gate because Ruff 0.14.10 would reformat two files. The independently
dispatched run `31134149702` passed its quality-gate job, but that dispatch used
a narrower changed-file scope and is not replacement evidence. Commit
`282b1a4d3` applies only the two reported formatter changes. A local
PR-merge-base-equivalent gate then reported `77 files already formatted`, Ruff
clean, `59 passed`, and a clean diff. New protected checks must run on the
published continuation head; queued work is not counted as passing.

The next exact-head PR run `31135497996` confirmed the formatting fix and then
exposed CI's pinned mypy 1.13 compatibility errors across six files. Commit
`1bc7f567c` resolves those errors with typed NumPy/Qt scalar boundaries,
literal narrowing for imported target kinds and analytics selections, and
distinct correlation/coefficient variables; it does not add blanket ignores.
The PR-equivalent 58-source-file set now passes both mypy 1.13 and the local
mypy 1.15, Ruff reports `77 files already formatted`, and `189` affected-domain
tests pass. Protected CI still needs to complete on the newly published head.

### Base synchronization and file-size recovery

The PR base advanced normally through #4212 merge
`8526f7e0ea7b08f7bd48423bf2416b2a822daf56` and #4209 merge
`4b659acc1f7fc183dff60daea2553009e82dbab9`. Local merge commit
`778be95a682998b7b2f71b3d68aa60b8c6f46891` synchronizes that exact base into
the child without rebasing, retargeting, or rewriting either parent.

The merge had one conflict in `flight_explorer_tab.py`: the child had already
split the shared speed-unit table into `flight_explorer_controls.py`, while the
parent still referenced its former local constant. Resolution retains the
child's extracted canonical table and typed Qt scalar locals, together with the
parent's Launch Direction/analytics contracts. The analytics handoff and its
expanded TypeScript parity test merged without conflict.

Failed File Size Budget run `31136702822`, job `92737550769`, reported three
files against the old base: `simulation_tab.py` at 774 LOC,
`plotting/catalog.py` at 533 LOC, and `main_window.py` at 521 LOC. After the
normal parent merges, the exact changed-only gate proved that the latter two
were base-owned and left only `simulation_tab.py` as a child violation. Commit
`50089b66a3eca3220d157dded040cc74d02c729a` separates controls and runtime
behavior without changing the public `SimulationTab` API. Final formatted
sizes are 402, 218, and 272 LOC respectively.

Exact post-sync evidence against
`origin/feat/4181-launch-monitor-registry@4b659acc1`:

- CI-equivalent changed-only 500-LOC check: 55 files scanned, zero violations.
- Repository module-size budget and `git diff --check`: passed.
- Mypy 1.13.0: 44 changed production files passed.
- Ruff 0.14.10 check/format: 59 changed Python files passed and already formatted.
- High-risk PyQt simulation/navigation suite: 135 passed.
- Shared flight/solver plus flight, playback, analytics, and help suite:
  230 passed, four expected Rust parity skips.
- Complete React suite: 70 files and 445 tests passed; TypeScript type-check,
  zero-warning ESLint, and the 147-module production build passed.

### Rendered design and error-state audit

Epic [#4234](https://github.com/D-sorganization/Tools/issues/4234) and child
issues #4235-#4239 capture a read-only computer-controlled review of the live
React application and standalone PyQt6 window. The epic is sequenced after the
current campaign and #4218, and consumes #4224/#4225 rather than duplicating
their plot and view-compositor contracts.

Confirmed React findings include a 1,091 px tab rail at a 390 px viewport,
30-35 px controls, non-semantic Details affordances, a single selected plot
canvas with fixed legends, silent 0 mph to 0.1 mph coercion, and acceptance of
-1 mph without visible or accessible validation while stale prior results
remain visible. Negative spin-axis input itself is confirmed working: -10 deg
produced -17.3 yd lateral, and the double-pendulum articulated skeleton rendered.

The reported 1280 x 768 PyQt Simulation defects are now corrected: the control
rails scroll vertically without horizontal overflow, wrapped forms reserve
readable editor heights, layer labels and engineering details collapse into
discoverable panels, key metrics remain visible, and the legend can be placed
outside, moved inside, or hidden. Native Flight continues to show side,
top-down, and 3D trajectories together. Automated full-window coverage now
includes 1024 x 700, 1269 x 731, 1280 x 768, and 125% Qt scaling. A broader
150%/200% platform matrix, keyboard traversal audit, and stable pixel-baseline
suite remain owned by #4235/#4239.

### 2026-08-07 toolstrip, plot-workspace, and parity continuation

The `feat/4218-toolstrip-workspace` continuation is published as
[draft PR #4279](https://github.com/D-sorganization/Tools/pull/4279) against
`feat/4181-launch-monitor-registry`, the current stacked base after PR #4217
was squash-merged there. It adds one
UI-neutral registry for 17 File/View/Tools commands, a strict versioned
workspace document with atomic file persistence, matched PyQt/React top
toolstrips, persistent module visibility/order, theme and shortcut surfaces,
and direct Impact/Swing/Flight navigation. File actions that do not yet have a
complete client adapter remain visibly disabled with a reason rather than
pretending to save incomplete state.

The same continuation corrects the interaction defects reported against the
live Swing and Plots views. Playback now has deterministic replay-from-end,
Restart, granular 0.05x through 4.00x speed, pause, and loop behavior. The
full swing path is opt-in so a persistent trail does not obscure the current
frame. Each managed plot now owns a distinct figure/canvas, zoom state,
Auto Fit action, wheel zoom, and independently movable or hideable legend;
the plot workspace presents all managed plots instead of reusing one selected
canvas. PyQt small-window testing caught the new playback editor compressing
below the 64 px readability floor; the explicit editor minimum fixes that case
and the three-case layout suite passes.

Two read-only cross-repository audits are now tracked as separate programs:

- [#4260](https://github.com/D-sorganization/Tools/issues/4260), with
  #4261-#4266, establishes one impact/flight authority and a machine-readable
  parity contract across Tools PyQt, Tools React, UpstreamDrift PyQt, and
  UpstreamDrift React.
- [#4267](https://github.com/D-sorganization/Tools/issues/4267), with
  #4268-#4276, defines qualified landing, bounce, skid, roll, and total-distance
  modeling with editable ground profiles and exact UpstreamDrift adapters.

The parity audit found that UpstreamDrift PyQt reuses Tools, while the
UpstreamDrift React launcher has no native Rate React route. UpstreamDrift's
Tools gitlink `ff4240217005e1415ca409fd124e50b64ee642d2` also predates the
current integration head by 184 commits, and its sibling/vendor resolution is
ambiguous. The ground audit found a useful existing fail-closed
`GroundModelResult` boundary plus reusable putting/terrain primitives, but no
qualified end-to-end ground solver. Before bounce can be correct, airborne
flight must terminate against physical terrain plus ball radius and preserve
the full terminal angular-velocity vector; the current relative launch-plane
event and spin-free trajectory state do neither. Those prerequisites are
explicit in #4269 and must not be hidden by UI-derived estimates.

The final local verification pass is green. The complete Rate-of-Closure and
shared swing-model run passed 890 tests with one expected skip because the
optional `swing_core` Rust wheel is not installed; the remaining 15 warnings
are the existing Hypothesis collection warning. React passed 89 files / 545
tests, zero-warning ESLint, TypeScript checking, and the production Vite build.
Ruff, Black, targeted mypy, `git diff --check`, and the repository structural
limits also pass: every changed production Python file is at most 400 lines and
every changed production Python function is at most 50 lines. Rendered PyQt
inspection confirmed independent plot canvases, responsive single-column
reflow at the tested desktop width, independent 125%/100% zoom state, working
Auto Fit, and the opt-in trail/playback controls. These are local validation
results only; they do not establish protected CI, review, merge, or release
status.

### 2026-08-07 variation export and completion audit continuation

The post-toolstrip branch `feat/4144-variation-export-continuation` is published
as [draft PR #4280](https://github.com/D-sorganization/Tools/pull/4280), based
on exact parent head `c36ca36e91f34fa849d2508708bf9dd6c0cdc392`. It keeps #4279 unchanged
while closing one remaining #4144 parity gap: selected scalar scatter data can
now be exported as CSV from both clients, retaining every raw trial, typed
outcome, and unavailable cell rather than only the finite points drawn on the
canvas. PyQt also has a bounded read-only raw-trial table matching the web
workflow, and the table population is shared with the matrix view.

The complete post-change local gates passed:

- Python/PyQt/shared swing suite: `890 passed, 1 skipped, 15 warnings`; the
  skip is the optional `swing_core` wheel and the warnings are the existing
  Hypothesis collection and empty polynomial-preview legend warnings.
- React: `89` files / `545` tests passed.
- Ruff check/format, Black, targeted mypy, TypeScript, zero-warning ESLint,
  the `166`-module Vite production build, and `git diff --check` passed.
- Every changed production file is below 400 lines and every changed
  production function is at most 50 lines.

A live GitHub/source reconciliation covered every requested epic in this
campaign. No epic yet satisfies its own definition of done: most implementation
is still on feature branches, #4119 is the only Rate platform PR targeting
`main` and is currently dirty, #4203 and #4279 remain draft/unstable, and only
formal club-builder child #4147 is closed. The variation request is
substantively implemented, but #4142/#4144 remain open because bounded
large-ensemble execution, nonlinear global sensitivity, localized execution,
the immutable UpstreamDrift consumer pin, protected CI, and default-branch
release are incomplete.

The literal universal-runner audit also found two uncovered many-evaluation
paths. Wind strategy analysis retains all paired outcomes but has no user
workflow or universal plot adapter; capability optimization retains aggregates
but not individual sample rows. The next safe model slice is a UI-neutral
scalar-ensemble contract with unique composite row IDs, unit-bearing variable
metadata, caller-defined cohorts, paired-finite scatter extraction, and exact
availability accounting. Wind integration must accept both its immutable
request and analysis so launch definitions and provenance are not inferred.
Issue #4199 already owns the required controls, scatter, strategy table,
progress/cancellation, and export workflow.

The first narrow #4199 implementation slice is published as
[draft PR #4281](https://github.com/D-sorganization/Tools/pull/4281) from branch
`feat/4199-wind-scalar-adapter`, stacked on exact PR #4280 head
`d71b0ea01b5659d3049ff05627c41f06481207e4`. Implementation commit
`4a28114aa` introduces an exact
cross-runtime `scalar-ensemble/v1` wire contract and pure wind-strategy
adapters. The contract preserves structured provenance, unit-bearing variable
definitions, caller-defined cohorts, RFC3986 composite identities, nullable
raw rows, and exact scatter availability. The adapters validate the immutable
request against the stored paired analysis, preserve completed,
nonconverged, and invalid outcomes, and never invoke a flight model. React has
an explicit mocked-integrator regression test for that boundary.

Current exact local evidence is 906 Python/PyQt/shared-swing tests passed with
one expected optional-Rust skip and 15 existing warnings, plus 91 React test
files / 555 tests passed. Ruff, formatting, Black, focused mypy, TypeScript,
zero-warning ESLint, the 166-module production build, `git diff --check`, and
the production module/function budgets pass. The adapter is plot-ready model
infrastructure, not an end-user workflow; #4199 remains open for worker,
progress/cancellation, client controls, strategy/scatter displays,
persistence, and exports.

### 2026-08-07 ground and four-surface audit refinement

The rolling-ground and cross-application parity requests remain tracked by the
existing [ground epic #4267](https://github.com/D-sorganization/Tools/issues/4267)
and [parity epic #4260](https://github.com/D-sorganization/Tools/issues/4260);
no duplicate epic or child issue is required. The latest exact-path audit and
acceptance refinements are attached to
[the ground epic](https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5222725556)
and [the parity epic](https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5222726010).

The scientific implementation order is contractual: #4268 defines the
surface/contact/trajectory/result transfer state, then #4269 corrects physical
terrain contact and preserves terminal full angular velocity. Only then may
#4270/#4271 qualify the 3D impulse, repeated bounce, skid, and pure-roll
phases. Carry remains first physical contact. Final downrange, final lateral,
horizontal displacement, surface path length, and launch-monitor-style total
distance are distinct quantities; no implementation may silently assume
`total distance = carry + roll distance`.

Reusable UpstreamDrift scope is deliberately narrow: its split terrain
material/elevation/normal/region package can feed a one-way versioned DTO
adapter. Current scalar landing, heuristic putting-roll, duplicate legacy
terrain, and Rust tangential-loss implementations are reference material, not
the qualified physics authority. Upstream surface defaults remain illustrative
until citations, calibration, uncertainty, and applicability are recorded.

The parity matrix must distinguish seven product identities: standalone Rate
PyQt6 and React, the Upstream Rate PyQt provider and React route, Upstream Shot
Tracer PyQt6 and React, and the legacy Upstream ball-flight GUI. Current
Upstream `main` (`0782853295e005af68818617e4725eb980890f43`) pins Tools at
`ff4240217005e1415ca409fd124e50b64ee642d2`, exposes no native Rate React route,
and contains contradictory vendor-first and sibling-first Tools resolvers.
These facts are current audit evidence, not completion; #4260, #4267, and all
children remain open.

## Open release blockers

GitHub issue #4201 remains open. Its 2026-08-06 release checkpoint still
requires all of the following before any production-ready or merge claim:

- protected CI and required reviews for the combined stack;
- complete PyQt/React end-user workflows for desired-flight solving, solution
  families, capability profiles, and wind uncertainty, plus native aerial
  target objectives in the currently ground-only solver/variation paths;
- off-main-thread wind-ensemble execution with progress and cancellation;
- complete save/load/export integration;
- Rust/WASM trajectory parity and installed-package/UpstreamDrift pin checks;
- scientific validation, convergence, performance, and benchmark evidence;
- browser resize, high-DPI, keyboard, accessibility, reduced-motion, and visual
  regression coverage.

The metric catalog, inverse solver, solution families, capability optimizer,
and wind-uncertainty work must therefore be described as tested contracts/cores
unless and until their missing UI workflows are delivered. Spatial-target
editing, rendering, and persistence are end-user workflows; aerial optimization
remains an explicit fail-closed boundary.

## Next safe steps

1. Publish this child continuation only through a normal push after review,
   then require protected checks on that exact head; do not retarget,
   force-push, admin-merge, or bypass protected checks.
2. Keep epic #4218 and children #4219-#4225 sequenced after this
   ball-flight/variation/wedge campaign reaches its declared completion gate.
   The top-toolstrip/persistence work must not be used to hide #4217 release
   blockers or intermixed with this recovery diff.
3. After #4218, implement design-quality epic #4234 and children #4235-#4239.
   Preserve its confirmed rendered findings, explicit DPI gap, Current
   Calculation context, no-silent-coercion rule, accessibility contract, and
   cross-interface visual-regression requirements.
4. Add the missing UI workflows against the canonical shared Python/TypeScript
   contracts, with one visible-control-to-state integration test per control.
5. Add cancellation/progress, persistence/export migrations, Rust/WASM golden
   parity, performance budgets, and Playwright visual/accessibility coverage.
6. Verify a clean installed package and the exact UpstreamDrift dependency pin.
7. Rerun every recorded gate, inspect protected GitHub checks/reviews, and keep
   #4201 open until every acceptance criterion has current evidence.

## 2026-08-07 responsive wind workflow checkpoint

Branch `feat/4199-wind-workflow` is published as
[draft PR #4282](https://github.com/D-sorganization/Tools/pull/4282) at exact
implementation head `fdcc25008`. It is stacked on exact draft PR #4281 head
`8b8690e8760d82ba814e8d95588d2540d28a6759`; do not extend, retarget, rewrite,
or merge ahead of #4281.

The slice delivers matched PyQt6 and React current-launch wind-strategy
workflows on the shared `wind-strategy-analysis/v2` and
`scalar-ensemble/v1` authorities. It adds off-GUI-thread/off-main-thread
execution, exact progress, cancellation and teardown, canonical target reuse,
all-variable cohort-aware scatter, null-preserving generic CSV, explicit
availability, captured calculation basis, and stale-result invalidation. The
managed plot controls reset toolbar history and expose Auto Fit, zoom, and
legend placement. React data marks are clipped to the plot region and the
axes have numeric ticks/gridlines. Its workspace is genuinely code-split,
not hidden behind a raised bundle-warning threshold.

Native-window QA at 1280 x 768 found and closed two late usability gaps. Ball
flight now has an accessible Loop control in both clients and wraps without
creating a second timer/animation frame. The PyQt wind panel now uses compact
two-column Setup and plot-first Results views, switches to Results after a
successful run, and leaves run/cancel/export and progress/status continuously
available. A live five-trial run completed 5/5 with the captured basis,
summary, scatter, native pan/zoom, Auto Fit, and legend placement visible.
The in-app browser connection refused localhost navigation under its URL
policy, so React visual evidence remains the full component suite and
production build rather than a claimed live-browser pass.

Current primary validation is:

- Python/PyQt/shared swing: `1350 passed, 5 skipped, 15 warnings`;
- React: `94` files / `566` tests, plus focused playback and wind passes;
- Rust swing core: `12 passed`;
- Ruff, Black, focused mypy, TypeScript, zero-warning ESLint, production Vite
  build, structural line/function budgets, and `git diff --check`: passed.

The five Python skips are the absent optional `swing_core` and `tools_core`
wheel fast paths, not failures. The two warning classes are established
Hypothesis collection configuration and the empty polynomial preview legend.
Hosted CI, required review, mergeability, and exact deployed/default-branch
state remain unproven until the new child PR is published and protected checks
finish.

The independent rolling-ground audit refined epic #4267 at
<https://github.com/D-sorganization/Tools/issues/4267#issuecomment-5223106106>.
It defines carry, final coordinates, launch-monitor total displacement, and
bounce/skid/roll/ground path lengths separately; requires full angular state
and arbitrary-normal physical contact; and restricts UpstreamDrift terrain
reuse to a one-way versioned adapter. The four-surface audit refined #4260 at
<https://github.com/D-sorganization/Tools/issues/4260#issuecomment-5223106465>:
CI must prove the complete capability by `tools.pyqt6`, `tools.react`,
`upstreamdrift.pyqt6`, and `upstreamdrift.react` Cartesian product with
commit-fresh evidence. A launcher/native-window handoff is not parity.

The next universal-ensemble slice is the capability optimizer. Its exact
streaming observation/cancellation/scalar-adapter contract is recorded at
<https://github.com/D-sorganization/Tools/issues/4197#issuecomment-5223170071>.
Keep the ordinary optimization result compact, stream every attempted sample
in deterministic order, preserve evaluator metrics and reasons, and never
invent outputs for no-impact or failed rows.

### 2026-08-07 protected-CI repair and ground/parity audit

PR #4282 initially failed the hosted Python 3.12 delta mypy gate because the
wind lifecycle mixin and `QWidget` exposed incompatible `closeEvent`
signatures. Commit `424b4c395370aea26069386c070a65f7abe885bc` moves the Qt
override onto a concrete `WindStrategyGroupBox` and leaves the reusable mixin
responsible only for cancellation/join behavior. Fresh Python 3.12 mypy
passes for all 11 changed source files; Ruff, format, diff validation, and the
19 focused wind-panel/worker/playback tests also pass. This is a scoped CI
repair, not evidence that the still-queued protected stack is merge-ready.

The current remote UpstreamDrift audit basis is `main` at
`0782853295e005af68818617e4725eb980890f43`. Reusable ground assets exist in
its Rust contact kernel, split terrain/material package, compressible-turf
helpers, and putting roll engine, but none is a qualified drop-in. Material
round trips lose seven physical fields, the elevation-grid boundary contract
has two failing cases, terminal flight spin is not exported as a full vector,
and the Rust contact result uses scalar spin and a per-unit-mass energy value
labelled as joules. Tools must own a strict, versioned target-frame
flight-to-ground request/result authority; UpstreamDrift may contribute only a
one-way explicit adapter.

The parity matrix remains materially incomplete. Tools PyQt is the broadest
native surface; Tools React still has reduced impact/flight model authority;
UpstreamDrift PyQt is an external launcher; and UpstreamDrift React has no Rate
of Closure route. A separate generic simulator, copied TypeScript physics,
or launcher tile does not satisfy parity. Required next evidence is a
commit-fresh capability-by-surface manifest backed by shared golden fixtures,
one authoritative Tools physics contract, thin UI adapters, and an immutable
UpstreamDrift Tools pin.

### 2026-08-07 capability-observation continuation

Active branch `feat/4197-capability-observer` is based exactly on PR #4282
head `6e3c1029f1f3a80ae09020ef7d0afacb3c0d5484`. It must remain a normal
stacked child of `feat/4199-wind-workflow`; do not retarget, rewrite, or merge
it ahead of that parent.

The branch is published as
[draft PR #4283](https://github.com/D-sorganization/Tools/pull/4283). Its
validated implementation/hardening head is
`5c6073bd68ed4c8f23b343d4d11c2dc4277ea246`; this handoff-only continuation
will advance that head without changing the tested runtime behavior.

The optimizer now accepts optional synchronous observation and cooperative
cancellation hooks without retaining traces in `OptimizationResult`. Every
attempt emits one immutable `capability-sample-observation/v1` record in exact
candidate/club/sample order. Python and TypeScript normalize evaluator
exceptions, malformed results, no-impact, nonconvergence, and missing landing
metrics identically, preserve all valid evaluator metrics and provenance, and
never expose raw exception text. Cancellation is checked before the next
evaluator call and reports exact attempted/total counts.

The app-layer adapters convert streamed observations into the shared
`scalar-ensemble/v1` authority. They declare the complete scalar flight
catalog, preserve unavailable outputs as null, include nominal and perturbed
parameters plus target diagnostics, require a contiguous zero-based prefix,
and reject overflow before retaining a row. TypeScript deep-parses and
freezes caller input before storage. Stable JSON ordering is Unicode
code-point based in both runtimes; ASCII and Unicode parity fixtures hash to
`df36f765afdf508d00a3d264911ce5b6f07e25da3744b187596d67487ea3be5f`
and `18086b5e97d576598bbfa63407b6eda786a3a7ce20509654de282400bd32efd0`.

Current local evidence on this branch is 120 Python flight/adapter tests
passed with four expected optional `tools_core` skips, and 96 React files / 580
tests passed. Python 3.12 mypy, Ruff, Black, TypeScript, zero-warning ESLint,
the Vite production build, structural budgets, and `git diff --check` pass.
This completes the stream/adapter contract slice of #4197, not its remaining
end-user optimization workflow or the wider release epic.

Independent pre-publication review then found four fail-closed contract gaps,
all corrected before opening a PR: native Python/JavaScript number formatting
was not byte-stable at IEEE rounding and exponent edges; Unicode title-casing
could derive different labels; public observations admitted impossible
status/metric combinations; and the TypeScript declaration signature could
collide when identifiers contained its delimiters. The replacement canonical
writer emits code-point-sorted JSON with raw numeric tokens, fixed 11-decimal
half-away rounding, decimal integer-valued magnitudes, and normalized negative
zero. ASCII-only initial-letter label casing, strict landing/incomplete metric
invariants, and structural declaration comparison now match in both runtimes.

Adversarial regression coverage includes binary half boundaries, `1e-12`,
`1e-11`, large integer-valued magnitudes, negative zero, Unicode identifiers,
delimiter-bearing declarations, non-finite inputs, and every effective/source
status combination. Updated evidence is 135 Python flight/adapter tests passed
with four expected Rust-wheel skips and 96 React files / 584 tests passed, plus
Python 3.12 mypy, Ruff, Black, TypeScript, ESLint, Vite build, structural
budgets, and diff checks. The initial implementation commit was
`43ad5e35be299f2ab11260784ee707fc5721fd2e`; corrections are committed at
`5c6073bd68ed4c8f23b343d4d11c2dc4277ea246` and published in draft PR #4283.
Protected CI, reviews, and every parent PR remain required.

The first hosted CI Standard run on PR #4283 reached delta mypy after checkout,
dependency installation, Ruff, and formatting passed. With unchanged imports
skipped, mypy treated the request fields used by the new private runtime as
`Any` and rejected `_OptimizationContext.total_count` for returning an implicit
`Any`. The request contract already guarantees positive integer operands; the
scoped fix makes the return boundary explicit with `int(...)`. The exact
seven-file Python 3.12 CI mypy command, Ruff/format, diff check, and the full
135-test flight/adapter suite now pass (four optional Rust-wheel skips). This
fix and handoff update are committed together as
`60ac5b46c78988225862d9b89a33ddc3656a3413`, now present in the propagated
capability ancestry.

### 2026-08-07 strict flight-to-ground contract continuation

Active worktree
`C:\Users\diete\Repositories\Tools-worktrees\ground-transition-contract` on
branch `feat/4268-ground-contract` starts exactly at protected draft PR #4283
head `60ac5b46c78988225862d9b89a33ddc3656a3413`. It is the stacked implementation
for [issue #4268](https://github.com/D-sorganization/Tools/issues/4268) under
ground-model epic #4267. The implementation and this durable handoff update are
committed together as `0d6f5d0b879ce3456c990c08b17d6df4185c4a8f`.

The new self-facaded `shared.python.swing_sim.ground` package owns strict
`flight-to-ground-request/v1` and `flight-to-ground-result/v1` contracts. Every
record is frozen, SI-only, and explicit about the canonical target frame. A
request carries two full signed 3D flight states that bracket physical
sphere/terrain contact, ball radius, mass, rotational inertia factor, complete
planar surface geometry/material data, provider/version identity, calibration,
and reproducibility provenance. It rejects non-finite or Boolean numbers,
unknown nested fields, unsupported versions/units/frames, non-unit or downward
normals, non-incoming contact, and states that do not straddle the physical
surface gap.

Results distinguish carry, bounce-air, skid, roll, accumulated surface path,
final downrange/offline, and launch-to-final horizontal total distance. Ordered
phase samples, event ledgers, status/termination matrices, warnings,
calibration, and provenance fail closed: failed/unavailable results cannot
fabricate trajectory summaries; rest samples cannot still move or spin; event
bounce counts and trajectory-derived distance summaries must agree. The only
legacy projection is the explicit one-way `to_ground_model_result` adapter,
which accepts complete qualified results and never infers total or roll from
carry.

Machine-readable Draft 2020-12 request/result schemas, deterministic compact
serialization, explicit current-version migration gateways, a shared
Python/TypeScript/Rust/WASM golden fixture, contract documentation, and a pinned
public API are included. The local gate is green: 45 focused contract/API/
schema/migration/parity tests and the full Python 3.12 flight-plus-ground suite
(180 passed, four expected optional Rust-wheel skips), plus Ruff, formatting,
production mypy, schema meta-validation, structural file/function budgets, and
diff checks. The Python 3.12 environment reports the pre-existing SciPy/NumPy
compatibility warning; no new ground test warning is introduced.

Independent pre-publication review then found four release blockers before any
commit or PR: Python-native JSON number spelling was not cross-runtime stable;
JSON Schema integers and runtime integer parsing disagreed on values such as
`64.0`; direct constructors could accept invalid nested records; and a plane
could move along its normal without a reference epoch while zero-speed contact
was classified as incoming. The fixes reuse the shared 11-decimal canonical
numeric writer, normalize all contract floats and integral JSON numbers, pin
adversarial numeric tokens in the golden fixture, validate every nested record
at the public constructor boundary, restrict v1 surface motion to the tangent
plane, and require both bracket states to have strictly incoming relative normal
velocity. First-contact event/time/position/output-state identity and complete
event-range checks are also enforced.

Two subsequent adversarial reviews found additional fail-closed gaps. Explicit
phase/event transitions and status/termination pairings now prevent regressions;
terminal event time, position, linear/angular state, phase, and completion are
bound to the final trajectory point; duplicate JSON object keys are rejected at
every nesting depth; and the target-frame origin and post-first-contact bounce
count are unambiguous. Event ledgers preserve signed pre/post angular state,
unavailable results carry typed field/reason/provenance records, raw physical
and relational bounds are checked before canonical rounding, and unsafe or
oversized integers, noncanonical edge whitespace, and surrogate text fail
closed with typed validation errors. All files and functions were split back
under the repository's 400-line/50-line/four-parameter limits. Two final
independent re-reviews found no remaining publication blocker in #4268 scope.

Do not connect this contract to current flight output by substituting initial
spin or a launch-plane crossing. Issue #4269 must first propagate full terminal
angular velocity and two states bracketing ball-radius/terrain contact across
Python, TypeScript, Rust, and WASM. UpstreamDrift remains a one-way adapter
consumer; Tools must not import it, and its lossy terrain material round trip and
elevation-grid boundary defects require separate repair evidence.

New visualization issue
[#4284](https://github.com/D-sorganization/Tools/issues/4284) is a child of
toolstrip/workspace epic #4218. It tracks bounded clubhead camera following and
Face On, Down the Line, and Overhead snap views with canonical frame definitions,
per-viewport state, PyQt/React parity, playback/zoom interaction coverage, and
rendered computer-control QA.

Draft PR #4285 initially failed only the CI Standard changed-test assertion
gate because its fixture-only package marker and deterministic record builder
live beneath a `tests` directory. Both files are now explicitly allowlisted by
exact repository path in `scripts/test_assertion_allowlist.txt`; behavioral test
modules remain subject to the AST assertion gate. Reproduce this narrow check
from the PR worktree by diffing Python paths against
`feat/4197-capability-observer` and passing that list to
`scripts/check_test_assertions.py --changed-files`. This gate repair and the
handoff update must be committed and pushed together as a normal follow-up
commit; do not amend or force-push the published contract commit.
The next protected run exposed two `detect-secrets` false positives in each
runtime's cross-language SHA-256 parity assertions. They are deterministic test
digests, not credentials. Mark the four exact constants with the scanner's
`pragma: allowlist secret` annotation; do not add broad path exclusions or
rewrite the baseline. Re-run the scanner normalization gate, focused parity
tests, lint, and diff checks. Commit this CI repair with this handoff update and
push normally on `feat/4197-capability-observer` before propagating the parent
head through the protected stack. That repair is parent commit
`49612946138b1021f80c9f8d2a4d06f1610825db`; this child now merges it normally
without rewriting either published branch.

Issue #4269 branch `feat/4269-flight-ground-transfer` now merges protected
contract head `3235af71150a774954e7673fc81d7179330fbe76` without rewriting the
stack. Keep its cross-runtime transfer implementation uncommitted until the
post-repair independent review and complete Python/TypeScript/Rust/PyO3/WASM
gates are green.

### 2026-08-07 flight-to-ground physical transfer continuation

Issue #4269 continues from alignment merge `13184096e` in
`C:\Users\diete\Repositories\Tools-worktrees\flight-ground-transfer`. Python
and TypeScript now preserve full signed angular state, require explicit
launch-origin evidence, and qualify sphere contact against the configured
launch-relative terrain plane. Python exposes `simulate_to_surface` for built-in
native models without breaking the legacy `simulate` contract; the web RK4 path
rejects more than 50,000 synchronous steps before entering its loop and uses an
exact partial final step rather than exceeding the requested horizon.

Rust, PyO3, and WASM accept the complete `flight-to-ground-request/v1` record,
including surface material/provider data, calibration, provenance, ball data,
and the strict incoming time-ordered bracket. Rust retains its raw crossing
bracket in transfer-event evidence; Python and TypeScript intentionally use the
exact zero-gap interpolated contact as the v1 first-penetrating state. Tee height
remains a vertical ground-to-ball-bottom measure and terrain elevation remains
observable. At this pre-publication checkpoint, the
implementation/specification/handoff commit became
`d2d3d0f53a78aa863574afe43290a29c48318d94`; the following review record and
current handoff supersede the then-pending publication instruction.

The second independent review found three real blocker classes: approximate
Python origins/malformed chronology, noncanonical Rust wire tokens, and
fixed-step runtimes exceeding or truncating their requested horizon. All are
now repaired with adversarial tests. Current evidence is 208 Python tests using
the exact rebuilt CPython 3.12 wheel with no skips, 603 web tests, 160 Rust
tests, exact PyO3/Python canonical output, PyO3 and wasm32 checks, production web
build, and a completed `wasm-pack build`. The final independent closure audit
found no P0-P2 issue and declared #4269 locally publication-ready.

Full-crate Clippy warnings remain confined to pre-existing unrelated electrode,
SCADA, signal, and math modules; no `flight_ground` warning is present. The
existing local SciPy/NumPy compatibility warning also remains environmental.
All new source files are below 400 lines. The oversized append-only SPEC and
handoff registries plus the preserved Waterloo and `from_imperial` public
signatures predate #4269 and are explicitly retained for compatibility.

## 2026-08-09 Issue #4273 qualified study foundation

The next local child branch, `feat/4273-ground-study-projection`, starts from
the exact PR #4305 head `a35fc8aac0cbc2aeeef757fd1d1c518987f2355c`.
It adds the strict `ground-study-projection/v1` boundary described in
`docs/specs/GROUND_RESULT_STUDIES.md`: exact summary preservation; request,
surface, model, and canonical source-result identity; complete ball/surface and
evidence-bearing profile bindings; typed warnings; observed endpoint geometry;
intrinsic arbitrary-plane landing-target misses; fail-closed objective
eligibility; and deterministic canonical persistence. The wire parser
re-derives summary/endpoints, sphere/plane contact, target miss, and
profile/surface coherence rather than trusting stored assertions. Valid partial
airborne endpoints remain censored with typed target unavailability and no
invented surface projection.
The request digest is caller-context evidence rather than an attested source
binding because `ground-result/v1` carries no producing-request fingerprint.
Only the available ID, surface/frame, calibration, and provenance compatibility
checks are claimed; exact request/result attribution remains a follow-up.
The record embeds the exact result calibration/provenance and independently
requires measured/literature model calibration with positive confidence for
solver admission. Estimated, unvalidated, or zero-confidence model evidence
fails closed; provenance is retained without claiming producer certification.
The older direct result-to-metric adapter is deprecated and removed from the
ground facade. It remains module-level compatibility code only and cannot be
used as the qualification-sensitive path because it has no profile binding.

Exact repair commit `940563f222065c4f343b587699c52062c6e1db59`
passes 194 ground tests, 27 flight-first import/result/transfer tests, and an
independent 75-test adversarial audit of calibration, provenance, strict wire,
lazy imports, facade exports, and deprecated compatibility behavior.
No material handoff change beyond correcting the deprecated compatibility
adapter's rejection wording so it does not imply profile qualification.

This is a bounded foundation for issue #4273, not completion of the issue or
ground epic. No ensemble/variation/wind/optimizer/UI/compiled/Upstream consumer
is claimed. Keep both #4273 and #4267 open until those later adapters and
protected release evidence exist. The branch must remain a normal fast-forward
child of #4305 and may be published only after its complete local gates and
independent review are recorded.

Implementation commit `0de714842cf4cd1207944044c883c2d8dc83a7ba`
passed independent adversarial review. After normally merging current #4305
head `a35fc8aac0cbc2aeeef757fd1d1c518987f2355c`, the tree passes all 192 ground
tests and 47 focused projection/state/wire/API tests.

Draft PR #4306 publishes the normal stack child on
`feat/4273-ground-study-projection`, targeting
`feat/4272-ground-material-profiles`; its creation head was
`6a1b2f76160de0535fca2126958934c53ad5ed25`. This is protected-integration
evidence only after required checks and review complete, so #4273/#4267 remain
open.

The next local #4273 continuation,
`feat/4273-ground-study-result-adapter`, started from PR #4306 creation head
`17473948f1ce5837bd5b55618d5393b0d8575188` and normally includes current head
`d44edeb4119048fe7a3f8ccfdcae81c8771561e8`. It adds a one-way qualified-study
adapter for the existing total/roll/bounce/final-offline DTO. The adapter fails
closed on ineligible studies, does not treat target miss as missing physics,
and documents that the legacy DTO is provenance-lossy. It has no carrier or
protected evidence yet.
Exact reviewed evidence is commit
`6c296ab35471fc8d2070d229f2921d200f7defdb`: 198 ground tests, 27 flight-first
import/result/transfer tests, and 44 focused adapter/compatibility/API tests
pass. Independent re-review found no remaining publication blocker.
Draft PR #4307 publishes this child against
`feat/4273-ground-study-projection` from creation head
`dac35e3fd61ee8af80dc8c2262da31ea274dbb1d`. Keep #4273 and #4267 open; the PR
has no protected release evidence until required checks and review finish.

A post-publication flight-first import gate found a facade cycle introduced by
the new study exports: flight loaded the ground facade, study loaded the solver
package, and solver returned to the partially initialized flight facade. PR
#4306 repairs this at the dependency boundary by importing `GroundSurfaceProfile`
and `GroundContactState` directly from `ground.contract_types` in the two flight
consumers and lazily resolving solver-dependent study facade exports. Later
children must receive this only through normal ancestry.

## 2026-08-09 PR #4302 deterministic-digest scanner repair

The protected run at exact PR #4302 head
`920c46dee688815691e251777142126bf1489b1a` found one actionable scanner-only
failure: the committed impact golden fixture's public SHA-256 assertion was
classified as a high-entropy secret. The assertion now carries only the exact
inline `pragma: allowlist secret` annotation used elsewhere in this campaign.
The digest, fixture, physics, scanner scope, and baseline remain unchanged.

Commit and push this bounded repair normally before propagating it into #4304,
#4305, or later ground children. The contemporaneous file-size cancellation
occurred in checkout and remains infrastructure evidence, not a code failure.

## 2026-08-09 Issue #4273 scalar ensemble continuation

Local branch `feat/4273-ground-study-scalar-adapter` starts from exact PR #4307
head `de6ea15290f6b3c5c49bd436b846baa8f6cb752b`. It adds an explicit-identity,
bounded adapter from `ground-study-projection/v1` records into the shared
`scalar-ensemble/v1` plot/export contract. The adapter sorts by caller-supplied
series/trial identity, rejects duplicates and overflow without truncation,
retains complete and censored numeric observations, and exposes failed or
unavailable rows with null scalars rather than inventing outcomes. Partial
airborne studies retain first-contact target evidence while final-target values
stay null with the typed reason.

Row attributes preserve a whole-study digest, request-context and result
digests, exact target geometry, result calibration and provenance, surface/frame
and material-profile identity, qualification
and operating condition, solver eligibility reasons, and target availability.
Missed targets and numeric but unqualified results remain analyzable. This
slice does not implement ensemble execution, optimization composition, plots,
UI, compiled runtimes, or downstream parity and cannot close #4273 or #4267.
Require focused/shared scalar tests, the full ground suite, static and campaign
gates, independent review, and normal protected review before integration.
Publish only as a draft child of `feat/4273-ground-study-result-adapter`.

Independent re-review found all semantic and static blockers resolved at exact
implementation commit `b71bf88b6ed888248ad152f69a2bd2de3892e256` after 198
ground and 19 adapter/shared-scalar tests plus Ruff, Black, MyPy, manifest,
documentation, assertion, file-size, structural, and diff gates. Draft PR
#4308 publishes that implementation against unchanged parent branch
`feat/4273-ground-study-result-adapter` at
`de6ea15290f6b3c5c49bd436b846baa8f6cb752b`. Protected CI/review and all
remaining #4273/#4267 scope are still open.

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

## 2026-08-10 Issues #4273/#4275 bounded reference execution

Branch `feat/4275-ground-reference-execution` starts from exact draft PR #4308
head `c8ebf422669992c4a33db661b0c37dfe72b580ae`. It adds the narrow canonical
Python orchestration missing between the existing bounce, static-plane
skid/roll, and result-composition contracts. The executor calls each phase once,
passes the same cooperative cancellation hook to both, and returns a public
result only when the native phase outcomes already have an honest v1 mapping.

Rest, finite-domain exit, time-limit, and event-limit suffixes compose after a
settled-to-skid prefix. Cancellation raises a distinct typed signal. Bounce
time/event limit, no-recontact, and numerical failure, plus suffix step-limit,
unsupported-surface, and numerical failure, raise typed fail-closed evidence
with phase, native reason, and request fingerprint. Composition rejection is
also typed and retains the original exception as its cause. No terminal state
is relabeled.
The current skid/roll implementation does not emit its reserved
`numerical_failure` enum; native numerical exceptions propagate, and the
coordinator intentionally avoids a broad `ValueError` catch that would also
swallow resolver, configuration, or callback contract errors.

The shared golden fixture records an exact complete bounce/skid/roll/rest
request and result with canonical digests. Focused tests also cover repeat-run
byte determinism, representable censored outcomes, exact controls, callback
continuity, and public exports. This is a bounded #4273/#4275 continuation, not
ground epic completion: UI, ensembles, production-qualified material data,
changing normals/regions, compiled-runtime parity, and UpstreamDrift consumers
remain open under #4274/#4276/#4267. Require complete local and independent
review gates, then publish only as a stacked draft child of PR #4308 without
retargeting or rewriting ancestors.

Independent exact-tree re-review declares this bounded slice READY after the
self-contained execution fixture and pre-physics resolver DbC blockers were
fixed. Current evidence is 219 ground tests, 44 focused executor/API tests on
CPython 3.12 and isolated 3.10, 26 flight contract/result/transfer tests,
pinned Ruff 0.14.10, changed-file Black, pinned MyPy 1.13, campaign manifest
plus eight contracts, documentation governance, changed-test assertions,
400-line file, 50-line function, four-parameter signature and diff gates, both
import orders, and the pinned fixture bytes. Repository-wide Black separately
reports only unchanged inherited `ground/study_wire.py` and
`ground/tests/test_profile_contract.py`; no unrelated formatting expansion is
part of this carrier.

Draft PR #4309 publishes exact independently reviewed implementation head
`c93c6f36d361f4c129d702565a9330149e175557`, targeting unchanged parent
`feat/4273-ground-study-scalar-adapter` at
`c8ebf422669992c4a33db661b0c37dfe72b580ae`. This publication-only continuation
adds the carrier and immutable local evidence to the campaign manifest. It does
not claim protected CI, review, merge, issue closure, or epic completion.

## 2026-08-10 Issue #4274 strict ground-result playback slice

The local `feat/4274-ground-playback` branch starts from exact PR #4309 head
`51492c3ddc8b15b1358434da9b29f600261c918a`. It introduces first-class Ground
Playback in the standalone PyQt6 and React workspaces while preserving the
strict execution boundary: clients import an exact
`flight-to-ground-result/v1` and never run or imitate ground physics. Import is
bounded to 5 MiB and 100,000 trajectory samples, validates a candidate before
state replacement, and retains the prior valid result after rejection.

The two clients use the same golden result and aligned absolute-time semantics:
interpolation is limited to one declared phase, and phase transitions hold the
preceding exact state until the next exact sample. Controls expose replay,
pause, exact stepping, phase jumps, scrub, loop, granular speed, and camera
reset. Locked physical axes, orbit/zoom, carry and complete/observed terminal
markers, and trajectory/event/warning/calibration/provenance tables preserve
the result evidence. Since v1 carries no surface geometry, both clients show
neutral axes and explicitly avoid claiming a terrain plane.

Evidence at handoff is 872 passed full Rate Python tests, 672 passed full React
tests, and 9 focused adapter/Qt tests. Pinned Ruff 0.14.10, Black, pinned MyPy
1.13, React lint/type/build, changed-file policy, assertion, size,
documentation, and diff gates pass. Standalone Playwright Chromium verified
desktop and narrow import/playback/zoom behavior, canvas containment, and zero
horizontal overflow at 1440x900 and 520x900. An offscreen PyQt render at the
supported 1024x700 minimum verified the compact two-row controls and usable 3D
viewport.

This does not complete #4274 or #4267. Surface editors, exact terrain meshes,
comparison overlays, workspace persistence/export, ensembles, inverse
optimization, Rust/WASM execution, and UpstreamDrift consumer parity remain
open. Require independent exact-tree review plus ordinary protected CI before
publication; no push or PR is part of this local slice.
