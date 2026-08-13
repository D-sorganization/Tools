# AGENT_HANDOFF — rate_of_closure

## 2026-08-11 Regional surfaces receive current skid/roll parent

Open draft PR `#4332` retains
`feat/4271-regional-surface-transitions` over
`feat/4271-ground-skid-roll`. Exact remote regional-surface child
`04ccf08dd990de1cd056a3420e67772773a4be2e` is merged first and exact current
skid/roll parent `3f861c1faaa7455faee92eaa7f813174667208e0` second by an ordinary no-ff
merge. Child-owned regional transition production, tests, fixtures, contracts,
evidence, and limitations remain intact while inheriting current skid/roll,
impact/bounce, flight-transfer, and strict-ground ancestry.

Focused local qualification passes all `121` Python ground/regional tests,
`20` focused React ground-contract/transfer tests, and all `137` `tools-core`
Rust tests. TypeScript, zero-warning ESLint, the 189-module Vite build, Rust
format and warning-denied clippy, pinned Ruff 0.14.10 across 14 changed Python
files, pinned MyPy 1.13 and Bandit medium/high across 12 production files, the
500-LOC changed-file gate, manifest validation plus eight contracts,
documentation governance, exact child-tree preservation, history/marker, and
diff checks are green.

This candidate remains unpublished and `not_released`. Changing normals,
discontinuous geometry or surface velocity, regional UI, compiled regional
physics, UpstreamDrift parity, protected exact-head CI, review, approval,
dependency integration, and release remain open.

## 2026-08-11 Skid/roll receives current impact/bounce parent

Open draft PR `#4304` retains `feat/4271-ground-skid-roll` over
`feat/4270-ground-impact-bounce`. Exact remote skid/roll child
`0ea6740965068542e9d8c7449e06ec07d88969e0` is merged first and exact current
impact/bounce parent `62beb3e1aa951645fd556a53f9cbae4bb46c47e5`
second by an ordinary no-ff merge. Child-owned skid/roll production, contracts,
tests, fixture, and limitations remain intact while inheriting current impact,
flight-transfer, and strict-ground ancestry.

Local verification passes `154` focused ground/flight tests, all eight manifest
tests, and all `137` `tools-core` Rust tests. Pinned Ruff 0.14.10 covers the 18
PR-delta Python files; pinned MyPy 1.13 and Bandit medium/high cover 12
production files. Rust format/clippy, manifest, docs, exact child-tree,
history/marker, and diff gates are green.

This local model slice remains `not_released`. Regional/changing-normal
surfaces, deformation, torsional damping, roll-to-skid transitions, UI,
compiled physics, publication, protected exact-head CI, review, approval,
downstream integration, and release remain open.

## PR #4302 corrected-flight-transfer propagation

Exact corrected #4288 parent
`247215422a6d4b677552955b4923bc609a553259` is incorporated into
`feat/4270-ground-impact-bounce` by the normal merge containing this handoff.
PR #4302 keeps base `feat/4269-flight-ground-transfer`; neither branch was
rebased, retargeted, rewritten, or force-pushed. The child retains passive
impact, deterministic repeated hops, exact contact, capture-to-skid, bounded
cancellation/failure, and airborne evidence while inheriting corrected flight
transfer, deterministic workspace timestamps, and canonical `swing_sim`
identity.

This remains a partial `not_released` slice. Issue #4271 retains skid, roll,
rest, total distance, and final ground results; terrain deformation, UI,
TypeScript bounce physics, compiled bounce runtimes, and UpstreamDrift adapters
remain excluded. Protected CI, independent review, and normal stack collapse
remain open.

Merged-tree validation is `987` Python tests, `106` React files / `661` tests,
and the complete `tools-core` Rust suite at `137` tests (`111` unit, `20`
transfer, `6` wire). The affected cross-version suite is also `146` tests on
real CPython 3.10.20. The 189-module Vite production build, TypeScript,
zero-warning ESLint, Ruff check/format across 90 files, pinned mypy 1.13 across
all 17 ground and nine transfer production modules, Rust workspace format plus
warning-denied `tools-core` clippy, campaign-manifest validator plus eight
contracts, documentation governance, 11-file 500-LOC budget, marker scan, and
diff checks are clean. Hosted checks and review apply to the new exact merge
head only.


## PR #4288 corrected-ground propagation

Exact corrected #4285 parent
`788aa547651a3685a363ea401824a5d81477bafb` is incorporated into
`feat/4269-flight-ground-transfer` by the normal merge containing this handoff.
PR #4288 keeps base `feat/4268-ground-contract`; neither branch was rebased,
retargeted, rewritten, or force-pushed. The child retains its qualified
cross-runtime terminal-state/contact transfer and inherits the corrected UTC
parser plus canonical `swing_sim` import identity.

This is ancestry propagation, not bounce, skid, roll, terrain response, total
distance, or UI delivery. Protected CI, independent review, and normal stack
collapse remain open.

Merged-tree validation is `951` Python tests, `106` React files / `661` tests,
and `26` focused Rust transfer/wire tests. The affected cross-version suite is
also `110` tests on real CPython 3.10.20. The 189-module Vite production build,
TypeScript, zero-warning ESLint, Ruff check/format across 82 files, pinned mypy
1.13 across the six-file transfer and 13-file parent production namespaces,
Rust workspace format plus warning-denied `tools-core` clippy, campaign-manifest
validator plus eight contracts, documentation governance, 13-file 500-LOC
budget, marker scan, and diff checks are clean. Hosted checks and review apply
to the new exact merge head only.


> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-11

## 2026-08-11 PR #4332 current-parent ancestry candidate

The clean dedicated `feat/4271-regional-surface-transitions` worktree starts
from exact live PR #4332 child `1a48d749af508843fac2a5102f4dd56294429bda`
and normally merges exact newly published `feat/4271-ground-skid-roll` parent
`0ea6740965068542e9d8c7449e06ec07d88969e0` as its second parent. PR #4332
keeps base `feat/4271-ground-skid-roll`; neither branch is rebased, retargeted,
rewritten, force-pushed, or published by this reconciliation. The only textual
merge conflict is the independent SPEC-version collision. SPEC 1.14.27 retains
the regional child record after the parent's 1.14.26 entry; production physics,
contracts, schemas, numerical ordering, and public APIs merge without conflict.

The child retains bounded coplanar material overlays, explicit precedence,
exact quadratic boundary splitting, base-edge precedence, state and energy
continuity, strict `surface_transition` evidence, request-bound transition
limits, and randomized piecewise-analytic coverage. The current parent retains
its reviewed impact, bounce, skid, roll, resistance, qualified-rest, edge,
composition, and passive-ledger behavior. Changing normals, height or
surface-velocity discontinuities, terrain deformation, torsional-spin damping,
roll-to-skid transitions, regional UI, compiled regional physics, downstream
parity, protected CI, review, normal stack integration, and main release remain
open. This local merge is not release evidence and requires independent review
before an ordinary fast-forward publication.

Merged-tree qualification is `121` focused Python ground/regional tests and
`5` focused React ground-contract tests passing. Pinned Ruff 0.14.10 check and
format pass all `41` ground Python files; pinned MyPy 1.13 passes all `28`
ground production modules; and Bandit reports no medium/high finding. The
campaign manifest validator and all `8` manifest contracts, documentation
governance, changed-production policy, minimum test contract, both parent and
child diff checks, and the official 500-LOC changed-file gate (`14` files,
zero violations) pass. Production maxima are `392` lines per module, `46`
lines per function, and `4` parameters excluding `self`/`cls`. The manifest
now records open PR #4332 at its still-published child head and open PR #4304
at its exact newly published parent head; neither record claims protected or
main-release evidence.

## 2026-08-11 Ground-impact bounce receives current flight-transfer parent

Open draft PR `#4302` retains `feat/4270-ground-impact-bounce` over
`feat/4269-flight-ground-transfer`. Exact remote bounce child
`cf6e72bad98e5f36f782254942e6895b8b71e670` is merged first and exact current
transfer parent `c39297d37a17d5d8d3520ed62c5563cdcc609cab` second by an
ordinary no-ff merge. The child-owned impact and repeated-bounce contracts,
tests, fixture, and explicit limitations remain intact while inheriting current
strict-ground and cross-runtime flight-transfer ancestry.

Three inherited PR-delta Python files receive only their pinned Ruff 0.14.10
layout. Analysis, persistence, and contract assertion behavior are unchanged.

Local verification passes `1129` Rate/ground/flight Python tests, `25` focused
React transfer tests, and `137` Rust tests. TypeScript, zero-warning ESLint, the
189-module Vite build, Rust format/clippy, pinned Ruff 0.14.10, pinned MyPy
1.13, Bandit medium/high, manifest parsing, history preservation, and diff
checks are green.

This is a local `not_released` candidate. Skid, roll, rest, total distance,
final ground results, UI, TypeScript bounce physics, and compiled bounce
runtimes remain outside this slice. Publication, protected exact-head CI,
review, approval, dependency integration, and release remain open.

## PR #4285 workspace timestamp propagation

Exact corrected #4282 parent
`5f77af4add23547a21cc3fabce98ae9ad4260427` is incorporated into
`feat/4268-ground-contract` by the normal merge containing this handoff. PR
#4285 keeps base `feat/4197-capability-observer`; neither branch was rebased,
retargeted, rewritten, or force-pushed. The child retains its strict
flight-to-ground contract and inherits the deterministic Python 3.10-3.12 UTC
parser plus complete variation, scalar, wind, capability, and release ancestry.

The preceding exact-head Python 3.12 CI lane loaded embedded
`src.shared.python.swing_sim` tests and canonical `shared.python.swing_sim`
imports as distinct package trees, producing ground/impact collection errors.
The shared alias registry now coalesces that package root. A subprocess identity
contract failed before the fix and now passes together with both affected
public-API suites. The file-size job was cancelled in checkout before its
budget step and is not a code failure.

Merged-tree evidence is `915` Rate/ground/impact Python tests, all `12` shared
alias architecture tests, `28` focused tests on CPython 3.12, `28` direct
workspace compatibility tests on real CPython 3.10.20, and `104` React files /
`642` tests. The production build, TypeScript, zero-warning ESLint, Ruff,
format, pinned mypy 1.13, campaign manifest and eight contracts, docs
governance, 500-LOC budget, marker scan, and diff checks pass.

## 2026-08-11 PR #4304 current-parent ancestry candidate

The clean dedicated `feat/4271-ground-skid-roll` worktree first fast-forwarded
to exact live PR #4304 child `52d9a6a978d8e6b8b19ef92f02f265c9058b00ad`,
then normally merged exact live `feat/4270-ground-impact-bounce` parent
`cf6e72bad98e5f36f782254942e6895b8b71e670`. The PR base remains unchanged;
neither branch was rebased, retargeted, rewritten, force-pushed, or published.
The automatic merge was conflict-free. Its only child-tree change before this
documentation reconciliation is deterministic formatting in the existing
skid/roll regression test; production physics, contracts, schemas, numerical
ordering, and public APIs remain unchanged.

Local qualification on the merge candidate is `115` focused ground
tests passing on CPython 3.13, pinned Ruff 0.14.10 clean across the ground
package and tests, and pinned MyPy 1.13 clean across all `25` ground production
modules. The manifest validator and all `8` manifest tests, documentation
governance, exact-parent and exact-child diff checks, changed-production policy,
the official changed-file 500-LOC budget, and mandatory production limits of
400 lines per module, 50 lines per function, and four parameters all pass. The
local merge is a PUBLISH candidate for independent review only; protected
exact-head CI, approval, normal child propagation, and publication remain open.

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
## 2026-08-11 Flight-ground transfer receives current strict-ground parent

Open draft PR `#4288` retains `feat/4269-flight-ground-transfer` over
`feat/4268-ground-contract`. Exact remote child
`df716ac740beb9d47dced66f867dc3dde1fa5367` is merged first and exact current
parent `584e4e53a7990053a346938c4e45538786887ec7` second by an ordinary
no-ff merge. All transfer production/tests/fixtures and Python, TypeScript,
Rust, PyO3, and WASM contracts remain intact. Only append-only handoff/SPEC
records conflict, and both dated histories are retained. Focused verification
passes 85 Python, 25 React, and 137 Rust tests; pinned Ruff 0.14.10, MyPy 1.13,
Bandit, TypeScript, exact-delta ESLint, the production web build, Rust format,
docs, strict manifest, diff, exact transfer-tree, and append-only preservation
gates pass. Publication, protected exact-head CI, review, approval, and release
  remain open.

## 2026-08-11 PR #4288 append-only history preservation repair

This documentation-only follow-up to exact local candidate
`1eb4188ff1a87a3d8e9a55aeca68194ca04e0bb4` restores every omitted parent
handoff section and specification row and repairs the interleaved paragraph
below. Production code, tests, schemas, the campaign manifest, child-first
merge topology, and #4288's `feat/4268-ground-contract` base are unchanged.
Publication, protected exact-head CI, approval, dependency integration, and
release remain open.

## 2026-08-11 PR #4288 current-ground-parent ancestry candidate

The clean `feat/4269-flight-ground-transfer` reconciliation starts from exact
published child `afb1bdcfc6701caaf1f7bc3497a6a37dd9698c14` and normally merges
exact newly published #4285 parent
`e09ab96280cab363b36bd0e7db4a3cd064dc2527` second without changing #4288's
base. The child's bounded `flightIntegrator.ts` facade is preserved instead of
restoring the parent's superseded inline loop, while kinetics resolves to the
parent's split validated implementation. Seven child-owned Python files receive
pinned Ruff 0.14.10 normalization with no AST change; the preserved TypeScript
facade uses standard line endings. Bounce, skid, roll, terrain response, total
distance, UI integration, protected evidence, and release remain open.

> Update this append-only handoff with every implementation commit and every
> push to `main`. Last updated: 2026-08-11.

## 2026-08-11 Ground contract receives current capability carrier

Open draft PR `#4285` retains branch `feat/4268-ground-contract` and base
`feat/4197-capability-observer`. Exact remote ground-contract child
`e09ab96280cab363b36bd0e7db4a3cd064dc2527` is merged first and exact current
capability-observer parent `d0b827667e61f0583e1dad0b3cbbca6819624d3c`
second by an ordinary no-ff merge. All strict flight-to-ground production,
tests, schemas, fixtures, dependencies, and fail-closed limitations are
preserved. Only append-only handoff/SPEC files conflict; their dated records
from both parents remain. Publication, protected exact-head CI, review,
approval, release, and downstream #4288 propagation remain open. Focused
verification passes 214 Python/PyQt tests, pinned Ruff 0.14.10 across 27
PR-delta Python files, pinned MyPy 1.13 across 20 production files, and Bandit's
medium/high threshold. Docs, strict manifest, diff, exact ground-tree, and
both-parent append-only preservation gates pass.

## 2026-08-11 Merged capability-observer carrier receives current wind workflow

PR `#4283` is merged, while its former head branch
`feat/4197-capability-observer` remains the base of open draft PR `#4285`.
Exact remote/local carrier child
`b2baf6ef615b0d756a86ac4ca7eef2fc583210ee` is merged first and exact published
#4282 parent `3faf9e9493c9ba37168d9f65ab10cafeedc2a72f` second by a normal
no-ff merge. Since the carrier is already the first parent of #4282, the source
merge is conflict-free: the complete capability-observer implementation and
history remain present while the exact current wind workflow and manifest tree
are inherited. Only handoff/SPEC carrier records differ from #4282. All 86
focused Python/PyQt capability, wind, and manifest tests and all 86 tests in 12
focused React capability/wind suites pass. React type-check, zero-warning lint,
and the production build pass. Docs, manifest, and exact-parent diff gates pass;
changed-Python static gates are not applicable to the documentation-only exact
delta. This local audit does not authorize publication; review,
protected CI, and
later dependency propagation into #4285 remain open.

## 2026-08-11 Wind workflow receives newly published scalar-adapter parent

PR `#4282` keeps branch `feat/4199-wind-workflow` and base
`feat/4199-wind-scalar-adapter`. Exact published child
`b2baf6ef615b0d756a86ac4ca7eef2fc583210ee` is merged first and exact current
parent `9321c1d2e091b8c7e5a4a83aa9ad726290e7fb5a` second. Implementation and test
paths merge automatically: responsive PyQt6/React wind execution,
progress/cancellation, controls, scatter, persistence/export, consolidated
capability workflow, and strict campaign release authority remain intact while
the parent contributes current scalar-adapter, variation, workspace,
launch-monitor/D-plane, split-kinetics, and Qt repairs. Only append-only
handoffs and SPEC conflict textually; both histories are preserved. All 94
focused Python/PyQt wind, responsive-layout, navigation, playback, and manifest
tests pass, as do all 44 tests in eight focused React wind/responsive suites.
Docs governance and strict campaign-manifest generation/validation pass.
Pinned Ruff checks/formats all 57 changed Python files; pinned MyPy accepts 38
production files; pinned Bandit finds no medium/high issues in 37 scanned source
files; the 500-LOC, changed-Python policy, test-contract/assertion, diff, React
type-check/lint, and production-build gates pass. Review, ordinary publication,
protected exact-head CI, unresolved threads, dependency order, and release
remain open.

## 2026-08-11 Wind workflow receives current scalar-adapter parent

PR `#4282` remains on `feat/4199-wind-workflow`, based on
`feat/4199-wind-scalar-adapter`. Exact remote child
`29e15d6ff631f7f30afcc745be783f2e716d7dcf` normally merges exact published
parent `a7dce5f89b483303938f518b74b4028a1c68ba81` second, preserving child-first
history and the existing base. The responsive PyQt6/React workflow and
consolidated capability stack inherit the parent's split kinetics facade,
Ground/Tee parity, and 252-line public/213-line private wind-adapter repair.

The only source conflict is an obsolete formatting-only edit to the pre-split
kinetics monolith and resolves wholly to the validated parent facade. Both
append-only documentation histories remain present. Seven surviving
child-owned paths receive only repository-pinned Ruff formatting. The exact
skipped-import MyPy profile also replaces one now-unused cache-return ignore
with an explicit `KineticsSeries | None` cast; the cached object and runtime
behavior are unchanged. Protected exact-head CI, independent review,
unresolved-thread checks, publication, and dependency order remain release
gates.

Local evidence is 74 focused Python/PyQt wind, kinetics, and Ground/Tee tests;
29 focused React tests; and 8 campaign-manifest tests. TypeScript, focused
zero-warning ESLint, Ruff check/format, pinned MyPy 1.13, Bandit's medium/high
threshold, strict wind module/signature/function budgets, manifest,
documentation, policy, SPEC, and diff gates pass.

## 2026-08-11 Ground contract receives current workflow/base ancestry

Draft PR `#4285` keeps branch `feat/4268-ground-contract` and base
`feat/4197-capability-observer`. Exact published child
`7a38b9838d743e051a6900620c4d6e754582aa89` is merged first with exact current
base/workflow head `b2baf6ef615b0d756a86ac4ca7eef2fc583210ee` second. The base contains
merged #4283 head `9bbb98e16e435a0d4c74153b909f2ebfefbbce7a`; no rebase, retarget,
force-push, parent rewrite, or publication is used.

The child's strict UI-neutral `flight-to-ground/v1` schemas, canonical fixture,
migrations, legacy-result adapter, explicit dependency, enum compatibility,
and fail-closed validation remain authoritative. The parent contributes the
current responsive wind/capability ancestry, Ground/Tee parity, and split
kinetics modules. Its facade wholly resolves the sole formatting-only source
conflict; both append-only documentation histories remain present.

The ground branch's contract-checked scalar extraction repair remains intact.
Because it makes the plotting catalog part of the effective delta, the former
459-line registry now delegates immutable ordered rows to focused private
scalar/series modules. The public catalog facade is 61 lines; every changed
production module is at most 400 lines, every function at most 50 lines, and
every signature at most four non-receiver parameters. Stable keys, display
order, units, categories, extractor behavior, and public objects do not change.

Complete local regression evidence is 1,539 Rate/shared-swing Python tests
passed with one explicit optional Rust-wheel skip; 106 React files / 648 tests,
TypeScript, zero-warning ESLint, and the 188-module Vite build passed; and all
12 `swing-core` Rust tests passed. The focused ground, Ground/Tee parity,
kinetics, wind adapter, compatibility, and manifest set passes 146 tests; 57
additional plotting/catalog/variation/kinetics regressions pin the structural
split. Pinned Ruff 0.14.10, MyPy 1.13, Bandit 1.7.7, manifest, documentation,
SPEC, policy, diff, and structural gates are required again on the committed
candidate before publication.

This is an unpublished local candidate. It does not claim bounce, skid, roll,
terrain, total distance, ground UI, Rust/WASM ground delivery, protected CI,
approval, or release. Independent review and fresh protected checks are
required before normal publication and later #4288 propagation.

## 2026-08-10 Repaired scalar-adapter propagation into wind workflow

Draft PR `#4282` remains on `feat/4199-wind-workflow`, based on
`feat/4199-wind-scalar-adapter`. It now normally incorporates repaired parent
head `d6fb04e07c2a625412e9208b07103acdc42c621b` after that head's quality gate
passed. The merge had no wind-workflow production/test conflict and used no
rebase, retarget, force-push, or history rewrite. Twenty-five focused tests and
the documentation, size, and whitespace gates pass locally; protected CI,
review, and later-stack propagation remain outstanding.

## 2026-08-10 Exact-head manifest typing repair

PR #4282 exact head `aa6eeffb0395f7ed7954f2315b1c625cada552d8`
failed hosted quality-gate run `31395741841` only at pinned-mypy
`scripts/rate_campaign_manifest.py:335:5 [no-any-return]`. The CI profile skips
import analysis, so the Pydantic validator result becomes `Any`; an explicit
local `CampaignManifest` annotation now retains the validated return contract
without changing parsing, validation, or error behavior. Ruff and format had
already passed on the failed head. Publish this repair normally, then propagate
its new exact head into #4285; do not retry the obsolete run.

## 2026-08-10 Wind workflow receives exact scalar-adapter parent

PR `#4282` keeps branch `feat/4199-wind-workflow`, base
`feat/4199-wind-scalar-adapter`, and original child first parent
`5f77af4add23547a21cc3fabce98ae9ad4260427`. Exact parent
`4a793c4c3f19aad43a3c215800b266be487ace49` merges normally second. The
responsive PyQt6/React wind strategy workflow, consolidated capability stack,
and campaign release authority remain additive with the parent's complete
variation, workspace, compatibility, and strict scalar-adapter history.

The parent's extracted navigation-state tuple is the single authority and now
includes the child-owned `capability_optimization` ID. This preserves safe
legacy visibility migration and prevents a persisted pre-capability workspace
from hiding the new tab. Runtime-only UI and capability invariants now raise
explicit errors under optimized Python, and the safe fixed-argument builder
subprocess is annotated for Bandit. The skipped-import manifest return repair
is recorded above. SPEC 1.14.15 is the current exact-head repair entry.
Verification passes 73 focused Python tests, the complete 1,657-test
Rate/shared-swing/golf-club matrix with two explicit optional build123d skips,
all 643 React tests across 105 files plus type-check/lint/build, 12 Rust tests,
31 real-CPython-3.10 compatibility tests, and 43 post-review invariant/GUI/
manifest tests. Ruff/format, pinned MyPy 1.13 across 37 changed production
files, Bandit, exact-parent size, manifest, docs, minimum-test, assertions,
CI-policy secrets, compilation, and diff gates pass. Independent re-review
found no actionable findings after 94 Python workflow/navigation tests, 20
manifest/GUI tests, 61 React workflow tests, and independent static checks.
Protected CI/review and dependency-ordered propagation remain release gates.

## Current continuation

The active local continuation is integrated directly on the existing PR #4282
carrier `feat/4199-wind-workflow`, starting from original published head
`5f77af4add23547a21cc3fabce98ae9ad4260427` and normally incorporating exact
published #4281 parent `4a793c4c3f19aad43a3c215800b266be487ace49`.
The base remains `feat/4199-wind-scalar-adapter`; no branch was rebased,
retargeted, force-pushed, or published by this continuation. It adds strict
cross-runtime capability parsing, reliable signed decimal entry, complete
ranked diagnostics and result exports, quantitative React scatter annotations,
package-safe static-web release entrypoints, and the strict
`rate-of-closure-campaign/v1` release authority. It also carries the corrected
parent's Python 3.10 compatibility, variation-export, scalar-ensemble, and
wind-adapter history. Normal publication does not establish a protected
release; CI, review, and downstream evidence remain required.

The direct web launcher dynamically loads the root bootstrap module without a
launcher-local `sys.path` mutation; its subprocess delegation test and the
changed-production-Python policy guard cover this release entrypoint.

Use these authorities together:

- `docs/release/rate_of_closure_campaign.v1.json` — machine state;
- `scripts/rate_campaign_manifest.py` — validation and generated JSON Schema;
- `docs/release/RATE_OF_CLOSURE_CAMPAIGN_MANIFEST.md` — update procedure;
- `docs/development/RATE_OF_CLOSURE_CAMPAIGN_HANDOFF.md` — detailed history;
- `docs/specs/*.md` — scientific and interoperability contracts.

The manifest is intentionally normalized: programs reference shared carrier
and test-evidence records. It fails on missing primary issues, undeclared
references, duplicate IDs, malformed SHAs, absent repository evidence,
placeholders, and contradictory release claims.

## Carrier state

The capability stack was collapsed top-down on 2026-08-09:

1. #4294 Shot Optimizer UI merged into `feat/4197-capability-flight-evaluator`;
2. #4289 evaluator merged into `feat/4197-capability-observer`;
3. #4283 observer merged into `feat/4199-wind-workflow`.

These were feature-parent merges, not protected releases. Current carrier
#4282 descends from #4281 → #4280 → #4279 → #4203 → #4202 → wedge and
variation parents. Exact published #4281 parent
`4a793c4c3f19aad43a3c215800b266be487ace49` is incorporated locally. Preserve
that dependency order and use normal merges only.

Outer platform PR #4119 targets `main` and still needs reconciliation. PR #4133
impact-interval dynamics merged into a historical feature parent after that
parent had already propagated, so its source and tests must be explicitly
reconciled before #4130 can close.

## Implemented on the feature stack

- PyQt6 and React variation workspaces: selectable input/contact/impact/shot
  scatter, matrix, landing dispersion, every-trial 3D swing arcs, quiet zones,
  linked selection, bounded raw rows, and lossless exports.
- Matched toolstrip/workspace controls, module visibility, independent plots,
  legend placement, zoom/autofit, replay, loop, and granular playback rate.
- Launch-monitor convention, D-plane, spatial target, inverse flight, wind,
  interactive flight playback, and capability-optimization contracts/workflows.
- Wedge kinematics, clearance, turf proxy, impact visualization, and
  forgiveness-analysis slices.

## Current limitations

- No complete campaign program has a qualified `main` release SHA.
- Capability v1 is still-air carry to first ground crossing; it excludes wind,
  bounce, roll, and total distance in that evaluator.
- Ground contracts and flight transfer are open carriers; qualified bounce,
  skid, roll, profiles, total-distance optimization, UI, and Rust/WASM parity
  remain.
- UpstreamDrift React has no native Rate route, and the Tools source pin and
  resolver behavior are not a qualified immutable parity boundary.
- Camera tracking/snap views (#4284), complete persistence adapters, high-DPI,
  keyboard, reduced-motion, and visual-regression matrices remain open.
- Local/rendered evidence is not protected CI or installed-package evidence.

## Most recent evidence

- Corrected-parent propagation: 62 focused wind/scalar/variation and
  compatibility tests pass on Python 3.11 and real CPython 3.10.20; 8 React
  files / 35 tests, TypeScript, and focused zero-warning ESLint pass. The real
  3.10 run found and now guards #4282's child-owned capability-observation
  `StrEnum` boundary. Ruff check/format passes 15 focused files, pinned mypy
  1.13 passes 10 production modules, and nine campaign manifest/parity
  contracts pass.
- Composed local continuation: 828 Rate Python/PyQt tests and 104 React files /
  642 tests passed; TypeScript, zero-warning ESLint, and the 188-module Vite
  production build passed. The manifest, schema JSON, Ruff, targeted mypy, and
  nine manifest/parity contracts pass on implementation head `2c1a77baa`.
  Hosted CI has not run on this composition.
- Hosted quality-gate run `31340032608` subsequently found mypy 1.13
  `no-any-return` findings at the Pydantic loader and Qt elapsed-timer adapter
  under `--follow-imports=skip`; both boundaries now narrow their return values
  explicitly. The exact Python 3.12/mypy 1.13 delta is clean across 54 files;
  Ruff, 62 focused regression tests, and eight campaign-manifest tests also
  pass locally.
- Consolidated capability head `c1827bbdc`: 1,426 Python Rate/shared-swing
  tests and 624 React tests passed locally.
- Variation continuation `d71b0ea01`: 890 Python/PyQt/shared-swing tests and
  545 React tests passed locally, with one optional Rust skip.
- Toolstrip head `c36ca36e9`: the same 890/545 local suite totals passed.
- Hosted #4282 head `3186a265b1`: `swing_core` built, but cached plugin loading
  failed before parity collection; this is failed evidence, not a model pass.
- CI-isolation head `18fe89201`: focused workflow regression passes locally;
  the hosted non-skipped parity run is still required.

## Validation

```powershell
python scripts/rate_campaign_manifest.py
python -m pytest tests/rate_of_closure/test_campaign_release_manifest.py -q
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest tests/rate_of_closure src/shared/python/swing_sim -q
cd src/rate_of_closure/web
npm test -- --run
npm run type-check
npm run lint
npm run build
```

## 2026-08-11 Wind workflow receives current scalar-adapter parent

PR `#4282` remains on `feat/4199-wind-workflow`, based on
`feat/4199-wind-scalar-adapter`. Exact remote child
`29e15d6ff631f7f30afcc745be783f2e716d7dcf` normally merges exact published
parent `a7dce5f89b483303938f518b74b4028a1c68ba81` second, preserving child-first
history and the existing base. The responsive PyQt6/React workflow and
consolidated capability stack inherit the parent's split kinetics facade,
Ground/Tee parity, and 252-line public/213-line private wind-adapter repair.

The only source conflict is an obsolete formatting-only edit to the pre-split
kinetics monolith and resolves wholly to the validated parent facade. Both
append-only documentation histories remain present. Seven surviving
child-owned paths receive only repository-pinned Ruff formatting. The exact
skipped-import MyPy profile also replaces one now-unused cache-return ignore
with an explicit `KineticsSeries | None` cast; the cached object and runtime
behavior are unchanged. Protected exact-head CI, independent review,
unresolved-thread checks, publication, and dependency order remain release
gates.

Local evidence is 74 focused Python/PyQt wind, kinetics, and Ground/Tee tests;
29 focused React tests; and 8 campaign-manifest tests. TypeScript, focused
zero-warning ESLint, Ruff check/format, pinned MyPy 1.13, Bandit's medium/high
threshold, strict wind module/signature/function budgets, manifest,
documentation, policy, SPEC, and diff gates pass.

## 2026-08-10 Final repaired workflow ancestry in strict ground contract

Draft PR `#4285` remains on `feat/4268-ground-contract`, based on
`feat/4197-capability-observer`. Its original head
`e5bcbd1096d3be1f621a805c9d9f3fd321e375a5` normally incorporates exact
quality-green #4282 head `1e82f15026786ea0b08f78f4c001590ddce9ff39`
second. Ground-contract source and tests had no conflict. Protected CI and
review must pass before the new exact head propagates normally into #4288.

## 2026-08-10 Final repaired strict-contract ancestry in flight transfer

Draft PR `#4288` remains on `feat/4269-flight-ground-transfer`, based on
`feat/4268-ground-contract`. Its original head
`108a841b1378c992defd3c7b7ee263d41a6c8b24` normally incorporates exact
quality-green #4285 head `a93edb4bfd6a8dc9334122cd2ae660983d5bf424`
second. Transfer source and tests had no conflict. Fresh protected CI and review
must pass before any release claim.

## PR #4288 exact repaired-ground propagation

Original child `247215422a6d4b677552955b4923bc609a553259` normally incorporates
exact repaired #4285 parent `e5bcbd1096d3be1f621a805c9d9f3fd321e375a5`
second in the merge on `feat/4269-flight-ground-transfer`. PR #4288
keeps base `feat/4268-ground-contract`; neither branch was rebased, retargeted,
rewritten, or force-pushed. The child retains its qualified cross-runtime
terminal-state/contact transfer and inherits the corrected UTC parser,
canonical `swing_sim` import identity, hosted-mypy manifest repair, and the
complete variation, wind, capability, and campaign ancestry.

This is ancestry propagation, not bounce, skid, roll, terrain response, total
distance, or UI delivery. Protected CI, independent review, and normal stack
collapse remain open.

Exact composed-tree verification is 1,080 Python tests passed with six explicit
optional installed-`tools_core` wheel skips, 107 React files / 662 tests,
26 direct Rust transfer/wire tests, TypeScript, zero-warning ESLint, and the
189-module production build. Ruff check/format passes all 28 changed Python
files; pinned mypy 1.13 passes all 21 changed production files. Campaign
manifest, documentation governance, and diff checks pass. The missing local
wheel remains an explicit installed-package release boundary and is not
misreported as accelerated parity evidence.

## 2026-08-10 Exact repaired #4282 propagation into PR #4285

Original child `788aa547651a3685a363ea401824a5d81477bafb`
normally incorporates exact repaired #4282 parent
`686016196a2f895058b8a566dff103a0fd32cd10` second on
`feat/4268-ground-contract`. PR #4285 keeps base
`feat/4197-capability-observer`; neither branch was rebased, retargeted,
rewritten, or force-pushed. The parent contains merged capability PR #4283,
its exact observer head, and the exact-hosted-mypy manifest repair. The child
retains its strict flight-to-ground contract while acquiring the latest
deterministic variation, workspace, wind, capability, and release ancestry.

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

## 2026-08-10 Exact repaired #4282 propagation into PR #4285 (continued)

The inherited ground descendant passes 1,703 Python tests with two optional `build123d`
skips, 643 React tests across 105 files plus type-check/lint/build, 12 Rust
tests, and 77 real-CPython-3.10 ground and compatibility tests. Ruff/format pass
78 changed Python files; pinned mypy and Bandit pass 52 changed production
files. Manifest, docs, minimum-test, assertions, 500-LOC, changed-file secrets,
Python 3.10 compilation, and diff checks are clean.

The repaired parent explicitly types the Pydantic manifest return. The composed
tree also normalizes contract-checked scalar plot values with `float`, replaces
the command-state `assert` with a runtime invariant error, and documents the
alias finder's intentional optional-probe `B112` security boundary. Protected
current-head CI and required review remain release gates. The exact repaired
ground head is now incorporated into #4288; normal propagation of the reviewed
#4288 head into #4298 is the next ancestry gate.

## 2026-08-09 Ground parent reconciled with corrected wind carrier

Draft PR #4285 keeps base `feat/4197-capability-observer` and normally
incorporates exact corrected wind carrier
`bb101cedd555d07d493aae998b46050c68660cdd`. No branch was rebased,
retargeted, force-pushed, or published. The strict ground contract remains
UI-neutral and fail closed; the merge adds no bounce, skid, roll, terrain
profile, total distance, or presentation claim.

This branch is now the carrier-reconciled ground parent. PR #4288 must merge
the resulting exact #4285 head normally before its transfer logic is retested
or published. That descendant merge will carry both the ground contract and
the explicit observer-to-wind ancestry reconciliation without changing either
PR base.

Focused evidence is 89 ground/compatibility/wind tests on Python 3.11 and 89
on real CPython 3.10.20. Ruff check/format passes 34 focused files; pinned mypy
1.13 passes 23 production modules; the inherited campaign manifest validates
and its nine manifest/parity tests pass. Ground production modules remain
within the 400-line/50-line structural budgets and contain no placeholder
markers.

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
## Inherited current-parent handoff history

 Python/TypeScript SHA-256 parity digests as high-entropy strings. These are
 expected deterministic test outputs, not secrets. Use only exact inline
 `pragma: allowlist secret` annotations on the four digest constants; preserve
 the repository baseline and scanner scope. Commit the annotations and this
 handoff together on the parent capability-observer branch, then re-run the
 protected stack without force-pushing or bypassing the gate. Parent commit
 `49612946138b1021f80c9f8d2a4d06f1610825db` is now propagated into this child
 by a normal merge commit.
## Inherited #4281 handoff history

## 2026-08-11 Wind scalar adapter receives current variation parent

PR `#4281` keeps branch `feat/4199-wind-scalar-adapter` and base
`feat/4144-variation-export-continuation`. Exact published child
`a7dce5f89b483303938f518b74b4028a1c68ba81` is merged first and exact current
parent `e6c7460a01082631565fb9ed48aa32538bd7772c` second. Implementation and test
paths merge automatically: Python/TypeScript scalar-ensemble provenance,
availability, deterministic-scenario, wind-strategy adapter, and strict
module-budget behavior remain intact while the parent contributes current
variation visualization/export, workspace, launch-monitor/D-plane ancestry,
split kinetics, and behavior-preserving Qt return narrowing. Only append-only
handoffs and SPEC conflict textually; both histories are preserved. All 77
focused wind/variation Python tests and all 20 tests in the five focused React
wind/variation suites pass before the merge commit. Pinned Ruff checks/formats
all five changed Python files; pinned MyPy and Bandit accept all three production
files; both 500- and strict 400-LOC, docs, changed-Python policy,
test-contract/assertion, diff, React type-check/lint, and production-build gates
pass. The later campaign-manifest artifact/checker/test exist on neither side of
this pre-manifest stack and are not applicable. Review, ordinary publication,
protected exact-head CI, unresolved threads, dependency order, and release
remain open.

## 2026-08-11 Wind scalar-adapter module-budget repair

The 403-line `variation/wind_strategy_plot_adapter.py` exceeded the mandatory
400-line ceiling. Request/analysis validation now lives in the private
`variation/_wind_strategy_plot_validation.py` collaborator, leaving a 252-line
public adapter and a 213-line validator. A frozen, slotted scenario-expectation
object keeps the private scenario validator at two cohesive parameters; every
production function is at or below four parameters and 50 lines. The public
imports, dataset schema,
variable/stage/category/cohort order, composite row identities, row values,
typed availability, attributes, validation messages, and no-flight-rerun
boundary are unchanged.

The existing 14-test Python adapter suite and 16 focused React wind tests cover
the behavior-preserving extraction; Ruff/format, pinned MyPy 1.13, Bandit, and
the explicit strict line budget pass locally. Protected exact-head CI,
independent review, unresolved-thread checks, ordinary publication, and stack
order remain release gates.

## 2026-08-11 Wind scalar adapter receives reviewed variation parent

PR `#4281` stays on `feat/4199-wind-scalar-adapter`, based on
`feat/4144-variation-export-continuation`. Exact remote child
`247046d55afcad3e6cd4f8029f854856c427f59c` normally merges exact reviewed
parent `3337945699966b63cb5cd8e52d7c3b194315e911` in child-first order,
preserving both histories and every commit. Python/TypeScript scalar-ensemble
provenance, typed availability, deterministic-scenario rules, and wind-strategy
plot-adapter behavior remain unchanged. The parent supplies selected-scatter
export, linked variation views, the kinetics split, Ground/Tee parity,
workspace/toolstrip behavior, and protected Ruff normalization. The only
feature-code conflict referenced obsolete monolithic kinetics and is resolved
to the validated parent `pendulum.sample(...)` facade. This local merge still
requires independent review, fresh exact-head protected CI, unresolved-thread
checks, dependency order, and ordinary publication.
## 2026-08-11 Variation export receives current workspace parent

PR `#4280` keeps branch `feat/4144-variation-export-continuation` and base
`feat/4218-toolstrip-workspace`. Exact published child
`3337945699966b63cb5cd8e52d7c3b194315e911` is merged first and exact current
parent `efbca84095b617b4018732f7802c2da3f0525387` second. Implementation and test
paths merge automatically: selected-scatter CSV parity, typed unavailable
outcomes, bounded accessible tables, linked selection, and all-trial arc
analysis remain intact while the parent contributes current workspace,
launch-monitor/D-plane ancestry, split kinetics, and behavior-preserving Qt
return narrowing. Only append-only handoffs and SPEC conflict textually; both
histories are preserved. All 56 focused Python variation tests and all 10 tests
in the three focused React variation suites pass before the merge commit.
Pinned Ruff checks/formats all five changed Python files; pinned MyPy and Bandit
accept all four production files; docs, 500-LOC, changed-Python policy,
test-contract/assertion, diff, React type-check/lint, and production-build gates
pass. The later campaign-manifest artifact/checker/test exist on neither side of
this pre-manifest stack and are not applicable. Review, ordinary publication,
protected exact-head CI, unresolved threads, dependency order, and release
remain open.
## 2026-08-11 Variation export receives reviewed workspace parent

PR `#4280` stays on `feat/4144-variation-export-continuation`, based on
`feat/4218-toolstrip-workspace`. Exact remote child
`668ba96746f79f7a12e8092161bd610054197f58` normally merges exact reviewed
parent `ccd0e026c580c93038fdf5c59d5d452a85ba27a0` in child-first order,
preserving both histories and every commit. Selected-scatter CSV parity, typed
unavailable outcomes, bounded accessible tables, linked selection, and
all-trial arc analysis remain unchanged. The parent supplies its validated
kinetics split, Ground/Tee parity contracts, protected Ruff normalization, and
workspace/toolstrip implementation. The only feature-code conflict referenced
obsolete monolithic kinetics source and is resolved to the current
`pendulum.sample(...)` façade. Seven duplicate child automation edits do not
match protected Ruff 0.14.10 output and are normalized to the exact reviewed
parent blobs while their commit remains in history. This local reconciliation
still requires independent review, exact-head protected CI, unresolved-thread
checks, dependency order, and ordinary publication.
## 2026-08-11 Workspace child receives current #4203 parent

PR `#4279` keeps branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. Exact published child
`ccd0e026c580c93038fdf5c59d5d452a85ba27a0` is merged first and exact current
parent `7abce9ad767fe8311da66a1e5998b892ea3ca9de` second. Implementation and
test paths merge automatically: the child workspace/toolstrip, visibility,
navigation, playback, and independent-plot behavior remain intact while the
parent contributes current launch-monitor/D-plane ancestry, split kinetics,
and behavior-preserving Qt return narrowing. Only append-only handoffs and
SPEC conflict textually; both histories are preserved. All 142 exact
PR-delta Python tests and all 32 tests in the eight changed React suites pass
before the merge commit. Pinned Ruff checks/formats all 27 changed Python
files; pinned MyPy and Bandit accept all 18 production files; docs, 500-LOC,
changed-Python policy, test-contract/assertion, diff, React type-check/lint, and
production-build gates pass. The later campaign-manifest artifact/checker/test
exist on neither side of this pre-manifest stack and are not applicable.
Review, ordinary publication, protected exact-head CI, unresolved threads,
dependency order,
and release remain open.
## 2026-08-11 Remote workspace history reconciled locally

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. Exact local head
`0b22c401a26c31441a599d8d9b39de123706e7ea` ordinarily merges divergent
remote head `61fe2d556a5413e525d958612ccfd57e65b8d5a2`, preserving every commit in
both histories. The remote automation commit is formatting-only: 15 of 23
paths were already identical through the parent, while its seven unique edits
are normalized back to protected Ruff 0.14.10 output. Its one content conflict
referenced the obsolete pre-split kinetics implementation. The reconciled tree
keeps the current `pendulum.sample(...)` façade path and therefore preserves
the kinetics split, physics, frames, units, imports, schemas, and UI behavior. Workspace/toolstrip,
visibility, navigation, playback, and independent plot behavior are unchanged.
Publication remains blocked on independent review, fresh exact-head protected
CI, unresolved-thread checks, and dependency order.

## 2026-08-11 Workspace child receives hosted MyPy repair

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. Exact child
`7806a16f58e1c6999d32f0127a187fbb21f839a1` normally merges exact published
parent `3796b49e40b677fbac4e05739f8be49f905df2cb`. No feature-code conflict
exists: the inherited production delta is four static `numpy.ndarray` casts,
while workspace/toolstrip behavior and runtime arrays, physics, frames, units,
imports, and UI behavior remain unchanged. This local merge does not satisfy
fresh protected CI, required review, unresolved-thread, dependency, or release
gates.

## 2026-08-11 workspace child receives latest repaired parent

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. A normal merge incorporates exact parent
`0216a547aa79727091a2939b96e779e8ddbd7304` into exact child
`61b7f48b5aeb7d57246b4963da3df086e79cbe15`. There are no feature-code
conflicts: workspace/toolstrip, visibility, navigation, playback, and plot
behavior remain unchanged while the parent kinetics façade/dynamics/series
split and pinned formatting are inherited. This local merge does not satisfy
fresh protected CI, required review, unresolved-thread, dependency, or release
gates.

## 2026-08-11 Append-only SPEC preservation repair

Independent review found that the local #4203 reconciliation omitted four
exact D-plane parent SPEC rows dated 2026-08-10 (versions 1.13.11, 1.13.9,
1.13.7, and 1.13.6). They are restored verbatim in a documentation-only
follow-up. Production behavior, tests, ordered parents, base, and local quality
evidence are unchanged; re-review, normal publication, and protected CI remain
open.

## 2026-08-11 Current D-plane parent reconciliation

PR #4203's published child `217e36dc93d30f79826847f958fbcd10805e58ed`
is being normally merged with exact current D-plane base
`f3363aa88868f6a5c7e9ccfc682a9eca014e86c1`. The base and source branch stay
unchanged. The only conflict is the parent's formatting of an older monolithic
kinetics expression: the split facade keeps its typed pendulum accessor and
inherits the parent's geometry explanation. No launch-monitor, D-plane, or
kinetics behavior is intentionally changed. Four Qt-stub `Any` returns exposed
by the exact MyPy 1.13 delta are also narrowed to their declared `bool`/`str`
contracts without changing values. Focused/full validation, independent
review, normal publication, and protected CI remain open.

## 2026-08-11 Exact-head format completion

Protected CI on #4203 head `7d69a545ae555679f0318940e67c1786626d6794`
failed only Ruff formatting. The pinned 0.14.10 reproduction found eleven
noncompliant changed Python files: four inherited files plus seven altered by
the automated pre-commit repair. The pending repair intentionally changes no
launch-monitor, kinetics, simulation, or UI behavior and requires independent
review plus fresh protected CI before integration.

## 2026-08-11 Hosted MyPy NumPy-return repair

CI Standard run `31477542889`, job `93734652129`, found four
`no-any-return` errors after the kinetics façade extraction at exact PR #4203
head `0216a547aa79727091a2939b96e779e8ddbd7304`. The NumPy stubs widen
`linalg.norm`, `concatenate`, and matrix projection results to `Any` even
though the runtime operations return arrays. The private series/dynamics
modules and public façade now use explicit `cast(np.ndarray, ...)` boundaries.
These casts are static only: array identity, values, dtype, shape, physics,
frames, units, imports, and UI behavior are unchanged. Verification must cover
the complete PR changed-source MyPy set and both focused and full Rate tests.
Current local evidence is 102 changed-source MyPy files, 28 focused tests, and
all 701 Rate tests passing; complete-delta Ruff/format and Bandit are also
green.

## 2026-08-11 Kinetics façade extraction

The strict current-head delta reproduces a protected size failure that the
PR-base comparison hides: `kinetics.py` is changed versus `HEAD~1` and is 646
LOC, above the ungrandfathered 500-LOC maximum. The public module is now a
222-LOC run adapter/facade; its pure double-pendulum dynamics are in the
205-LOC `_kinetics_dynamics.py`, and the immutable `KineticsSeries`/DbC
contract is in the 131-LOC `_kinetics_series.py`. Existing public imports and
the `_reaction_forces` seam are the same implementation objects. A RED-first
identity test plus the existing physics, energy, force, parity-fixture,
presentation, and PyQt tests protect behavior. No physics, frame, SI-unit,
schema, UI, or stack-order change is intended; the commit remains local only.
Focused kinetics/presentation/PyQt verification is 28 passing tests; the
complete Rate-of-Closure Python regression suite is 701 passing tests.

## 2026-08-11 #4203 pinned-Ruff formatting repair

No material handoff change: current-head CI Standard run `31468208320`, job
`93705508050`, identified eight files that differ from repository-pinned Ruff
0.14.10 output. This commit changes formatting only; Rate physics, behavior,
frames, persistence schemas, PyQt/React contracts, stack bases, and dependency
order remain unchanged. Fresh protected CI and review are still required.

## 2026-08-10 Ground/Tee parity child receives repaired parent

Ready PR `#4325` keeps branch `feat/4143-tee-parity-fixture` and base
`feat/4181-launch-monitor-registry`. It normally merges exact repaired parent
head `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76` without any production/test
code conflict or history rewrite. Shared parity and rendered-evidence
contracts remain intact; fresh protected CI, review, dependency order, and
#4143 release remain open.

## 2026-08-10 #4143 Python/React Golden Ball-Setup Parity

The isolated `feat/4143-tee-parity-fixture` child begins at exact draft PR
#4203 head `31cbc007d4c85b5479b7cd0fb0969124eab2af67`. One versioned JSON fixture now
drives both Python and React ball-support tests with explicit SI units and the
ground-plane-to-ball-bottom height reference. The shared cases cover Driver
and non-Driver defaults, user overrides, Ground zero effective height,
derived ball-center geometry and serialization, negative/NaN/infinite height
rejection, and backward-compatible migration of a legacy run without
`ball_setup`.

Evidence is 18 passing Python tee/parity tests and 24 passing React
tee/persistence/parity tests, plus green TypeScript, ESLint, Vite production
build, Ruff check, and Ruff format. Production model and UI code are unchanged.
Recorded visual evidence now adds 1600 x 1200 Playwright Driver/Tee and
rerun-Ground captures plus 1400 x 900 hidden-window PyQt captures. The browser
run asserts control state, diagram geometry, and zero console/page errors; the
desktop run asserts canonical center/artist state, nonblank output, and
different Ground/Tee digests without pixel absolutes. The exact artifacts and
versioned manifests are in
`C:\Users\diete\AppData\Local\Temp\rate-4143-visual-evidence-8050eeba`.
Do not close #4143: protected current-head CI/review and release to `main`
remain. This exact parent predates the strict campaign manifest on a divergent
branch, so no downstream manifest was copied into this bounded child.

## 2026-08-10 Wind scalar child receives second repaired parent

PR `#4281` retains branch `feat/4199-wind-scalar-adapter` and base
`feat/4144-variation-export-continuation`. It normally merges exact repaired
parent `b90e5021a59e2081415b51ef29fbed06377bc201` without scalar-adapter code
conflict or history rewrite. Fresh child CI, review, and dependency order
remain required.
Post-reconciliation evidence is 25 focused D-plane/impact tests plus docs,
changed-file-size, and whitespace gates.

## 2026-08-10 Variation export child receives exact workspace parent

PR `#4280` retains branch `feat/4144-variation-export-continuation`, base
`feat/4218-toolstrip-workspace`, and original child first parent
`f90836e342efc8be624739802375af2876d11e5f`. Exact parent
`6717e9e09d507dbc24bedb36177f1cdf0b4fd90b` merges normally as the second
parent. All variation visualization/export source merged without conflict:
selected scatter CSV, typed unavailable rows, accessible bounded raw tables,
linked trials, focused PyQt widgets, and React parity remain additive with the
parent workspace/toolstrip/playback/plot/navigation repairs. SPEC 1.14.12 is
the unique combined child entry. Staged review exposed a stale linked-trial
selection when a smaller rerun replaced the active result. Corrected PyQt
views clear result-local selection before repopulating and validate all public
setter indices; React clears on result identity change and shares only bounded
selections with linked views. Thirty-seven focused variation tests and the
complete Rate/shared-swing/golf-club matrix pass 1,528 tests with two explicit
optional build123d skips; all 546 React tests across 90 files pass;
TypeScript, ESLint, and production build are green. Exact-parent
Ruff/Ruff-format/Black, pinned MyPy 1.13 across four Python production modules,
Bandit, file-size, docs, minimum-test, assertion, detect-secrets, diff, real
CPython 3.10.20 compilation, and 30 compatibility regressions pass. There is
no Rust delta from the exact parent. Independent staged re-review found no
actionable findings, including the first replacement-render React boundary.
Protected CI/review remain release gates after an ordinary guarded push.

## 2026-08-10 Variation child receives second repaired workspace parent

PR `#4280` retains branch `feat/4144-variation-export-continuation` and base
`feat/4218-toolstrip-workspace`. It normally merges exact repaired parent head
`61b7f48b5aeb7d57246b4963da3df086e79cbe15` without feature-code conflict or
history rewrite. Variation/export behavior remains unchanged; fresh child CI,
review, and dependency order remain required.
Post-reconciliation evidence is 25 focused D-plane/impact tests plus docs,
changed-file-size, and whitespace gates.

## 2026-08-10 Wind scalar child receives exact variation parent

PR `#4281` keeps branch `feat/4199-wind-scalar-adapter`, base
`feat/4144-variation-export-continuation`, and original child first parent
`cf52529b1e68479321bb93b1be3d59c77f782008`. Exact parent
`8bcc49fc4e16e5e43be0b7f0f03c3017d5b79d0c` merges normally second. The
child's scalar-ensemble wire contract, provenance-preserving wind rows,
availability accounting, and plot adapter remain additive with the parent's
variation visualization/export and result-local selection repairs. Python and
TypeScript reject wind scenarios unless their per-trial provenance is exact,
shear/turbulence are zero, seed is zero, and gusts are empty. SPEC 1.14.13 is
the unique combined child entry. Twenty-seven focused Python tests, the
1,549-test complete Rate/shared-swing/golf-club matrix with two explicit
optional build123d skips, and all 556 React tests across 92 files pass.
TypeScript, ESLint, production build, Ruff/format, pinned MyPy, Bandit, size,
docs, minimum-test, assertions, detect-secrets, diff, and CPython 3.10 compile
gates pass. Independent re-review found no actionable findings after checking
the exact provenance derivation and all five deterministic-scenario
regressions; protected CI/review remain release gates.

## 2026-08-10 Variation child timestamp propagation

The #4280 continuation now carries exact parent
`05383d333b6fd87eaf5e37305476f50b505c2c2e` through a normal merge while
preserving its original base and complete variation export/accessibility
implementation. The inherited workspace parser now has one strict UTC grammar
and consistent zero- through six-digit fractional-second behavior across
Python 3.10-3.12. The child variation behavior is unchanged.

SPEC 1.14.10 records the parent compatibility repair and 1.14.11 records the
child variation delta. Keep this handoff, the root and campaign handoffs, and
the reconciled SPEC in the merge commit. The combined evidence is `778 passed`
for the Rate suite, `27 passed` on real CPython 3.10.20, and `1 file / 8 tests`
for focused React variation, with TypeScript, focused ESLint, Ruff, format, and
pinned mypy 1.13 clean. Normal publication and dependency-ordered propagation
remain separate gates.

## 2026-08-10 Workspace child receives second repaired parent

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. It normally merges exact repaired parent
head `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76`; there is no feature-code
conflict, rebase, retarget, or history rewrite. The inherited explicit ndarray
boundaries do not alter numerical behavior. Fresh child CI, review, and
dependency order remain required.
Verification after reconciliation is 25 focused D-plane/impact tests plus
docs-governance, changed-file-size, and whitespace gates.

## 2026-08-10 Workspace child receives exact repaired parent

PR `#4279` keeps branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. Its original head
`05383d333b6fd87eaf5e37305476f50b505c2c2e` normally merges exact parent
`31cbc007d4c85b5479b7cd0fb0969124eab2af67` in that order. The combined
shell retains the child's top toolstrip, persistent module visibility,
reorderable workspace, granular playback, path trail, and independent plot
controls. Parent repairs remain additive: the workspace mixin reuses the
canonical navigation constants, and the simulation controls reuse the single
persisted `ImpactLayerControls` checkbox map without duplicating UI state.

Focused and broad verification is green: 1,339 Python tests pass with six
explicit optional/Rust-wheel skips; 545 React tests pass, as do type-check,
lint, and production build. Forty navigation/simulation GUI tests pass after
the final formatting change. Exact-parent Ruff/Ruff-format/Black, pinned MyPy
1.13 for 18 production modules, Bandit, file-size, docs, minimum-test,
assertion, detect-secrets, diff, CPython 3.10.20 compilation/import, and 30
compatibility regressions pass. There is no Rust delta from the exact parent.
The merge commit must keep this file, the root handoff, campaign handoff, and
`SPEC.md` together. Independent staged review found no actionable findings
after 76 additional focused PyQt/navigation/workspace tests; exact-head
protected CI and required review remain release gates after a guarded ordinary
push.

## 2026-08-10 Workspace fractional-timestamp compatibility repair

The Python 3.10 lane on descendant PR #4281 revealed that the strict workspace
ordering test reached `datetime.fromisoformat` with a one-digit fractional
second. Python 3.10 rejects that spelling before comparing timestamps, although
newer supported interpreters accept it. PR #4279 is the earliest owning branch.
The workspace validator now enforces one anchored UTC grammar and parses every
losslessly representable zero- to six-digit fraction consistently, rejecting
greater precision rather than allowing interpreter-dependent truncation. It
does not modify the persisted spelling, UTC requirement, schema, or ordering
rule.

Evidence is `778 passed` for the full Rate suite, `45 passed` across the
compatibility and workspace document tests on the local interpreter, and
`27 passed` under real CPython 3.10.20. Ruff, format, pinned mypy 1.13, docs
governance, and the 400-line budget pass. Keep this handoff, the root handoff,
campaign handoff, and SPEC entry in the same implementation commit; verify the
exact remote head before a normal push and propagate through later descendants.

## 2026-08-09 Universal Variation Visualization Continuation

Branch `feat/4144-variation-export-continuation` remains based on
`feat/4218-toolstrip-workspace` and now contains exact corrected parent
`3f67ed466fefc8991db9c4409f921f25e1c37142` through the normal merge containing
this handoff. The core PyQt6/React variation workspace has selectable
input/contact/impact/shot scatter, a scatter matrix with marginals, landing
dispersion, and an all-trial 3D swing-arc overlay with reference trace,
principal spread, RMS variability, quiet zones, filtering, and linked trial
selection.

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

The current local continuation normally incorporates exact corrected parent
`3f67ed466fefc8991db9c4409f921f25e1c37142` while preserving #4280's
`feat/4218-toolstrip-workspace` base. Its current focused evidence is `46`
Python 3.11 tests and `19` real-Python-3.10.20 tests with PyQt6 installed,
including the merged compatibility and variation widget contracts. React
passes `1 file / 8 tests`, TypeScript, and focused zero-warning ESLint. Ruff,
format, pinned mypy 1.13, documentation governance, ancestry/SPEC assertions,
and diff checks pass. SPEC 1.14.11 is the unique child entry for the
scatter-export and accessible-row-table source change. This propagation is
local only and has not been pushed.

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
on exact corrected draft PR #4280 head
`38ed58cab96842a3007e76a855db83ee2452b8fd`. That parent is incorporated by a
normal local merge while preserving base
`feat/4144-variation-export-continuation`; no branch was rebased, retargeted,
force-pushed, or published by this continuation.
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

The original broad local gates remain green: 906 Python/PyQt/shared-swing tests
passed with one expected optional-Rust skip and 15 existing warnings; 91 React
files / 555 tests passed. After corrected-parent propagation, focused evidence
is `49 passed` on Python 3.11 and `49 passed` on real CPython 3.10.20 with
PyQt6/scientific dependencies present, plus `2 React files / 10 tests`,
TypeScript, and zero-warning focused ESLint. Ruff check/format passes all four
child Python files; pinned mypy 1.13 passes both production modules and the
wind-adapter test after its status helper received an exact `Literal` type.
The function budget passed at that historical head, but the 403-line wind
adapter did not satisfy the mandatory below-400 module ceiling; the 2026-08-11
repair above is the first point at which that module claim is true. SPEC
1.14.11 uniquely records the child change above parent 1.14.10. This
contract/adapter does not complete #4199: the next slice still needs background
execution, progress and cancellation, PyQt/React scatter/strategy UI,
persistence, and export wiring.

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

## 2026-08-09 Workspace Python 3.10 compatibility completion

The #4279 continuation now contains exact corrected parent
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` through normal local merge
`a340fabefa443d47325c5538f342683b38c01ade`, without rebasing, retargeting,
force-pushing, or publication. The workspace command and view enums use the
shared `StrEnum` runtime compatibility contract with native enum typing kept
under `TYPE_CHECKING`; `_workspace_validation` uses the shared `UTC` value.
Stable identifiers, enum values, UTC serialization, schemas, and behavior are
unchanged.

The combined regression inspects the runtime branch for all nine inherited
and child string-enum modules plus both UTC modules and directly executes the
three child workspace modules. Focused Python 3.11 evidence is `126 passed`.
Real CPython 3.10.20 evidence is `14 passed` plus successful dotted imports of
10 Rate/shared targets. Ruff check/format is clean, and pinned mypy 1.13 passes
all 11 affected production files under the changed-file CI contract. Keep the
root, tool, campaign handoffs and SPEC 1.14.9 entry in the implementation
commit; protected CI and review remain required after a normal push.

## 2026-08-09 Workspace stack parent propagation

Draft PR #4279 keeps base `feat/4181-launch-monitor-registry`. Exact published
parent head `08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` is carried by a normal
merge without rebasing, retargeting, force-pushing, or publishing this local
continuation. Source changes applied cleanly; overlapping handoff/specification
history was reconciled monotonically. The result retains the parent's facade
and Python 3.10 repairs plus the child's complete File/View/Tools registry,
strict workspace document, matched PyQt6/React toolstrips, module
visibility/order, playback, and independent plot controls.

The reconciled ancestry passes `126` focused Python/PyQt tests and `8 React
files / 32 tests`. Ruff check/format passes all 28 relevant Python files, and
CI-pinned mypy 1.13 passes the 18 changed production files and both facade
tests. Mypy caught an untyped Qt `isChecked()` result in the child's
`legend_visible()` boundary; the method now returns an explicit `bool` without
changing runtime behavior. The affected simulation GUI rerun is `29 passed`;
documentation governance and diff checks pass. Keep this handoff, the root
handoff, the campaign handoff, and `SPEC.md` together in the merge commit.

## 2026-08-10 Second D-Plane Parent Repair Propagation

Draft PR `#4203` keeps base `feat/4189-dplane` and normally merges exact
repaired parent head `7d8d2f06dc797021d01939691e58f8425b652b33` without
rewriting either branch. The inherited change makes the two private D-plane
NumPy return boundaries explicit for pinned MyPy while preserving numerical
semantics, frames, schemas, and UI behavior. The parent quality gate is green;
child protected CI, review, dependency order, and #4189 closure remain open.
Post-reconciliation evidence is 25 focused D-plane/impact tests plus docs,
changed-file-size, and whitespace gates. Local pinned-type evidence remains the
parent's green hosted quality gate: Windows MyPy 1.15 encounters incompatible
installed NumPy stub syntax and WSL is unavailable with `E_FAIL`, so neither
local attempt is represented as green.

## 2026-08-10 Launch-Registry Child Propagation

Draft PR `#4203` (`feat/4181-launch-monitor-registry`) retains its
`feat/4189-dplane` base and normally merges original child head
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` with exact parent head
`b443fdbed7064c5db0320106013c8413e3e24356`.

The reconciliation preserves the child branch's responsive
`SimulationViewControlsMixin` and delegates its persisted D-plane layer
checkboxes to the parent's `ImpactLayerControls` helper. The compatibility
mapping and helper mapping are the same object, which prevents duplicate state
while retaining existing UI automation and persistence behavior. Both PyQt
modules remain below the 500-line limit.

The original child already exceeded the protected budget in swing sources,
the plotting catalog, and the PyQt main window. Focused modules now own the
triple-pendulum model, immutable plotting-variable contract, and versioned tab
state. Compatibility imports remain the same objects, while the former
monoliths fall to 282, 459, and 494 lines. Focused evidence is green across 36
PyQt simulation/layout tests, 38 plotting/navigation tests, and 21 simulation
source/export tests. Combined verification is green across 1,249 Python tests
with six explicit skips, 521 React tests and all web gates, 12 Rust tests,
real CPython 3.10 checks, scoped Ruff/Black/pinned MyPy, and repository
governance/security checks. The changed-file 500-line gate passes all 107
candidate files. A full-tree audit still reports untouched `kinetics.py` and
`torque_profile_panel.py`; neither differs in this propagation. Independent
staged review found no actionable findings after 95 additional focused tests.
Current-head protected CI and required repository review remain release gates.

## 2026-08-09 Python 3.10 enum import repair

PR #4203 now owns the earliest fix for seven inherited Rate/shared swing
modules that imported Python 3.11's `enum.StrEnum` directly. Runtime code uses
the established shared compatibility helper while mypy sees the native enum
only inside `TYPE_CHECKING`. This preserves wire values and enum semantics on
supported interpreters and lets the repository's hosted Python 3.10 lane
collect the modules. The focused evidence is 64 tests, Ruff/format, pinned
mypy 1.13 across eight changed files, and a real CPython 3.10.20 source/runtime
probe. The exact published #4203 head is
`ab7de5a47977417e02926c3fbc7476002e82b690`; propagate it through the existing
stack without changing bases.

A follow-up scan found the torque-profile controller importing
`datetime.UTC` directly. That parent-owned Python 3.11 boundary now uses
`shared.python.compatibility.UTC`; persisted timestamps and torque-profile
behavior are unchanged. Re-run the focused torque-profile UI tests and the
real Python 3.10 compatibility probe before propagation.

## 2026-08-09 Launch-registry parent CI repair

PR #4203 exact-head run `31199764932` reached the Python test lanes but failed
during Linux collection, before behavioral assertions. The two in-package
flight/solver facade tests were collected as `src.shared...` modules while
their absolute aliases crossed into the editable `shared...` namespace. They
now import their sibling facade package relatively, preserving the pinned
public API contract and production behavior. Reproduce with pytest
`--import-mode=importlib`; keep the separate Rust missing-`libpython3.11`
failure classified as runner infrastructure. Publish normally and propagate
the repaired parent through the existing stack without changing PR bases.
Verification is `12 passed` on Windows and `12 passed` on WSL Python 3.11
under importlib collection; Ruff/format and exact mypy 1.13 pass for both
changed modules. The dataclass metadata assertion remains active behind an
explicit test-only `Any` introspection boundary.

## 2026-08-10 PR #4202 D-plane ndarray typing repair

On top of verified published head
`b443fdbed7064c5db0320106013c8413e3e24356`, the two private D-plane ndarray
helpers now bind NumPy results to explicit ndarray locals before returning
them. This closes the exact MyPy 1.13 `no-any-return` findings from CI Standard
run `31384810375`, job `93442745760`, without changing numerical operations,
DbC validation, reference frames, persistence, rendering, or public contracts.

The exact MyPy failure was reproduced before editing and is green after the
repair. Twenty-four focused D-plane and impact integration tests pass, along
with seven metadata/pre-push contract tests, scoped Ruff, Ruff format, Black,
docs governance, minimum-test, module-size, changed-file-size, and diff checks.
Three unrelated CI-workflow contract tests still expect later toolcache/env
steps not present on this older branch; no workflow file is in scope. This
local commit does not push, retarget, change draft state, or claim issue
completion; protected CI, review, and parent-first release order remain
mandatory.

## 2026-08-10 D-Plane Child Propagation

Draft PR `#4202` (`feat/4189-dplane`) retains its
`feat/4162-wedge-impact-visualization` base and normally merges original child
head `b4abec03bccfbdd87ddf91427159c5c2332c21dd` with exact parent head
`6704a3e541a3e74c28b4a284530d1a21269dd340`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the frame-explicit D-plane contract.

Persisted D-plane layer controls now live in a focused helper so
`simulation_view.py` again satisfies the protected 500-line budget while
retaining the existing UI-automation compatibility seam. Combined verification
is green: 93 focused and 825 scoped Python tests (two optional `build123d`
skips), 360 React tests and all web gates, real CPython 3.10.20
compilation/UTC, scoped Ruff/Black/MyPy, and repository governance gates. The
exact parent's 12 unchanged `swing-core` tests remain applicable because this
child has no Rust delta. The 17-error broad MyPy Qt/NumPy baseline in 11
untouched files remains separate. Protected CI and required review remain
release gates.

## 2026-08-10 Impact-Visualization Child Propagation

Draft PR `#4179` (`feat/4162-wedge-impact-visualization`) retains its
`feat/4166-wedge-turf-physics` base and normally merges original child head
`0eb804e70887c788421332369e42792411aff55a` with exact parent head
`bfa83aedc88ead380babc73a699377d98b971006`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the exact-event scene contract.

Combined verification is green: 58 focused and 739 scoped Python tests (two
optional `build123d` skips), 347 React tests and all web gates, real CPython
3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository governance
gates. The exact parent's 12 unchanged `swing-core` tests remain applicable
because this child has no Rust delta. The 17-error broad MyPy Qt/NumPy baseline
in 11 untouched files remains separate. Protected CI and required review
remain release gates.

## 2026-08-10 Turf-Physics Child Propagation

Draft PR `#4178` (`feat/4166-wedge-turf-physics`) retains its
`feat/4161-wedge-ground-clearance` base and normally merges original child
head `aaae3f73e17dbfaad5cca1dc6f49559b3aebe9d5` with exact parent head
`9ea93e92563280ec34bca682ad44d7409edd7a02`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the provenance-gated turf model.

Combined verification is green: 56 focused and 732 scoped Python tests (two
optional CAD-dependency skips), real CPython 3.10.20 checks, scoped
Ruff/Black/MyPy, and repository governance gates. The unchanged TypeScript and
Rust surfaces retain the exact parent's green 345 React and 12 Rust test
evidence. The 17-error broad MyPy Qt/NumPy baseline in 11 untouched files
remains separate. Protected CI and required review remain release gates.

## 2026-08-10 Ground-Clearance Child Propagation

Draft PR `#4174` (`feat/4161-wedge-ground-clearance`) keeps its
`feat/4163-impact-inspector` base and normally merges original child head
`880a6465fc872cf3d6650283db154ddc41793a31` with exact parent head
`9ddaff3b6bca542fd7a2befc7d7b0ae53910a60a`. The inherited Python 3.10 UTC
repair and AST guard remain intact beside the ground-clearance analysis.

Combined verification is green: 56 focused and 703 scoped Python tests (two
optional `build123d` skips), 345 React tests and all web gates, 12 Rust tests,
real CPython 3.10.20 compile/UTC checks, scoped Ruff/Black/MyPy, and repository
governance checks. The existing 17-error broad MyPy Qt/NumPy baseline across
11 untouched files is documented, not expanded. Current-head protected CI and
required review remain pending.

## 2026-08-10 Python 3.10 Repair Propagation

Draft child PR `#4173` (`feat/4163-impact-inspector`) retains its
`feat/4144-variation-visualizations` base and normally merges original child
head `3c43955aaeb3964ff8c3ef2748d626baae518b76` with exact parent head
`22b66b560652b78de84141344c4ddd9a92a83b26`. This carries the shared
Python 3.10-compatible UTC export and the source-wide AST guard into the wedge
impact inspector without changing the persistence schema or user-visible
timestamp format.

Combined-stack verification is green across 63 focused Python tests, all 562
Rate tests, all 334 React tests, TypeScript/ESLint/Vite gates, 12 `swing-core`
tests, real CPython 3.10.20 compile/UTC checks, Ruff/Black, focused pinned MyPy
1.13, and repository governance checks. The broad MyPy sweep retains 17
pre-existing Qt/NumPy typing findings in 11 untouched files. The PR must remain
draft until its exact-head protected checks complete and required review
approves. Do not retarget, rewrite, force-push, admin-merge, or count
infrastructure failures as passing evidence.

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

Run Ruff format/check and pinned delta mypy on every changed Python file. Do not
rewrite parents, force-push, bypass checks, infer unavailable values, or close
#4201 until exact protected merge, installed package, downstream pin, science,
performance, accessibility, documentation, and rollback evidence all exist.

## 2026-08-11 parent sections preserved during #4282 reconciliation

## 2026-08-11 Workspace child receives hosted MyPy repair

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. Exact child
`7806a16f58e1c6999d32f0127a187fbb21f839a1` normally merges exact published
parent `3796b49e40b677fbac4e05739f8be49f905df2cb`. No feature-code conflict
exists: the inherited production delta is four static `numpy.ndarray` casts,
while workspace/toolstrip behavior and runtime arrays, physics, frames, units,
imports, and UI behavior remain unchanged. This local merge does not satisfy
fresh protected CI, required review, unresolved-thread, dependency, or release
gates.

## 2026-08-11 workspace child receives latest repaired parent

PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. A normal merge incorporates exact parent
`0216a547aa79727091a2939b96e779e8ddbd7304` into exact child
`61b7f48b5aeb7d57246b4963da3df086e79cbe15`. There are no feature-code
conflicts: workspace/toolstrip, visibility, navigation, playback, and plot
behavior remain unchanged while the parent kinetics façade/dynamics/series
split and pinned formatting are inherited. This local merge does not satisfy
fresh protected CI, required review, unresolved-thread, dependency, or release
gates.

## 2026-08-11 Append-only SPEC preservation repair

Independent review found that the local #4203 reconciliation omitted four
exact D-plane parent SPEC rows dated 2026-08-10 (versions 1.13.11, 1.13.9,
1.13.7, and 1.13.6). They are restored verbatim in a documentation-only
follow-up. Production behavior, tests, ordered parents, base, and local quality
evidence are unchanged; re-review, normal publication, and protected CI remain
open.

## 2026-08-11 Current D-plane parent reconciliation

PR #4203's published child `217e36dc93d30f79826847f958fbcd10805e58ed`
is being normally merged with exact current D-plane base
`f3363aa88868f6a5c7e9ccfc682a9eca014e86c1`. The base and source branch stay
unchanged. The only conflict is the parent's formatting of an older monolithic
kinetics expression: the split facade keeps its typed pendulum accessor and
inherits the parent's geometry explanation. No launch-monitor, D-plane, or
kinetics behavior is intentionally changed. Four Qt-stub `Any` returns exposed
by the exact MyPy 1.13 delta are also narrowed to their declared `bool`/`str`
contracts without changing values. Focused/full validation, independent
review, normal publication, and protected CI remain open.

## 2026-08-11 Exact-head format completion

Protected CI on #4203 head `7d69a545ae555679f0318940e67c1786626d6794`
failed only Ruff formatting. The pinned 0.14.10 reproduction found eleven
noncompliant changed Python files: four inherited files plus seven altered by
the automated pre-commit repair. The pending repair intentionally changes no
launch-monitor, kinetics, simulation, or UI behavior and requires independent
review plus fresh protected CI before integration.

## 2026-08-11 Hosted MyPy NumPy-return repair

CI Standard run `31477542889`, job `93734652129`, found four
`no-any-return` errors after the kinetics façade extraction at exact PR #4203
head `0216a547aa79727091a2939b96e779e8ddbd7304`. The NumPy stubs widen
`linalg.norm`, `concatenate`, and matrix projection results to `Any` even
though the runtime operations return arrays. The private series/dynamics
modules and public façade now use explicit `cast(np.ndarray, ...)` boundaries.
These casts are static only: array identity, values, dtype, shape, physics,
frames, units, imports, and UI behavior are unchanged. Verification must cover
the complete PR changed-source MyPy set and both focused and full Rate tests.
Current local evidence is 102 changed-source MyPy files, 28 focused tests, and
all 701 Rate tests passing; complete-delta Ruff/format and Bandit are also
green.

## 2026-08-11 Kinetics façade extraction

The strict current-head delta reproduces a protected size failure that the
PR-base comparison hides: `kinetics.py` is changed versus `HEAD~1` and is 646
LOC, above the ungrandfathered 500-LOC maximum. The public module is now a
222-LOC run adapter/facade; its pure double-pendulum dynamics are in the
205-LOC `_kinetics_dynamics.py`, and the immutable `KineticsSeries`/DbC
contract is in the 131-LOC `_kinetics_series.py`. Existing public imports and
the `_reaction_forces` seam are the same implementation objects. A RED-first
identity test plus the existing physics, energy, force, parity-fixture,
presentation, and PyQt tests protect behavior. No physics, frame, SI-unit,
schema, UI, or stack-order change is intended; the commit remains local only.
Focused kinetics/presentation/PyQt verification is 28 passing tests; the
complete Rate-of-Closure Python regression suite is 701 passing tests.

## 2026-08-11 #4203 pinned-Ruff formatting repair

No material handoff change: current-head CI Standard run `31468208320`, job
`93705508050`, identified eight files that differ from repository-pinned Ruff
0.14.10 output. This commit changes formatting only; Rate physics, behavior,
frames, persistence schemas, PyQt/React contracts, stack bases, and dependency
order remain unchanged. Fresh protected CI and review are still required.

## 2026-08-10 Ground/Tee parity child receives repaired parent

Ready PR `#4325` keeps branch `feat/4143-tee-parity-fixture` and base
`feat/4181-launch-monitor-registry`. It normally merges exact repaired parent
head `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76` without any production/test
code conflict or history rewrite. Shared parity and rendered-evidence
contracts remain intact; fresh protected CI, review, dependency order, and
#4143 release remain open.

## 2026-08-10 #4143 Python/React Golden Ball-Setup Parity

The isolated `feat/4143-tee-parity-fixture` child begins at exact draft PR
#4203 head `31cbc007d4c85b5479b7cd0fb0969124eab2af67`. One versioned JSON fixture now
drives both Python and React ball-support tests with explicit SI units and the
ground-plane-to-ball-bottom height reference. The shared cases cover Driver
and non-Driver defaults, user overrides, Ground zero effective height,
derived ball-center geometry and serialization, negative/NaN/infinite height
rejection, and backward-compatible migration of a legacy run without
`ball_setup`.

Evidence is 18 passing Python tee/parity tests and 24 passing React
tee/persistence/parity tests, plus green TypeScript, ESLint, Vite production
build, Ruff check, and Ruff format. Production model and UI code are unchanged.
Recorded visual evidence now adds 1600 x 1200 Playwright Driver/Tee and
rerun-Ground captures plus 1400 x 900 hidden-window PyQt captures. The browser
run asserts control state, diagram geometry, and zero console/page errors; the
desktop run asserts canonical center/artist state, nonblank output, and
different Ground/Tee digests without pixel absolutes. The exact artifacts and
versioned manifests are in
`C:\Users\diete\AppData\Local\Temp\rate-4143-visual-evidence-8050eeba`.
Do not close #4143: protected current-head CI/review and release to `main`
remain. This exact parent predates the strict campaign manifest on a divergent
branch, so no downstream manifest was copied into this bounded child.

## 2026-08-10 Wind scalar child receives second repaired parent

PR `#4281` retains branch `feat/4199-wind-scalar-adapter` and base
`feat/4144-variation-export-continuation`. It normally merges exact repaired
parent `b90e5021a59e2081415b51ef29fbed06377bc201` without scalar-adapter code
conflict or history rewrite. Fresh child CI, review, and dependency order
remain required.
Post-reconciliation evidence is 25 focused D-plane/impact tests plus docs,
changed-file-size, and whitespace gates.

## 2026-08-10 Variation export child receives exact workspace parent

PR `#4280` retains branch `feat/4144-variation-export-continuation`, base
`feat/4218-toolstrip-workspace`, and original child first parent
`f90836e342efc8be624739802375af2876d11e5f`. Exact parent
`6717e9e09d507dbc24bedb36177f1cdf0b4fd90b` merges normally as the second
parent. All variation visualization/export source merged without conflict:
selected scatter CSV, typed unavailable rows, accessible bounded raw tables,
linked trials, focused PyQt widgets, and React parity remain additive with the
parent workspace/toolstrip/playback/plot/navigation repairs. SPEC 1.14.12 is
the unique combined child entry. Staged review exposed a stale linked-trial
selection when a smaller rerun replaced the active result. Corrected PyQt
views clear result-local selection before repopulating and validate all public
setter indices; React clears on result identity change and shares only bounded
selections with linked views. Thirty-seven focused variation tests and the
complete Rate/shared-swing/golf-club matrix pass 1,528 tests with two explicit
optional build123d skips; all 546 React tests across 90 files pass;
TypeScript, ESLint, and production build are green. Exact-parent
Ruff/Ruff-format/Black, pinned MyPy 1.13 across four Python production modules,
Bandit, file-size, docs, minimum-test, assertion, detect-secrets, diff, real
CPython 3.10.20 compilation, and 30 compatibility regressions pass. There is
no Rust delta from the exact parent. Independent staged re-review found no
actionable findings, including the first replacement-render React boundary.
Protected CI/review remain release gates after an ordinary guarded push.

## 2026-08-10 Variation child receives second repaired workspace parent

PR `#4280` retains branch `feat/4144-variation-export-continuation` and base
`feat/4218-toolstrip-workspace`. It normally merges exact repaired parent head
`61b7f48b5aeb7d57246b4963da3df086e79cbe15` without feature-code conflict or
history rewrite. Variation/export behavior remains unchanged; fresh child CI,
review, and dependency order remain required.
Post-reconciliation evidence is 25 focused D-plane/impact tests plus docs,
changed-file-size, and whitespace gates.

## 2026-08-10 Wind scalar child receives exact variation parent

PR `#4281` keeps branch `feat/4199-wind-scalar-adapter`, base
`feat/4144-variation-export-continuation`, and original child first parent
`cf52529b1e68479321bb93b1be3d59c77f782008`. Exact parent
`8bcc49fc4e16e5e43be0b7f0f03c3017d5b79d0c` merges normally second. The
child's scalar-ensemble wire contract, provenance-preserving wind rows,
availability accounting, and plot adapter remain additive with the parent's
variation visualization/export and result-local selection repairs. Python and
TypeScript reject wind scenarios unless their per-trial provenance is exact,
shear/turbulence are zero, seed is zero, and gusts are empty. SPEC 1.14.13 is
the unique combined child entry. Twenty-seven focused Python tests, the
1,549-test complete Rate/shared-swing/golf-club matrix with two explicit
optional build123d skips, and all 556 React tests across 92 files pass.
TypeScript, ESLint, production build, Ruff/format, pinned MyPy, Bandit, size,
docs, minimum-test, assertions, detect-secrets, diff, and CPython 3.10 compile
gates pass. Independent re-review found no actionable findings after checking
the exact provenance derivation and all five deterministic-scenario
regressions; protected CI/review remain release gates.

## 2026-08-10 Variation child timestamp propagation

The #4280 continuation now carries exact parent
`05383d333b6fd87eaf5e37305476f50b505c2c2e` through a normal merge while
preserving its original base and complete variation export/accessibility
implementation. The inherit…4250 tokens truncated…7417e02926c3fbc7476002e82b690`; propagate it through the existing
stack without changing bases.

A follow-up scan found the torque-profile controller importing
`datetime.UTC` directly. That parent-owned Python 3.11 boundary now uses
`shared.python.compatibility.UTC`; persisted timestamps and torque-profile
behavior are unchanged. Re-run the focused torque-profile UI tests and the
real Python 3.10 compatibility probe before propagation.

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
