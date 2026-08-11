# AGENT_HANDOFF — Tools

## 2026-08-11 local flight-through-regional-ground pipeline candidate

Local branch `feat/4271-flight-regional-ground-pipeline` starts from exact
published PR #4359 head `e53c6fb1bd273292c02085ee5d0a2b5497820871`
without modifying its published worktree. Audit found that the existing
regional envelope is exact only after `SETTLED_TO_SKID`; bounce time/event
limits and no-recontact cannot be mapped to its failure enum honestly.

The child adds `execute_regional_ground_from_flight`, which validates exact
flight, launch, transfer, plan, and regional-option records, capture speed, and
launch-relative plan/base-surface equality before bounce physics. It composes
only the existing flight-to-bounce and regional-ground executors. The new
strict bounded `flight-regional-ground-pipeline/v1` in-memory result preserves
the exact bounce pair, ground and bounce-input digests, plan/digest/provenance,
and existing regional envelope. Regional evidence is required exactly for a
settled bounce; every non-settled bounce reason remains native and forbids
downstream evidence. Canonical regional-plan hashing is now one shared helper.

RED recorded the missing module/result/exports, GREEN passed 17 pipeline/public
contract tests, and REFACTOR passed 39 pipeline/public/regional contract tests.
The complete flight-plus-ground suite is green for 377 tests. Ruff
check/format, scoped Black, protected changed-file and import-following MyPy,
Bandit, placeholder and diff checks, documentation governance,
blocking-quality policy, minimum-test and test-assertion contracts,
changed-Python policy, both LOC policies, campaign-manifest validation, and 11
manifest/layout tests are green. Protected skipped-import MyPy required only
explicit result casts at the already dynamic wire-parser boundaries; runtime
and canonical bytes are unchanged. Standalone Black reports one inherited
formatting preference in `test_contract_api.py`; repository-authoritative Ruff
is green and that file's delta is limited to the required public API entries.

This is an unpushed, `not_released` candidate. It adds no new wire schema or
migration and no PyQt6/React, TypeScript/Rust/WASM, persistence, playback,
calibration, target/solver/variation, or downstream integration. Keep #4271,
#4273, and #4267 open.

## 2026-08-11 PR #4359 flight-to-bounce composition

Ready-for-review PR [#4359](https://github.com/D-sorganization/Tools/pull/4359)
on `feat/4270-flight-bounce-execution` starts from exact clean published Tools
#4357 head `c492b52f9f7615c5bc38e780965167cc8f64327c`. Its reviewed implementation
commit is `869b626e2d3ebd4097ae76b8fc9720cda6696947`.
It adds the shared Python `execute_repeated_bounce_from_flight` facade. The
facade requires exact flight, launch, and transfer records; validates the
callback and capture threshold before transfer; and composes only the existing
`build_ground_simulation_request`, `RepeatedBounceRequest`, and
`execute_repeated_bounce_request` authorities. Typed transfer errors propagate
unchanged, while successful and cancelled results retain the existing request,
surface, frame, model, fingerprint, and joint execution-input identities.

RED-GREEN evidence recorded the missing module and public export before the
implementation. Independent follow-up coverage now proves exact message,
field, and reason propagation plus zero executor calls for no physical contact,
grazing contact, and missing terminal angular state. The focused contract suite
is green for 17 tests and the complete flight-plus-ground suite is green for
365 tests. Ruff check/format, scoped
Black, protected and import-following MyPy, Bandit, placeholder and diff
checks, documentation governance, blocking-quality policy, minimum-test and
test-assertion contracts, changed-Python policy, module-size policy, campaign
manifest validation, and 11 manifest/layout tests are green. Standalone Black
reports one inherited formatting preference in unchanged surrounding syntax
of `test_contract_api.py`; repository-authoritative Ruff is green and the
file's only delta is the required public-API entry. The committed changed-file
size gate is also green with zero violations across four changed Python files.

The ready PR is mergeable, but protected checks and review are not yet complete,
so it remains `not_released`. It changes no PyQt6/React UI,
TypeScript/Rust/WASM runtime, persistence, playback, camera behavior,
regional-material chaining, skid/roll completion, or total-distance contract.
Issues #4270 and #4267 remain open; no issue-completion claim is made.

## 2026-08-11 PR #4357 repeated-bounce request execution binding

Ready-for-review PR [#4357](https://github.com/D-sorganization/Tools/pull/4357)
on `feat/4270-repeated-bounce-execution` starts exactly from
published #4356 head `2387430fc78baa92ba122c7ad008a498118bf62d` and
is published at implementation head
`cf54d3528a71fd429ad19f53f04e4a1a84495097`. It adds one UI-neutral Python
`execute_repeated_bounce_request` boundary: exact validated request input,
callable-or-`None` cancellation, settings derived from the request-bound
capture threshold, invocation of the existing Python physics authority, and
the existing identity-validated request/result pair as output. No wire schema,
physics law, TypeScript runtime, or UI surface changed.

TDD recorded the expected missing-public-executor failure before the binding
was added. Qualification is green for 28 focused contract tests, the complete
189-test ground suite, 11 campaign-manifest/layout tests, Ruff check and format,
Black, the protected changed-file MyPy profile, Bandit, placeholder and diff
checks, documentation governance, blocking-quality policy, minimum-test and
test-assertion contracts, changed-Python policy, module-size policy, and the
campaign-manifest validator. A non-authoritative import-following MyPy probe
still reports three inherited redundant casts in unchanged `bounce_wire.py`,
`regional_plan_records.py`, and `regional_plan_wire.py`; the protected
`--follow-imports=skip` profile is green for the new production module.

This remains `not_released`; protected checks and review are pending. UI request construction
and invocation, persistence, playback, TypeScript or compiled physics,
regional-material chaining, measured terrain calibration, downstream parity,
protected exact-head evidence, review, approval, release, and issue/epic
completion remain open.

The exact open dependency chain from #4203 through #4357 is now ready for
review without base or history changes. The 2026-08-11 release reconciliation
found no current-head failing check, but every open PR still had queued
protected contexts and no submitted approval. The release manifest now records
the reconciled parent heads and #4357; it remains evidence-only and does not
claim protected completion or release.

## 2026-08-11 PR #4356 published current-parent propagation

Published ready-for-review PR `#4356` remains on
`feat/4270-repeated-bounce-request-wire` with base
`feat/4271-repeated-bounce-wire`. Exact current child
`23897eac03e8a3edf4a37855f0ba05e8c2527986` is the first parent and exact
published PR #4355 head `a04d14e9308990e676e8c90ddb1d80e368dd1387`
is the second parent of normal no-ff merge
`345c329e6b6e3fc7a8fc981abf65795f356b94cf`. The child's complete strict
cross-runtime repeated-bounce request envelope, canonical ground-request and
joint-execution-input digests, exact request/result identity pairing, shared
golden corpus, adversarial capture-speed digest follow-up, and live-PR handoff
remain intact while inheriting the complete #4355 result-wire and cancellation
evidence, both typed-Boolean protected-MyPy repairs, and all regional/ground
ancestry.

Local qualification is complete: 1,099 Python tests, 116 React files with 738
tests, complete Cargo workspace tests, focused 64-Python/53-React coverage,
Ruff check/format and Black on all four child-delta Python files, protected
MyPy on two child production modules plus the coherent 37-module ground
profile, Bandit on both production modules, a clean placeholder scan,
TypeScript, zero-warning ESLint, the 204-module Vite build, Rust
formatting/clippy, both repository size budgets, the manifest validator and
eight manifest tests, and every repository governance gate are green. All
eight child feature/spec/test files remain byte-exact; the parent-only
result-wire files and both inherited typed-Boolean repairs also remain exact.
Known warnings remain the Hypothesis cache ignore, empty polynomial legend,
Node local-storage flag, and 528.82 kB Vite chunk. The propagation was normally
published, and exact heads #4351 through #4356 were marked ready for review
without rebasing, retargeting, rewriting, force-pushing, merging, or changing
their bases. On the first protected checkpoint, #4356 had one successful
quality check, four skipped checks, twelve queued checks, no failure, and no
review. UI request construction, executor invocation, persistence, playback,
measured calibration, compiled and downstream parity, protected completion,
review, approval, dependency integration, release, and issue completion remain
open.

## 2026-08-11 PR #4355 current-parent propagation candidate

This no-publish candidate keeps PR `#4355` on
`feat/4271-repeated-bounce-wire` with base
`feat/4271-regional-trajectory-export`. Exact current child
`b67af52226fa6334dd3570cf650aebeaf81912fc` is the first parent and exact
published PR #4354 head `97925e4803f4fbd72d576eb1c11c47f8e61b0b66`
is the second parent of a normal no-ff merge. The child's complete strict
cross-runtime repeated-bounce result-wire contract, canonical golden corpus,
phase/chronology/energy invariants, and pre-contact cancellation follow-up
remain intact while inheriting regional trajectory inspection/export, both
typed-Boolean protected-MyPy repairs, and the complete regional/ground ancestry.

Local qualification is complete: 1,078 Python tests, 115 React files with 719
tests, complete Cargo workspace tests, focused 43-Python/34-React coverage,
Ruff check/format on all seven child-delta Python files, protected MyPy on five
child production modules plus the coherent 36-module ground profile, Bandit on
the five production modules, a clean placeholder scan, TypeScript,
zero-warning ESLint, the 204-module Vite build, Rust formatting/clippy, both
LOC budgets, the manifest validator and manifest tests, and every repository
governance gate are green. All 12 child feature/test files and both inherited
typed-Boolean repairs remain byte-exact. Standalone Black is non-authoritative
by repository policy and reports one advisory formatting difference in the
audited child `bounce_types.py`; authoritative Ruff is green, so that child
file remains exact. Known warnings remain the Hypothesis cache ignore, Node
local-storage flag, and 528.82 kB Vite chunk. No branch has been rebased,
retargeted, rewritten, force-pushed, or published. Request construction,
executor invocation, persistence, playback, measured calibration, compiled
and downstream parity, protected exact-head evidence, review, approval,
dependency integration, release, and issue completion remain open.

## 2026-08-11 PR #4354 current-parent propagation candidate

This no-publish candidate keeps PR `#4354` on
`feat/4271-regional-trajectory-export` with base
`feat/4271-regional-event-inspection`. Exact current child
`99b0739bdc3ece814ed6039e6ba31f7ac38c0227` is the first parent and exact
published PR #4353 head `e0433adbc3c82272745d098867f261462a790d08`
is the second parent of a normal no-ff merge. The child's matched bounded
PyQt6/React raw-trajectory inspection and canonical semantic-lossless evidence
export remain intact while inheriting ground-event and regional-transition
ledger inspection, the complete qualified result projection, the explicit
Boolean local required by protected delta-MyPy, embedded-plan execution and
provenance, request-I/O boundaries, and complete regional physics ancestry.

Local qualification is complete: 1,058 Python tests, 114 React files with 700
tests, complete Cargo workspace tests, focused 6-Python/8-React coverage, Ruff
check/format on the three child-delta Python files, protected MyPy on two child
production modules plus the coherent 35-module ground profile, Bandit on the
two child production modules, TypeScript, zero-warning ESLint, the 204-module
Vite build, Rust formatting/clippy, both LOC budgets, the manifest validator
and manifest tests, and every repository governance gate are green. Protected
delta-MyPy found the same skipped-import `no-any-return` boundary as #4351 in
the new evidence exporter; its helper result is now assigned to an explicit
Boolean local with no runtime or canonical-byte change. The other seven child
feature/test files and inherited Boolean-local repair remain byte-exact.
Standalone Black is non-authoritative by repository policy and its Python 3.13
runner cannot safety-parse the inferred 3.14 target; authoritative Ruff is
green. Known warnings remain the Hypothesis cache ignore, empty polynomial
legend, Node local-storage flag, and 528.82 kB Vite chunk. No branch has been
rebased, retargeted, rewritten, force-pushed, or published. Input construction,
UI executor invocation, interpolation/playback, calibration workflows,
compiled regional physics, downstream parity, protected exact-head evidence,
review, approval, dependency integration, release, and issue completion remain
open.

## 2026-08-11 PR #4353 current-parent propagation candidate

This no-publish candidate keeps PR `#4353` on
`feat/4271-regional-event-inspection` with base
`feat/4271-regional-result-readback`. Exact current child
`7fc00f43561c31923b74563bc2bf6caf89bbc9eb` is the first parent and exact
published PR #4352 head `12fc80798d2a15b44c0215688ffb031dd99cbdd1`
is the second parent of a normal no-ff merge. The child's matched bounded
PyQt6/React inspection of validated ground-event and regional-transition
ledgers remains intact while it inherits the complete qualified result
projection, the explicit Boolean local required by protected delta-MyPy,
embedded-plan execution/provenance and request-I/O boundaries, complete
regional physics ancestry, capability-only extended finite-float serializer,
and default ground safe-number boundary.

Local qualification is complete: 1,057 Python tests, 113 React files with 698
tests, complete Cargo workspace tests, focused 76-Python/38-React coverage,
Ruff check/format on the three child-delta Python files, protected MyPy on two
child production modules plus the coherent 35-module ground profile, Bandit on
the two child production modules, TypeScript, zero-warning ESLint, the
203-module Vite build, Rust formatting/clippy, both LOC budgets, the manifest
validator and eight manifest tests, and every repository governance gate are
green. Child feature bytes and the inherited Boolean-local repair are exact;
conflict-marker and diff checks are clean. Non-failing warnings are limited to
the known Hypothesis cache ignore, empty polynomial legend, Node local-storage
flag, and 526.79 kB Vite chunk. No branch has been rebased, retargeted,
rewritten, force-pushed, or published. Trajectory-sample inspection, lossless
export, UI executor invocation, playback, calibration workflows, compiled
regional physics, downstream parity, protected exact-head evidence, review,
approval, dependency integration, release, and issue completion remain open.

## 2026-08-11 PR #4352 current-parent propagation candidate

This no-publish candidate keeps PR `#4352` on
`feat/4271-regional-result-readback` with base
`feat/4271-regional-execution-ui`. Exact current child
`10fdac4860035fd5c845a621752e93688e2e674e` is the first parent and exact
published PR #4351 head `4024c8a1ad2d3871c6b06ef6369250a873789c39`
is the second parent of a normal no-ff merge. The child's complete matched
PyQt6/React qualified result projection remains intact while it inherits the
current bounded evidence import/readback, the explicit Boolean local required
by protected delta-MyPy, embedded-plan execution/provenance and request-I/O
boundaries, complete regional physics ancestry, capability-only extended
finite-float serializer, and default ground safe-number boundary.

Local qualification is green: all `1,057` combined Rate-of-Closure and shared-
ground Python tests, all `113` React files / `697` tests, and the complete Cargo
workspace pass. Focused result/readback/execution/I/O/capability coverage
passes `76` Python and `37` React tests. Pinned Ruff 0.14.10 check/format passes
all three child-delta Python files; isolated-import strict MyPy passes both
child production modules and the coherent 35-module ground profile passes with
inherited imports skipped and only the parent's documented `redundant-cast`
code disabled; Bandit passes both child production files. TypeScript, zero-
warning ESLint, the 202-module Vite build, Rust format and warning-denied
clippy, both 400- and 500-LOC gates, the campaign validator and eight manifest
tests, docs/tool-manifest/blocking-gate/assertion/minimum-test governance,
child-feature and inherited Boolean-local byte checks, marker scans, and diff
checks pass. Existing Hypothesis ignored-cache, polynomial-generator empty-
legend, Node local-storage option, and 523.34 kB Vite chunk warnings remain
non-failing.

No branch has been rebased, retargeted, rewritten, force-pushed, or published.
UI executor invocation, trajectory/event tables, playback, calibration
workflows, compiled regional physics, downstream parity, protected exact-head
evidence, review, approval, dependency integration, release, and issue
completion remain open.

## 2026-08-11 PR #4351 current-parent propagation candidate

This no-publish candidate keeps PR `#4351` on
`feat/4271-regional-execution-ui` with base
`feat/4271-regional-execution-binding`. Exact current child
`351a3051e9093c6b80cabf0f1db04aeeb15abfac` is the first parent and exact
published PR #4350 head `98f86990e9225903fbe84cd1f267ed38ef0a15d8`
is the second parent of a normal no-ff merge. The child's matched bounded
PyQt6/React evidence import and readback, including the explicit Boolean local
required by protected delta-MyPy, remain intact while inheriting the parent's
embedded-plan execution/provenance contract, request I/O, complete regional
physics ancestry, capability-only extended finite-float serializer, and
default ground safe-number boundary.

Local qualification is green: all `1,056` combined Rate-of-Closure and shared-
ground Python tests, all `113` React files / `696` tests, and the complete Cargo
workspace pass. Focused evidence/readback/execution/I/O/capability coverage
passes `75` Python and `36` React tests. Pinned Ruff 0.14.10 check/format passes
all six child-delta Python files; isolated-import strict MyPy passes all five
child production modules, preserving the Boolean-local repair; the coherent
35-module ground profile passes with inherited imports skipped and only the
parent's documented `redundant-cast` code disabled; and Bandit passes those
five production files. TypeScript, zero-warning ESLint, the
202-module Vite build, Rust format and warning-denied clippy, both 400- and
500-LOC gates, the campaign validator and eight manifest tests,
docs/tool-manifest/blocking-gate/assertion/minimum-test governance,
child-feature byte checks, marker scans, and diff checks pass. Existing
Hypothesis ignored-cache, polynomial-generator empty-legend, Node local-storage
option, and 521.54 kB Vite chunk warnings remain non-failing.

No branch has been rebased, retargeted, rewritten, force-pushed, or published.
UI executor invocation, playback, compiled regional physics, downstream
parity, protected exact-head evidence, review, approval, dependency
integration, release, and issue completion remain open.

## 2026-08-11 PR #4350 current-parent propagation candidate

This no-publish candidate keeps PR `#4350` on
`feat/4271-regional-execution-binding` with base
`feat/4274-regional-plan-io`. Exact current child
`dfb4b97481f187ff3594eceb08c427f650aca4e3` is the first parent and exact
published PR #4342 head `de66a851aa5dded680279cf9a2b25a5094966593`
is the second parent of a normal no-ff merge. The child's embedded-plan
execution/provenance envelope, executor authority, transition binding,
cross-runtime fixtures, and frozen base-result boundary remain intact while it
inherits the parent's current request I/O, matched editors, complete regional
physics ancestry, capability-only extended finite-float serializer, and
default ground safe-number boundary.

Local qualification is green: the combined Rate-of-Closure and shared-ground
Python suites pass all `1,052` tests; the complete React suite passes `111`
files / `692` tests; and the complete Cargo workspace passes. Focused
execution/I/O/capability coverage passes `71` Python and `36` React tests.
Pinned Ruff 0.14.10 check/format passes all seven child-delta Python files;
isolated-import strict MyPy passes the four execution modules and the coherent
35-module ground profile passes with only the parent's documented
`redundant-cast` code disabled. Bandit passes all five child production files.
TypeScript, zero-warning ESLint, the 199-module Vite build, Rust format and
warning-denied clippy, both 400- and 500-LOC gates, the campaign validator and
eight manifest tests, docs/tool-manifest/blocking-gate/assertion/minimum-test
governance, child-feature byte checks, marker scans, and diff checks pass.

The first CPU-contended full Python run recorded `1,051` passes and one
Hypothesis input-generation `too_slow` health check; that property passed alone
and all `1,052` tests passed in the single uncontended rerun. No branch has been
rebased, retargeted, rewritten, force-pushed, or published. Execution UI and
playback, compiled regional physics, downstream parity, protected exact-head
evidence, review, approval, dependency integration, release, and issue
completion remain open.

## 2026-08-11 PR #4342 current-parent propagation candidate

This no-publish candidate keeps PR `#4342` on
`feat/4274-regional-plan-io` with base `feat/4274-regional-surface-ui`.
Exact current child `c1f47f2ef68b3db102da5416aaac17a40f675207` is the first
parent and exact reviewed local #4339 candidate
`db335937afc4b587d235eb705e315f577519c5e6` is the second parent of a
normal no-ff merge. Child-owned canonical request import/export, bounded UTF-8,
native atomic save, browser-qualified download, tests, and limitations remain
intact while inheriting current editor, wire, regional-physics, and complete
ground-model ancestry.

The merge also resolves one cross-runtime compatibility edge without weakening
the ground contract. The default shared canonical encoder still rejects floats
or integers outside JavaScript's safe range. A separately named extended
finite-float policy reuses the same recursive encoder only through the
capability-observation facade; integers remain safe-range bounded and integral
finite doubles such as `1e20` and `1e21` emit exact exponent-free tokens that
match the TypeScript capability serializer. Non-finite values still fail
closed.

Local qualification is green: all `909` Rate-of-Closure Python tests, all
`110` React files / `686` tests, and the complete Cargo workspace pass. The
focused compatibility/regional-I/O slices pass `47` Python and `12` React
tests. Pinned Ruff 0.14.10 check/format passes `17` changed Python files; pinned
MyPy 1.13 and Bandit pass `12` changed production files. TypeScript,
zero-warning ESLint, the 199-module Vite build, Rust format and warning-denied
clippy, both 400- and 500-LOC changed-file gates, manifest/docs/blocking-gate/
assertion/minimum-test governance, marker scans, and diff checks pass. An
untouched manual-delivery UI test timed out once during the first concurrent
full run, then passed alone and in the single complete rerun.

No branch has been rebased, retargeted, rewritten, force-pushed, or published.
Execution and playback, result interchange, measured calibration, model-input
persistence, changing geometry or velocity, TypeScript/compiled regional
physics, downstream parity, protected exact-head evidence, approval,
dependency integration, and release remain open.

## 2026-08-11 PR #4339 current-parent propagation candidate

This no-publish candidate keeps PR `#4339` on
`feat/4274-regional-surface-ui` with base
`feat/4271-regional-wire-contract`. Exact current child
`d21741e312b849a63f73cabf351a15d9de80fb94` is the first parent and exact
published PR #4335 head `8f933ed8dcb29e55ece4ec6bb1e60813f6794d57`
is the second parent of a normal no-ff merge. The matched PyQt6/React regional
surface editors retain their validation, invalidation, engineering hints, and
strict request readback while inheriting the current regional wire, resolver,
regional physics, and complete ground-model ancestry. The extracted navigation
state remains canonical and now includes the child-owned `regional_surfaces`
module in the first-run and migration order.

Local qualification is green: the complete Rate-of-Closure Python suite passes
all `891` tests and the complete React suite passes `110` files / `678` tests.
Focused regional/ground/navigation coverage passes `177` Python tests and `14`
React tests; all `137` `tools-core` Rust tests pass. TypeScript, zero-warning
ESLint, the 198-module Vite build, Rust workspace format and warning-denied
clippy, pinned Ruff 0.14.10 across seven PR-delta Python files, pinned MyPy 1.13
across six production files, Bandit medium/high screening, both 400- and
500-LOC changed-file gates, manifest validation, documentation governance,
changed-test assertions, minimum-test contracts, child-feature byte checks,
conflict-marker scans, and diff checks pass. The test runners emit only the
existing Hypothesis ignored-cache and Node local-storage option warnings.

No branch has been rebased, retargeted, rewritten, force-pushed, or published.
Physics execution and playback, result interchange, measured calibration,
model-input persistence, changing geometry or surface velocity,
TypeScript/compiled regional physics, downstream parity, protected exact-head
evidence, approval, dependency integration, and release remain open.


## 2026-08-11 PR #4351 delta-MyPy boundary repair candidate

Protected CI on exact PR #4351 head
`fe463b5503a8c7b599a329da18bb690d008871cd` exposed a delta-root-dependent
typing boundary in `write_regional_surface_plan_request_atomic`. The CI profile
uses `MYPYPATH=src:src/python/src` and `--follow-imports=skip`, so the imported
atomic writer resolves as `Any` when it is not itself a MyPy root. A typed local
now preserves the declared Boolean return without reintroducing the cast that
becomes redundant when both modules are roots. Runtime validation, atomic file
semantics, canonical bytes, UI behavior, and physics are unchanged.

This is a local no-publish repair candidate. It must propagate normally through
descendants #4352, #4353, and #4354 after exact-head review; protected CI,
review, dependency ordering, and release remain open.


## 2026-08-11 regional execution current-parent reconciliation candidate

The clean `feat/4271-regional-execution-binding` worktree normally merges
exact reviewed child `012cdfc33ad1590f31a1cbb109f0b8bee8eee700` with exact newly
published PR #4342 parent `c1f47f2ef68b3db102da5416aaac17a40f675207`
as its second parent. The intended base remains
`feat/4274-regional-plan-io`; neither branch is rebased, retargeted, rewritten,
force-pushed, published, or opened as a PR by this reconciliation.

The child retains its remediated embedded-plan execution/provenance envelope,
executor authority, transition-to-plan binding, canonical cross-runtime
validation, and executor-produced evidence. The parent retains canonical
request I/O, the bounded engineering input helper, and its verbatim append-only
handoff/SPEC history. This local candidate is not protected or release evidence.


Merged-tree qualification is 143 focused Python ground tests and 24 focused
React execution/plan/editor tests passing. Pinned Ruff 0.14.10 check/format
passes all 50 ground files; pinned MyPy 1.13 passes all four execution modules
with redundant-cast warnings enabled and all 35 ground production modules with
only the parent's documented redundant-cast code disabled. Bandit reports no
medium/high finding. TypeScript, zero-warning ESLint, the 199-module production
build, campaign manifest and eight manifest tests, docs/tool-manifest
governance, changed-Python, minimum-test, 500-LOC, heading/SPEC preservation,
parent/child diff, and whitespace gates pass. Structural maxima are 376 lines
for TypeScript, 281 for Python, 43 per function, and four parameters.

A broader formatter sweep also reports three parent-only Rate test files that
current Ruff would reformat; the exact execution/ground scope is clean and this
child does not rewrite that published parent baseline. The React build retains
the existing nonblocking warning for its approximately 500 kB main chunk.


## 2026-08-11 regional execution independent-review remediation

Independent review of local commit `696a3ff8f124bebf6dc22ae0d584cf35f6d92843`
correctly rejected its permissive transition wire and synthetic golden
evidence. The follow-up embeds the exact regional plan in the v1 envelope,
recomputes its digest, enforces the executor producer/version, and validates
every ordered transition event and from/to region/surface pair against a real
boundary crossing in that plan. Python wire values now use the same canonical
safe-number, integral-number, vector, and nonblank-text policy as TypeScript.
Null-result cancellation/failure envelopes require an empty transition ledger
because no embedded result exists to substantiate transition evidence.

The shared golden document is generated from actual executor output and covers
representable, cancelled, and step-limit failed outcomes; a separate shared
adversarial corpus pins cross-runtime accept/reject parity. The frozen base-v1
fixture remains independently byte-pinned. No UI or new physics is included.

The unrelated baseline test
`test_stable_wire_uses_canonical_numeric_tokens_for_every_float` still fails
on its deliberately injected `1e20` value with `ValueError: canonical JSON
number exceeds cross-runtime safe range`; 10 sibling tests pass. The same
signature exists on exact parent `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`
and is not changed in this contract repair.

## 2026-08-11 regional ground execution binding

Local branch `feat/4271-regional-execution-binding` starts from exact current
PR #4342 head `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833` without rewriting
the plan-I/O parent. The UI-neutral `execute_regional_ground` boundary accepts
an exact ground request, settled bounce prefix, regional plan request, and
bounded execution options. The resolver is constructed only from the plan;
base-surface, request, prefix, digest, model, and transition identities fail
closed before evidence is accepted.

Strict `ground-regional-execution-result/v1` embeds representable frozen
ground-result v1 output and otherwise reports typed cancellation/failure with
no fabricated result. It carries canonical request/plan SHA-256 values, plan
and executor provenance, model identity, exact ordered from/to region+surface
transitions, and the coplanar/static limitations. Python executes the existing
solver/composer; TypeScript only parses/serializes. No UI controls, compiled
regional physics, UpstreamDrift consumers, protected CI/review, or #4271
completion are claimed.



## 2026-08-11 PR #4342 append-only preservation repair

Independent audit of local merge `e7fedfb18de1550eed3484ed2fc99d0baaecdca1`
found that three older parent ancestry sections had been summarized instead of
retained verbatim. This docs-only follow-up restores the full parent text below
without changing production code, tests, schemas, manifest state, PR base, or
release claims. The exact parent sections and SPEC rows are now governed by
byte-for-byte comparison against
`d21741e312b849a63f73cabf351a15d9de80fb94`.

## 2026-08-11 PR #4335 current-parent ancestry candidate

The clean dedicated `feat/4271-regional-wire-contract` worktree starts from
exact live PR #4335 child `74a053d2d544da9f44a88007660ad28c0127f285`
and normally merges exact newly published PR #4332 parent
`04ccf08dd990de1cd056a3420e67772773a4be2e` as its second parent. PR #4335
keeps base `feat/4271-regional-surface-transitions`; neither branch is rebased,
retargeted, rewritten, force-pushed, or published by this reconciliation.
Production physics, wire contracts, golden fixtures, numerical ordering, and
public APIs merge byte-exactly; only SPEC, manifest, and handoff records require
truthful reconciliation.

The child retains the strict cross-runtime regional-plan request/result wire
contract, canonical JSON/SHA-256 evidence, fail-closed parsing, and Python
resolver adapter. The parent retains bounded coplanar regional transitions and
the complete reconciled impact/bounce/skid/roll ancestry. Changing normals,
height or surface-velocity discontinuities, terrain deformation, torsional-spin
damping, roll-to-skid transitions, internal transition-ledger export, regional
UI, compiled or TypeScript regional physics, downstream parity, protected CI,
review, normal stack integration, and main release remain open. This local
merge is not release evidence and requires independent review before an
ordinary fast-forward publication.

Merged-tree qualification is `132` focused Python ground tests and `9`
focused React regional/ground-contract tests passing. Pinned Ruff 0.14.10 check
and format pass all `45` ground Python files. Pinned MyPy 1.13 passes the exact
two-file isolated CI boundary and all `31` production modules under the coherent
whole-package profile that disables only redundant-cast warnings; those casts
remain required by the isolated protected profile. Bandit reports no
medium/high finding. Documentation governance, manifest validation/layout and
all `8` manifest contracts, changed-production and minimum-test policies, the
official 500-LOC PR-delta gate (`6` files, zero violations), diff checks, and
mandatory maxima of `392` module lines, `46` function lines, and `4` parameters
all pass.

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


## 2026-08-11 PR #4342 current-parent reconciliation candidate

The clean `feat/4274-regional-plan-io` worktree normally merges exact
published PR #4342 child `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`
with exact newly published PR #4339 parent
`d21741e312b849a63f73cabf351a15d9de80fb94` as its second parent. PR #4342
continues to target `feat/4274-regional-surface-ui`; neither branch is rebased,
retargeted, rewritten, force-pushed, or published by this reconciliation.
The merge preserves strict regional request import/export together with the
parent's frozen engineering-number specification and bounded helper.

The child widget module now delegates its canonical-precision controls to that
shared three-parameter helper while retaining the inclusive cross-runtime safe
number bounds and eleven-decimal presentation required by its I/O contract.
Merged-tree qualification is 53 focused Python/PyQt/shared-ground tests and 14
focused React tests passing. Pinned Ruff 0.14.10 check/format, MyPy 1.13 over
all ten affected production modules, Bandit medium/high screening, TypeScript,
zero-warning ESLint, and the 199-module production build pass. Campaign
manifest validation and all eight manifest tests, documentation and tool
manifest governance, changed-Python, minimum-test, exact-diff assertion,
500-LOC, parent/child diff, and whitespace gates pass. Mandatory structural
maxima are 396 module lines, 40 function lines, and three parameters.
The local merge is not protected or release evidence. Independent review,
ordinary fast-forward publication, fresh protected CI/approval, dependency
integration, and release remain required.

## 2026-08-11 PR #4339 structural helper repair candidate

This parent follow-up descends exact independently reviewed reconciliation
candidate `eedad6d23b517eb4c99d4ba9000ff6555101099f`. Independent review found
one publication blocker: the regional PyQt number-input helper accepted six
parameters, above the campaign's mandatory four-parameter limit. RED-first
tests bind the helper's signature and all observable widget settings. A frozen
`NumberInputSpec` validates finite positive step and finite ordered bounds
before widget construction; the three-parameter helper preserves field names,
values, units, ranges, increments, tooltips, and presentation order.

Eight focused Python/PyQt tests and 25 focused React tests pass; the complete
React suite passes all 108 files / 672 tests. Pinned Ruff check/format, MyPy,
Bandit medium/high screening, TypeScript, zero-warning ESLint,
documentation/manifest governance, changed-production, test-assertion,
minimum-test, 500-LOC, and diff gates pass. Mandatory maxima are 400 module
lines, 42 function lines, and three parameters. No schema, digest, regional
physics, UI behavior, PR base, protected evidence, or release boundary changes.

## 2026-08-11 PR #4339 current-parent ancestry candidate

The clean `feat/4274-regional-surface-ui` worktree normally merged exact
published child `cbb9c0a6bdc6a50f59f7a661139b9d53e1892980` with exact published
#4335 parent `9e01ccc3e891cc45907293751a192624195a77a5`, while preserving
#4339's `feat/4271-regional-wire-contract` base. Production UI, contract,
resolver, regional physics, and ground ancestry merged without conflict; only
SPEC, manifest, and handoffs were reconciled. Independent review, ordinary
publication, protected exact-head CI/approval, dependency integration, and
release remained open.

## 2026-08-11 PR #4342 delta-MyPy follow-up

Protected CI on exact PR #4342 head
`cffe349ac0a8054f1d168cb36684fd00bc5f8a49` identified one redundant `bool`
cast in the regional atomic-write adapter under the Linux delta-MyPy gate. The
cast is removed; the typed helper's direct Boolean return is preserved. This
changes no wire bytes, validation, file semantics, UI behavior, or physics.
Focused persistence tests and the CI-equivalent pinned MyPy command pass
locally; fresh protected CI/review and dependency ordering remain required.

## 2026-08-11 regional request I/O protected publication

Branch `feat/4274-regional-plan-io` is published normally as draft PR
[#4342](https://github.com/D-sorganization/Tools/pull/4342), targeting exact
PR #4339 branch `feat/4274-regional-surface-ui` at parent head
`cbb9c0a6bdc6a50f59f7a661139b9d53e1892980`. Its reviewed implementation
head is `d748e7a5ef3da5e6ce7737ff6829e0f14665fe97`; publication commits change
no runtime behavior. Protected CI, independent review, issue #4274, parent
ordering, integration, and release remain open.

## 2026-08-11 bounded regional request read follow-up

Independent review of local safe-number commit
`10c394f6b1fd2927e7f3b1f96cc097cae6bfd380` found that native request import
checked file size with `stat()` and then performed a separate unbounded text
read. A concurrently grown or replaced file could therefore allocate beyond
the 1 MiB wire cap before the strict parser rejected it.

Native import now opens one binary handle, reads at most the wire cap plus one
sentinel byte, rejects overflow, strictly decodes UTF-8, and only then delegates
to the unchanged canonical v1 parser. Tests simulate content growth after the
metadata check and reject invalid UTF-8 explicitly. The complete regional and
atomic persistence set is 28 tests passing. This changes no schema, digest,
physics, browser behavior, or write semantics. Static and governance evidence
is recorded in this commit: Ruff, Ruff format, Black, focused MyPy, manifest,
module-size, documentation-governance, and diff checks pass. The child remains
local and unprotected.

## 2026-08-11 regional request I/O safe-number follow-up

Independent review of local commit `e39edf4b50b1fb9811b0032bec4758c7a08c9b74`
found two cross-runtime representation gaps. The PyQt6 precedence spin box
silently narrowed otherwise valid v1 integers above 1,000,000, and Python
accepted integer-valued wire numbers above JavaScript's exact safe range.

The native editor now uses an exact decimal integer entry for precedence and
preserves every v1 value from zero through 9,007,199,254,740,991 without a
floating-point conversion. Shared Python canonical-number and ground-record
validation now reject magnitudes outside that same range, matching React while
retaining all finite fractional values inside it. PyQt6 floating-point editors
publish the same bounds. Regression coverage pins exact maximum-precedence
open/save round-trip and matched native/browser refusal of unsafe material
numbers. This changes no schema version, provenance algorithm, physics, or
browser filesystem limitation. Final local evidence is 132 shared ground tests,
21 regional editor/I/O tests, five atomic-workspace tests, 23 focused React
tests, the whole-shell tooltip sweep, and eight manifest tests passing. Ruff,
Ruff format, Black, focused MyPy, TypeScript, zero-warning ESLint, the 199-module
production build, module and changed-file size budgets, documentation
governance, and diff checks pass. The child remains local and unprotected.

## 2026-08-10 PR #4339 stale-validation invalidation follow-up

Rendered exact-head browser QA found that a validated one-overlay readback
remained visible after adding a second overlay. The broad PyQt6 suite also
found missing hover hints on the new controls. PyQt6 and React now clear both
canonical readback and prior validation state whenever any identity, interval,
material, or overlay-row draft value changes; PyQt6 exposes the explicit
"Changes not validated" pending state, and every PyQt6 interactive has a
specific engineering-context tooltip. RED-first regression tests cover the
dynamic-row path on both surfaces. Final local evidence is 870 Python/PyQt
tests and 672 React tests passing; Ruff, format, MyPy, TypeScript, zero-warning
ESLint, the 198-module production build, manifest/docs/file-size governance,
and diff checks are clean. The known polynomial-generator empty-legend warning
is unrelated. This changes no wire schema, provenance digest, physics, or
persistence boundary; fresh protected CI is still required before merge.

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-11.

## 2026-08-10 issue #4274 canonical regional request I/O local child

Local branch `feat/4274-regional-plan-io` started directly from published PR
#4339 head `a9ace5052bcd54b78f79b62d5d9ac26debedb4b1`, then fast-forwarded
normally to corrected exact parent `cbb9c0a6bdc6a50f59f7a661139b9d53e1892980`.
The parent's stale-validation invalidation and tooltip follow-up are preserved.
This child has not been pushed, opened as a PR, reviewed, merged, or released.

The matched Ground Surfaces editors now import and export the canonical
`ground-regional-material-plan-request/v1` document. PyQt6 uses native Open
and Save As dialogs plus a shared flush/fsync/atomic-replace UTF-8 writer;
cancel is a no-op, failed reads and writes preserve the last-known-good editor
or file, and the recent path advances only after success. React validates a
bounded browser File before committing state and downloads exact canonical
bytes while revoking its object URL. Browser filesystem limitations are
explicit, and workspace persistence remains separate.

Import accepts only editor producer/provider v1 evidence, the fixed target
frame origin/downrange axis, and one to eight rows. It never relabels external
evidence. An unchanged import round-trips the exact request and provenance;
after any edit, validation binds a fresh draft digest. No measured calibration
is synthesized and neither path executes physics.

RED-first coverage now includes exact-byte round-trip, native cancel and
atomic rollback, corruption, duplicate keys, oversize input, unsupported
producer/axis qualification, stale-provenance rebinding, transactional PyQt6
population, browser rollback, URL cleanup, and pre-allocation size rejection.
Focused evidence is 25 Python/PyQt tests and 12 React tests passing. With
adjacent wire and manifest coverage, 44 Python and 16 React tests pass.
Focused MyPy, Ruff, TypeScript, zero-warning ESLint, the 199-module production
build, release-manifest validation, and module budgets are clean.

Issue #4274 remains open. Measured calibration, workspace model-input
persistence, execution/playback, result interchange, visualization, changing
geometry, compiled regional physics, downstream parity, protected CI/review,
integration of this child into the protected stack, and release remain explicit
gaps.

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

## 2026-08-11 regional execution evidence UI continuation

Branch `feat/4271-regional-execution-ui` is a local, unpublished child of
exact published PR #4350 head
`dfb4b97481f187ff3594eceb08c427f650aca4e3`. It adds matched PyQt6 and React
import-only readback for strict Python-produced
`ground-regional-execution-result/v1` evidence. Acceptance is transactional,
bounded, strict, and requires the embedded plan to equal the currently valid
visible plan. Plan edits clear stale evidence. React does not run physics.

Local evidence is green: 207 expanded Python ground/plan/PyQt/layout tests and
111 React files / 690 tests passed, with strict MyPy, Ruff/format, TypeScript,
zero-warning ESLint, production build, manifest + eight manifest tests,
documentation governance, structural budgets, and diff checks. The build
retains the inherited 500 kB chunk advisory. Do not publish until an
independent review is complete. UI construction of a qualified ground request
and settled bounce prefix, executor invocation, playback, measured
calibration, compiled regional physics, downstream parity, protected evidence,
release, and issue completion remain open.

## 2026-08-11 complete regional result readback continuation

Branch `feat/4271-regional-result-readback` is a local unpublished child of
exact draft PR #4351 head
`fe463b5503a8c7b599a329da18bb690d008871cd`. It extends the matched import-only
PyQt6/React readback to every qualified summary/result field required for
honest user inspection: distinct carry/bounce/skid/roll/surface-path/total,
final downrange/offline, bounce count, ground time, terminal completion, model
and surface provider IDs/versions, calibration evidence, observed phases,
typed warnings, executor provenance, and qualification limits.

Null-result cancellation/failure keeps ground-only values unavailable. Partial
evidence retains the censored-endpoint warning and is not relabeled as rest.
No physics, executor invocation, trajectory/event tables, playback, compiled
parity, calibration workflow, or downstream integration is added.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 691 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 202-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The build retains the inherited
500 kB chunk advisory. Independently review before any GitHub write.

## 2026-08-11 regional execution ledger inspection continuation

Branch `feat/4271-regional-event-inspection` is a local unpublished child of
exact draft PR #4352 head
`10fdac4860035fd5c845a621752e93688e2e674e`. It adds matched PyQt6 and React
inspection tables for the frozen result's validated ground-event and regional-
transition ledgers. Event rows show explicit SI time, position, before/after
linear velocity and angular velocity, frame, sequence, and type. Transition
rows show their bound event, SI time/position, and from/to region and surface.

Both clients retain the complete accepted evidence while rendering at most 256
rows per ledger with honest count/truncation text. Null-result evidence exposes
empty tables. Partial endpoint warnings remain visible. No physics, trajectory-
sample table, export, executor invocation, playback, calibration workflow,
compiled parity, or downstream integration is added. Validate and independently
review before any GitHub write.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 692 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 203-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The build retains the inherited
500 kB chunk advisory.

## 2026-08-11 regional trajectory inspection and canonical export continuation

Branch `feat/4271-regional-trajectory-export` is a local, unpublished child of
exact published draft PR #4353 head
`7fc00f43561c31923b74563bc2bf6caf89bbc9eb`. It adds matched PyQt6 and React
inspection of the frozen envelope's already-validated raw ground trajectory:
SI time, phase, position, linear velocity, angular velocity, and frame. Both
clients retain the complete accepted envelope while presenting at most 256
samples with exact count/truncation disclosure.

Accepted evidence can be saved with the frozen canonical serializer. PyQt6
uses a bounded UTF-8 native atomic write; React downloads the same canonical
JSON and makes no atomic-filesystem claim. Export does not project, recompute,
or alter evidence. Native cancellation is a no-op; import and export failures
preserve the prior accepted evidence. No browser physics is introduced.

This child has not been pushed and has no PR. Before any GitHub write, finish
the recorded local gates and independent review. Ground-request and settled-
bounce-prefix construction, UI executor invocation, interpolation/playback,
measured calibration, compiled-runtime parity, downstream parity, protected
CI/review, release, and #4267/#4271 completion remain open.

Exact local gates are green: 209 expanded Python ground/plan/PyQt/layout tests,
112 React files / 694 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 204-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The inherited 500 kB build advisory
remains. Independent review is required before publication.

## 2026-08-11 repeated-bounce evidence wire boundary

Branch `feat/4271-repeated-bounce-wire` is a local, unpublished child of exact
regional trajectory/export candidate
`99b0739bdc3ece814ed6039e6ba31f7ac38c0227`. It adds the strict
`ground-repeated-bounce-result/v1` executor-input evidence boundary that was
missing between the qualified #4270 bounce solver and later #4271 regional
execution. Python serializes and parses the complete `RepeatedBounceResult`;
React provides an import-only parser and canonical serializer and executes no
browser physics.

The contract has exact keys at every level, rejects duplicate JSON object keys,
enforces a 1 MiB UTF-8 bound, accepts only the frozen SI target frame, and
validates request/model identities, the 64-character request fingerprint,
event/impact/post-state trajectory correspondence, additive energy-ledger
arithmetic, airborne segments, settled handoff, termination chronology, and
warnings through the canonical record validators. A shared fixture pins
byte-deterministic canonical numeric JSON and
SHA-256 `d8e7400632215220d3c5b1ccd7c57040f6023ebd72470b380b48b8f8fa99b9f9`.

This slice does not construct a ground request, execute bounce or regional
physics, invoke an executor from either UI, persist files, interpolate or play
back trajectories, or claim measured calibration, compiled parity, downstream
parity, protected evidence, release, or #4267/#4271 completion. Local gates are
green on the predecessor candidate: 162 Python ground tests, 113 React files /
712 tests, focused pinned MyPy, Ruff/format, TypeScript type-check, zero-warning
ESLint, and the 204-module production build. A narrow follow-up explicitly pins
valid pre-contact cancellation with empty evidence ledgers and zero elapsed
ground time; its focused Python/TypeScript and documentation/manifest gates pass
without changing production behavior. The inherited 500 kB build advisory
remains. Independent
review found four evidence-integrity blockers; all four were remediated with
matched adversarial tests. A final independent review of the complete
post-remediation diff remains required before any publication.

## 2026-08-11 repeated-bounce request wire and pairing boundary

Draft PR #4356 publishes branch `feat/4270-repeated-bounce-request-wire` at
exact head `5f71bc2d8e3527bc76fe4c7f331f9f10203a6491`, stacked on exact
published draft parent PR #4355 head
`b67af52226fa6334dd3570cf650aebeaf81912fc`. It reuses the strict embedded
`flight-to-ground-request/v1` rather than duplicating physical inputs and adds
bounded `ground-repeated-bounce-request/v1` Python/TypeScript import contracts.
Exact SI/frame/request/surface/model identities, canonical ground-request
SHA-256, capture threshold, and joint execution-input SHA-256 are fail-closed.

An exact pairing record checks the existing bounce result's request, surface,
frame, model/version, and ground-request fingerprint. Result v1 does not carry
the joint execution digest, so it cannot independently prove the capture
threshold; later executor evidence must preserve the paired request or digest.
The browser still runs no bounce physics. UI construction/invocation,
persistence, compiled physics, downstream parity, protected evidence, release,
and #4267/#4270 completion remain open. Protected CI for #4356 is pending.

Exact prepublication implementation gates at `9da44ec98709dfb0d92a23591698ea3bf2be6e5c`
are green: 183 Python ground tests; 114 React files / 731 tests; exact
changed-source MyPy with hosted `--follow-imports=skip`; Ruff and format;
TypeScript; zero-warning ESLint; the 204-module production build;
campaign-manifest validation and eight manifest tests; docs governance;
module-size budget; placeholder/quality scan; and diff checks. Published head
`5f71bc2d8e3527bc76fe4c7f331f9f10203a6491` adds explicit finite
capture-threshold digest-drift coverage; its focused request suites pass 21
Python and 19 TypeScript tests. The inherited
528.54 kB build advisory remains. A temporary whole-tree secret-baseline scan
found no finding in this slice's changed paths, but still reported two
parent-existing findings in unchanged regional-surface-plan tests while PR
#4355's protected detect-secrets job remained pending. No protected or release
claim is made.
