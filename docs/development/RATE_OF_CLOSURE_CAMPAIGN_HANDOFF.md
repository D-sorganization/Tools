# Rate of Closure Ball-Flight Campaign Handoff

## 2026-08-11 local #4192/#4273 post-ground spatial-target projection

- The clean local candidate starts from exact published PR #4361 head
  `81de044075a4f72c6da8fedb972437df79a06ab8` and leaves the parallel
  ground-playback slice untouched. Its UI-neutral adapter accepts only exact
  pipeline-or-transfer evidence and an exact existing `SpatialTarget`, while
  reusing #4361's promoted complete-rest qualifier and evidence attributes.
- Numerics exist only for regional `COMPLETE` plus ground `COMPLETE/REST` with
  a summary. The output records the sole ground-v1 `GroundFrame.TARGET`
  (x-downrange/y-up/z-right), retains final x/z, replaces terminal ball-center
  y exactly once with the target's declared course-surface elevation, and then
  delegates geometry and signed long/high/right residuals to
  `SpatialTarget.miss`. App- and flight-authored targets are equivalent.
- Aerial targets are typed `AERIAL_REQUIRES_FLIGHT_TRAJECTORY`; transfer,
  non-settled bounce, regional cancelled/failed/partial, non-rest,
  `LEFT_SURFACE`, missing-summary, and censored outcomes retain null numerics
  with exact availability/phase/reason/frame/model/digest evidence. A bounded
  ordered scalar ensemble exposes hold, miss, downrange, elevation, and
  lateral results with deterministic row identity and provenance.
- RED captured the absent module. Sixteen new focused tests and seven parent
  adapter tests pass; all 1,315 selected Rate/flight/ground tests pass with 14
  environment-only Hypothesis collection warnings and one inherited
  polynomial-generator legend warning. Strict MyPy, focused Ruff,
  Bandit, campaign-manifest validation and eight tests, documentation,
  blocking-quality, minimum-test, changed-Python, 400-line module-size,
  changed-test assertion, placeholder, and diff gates pass. No PR/protected
  release, editor/UI, persistence, solver/capability,
  aerial trajectory evaluation, compiled runtime, geometry, or physics is
  claimed. Keep #4192, #4273, and #4267 open.


## 2026-08-11 PR #4361 qualified regional-ground study adapter

- Ready-for-review PR [#4361](https://github.com/D-sorganization/Tools/pull/4361)
  starts from exact published PR #4360 head
  `74f1ceafd87f952a76917dc868baa6414f856144`; its independently reviewed
  implementation commit is `d71c43fdd729b35e1abe5573f41ed60201698608`.
- Contract and historical-worktree audits retained only the existing
  complete-rest qualification invariant and scalar taxonomy. The adapter
  reuses `to_ground_model_result`, `FlightMetricInputs`, and
  `ScalarEnsembleDataset`; it does not import the stale parallel study model
  or its numeric censored endpoints.
- Regional complete plus ground complete/rest/summary evidence can populate
  canonical total, roll, final-offline, and bounce-count metrics and distinct
  bounce-air/skid/surface-path/final-downrange detail. Carry remains distinct.
  Partial/left-surface, every non-settled bounce reason, regional
  cancelled/failed, missing-summary, and typed transfer-error outcomes keep
  null values with exact typed status/reason/model/digest attributes. An
  unqualified outcome clears stale ground metric inputs.
- Seven focused tests and 1,299 Rate/flight/ground tests pass. Ruff,
  strict MyPy, pinned Bandit, manifest validation plus 8 manifest tests,
  documentation governance, blocking-quality, minimum-test, default module
  size, and diff gates pass. Inherited main-relative assertion and 400-line
  findings do not include this 328-line production module or its
  assertion-bearing test.
- Solver/capability invocation, variation UI, wind strategy, persistence,
  TypeScript/compiled and four-surface parity, protected CI/review,
  protected release, and #4273/#4267 completion remain open.

## 2026-08-11 PR #4360 flight-through-regional-ground pipeline

- Ready-for-review PR
  [#4360](https://github.com/D-sorganization/Tools/pull/4360) on
  `feat/4271-flight-regional-ground-pipeline` starts from exact published PR
  #4359 head `e53c6fb1bd273292c02085ee5d0a2b5497820871`; its reviewed implementation
  commit is `090e835477d1f19614f37f978a1b8a0e2f50ae21`.
- Audit showed the regional envelope cannot honestly represent bounce
  time/event limits or no-recontact. The new UI-neutral composition validates
  exact inputs, capture, and launch-relative plan/base equality before physics,
  then delegates only to existing flight/bounce and regional-ground
  authorities.
- Its strict bounded versioned in-memory result retains the exact bounce pair,
  ground and joint bounce-input digests, plan/digest/provenance, and optional
  existing regional envelope. The envelope exists exactly for a settled
  bounce; all five non-settled reasons remain native and skip regional physics.
  Canonical plan hashing is centralized in the regional-plan authority.
- RED captured missing module/result/exports; GREEN passed 17 pipeline/public
  tests; REFACTOR passed 39 pipeline/public/regional tests. All 377 flight and
  ground tests pass. Ruff check/format, scoped Black, protected and
  import-following MyPy, Bandit, placeholder/diff checks, documentation,
  blocking-quality, minimum-test, test-assertion, changed-Python, both LOC,
  campaign-manifest, and 11 manifest/layout gates are green. Explicit casts at
  dynamic wire-parser boundaries satisfy protected skipped-import MyPy without
  runtime or canonical-byte changes. Standalone Black retains one inherited
  preference in `test_contract_api.py`; authoritative Ruff is green and its
  delta contains only required public API entries.
- The ready PR remains `not_released` pending protected checks and review. No
  wire/migration, clients, TypeScript/Rust/WASM, persistence, playback,
  calibration, study integration, protected evidence, or release is claimed.
  Keep #4271, #4273, and #4267 open.

## 2026-08-11 PR #4359 shared Python flight-to-bounce composition

- Ready-for-review PR
  [#4359](https://github.com/D-sorganization/Tools/pull/4359) on
  `feat/4270-flight-bounce-execution` starts from exact clean published Tools
  #4357 head `c492b52f9f7615c5bc38e780965167cc8f64327c`; its reviewed implementation
  commit is `869b626e2d3ebd4097ae76b8fc9720cda6696947`.
- The public `execute_repeated_bounce_from_flight` seam validates exact flight,
  launch, and transfer types plus callback and capture inputs before transfer,
  then composes the existing request builder, strict repeated-bounce request,
  and UI-neutral executor. It adds no physics and preserves typed transfer
  failures and the existing identity/digest evidence.
- RED-GREEN evidence captured the missing module/export first. Independent
  follow-up coverage proves exact transfer-error messages, fields, and reasons
  plus zero executor calls for no-contact, grazing, and missing-angular-state
  paths. Seventeen focused tests and all 365 flight-plus-ground tests pass.
  Ruff check/format, scoped
  Black, protected and import-following MyPy, Bandit, placeholder/diff checks,
  documentation and blocking-quality governance, minimum-test/test-assertion
  contracts, changed-Python and module-size policies, the campaign manifest,
  and 11 manifest/layout tests are green. Standalone Black retains one
  inherited advisory in `test_contract_api.py`; authoritative Ruff is green
  and its only delta is the public API entry. The committed changed-file size
  gate is green with zero violations across four changed Python files.
- The ready PR is mergeable, but protected checks and review are pending, so it
  remains `not_released`. PyQt6/React UI,
  TypeScript/Rust/WASM physics, persistence, playback, camera behavior,
  regional chaining, skid/roll completion, final distance, protected evidence,
  and #4270/#4267 completion remain explicitly open.

## 2026-08-11 PR #4357 repeated-bounce request execution binding

- Ready-for-review PR
  [#4357](https://github.com/D-sorganization/Tools/pull/4357) on
  `feat/4270-repeated-bounce-execution` starts exactly from published #4356
  head `2387430fc78baa92ba122c7ad008a498118bf62d` and is published at
  implementation head `cf54d3528a71fd429ad19f53f04e4a1a84495097`.
- The candidate adds the UI-neutral Python
  `execute_repeated_bounce_request` boundary. It accepts only the exact strict
  request plus callable-or-`None` cancellation, consumes the request-bound
  capture threshold through fixed-version settings, invokes the existing
  Python physics authority, and returns the existing identity-validated
  request/result pair. Schemas, physics, TypeScript, and UIs are unchanged.
- TDD recorded the expected missing-public-executor failure. Qualification is
  green for 28 focused contract tests, the complete 189-test ground suite, 11
  campaign-manifest/layout tests, Ruff check/format, Black, protected
  changed-file MyPy, Bandit, clean placeholder/diff checks, documentation and
  blocking-quality governance, minimum-test/test-assertion contracts,
  changed-Python policy, module-size policy, and campaign-manifest validation.
  A non-authoritative import-following MyPy probe reports three inherited
  redundant casts in unchanged ground modules; the protected
  `--follow-imports=skip` profile is green for the new production module.
- This remains `not_released`; protected checks and review are pending. UI request construction
  and invocation, persistence, playback, TypeScript/compiled physics,
  regional-material chaining, measured terrain calibration, downstream parity,
  protected exact-head evidence, review, approval, release, and issue/epic
  completion remain open.
- Exact open dependencies #4203 through #4357 are now ready for review without
  base or history changes. The release reconciliation found no current-head
  failing check, but every open PR still had queued protected contexts and no
  submitted approval. The manifest records the reconciled parent heads and
  #4357 without claiming protected completion or release.

## 2026-08-11 PR #4356 published current-parent propagation

- Ready-for-review PR `#4356` retains branch
  `feat/4270-repeated-bounce-request-wire` and base
  `feat/4271-repeated-bounce-wire`. Exact current child
  `23897eac03e8a3edf4a37855f0ba05e8c2527986` is first and exact published PR
  #4355 head `a04d14e9308990e676e8c90ddb1d80e368dd1387` is second in normal
  no-ff merge `345c329e6b6e3fc7a8fc981abf65795f356b94cf`.
- The child retains its strict cross-runtime repeated-bounce request envelope,
  canonical ground-request and joint-execution-input digests, exact
  request/result identity pairing, shared golden corpus, adversarial
  capture-speed digest follow-up, and live-PR handoff while inheriting the
  complete #4355 result-wire and cancellation evidence, both typed-Boolean
  protected-MyPy repairs, and all regional/ground ancestry.
- Local qualification is complete: 1,099 Python tests, 116 React files with
  738 tests, complete Cargo workspace tests, focused 64-Python/53-React
  coverage, Ruff check/format and Black on all four child-delta Python files,
  protected MyPy on two child production modules plus the coherent 37-module
  ground profile, Bandit on both production modules, a clean placeholder scan,
  TypeScript, zero-warning ESLint, the 204-module Vite build, Rust
  formatting/clippy, both repository size budgets, the manifest validator and
  eight manifest tests, and every repository governance gate are green. All
  eight child feature/spec/test files remain byte-exact; the parent-only
  result-wire files and both inherited typed-Boolean repairs also remain exact.
  Known warnings remain the Hypothesis cache ignore, empty polynomial legend,
  Node local-storage flag, and 528.82 kB Vite chunk. The propagation was
  normally published, and exact heads #4351 through #4356 were marked ready
  for review without rebasing, retargeting, rewriting, force-pushing, merging,
  or changing their bases. The slice remains `not_released`.
- The first protected #4356 checkpoint had one successful quality check, four
  skipped checks, twelve queued checks, no failure, and no review.
- UI request construction, executor invocation, persistence, playback,
  measured calibration, compiled and downstream parity, protected exact-head
  completion, review, approval, dependency integration, release, and issue
  completion remain open.

## 2026-08-11 PR #4355 current-parent propagation candidate

- PR `#4355` retains branch `feat/4271-repeated-bounce-wire` and base
  `feat/4271-regional-trajectory-export`. Exact current child
  `b67af52226fa6334dd3570cf650aebeaf81912fc` is first and exact published PR
  #4354 head `97925e4803f4fbd72d576eb1c11c47f8e61b0b66` is second in a normal
  no-ff merge.
- The child retains its complete strict cross-runtime repeated-bounce
  result-wire contract, canonical golden corpus, phase/chronology/energy
  invariants, and pre-contact cancellation follow-up while inheriting regional
  trajectory inspection/export, both typed-Boolean protected-MyPy repairs, and
  the complete regional/ground ancestry.
- Local qualification is complete: 1,078 Python tests, 115 React files with
  719 tests, complete Cargo workspace tests, focused 43-Python/34-React
  coverage, Ruff check/format on all seven child-delta Python files, protected
  MyPy on five child production modules plus the coherent 36-module ground
  profile, Bandit on the five production modules, a clean placeholder scan,
  TypeScript, zero-warning ESLint, the 204-module Vite build, Rust
  formatting/clippy, both LOC budgets, the manifest validator and manifest
  tests, and every repository governance gate are green. All 12 child
  feature/test files and both inherited typed-Boolean repairs remain
  byte-exact. Standalone Black is non-authoritative by repository policy and
  reports one advisory formatting difference in the audited child
  `bounce_types.py`; authoritative Ruff is green, so that child file remains
  exact. Known warnings remain the Hypothesis cache ignore, Node local-storage
  flag, and 528.82 kB Vite chunk. The candidate has not been rebased,
  retargeted, rewritten, force-pushed, or published and remains
  `not_released`.
- Request construction, executor invocation, persistence, playback, measured
  calibration, compiled and downstream parity, protected exact-head evidence,
  review, approval, dependency integration, release, and issue completion
  remain open.

## 2026-08-11 PR #4354 current-parent propagation candidate

- PR `#4354` retains branch `feat/4271-regional-trajectory-export` and base
  `feat/4271-regional-event-inspection`. Exact current child
  `99b0739bdc3ece814ed6039e6ba31f7ac38c0227` is first and exact published PR
  #4353 head `e0433adbc3c82272745d098867f261462a790d08` is second in a normal
  no-ff merge.
- The child retains matched bounded PyQt6/React raw-trajectory inspection and
  canonical semantic-lossless evidence export while inheriting ground-event
  and regional-transition ledger inspection, the complete qualified result
  projection, the explicit Boolean local required by protected delta-MyPy,
  embedded-plan execution/provenance and request-I/O boundaries, and complete
  regional physics ancestry.
- Local qualification is complete: 1,058 Python tests, 114 React files with
  700 tests, complete Cargo workspace tests, focused 6-Python/8-React coverage,
  Ruff check/format on the three child-delta Python files, protected MyPy on
  two child production modules plus the coherent 35-module ground profile,
  Bandit on the two child production modules, TypeScript, zero-warning ESLint,
  the 204-module Vite build, Rust formatting/clippy, both LOC budgets, the
  manifest validator and manifest tests, and every repository governance gate
  are green. Protected delta-MyPy found the same skipped-import `no-any-return`
  boundary as #4351 in the new evidence exporter; its helper result is now
  assigned to an explicit Boolean local with no runtime or canonical-byte
  change. The other seven child feature/test files and inherited Boolean-local
  repair remain byte-exact. Standalone Black is non-authoritative by repository
  policy and its Python 3.13 runner cannot safety-parse the inferred 3.14
  target; authoritative Ruff is green. Known warnings remain the Hypothesis
  cache ignore, empty polynomial legend, Node local-storage flag, and 528.82 kB
  Vite chunk. The candidate has not been rebased, retargeted, rewritten,
  force-pushed, or published and remains `not_released`.
- Input construction, UI executor invocation, interpolation/playback,
  calibration workflows, compiled regional physics, downstream parity,
  protected exact-head evidence, review, approval, dependency integration,
  release, and issue completion remain open.

## 2026-08-11 PR #4353 current-parent propagation candidate

- PR `#4353` retains branch `feat/4271-regional-event-inspection` and base
  `feat/4271-regional-result-readback`. Exact current child
  `7fc00f43561c31923b74563bc2bf6caf89bbc9eb` is first and exact published PR
  #4352 head `12fc80798d2a15b44c0215688ffb031dd99cbdd1` is second in a normal
  no-ff merge.
- The child retains matched bounded PyQt6/React inspection of validated ground-
  event and regional-transition ledgers while inheriting the complete qualified
  result projection, the explicit Boolean local required by protected delta-
  MyPy, embedded-plan execution/provenance and request-I/O boundaries, complete
  regional physics ancestry, capability-only extended finite-float serializer,
  and default ground safe-number boundary.
- Local qualification is complete: 1,057 Python tests, 113 React files with
  698 tests, complete Cargo workspace tests, focused 76-Python/38-React
  coverage, Ruff check/format on the three child-delta Python files, protected
  MyPy on two child production modules plus the coherent 35-module ground
  profile, Bandit on the two child production modules, TypeScript,
  zero-warning ESLint, the 203-module Vite build, Rust formatting/clippy, both
  LOC budgets, the manifest validator and eight manifest tests, and every
  repository governance gate are green. Child feature bytes and the inherited
  Boolean-local repair are exact; conflict-marker and diff checks are clean.
  Non-failing warnings are limited to the known Hypothesis cache ignore, empty
  polynomial legend, Node local-storage flag, and 526.79 kB Vite chunk. The
  candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- Trajectory-sample inspection, lossless export, UI executor invocation,
  playback, calibration workflows, compiled regional physics, downstream
  parity, protected exact-head evidence, review, approval, dependency
  integration, release, and issue completion remain open.

## 2026-08-11 PR #4352 current-parent propagation candidate

- PR `#4352` retains branch `feat/4271-regional-result-readback` and base
  `feat/4271-regional-execution-ui`. Exact current child
  `10fdac4860035fd5c845a621752e93688e2e674e` is first and exact published PR
  #4351 head `4024c8a1ad2d3871c6b06ef6369250a873789c39` is second in a normal
  no-ff merge.
- The child retains its complete matched PyQt6/React qualified result
  projection while inheriting bounded evidence import/readback, the explicit
  Boolean local required by protected delta-MyPy, embedded-plan execution and
  provenance, request I/O, complete regional physics ancestry, the capability-
  only extended finite-float serializer, and default ground safe-number
  boundary.
- Local qualification passes all `1,057` combined Rate/shared-ground Python
  tests, all `113` React files / `697` tests, the complete Cargo workspace,
  `76` focused Python result/readback/execution/I/O/capability tests, and `37`
  focused React tests. Pinned Ruff 0.14.10 check/format passes three child-delta
  Python files; isolated-import strict MyPy passes both child production
  modules and the coherent 35-module ground profile passes with inherited
  imports skipped and only the parent's documented `redundant-cast` code
  disabled; Bandit passes both production files. TypeScript, zero-warning
  ESLint, the 202-module Vite build, Rust format and warning-denied clippy, both
  LOC gates, the campaign validator and eight manifest tests,
  docs/tool-manifest/blocking-gate/assertion/minimum-test governance, child-
  feature and inherited Boolean-local byte checks, marker scans, and diff
  checks pass. Existing Hypothesis ignored-cache, polynomial-generator empty-
  legend, Node local-storage option, and 523.34 kB Vite chunk warnings remain
  non-failing.
- The candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- UI executor invocation, trajectory/event tables, playback, calibration
  workflows, compiled regional physics, downstream parity, protected exact-
  head evidence, review, approval, dependency integration, release, and issue
  completion remain open.

## 2026-08-11 PR #4351 current-parent propagation candidate

- PR `#4351` retains branch `feat/4271-regional-execution-ui` and base
  `feat/4271-regional-execution-binding`. Exact current child
  `351a3051e9093c6b80cabf0f1db04aeeb15abfac` is first and exact published PR
  #4350 head `98f86990e9225903fbe84cd1f267ed38ef0a15d8` is second in a normal
  no-ff merge.
- The child retains matched bounded PyQt6/React execution-evidence import and
  readback, including the explicit Boolean local required by protected
  delta-MyPy, while inheriting the embedded-plan execution/provenance contract,
  request I/O, complete regional physics ancestry, capability-only extended
  finite-float serializer, and default ground safe-number boundary.
- Local qualification passes all `1,056` combined Rate/shared-ground Python
  tests, all `113` React files / `696` tests, the complete Cargo workspace,
  `75` focused Python evidence/readback/execution/I/O/capability tests, and `36`
  focused React tests. Pinned Ruff 0.14.10 check/format passes six child-delta
  Python files; isolated-import strict MyPy passes all five child production
  modules and preserves the Boolean-local repair; the coherent 35-module ground
  profile passes with inherited imports skipped and only the parent's
  documented `redundant-cast` code disabled; and Bandit passes those five files.
  TypeScript, zero-warning ESLint, the 202-module Vite build, Rust format
  and warning-denied clippy, both LOC gates, the campaign validator and eight
  manifest tests, docs/tool-manifest/blocking-gate/assertion/minimum-test
  governance, child-feature byte checks, marker scans, and diff checks pass.
  Existing Hypothesis ignored-cache, polynomial-generator empty-legend, Node
  local-storage option, and 521.54 kB Vite chunk warnings remain non-failing.
- The candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- UI executor invocation, playback, compiled regional physics, downstream
  parity, protected exact-head evidence, review, approval, dependency
  integration, release, and issue completion remain open.

## 2026-08-11 PR #4350 current-parent propagation candidate

- PR `#4350` retains branch `feat/4271-regional-execution-binding` and base
  `feat/4274-regional-plan-io`. Exact current child
  `dfb4b97481f187ff3594eceb08c427f650aca4e3` is first and exact published PR
  #4342 head `de66a851aa5dded680279cf9a2b25a5094966593` is second in a normal
  no-ff merge.
- The child retains its embedded-plan execution/provenance envelope, executor
  authority, transition binding, cross-runtime fixtures, and frozen
  base-result boundary while inheriting current request I/O, matched editors,
  complete regional physics ancestry, the capability-only extended
  finite-float serializer, and the default ground safe-number boundary.
- Local qualification passes all `1,052` combined Rate/shared-ground Python
  tests, all `111` React files / `692` tests, the complete Cargo workspace,
  `71` focused Python execution/I/O/capability tests, and `36` focused React
  tests. Pinned Ruff 0.14.10 check/format passes seven child-delta Python
  files; isolated-import strict MyPy passes the four execution modules and the
  coherent 35-module ground profile passes with only the parent's documented
  `redundant-cast` code disabled. Bandit passes five child production files.
  TypeScript, zero-warning ESLint, the 199-module Vite build, Rust format and
  warning-denied clippy, both LOC gates, the campaign validator and eight
  manifest tests, docs/tool-manifest/blocking-gate/assertion/minimum-test
  governance, child-feature byte checks, marker scans, and diff checks pass.
- The first CPU-contended Python run produced `1,051` passes and one Hypothesis
  input-generation `too_slow` health check. The property passed alone and all
  `1,052` tests passed in the single uncontended rerun.
- The candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- Execution UI/playback, compiled regional physics, downstream parity,
  protected exact-head evidence, review, approval, dependency integration,
  release, and issue completion remain open.

## 2026-08-11 PR #4342 current-parent propagation candidate

- PR `#4342` retains branch `feat/4274-regional-plan-io` and base
  `feat/4274-regional-surface-ui`. Exact current child
  `c1f47f2ef68b3db102da5416aaac17a40f675207` is first and exact reviewed
  local #4339 candidate `db335937afc4b587d235eb705e315f577519c5e6` is
  second in a normal no-ff merge.
- Child-owned canonical request import/export, bounded UTF-8, native atomic
  save, browser-qualified download, tests, and limitations remain intact while
  inheriting current editor, wire, regional-physics, and complete ground
  ancestry.
- The default shared canonical encoder still rejects floats and integers beyond
  JavaScript's safe range. The capability-observation facade alone selects a
  separately named extended finite-float policy that reuses the shared
  recursion, keeps integers bounded, emits exact exponent-free `1e20` and
  `1e21` tokens matching TypeScript, and rejects non-finite values.
- Local qualification passes all `909` Rate-of-Closure Python tests, all `110`
  React files / `686` tests, the complete Cargo workspace, `47` focused Python
  compatibility/regional-I/O tests, and `12` focused React capability tests.
  Pinned Ruff 0.14.10 check/format passes `17` changed Python files; pinned
  MyPy 1.13 and Bandit pass `12` changed production files. TypeScript,
  zero-warning ESLint, the 199-module Vite build, Rust format and
  warning-denied clippy, both changed-file LOC gates, manifest/docs/blocking-
  gate/assertion/minimum-test governance, marker scans, and diff checks pass.
  One untouched manual-delivery UI test timed out in the first concurrent full
  run, then passed alone and in the single complete rerun.
- The candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- Execution/playback, result interchange, measured calibration, model-input
  persistence, changing geometry or velocity, TypeScript/compiled regional
  physics, downstream parity, protected exact-head evidence, review, approval,
  dependency integration, and release remain open.

## 2026-08-11 PR #4339 current-parent propagation candidate

- PR `#4339` retains branch `feat/4274-regional-surface-ui` and base
  `feat/4271-regional-wire-contract`. Exact current child
  `d21741e312b849a63f73cabf351a15d9de80fb94` is first and exact published
  PR #4335 head `8f933ed8dcb29e55ece4ec6bb1e60813f6794d57` is second in a normal
  no-ff merge.
- The matched PyQt6/React regional surface editors retain validation,
  invalidation, engineering hints, and strict request readback while inheriting
  current wire/resolver/regional-physics/ground ancestry. The extracted PyQt
  navigation-state contract includes `regional_surfaces` in default and legacy
  migration order.
- Local qualification passes all `891` Rate-of-Closure Python tests, all `110`
  React files / `678` tests, `177` focused regional/ground/navigation Python
  tests, `14` focused React editor/navigation/wire tests, and all `137`
  `tools-core` Rust tests. TypeScript, zero-warning ESLint, the 198-module Vite
  build, Rust format and warning-denied clippy, pinned Ruff 0.14.10 across seven
  PR-delta Python files, pinned MyPy 1.13 across six production files, Bandit
  medium/high, 400- and 500-LOC changed-file gates, manifest/docs/assertion/
  minimum-test governance, child-feature byte checks, conflict-marker scans,
  and diff checks pass. Existing Hypothesis ignored-cache and Node local-storage
  option warnings remain non-failing.
- The candidate has not been rebased, retargeted, rewritten, force-pushed, or
  published and remains `not_released`.
- Execution/playback, result interchange, measured calibration, model-input
  persistence, changing geometry or velocity, TypeScript/compiled regional
  physics, downstream parity, protected exact-head evidence, review, approval,
  dependency integration, and release remain open.


## 2026-08-11 PR #4351 delta-MyPy boundary repair candidate

- Exact PR #4351 head `fe463b5503a8c7b599a329da18bb690d008871cd`
  fails the protected changed-file MyPy profile because
  `--follow-imports=skip` makes the imported atomic writer `Any` at this root.
- The regional plan writer now assigns that call to an explicitly typed local
  before returning it. This remains valid when the helper is also a MyPy root,
  avoiding both `no-any-return` and a conditionally redundant cast.
- No request bytes, validation, cancellation behavior, atomic persistence,
  UI behavior, or physics changes. This is a local no-publish candidate;
  descendants #4352/#4353/#4354 still require ordered propagation and their
  own protected exact-head evidence.


## 2026-08-11 regional execution current-parent reconciliation candidate

- The clean `feat/4271-regional-execution-binding` worktree normally merges
  exact reviewed child `012cdfc33ad1590f31a1cbb109f0b8bee8eee700` with exact
  newly published PR #4342 parent
  `c1f47f2ef68b3db102da5416aaac17a40f675207` as its second parent.
- The intended base remains `feat/4274-regional-plan-io`; no rebase, retarget,
  rewrite, force-push, publication, PR creation, protected-evidence, or release
  claim is made.
- The child's remediated execution/provenance contract and the parent's I/O,
  helper, and verbatim append-only history are retained together.
- Qualification is 143 Python ground tests and 24 React
  execution/plan/editor tests passing. Ruff passes all 50 ground files; strict
  MyPy passes the four execution modules, and the 35-module ground profile
  passes with only the parent's documented redundant-cast warning disabled.
  Bandit, TypeScript, zero-warning ESLint, the 199-module build, manifest and
  eight manifest tests, docs/tool-manifest governance, changed-Python,
  minimum-test, 500-LOC, heading/SPEC preservation, diff, and whitespace gates
  pass. Structural maxima are 376 TypeScript lines, 281 Python lines, 43
  function lines, and four parameters.
- A broader formatter sweep identifies three parent-only Rate tests that
  current Ruff would reformat. Exact execution/ground formatting is clean;
  this child does not rewrite the published parent baseline. The existing
  approximately 500 kB React chunk warning remains nonblocking.

## 2026-08-11 regional execution review remediation

- Independent review rejected local commit `696a3ff8f...` because its Python
  transition wire was more permissive than TypeScript, its fixture was not
  executor output, executor identity was not fixed, and from/to identities
  were not bound to the referenced plan.
- The repaired v1 envelope embeds the exact plan, recomputes its digest, binds
  plan/source/base identities, fixes executor producer/version, and validates
  every ledger row against both its ground event and a real plan boundary
  crossing. Shared adversarial and executor-produced outcome fixtures cover
  cross-runtime canonical validation and representable/cancelled/failed states.
- Null-result cancellation/failure evidence requires an empty transition
  ledger because no embedded result exists to substantiate transition rows.
- Frozen base-result v1 remains unchanged. UI, compiled regional physics,
  downstream consumers, protected evidence, and issue completion remain open.
- Separate baseline: the capability-observation test with a `1e20` value fails
  on exact parent `8e1c7ccd...` and this child with `ValueError: canonical JSON
  number exceeds cross-runtime safe range`; it is not attributed to this work.

## 2026-08-11 regional execution/provenance child

- `feat/4271-regional-execution-binding` begins at exact current PR #4342
  head `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`; the parent is not
  rewritten or retargeted.
- `execute_regional_ground` validates exact request/prefix/plan identities,
  requires `plan.base_surface == request.surface`, creates the only resolver
  from that plan, and delegates to the established Python solver/composer.
- Strict `ground-regional-execution-result/v1` carries canonical input digests,
  plan/executor/model provenance, exact ordered from/to transition evidence,
  and fixed coplanar/static limits. Frozen ground-result v1 is embedded only
  for representable complete/partial outcomes; typed cancellation/failure has
  a null result rather than fabricated physics. TypeScript is wire-only.
- No new controls, playback, compiled regional solver, downstream consumer,
  protected evidence, release, or #4271 completion is claimed.



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

- The clean `feat/4274-regional-plan-io` worktree normally merges exact
  published PR #4342 child `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`
  with exact newly published PR #4339 parent
  `d21741e312b849a63f73cabf351a15d9de80fb94` as its second parent.
- PR #4342 retains base `feat/4274-regional-surface-ui`; no rebase, retarget,
  rewrite, force-push, publication, protected-evidence, or release claim is
  made by this reconciliation.
- Strict regional request import/export is preserved. The child widget module
  delegates canonical-precision controls to the parent's frozen
  `NumberInputSpec` and three-parameter helper while retaining inclusive safe
  number bounds and eleven-decimal presentation.
- Merged-tree qualification is 53 focused Python/PyQt/shared-ground tests and
  14 focused React tests passing. Ruff, MyPy, Bandit, TypeScript, zero-warning
  ESLint, the 199-module production build, campaign manifest and eight manifest
  tests, documentation/tool-manifest governance, changed-Python, minimum-test,
  exact-diff assertion, 500-LOC, parent/child diff, and whitespace gates pass.
  Structural maxima are 396 module lines, 40 function lines, and three
  parameters.
- Independent review, ordinary fast-forward publication, fresh protected
  CI/approval, dependency integration, and release remain required.

## 2026-08-11 PR #4339 structural helper repair candidate

- Parent head `d21741e312b849a63f73cabf351a15d9de80fb94` replaces the regional
  PyQt six-parameter input helper with a frozen validated `NumberInputSpec`
  and three-parameter helper without changing field behavior or ordering.
- Eight focused Python/PyQt tests, 25 focused React tests, and the complete 672
  React tests pass with Ruff, MyPy, Bandit, TypeScript, ESLint, build,
  documentation, manifest, size, structural, and diff gates clean.
- No schema, digest, regional physics, PR base, protected evidence, or release
  boundary changes.

## 2026-08-11 PR #4339 current-parent ancestry candidate

- The parent worktree normally merged exact editor child
  `cbb9c0a6bdc6a50f59f7a661139b9d53e1892980` with exact published #4335
  parent `9e01ccc3e891cc45907293751a192624195a77a5` while retaining the
  `feat/4271-regional-wire-contract` base.
- UI, wire, resolver, regional physics, and ground ancestry were preserved;
  protected exact-head review, integration, and release remained open.

## 2026-08-11 PR #4342 delta-MyPy follow-up

- Protected CI on exact head
  `cffe349ac0a8054f1d168cb36684fd00bc5f8a49` rejected one redundant Boolean
  cast in the regional atomic-write adapter.
- The cast is removed and the typed helper's direct return is unchanged.
  Focused persistence tests and the CI-equivalent pinned MyPy command pass.
- No wire, validation, file, UI, or physics behavior changes. Fresh protected
  CI/review and dependency ordering remain open.

## 2026-08-11 regional request I/O protected publication

- Branch `feat/4274-regional-plan-io` is published normally as draft PR
  [#4342](https://github.com/D-sorganization/Tools/pull/4342), targeting exact
  PR #4339 branch `feat/4274-regional-surface-ui` at parent head
  `cbb9c0a6bdc6a50f59f7a661139b9d53e1892980`.
- Reviewed implementation head
  `d748e7a5ef3da5e6ce7737ff6829e0f14665fe97` includes canonical PyQt6/React
  request import/export, safe-number parity, and bounded strict UTF-8 native
  reads. Publication documentation commits change no runtime code.
- Protected CI, review, #4274, dependency ordering, integration, and release
  remain open.

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

Status verified 2026-08-08. This isolated integration is published as draft
[PR #4217](https://github.com/D-sorganization/Tools/pull/4217). No source PR
branch was rewritten.

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

## 2026-08-11 Regional execution evidence readback continuation

The next bounded #4267/#4271 slice is local on
`feat/4271-regional-execution-ui`, based exactly on published PR #4350 head
`dfb4b97481f187ff3594eceb08c427f650aca4e3`. Matched PyQt6 and React plan
surfaces import and present strict Python executor evidence only when the
embedded regional plan exactly matches the current valid editor plan. The
readback exposes partial/complete/failure truth, termination, model, distinct
skid/roll/total metrics, transition count, and executor provenance. Invalid
imports preserve the last accepted evidence; plan changes invalidate it.

This is deliberately not browser physics and not UI executor invocation. No
ground request or settled bounce prefix is synthesized from illustrative
editor state. Local gates are green: 207 expanded Python tests and 111 React
files / 690 tests, strict MyPy, Ruff/format, TypeScript, zero-warning ESLint,
production build, manifest and eight manifest tests, docs governance,
structural budgets, and diff checks. The inherited 500 kB build advisory
remains. Complete an independent audit before publication; then create a normal
stacked draft child without retargeting or rewriting PR #4350.

## 2026-08-11 Complete regional result readback continuation

Local branch `feat/4271-regional-result-readback` starts exactly at draft PR
#4351 head `fe463b5503a8c7b599a329da18bb690d008871cd`. The next bounded result-
usability slice makes matched PyQt6/React evidence inspection complete for the
existing frozen result: carry, bounce-air, skid, roll, surface path, total,
final downrange/offline, bounce count, ground time, completion, model and
surface-provider identities/versions, calibration evidence, ordered observed
phases, typed warnings, executor provenance, and qualification limits.

Cancellation/failure cannot fabricate ground-only values, and partial results
remain visibly censored. React still executes no ground physics. Trajectory and
event tables, export/playback, executor input construction/invocation, measured
calibration, compiled parity, Upstream consumers, protected evidence, release,
and issue completion remain open.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 691 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 202-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The inherited 500 kB build advisory
remains. Independent review is required before publication.

## 2026-08-11 Regional execution ledger inspection continuation

Local branch `feat/4271-regional-event-inspection` starts exactly at draft PR
#4352 head `10fdac4860035fd5c845a621752e93688e2e674e`. The bounded matched-
client slice presents the already-validated ground-event and plan-bound
regional-transition ledgers without running or approximating physics. Events
retain sequence/type, frame, SI time/position, and before/after linear and
angular velocities. Transitions retain the matching event, SI time/position,
and from/to region and surface identities.

The complete accepted envelope remains resident, while each table renders a
maximum of 256 rows and discloses the exact total if truncated. Null results
show empty ledgers and partial endpoints remain censored. Trajectory-sample
inspection, lossless export, executor construction/invocation, playback,
measured calibration, compiled parity, Upstream consumers, protected evidence,
release, and #4267/#4271 completion remain open.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 692 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 203-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The inherited 500 kB build advisory
remains. Independent review is required before publication.

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
