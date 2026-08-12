# AGENT_HANDOFF — rate_of_closure

## 2026-08-12 #4378 deterministic static-inspection distribution

Branch `codex/4378-static-inspection-runtime` starts exactly from Tools PR
#4376 head `d14ce6f8ef2696ccaa8971443fa4df7e8d52f21f` and must target
`codex/4369-authority-restart-recovery`. Human review is approved; protected
dependency order and ordinary non-admin merging remain mandatory.
Published child PR: #4388.

The built React application now enters explicit `static_inspection` mode. That
mode publishes a stable false capability, performs zero authority queries, and
keeps prepare/run/cancel/status/result/recovery controls disabled. Existing
canonical import, validation, visualization, and evidence download remain local
inspection features. A malicious or unrelated origin-root `/api` response
cannot enable execution because the built App never delegates capability
discovery to `fetch`.

Browser bootstrap parses exactly one embedded JSON runtime descriptor before
mounting the App and renders a local fail-closed error for malformed metadata.
The separately packaged descriptor is the same manifest-hashed byte contract;
Python readiness requires `static_inspection` and binds its release revision to
the asset manifest.

The Vite build emits no public source maps. A post-build Node contract writes a
strict versioned runtime descriptor and a deterministic path-sorted asset
manifest with exact byte sizes, SHA-256 digests, media types, non-executable
flags, bounded totals, and explicit development-or-commit identity. It rejects
unsupported files, ambiguous revisions, symlinks, case collisions, and
post-manifest substitution. Python mirrors the exact manifest boundary,
rejects duplicate JSON fields and unsafe inventories, verifies link/reparse,
open-file identity, size, and digest constraints, and returns immutable bytes
instead of filesystem paths that a future gateway could reopen.

The root wheel includes only the declared built web resources under explicit
package-data rules. Local evidence builds a 72-file wheel inventory (71 declared
assets plus the manifest), excludes source maps, installs the exact wheel
without dependencies into a clean temporary environment, and resolves the same
71 immutable assets from package resources under an unrelated working
directory. This is distribution evidence for static inspection only.

The setuptools build hook refuses to include an existing web bundle unless an
exact `ROC_RELEASE_REVISION` matches a clean checkout and the resolved bundle.
Before copying package data it clears only the verified Rate-of-Closure web
staging subtree, preventing obsolete hashed chunks from a prior wheel build.
The pinned-action distribution workflow rebuilds the exact PR head, runs the
complete frontend gate, builds one wheel, verifies every wheel member, installs
without dependencies, and resolves from an unrelated Unicode working
directory. It also proves the intentional generic no-bundle wheel branch. A
source distribution or generic wheel is not a qualified web distribution;
exact-revision web wheels require the trusted clean Git build job.
The public repository is fetched credential-free at the exact event SHA on an
ephemeral hosted runner. This avoids the checkout action's cleanup failure on
the repository's intentionally unregistered historical gitlink and leaves no
repository credential for PR-controlled code.

Hosted exact-head run `31587364259` completed checkout and the full React gate,
then failed before Python test collection because the isolated package-side
step installed only Pytest while the repository's declared Pytest arguments
require the benchmark and xdist plugins. The workflow now installs pinned
`pytest-benchmark==5.2.3` and `pytest-xdist==3.8.0` with `pytest==9.0.2` before
running the distribution contract. This is CI-environment repair only; it does
not change product behavior, distribution contents, or the release contract.

The production same-origin gateway, browser Playwright qualification, Windows
authority-state ACL/reparse hardening, frozen web/PyQt artifacts, installers,
signing, SBOM/attestation, calibrated or compiled physics, downstream parity,
protected release, and issue/epic closure remain open under #4377 and
#4379-#4385.


## 2026-08-12 #4369 durable loopback authority recovery

Branch `codex/4369-authority-restart-recovery` starts exactly from approved
Tools PR #4375 head `9561dae8c048a511722619a7dfdaf065bb1667c7` and must
target `codex/4369-flight-recompute-cancellation`. Human review is approved;
normal dependency order, protected checks, and non-admin merge behavior remain
mandatory.

The source-run React launcher now injects a private, fixed authority state root.
The isolated Python authority owns a versioned SQLite/WAL store under a
process-lifetime file lock. It transactionally retains canonical job, status,
and complete-result bytes plus independent SHA-256 digests, rejects unknown
schema/version, lock contention, integrity failure, digest substitution,
impossible status/result pairs, and oversized state, and never persists the
ephemeral bearer token, raw exceptions, callbacks, threads, or partial rows.
Queued/running work recovered after process loss becomes terminal
`execution_failed/authority_restart`; a recovered cancel request becomes
cancelled. Startup never invokes or replays physics.

The React Ground Study surface adds an explicit **Recover retained status**
action for an exact accepted/imported job. Recovery performs capability,
status, and complete-result reads only. It never POSTs a job, cancels work,
restores confirmation, or substitutes browser physics. Completed result bytes
and terminal evidence survive hard authority termination; ambiguous or corrupt
state fails closed. PyQt continues using its direct worker, so this slice makes
no PyQt restart-recovery claim.

The `rate-of-closure-web` extra now declares the qualified SciPy and file-lock
runtime dependencies, and the root wheel includes `rotation_converter`, which
the installed authority import graph requires. A clean extra installation can
import the durable authority store without relying on the development checkout.

The first hosted #4376 quality gate exposed one pinned-MyPy-1.13-only inference
gap in the deterministic status-fixture stage matrix. The corrective follow-up
types that matrix against the shared wire literal union; fixture bytes and all
runtime behavior remain unchanged. The exact pinned CI MyPy command, fixture
check, Ruff/format, and manifest gates pass locally before the follow-up push.

The next exact-head #4376 gate passed checkout, dependencies, Ruff, format,
pinned MyPy, and quality policies, then Bandit rejected the retention cleanup's
dynamically sized parameter-placeholder SQL. The correction reads bounded
stored identifiers and deletes obsolete rows with one static parameterized
statement via `executemany`; transaction rollback and retained-row identity
remain covered. This is an actionable security-gate fix, not a CI retry.

Local gates are green: 1,252 Rate-of-Closure Python/PyQt tests plus one
Windows-skipped POSIX-permission test with eight workers; 137 React files and
909 tests; TypeScript; zero-warning ESLint; the production Vite build; 74
focused Python store/API/process tests plus the same POSIX skip; 49 focused
React contract/controller/UI tests; deterministic fixture regeneration;
Ruff check/format; changed-source MyPy; the 400-line module budget; and
`git diff --check`. The inherited polynomial-generator empty-legend warning
and existing Vite chunk-size advisory remain unchanged.

The campaign manifest records PR #4375's exact carrier head. This child cannot
self-record its future PR number. Static-host companion discovery, frozen
runtime qualification, PyQt direct-worker recovery, measured calibration,
compiled/TypeScript regional physics, UpstreamDrift parity, ancestor
integration, protected release, and closure of #4369/#4273/#4267 remain open.


## 2026-08-12 #4369 qualified flight-recompute cancellation

Branch `codex/4369-flight-recompute-cancellation` starts exactly from Tools PR
#4374 head `e1da8708c15f966afc926e20eae3fdf084ba8a16` and must target
`codex/4369-editor-job-preparation`. Human review is approved for this campaign
continuation; repository protections and required checks still apply.

The Waterloo/Penner surface solver and registered profile recomputation now
accept one optional keyword-only exact-Boolean cancellation callback. The job
manager's existing cancellation event is polled before integration, on adaptive
derivative boundaries, through dense-output materialization, after metrics,
during chunked retained-trajectory construction, and during canonical evidence
serialization. Cancellation publishes no partial result and maps to typed
zero-of-total cancellation. Raising or non-Boolean callbacks preserve their
original cause internally and map to the existing non-secret callback-failure
stage. Callback-free and always-false paths preserve existing canonical flight
digests and every job/wire schema.

Independent review found and this branch closed the original post-solver latency
gap, duplicate comparison-digest work, oversized profile module, and a SciPy-call-count
coupled regression test. Deterministic tests cover solver, post-solver digest,
manager shutdown, callback-defect, no-regional-physics, exact signature, facade,
and digest-parity behavior. Final local evidence is 1,233 Rate of Closure tests
with eight workers, 87 focused cancellation/profile/job/runner tests, manifest
validation plus eight manifest tests, Ruff check/format, changed-source typing,
module-size and `git diff --check` gates. This slice changes no React or PyQt
visual surface, so the parent PR #4374 React/PyQt visual and build evidence
remains the relevant UI gate.

The campaign manifest now records parent PR #4374's exact head. This child must
not self-record its future PR number; verify and add that carrier in the next
implementation commit. Durable authority restart recovery, static-host and
frozen-runtime qualification, measured regional calibration, compiled or
TypeScript regional physics, downstream UpstreamDrift parity, ancestor-stack
integration, protected release, and closure of #4369/#4273/#4267 remain open.



## 2026-08-11 #4369 current-editor job preparation

Local branch `codex/4369-editor-job-preparation` starts exactly from corrected
PR #4373 head `cca7c839f8fcaeab57d43fcb9d6f3df3b428e3c4`. It adds a strict
bounded preparation-request/v1 boundary and a Python-only registered-profile
builder that recomputes flight evidence, derives canonical trajectory/result,
input, provenance, qualified-plan, and job digests, and preserves the current
Ground/Tee setup. The authenticated preparation endpoint returns a canonical
job without retaining, enqueueing, or running it.

PyQt6 now accepts only a current successful Simulation hit; any relevant editor
change or failed/missed rerun preserves historical playback but invalidates that
run as preparation authority. Ground Study can prepare, review, save, and then
separately confirm/run the accepted job. Failures preserve the prior accepted
job/result and suppress private causes. The preview discloses flight settings,
transfer/capture bounds, callback-free regional settings, and the explicitly
UNVALIDATED zero-confidence editor calibration.

React owns and validates the same full launch snapshot, sends it with the exact
validated variation/surface request through the same-origin authority client,
strictly binds the returned job to the captured request, and transactionally
accepts it with confirmation cleared. Browser code performs no preparation or
execution physics, and preparation never auto-submits.

Final local evidence is 1,223 Rate of Closure Python/PyQt tests with eight
workers, 134 focused authority/profile/preparation/UI regressions, and a
post-review 49-test PyQt preparation/simulation rerun. React is green across
137 files and 905 tests, TypeScript, zero-warning ESLint, and a 229-module
production build. Ruff, format, Python 3.11/MyPy 2.1 changed-source typing,
module-size budget, manifest validation plus its eight tests, and `git diff
--check` are green. An independent final review found no remaining P0, P1, or
P2 findings. The default 14-worker full-suite stress run twice hit a loopback
poll timeout on this loaded workstation; both isolated tests and the complete
eight-worker suite passed. Compiled MyPy 1.13 under local Python 3.13 also hit
an internal cache-serialization assertion, so the protected pinned CI check
remains required. The campaign manifest records exact carrier heads through PR
#4373.

This child targets corrected PR #4373; verify its live PR number, exact head,
and protected checks after publication rather than relying on this commit to
self-record future GitHub state. Durable authority restart recovery,
static-host execution, frozen-runtime qualification, cooperative cancellation
inside flight recomputation, measured calibration, compiled/TypeScript regional
physics, UpstreamDrift consumers, ancestor integration, protected release, and
#4369/#4273/#4267 completion remain open.


## 2026-08-11 #4369 imported-job accessibility correction

Physical browser inspection of PR #4373 found that the hidden file inputs and
their visible proxy buttons both appeared as import actions in the accessibility
tree. The Ground Playback panel and contextual File menu now use truly hidden,
programmatically activated inputs, leaving each visible button as the sole
accessible action. Focused React coverage proves the single-action contract and
preserves strict file-import behavior across workspace navigation.

## 2026-08-11 #4369 PyQt6 toolstrip protocol correction

Hosted PR #4373 quality-gate job 94021363392 found that the concrete main
window implemented all regional-ground execution File callbacks while the
`ToolstripHost` structural protocol declared only the variation callbacks.
The protocol now includes Open Job, Save Job, Save Result, and Export Rows CSV;
this is a typing-only correction with no runtime behavior change. Reproduce with
the pinned MyPy 1.13 changed-source profile and `MYPYPATH=src:src/python/src`.

## 2026-08-11 #4369 integrated web-launch repair

Physical launch verification found that Vite 7 mutates each proxy adapter with
internal routing fields. The strict authority proxy builder now validates the
loopback URL and ephemeral token, then returns a fresh mutable server-owned
adapter instead of a frozen object. The token remains confined to Vite and is
not emitted to browser code. `authorityProxyConfig.test.ts` pins both validation
and framework extensibility; use `src/rate_of_closure/launch_web.py` for the
integrated authority-backed client. A physical launch returned HTTP 200 from
`http://localhost:5193/` and the proxied capability endpoint returned the exact
qualified `available=true`, `regional_ground_execution=true` v1 evidence.

## 2026-08-11 #4369 imported-job execution surfaces

React now owns one regional-ground execution workspace above workspace
navigation. Ground Playback mounts a visible strict imported-job surface with
exact identity/provenance/digest evidence, explicit confirmation, Run, Cancel,
progress, typed failure, ambiguous-request reconciliation, canonical job/result
downloads, and lossless scalar-row CSV export. Active or uncertain authority
ownership prevents job replacement, the same immutable authority job cannot be
silently resubmitted, and neither editor state nor TypeScript physics enters the
job. Contextual File commands share the same App-owned state.

PyQt6 now exposes a dedicated Ground Study module between Ground Surfaces and
Ground Playback. It uses bounded strict UTF-8 import, QThread execution and
cooperative cancellation, safe error presentation, exact job/result evidence,
and native atomic job/result/CSV writes. Close blocks until its worker is
cancelled and joined. Direct embedded construction remains unavailable unless
an authority is injected; the source standalone launcher injects the qualified
direct Python runner without Uvicorn, while frozen distributions remain
explicitly unqualified.

Focused evidence passes 76 PyQt navigation, toolstrip, controller, file,
workspace, standalone, registration, and manifest tests. React passes its complete
875-test suite across 135 files, strict TypeScript, zero-warning ESLint, and the
production build. Ruff, Black, pinned MyPy 1.13, and diff hygiene pass for the
Python delta. Current-editor job construction, restart recovery, static-host
execution, frozen-runtime qualification, compiled/downstream parity, protected
integration, release, #4369, #4273, and #4267 remain open.

## 2026-08-11 #4369 qualified headless authority admission

Independent post-admission review verified deterministic submit-versus-close
and exceptional-lifespan regressions, exact Python reason/detail typing, and
matching TypeScript detail/media-type enforcement. Treat older no-runner text
below as historical slice evidence rather than current service state.

The default environment factory now constructs one inseparable pair: the
qualified production runner and the exact true/true service capability. Exact
Python/TypeScript discriminants, capability/manager construction checks,
authenticated bounded readiness parsing, and cooperative manager shutdown
prevent false-ready and ready-without-runner states. The React controller uses
the qualified production capability directly; the test-only admission seam is
removed.

Combined xdist execution may register the two documented ground-variable
extensions before the shared registry contract test. That contract now pins
the five built-in launch entries as an ordered prefix, eliminating test-order
coupling without changing the registry or production variable definitions.

Independent review reproduced and closed submit-versus-close and exceptional
lifespan-exit races. Python and React now also share exact runtime reason,
detail-length, media-type, and bounded-body capability validation.

Complete local evidence passes 2,014 Python/PyQt/shared-simulation tests with
one optional Rust-wheel parity skip and 860 React tests across 132 files. The
pinned MyPy 1.13/Ruff 0.14.10 profiles, Ruff format, high-severity Bandit,
TypeScript, zero-warning ESLint, 214-module production build, fixture,
campaign-manifest, governance, and diff-hygiene gates pass as well.

Do not mistake service readiness for an editor workflow. There is still no
production execution-job constructor in either live client, the evidence
presentations remain unmounted/disabled, static React has no backend, PyQt
packaging has not qualified a Uvicorn helper, and job/result retention is not
durable. Implement strict imported-job execution next and preserve every open
security, persistence, compiled-parity, downstream, and protected-release gate.

## 2026-08-11 #4369 qualified fixture, runner, and matched presentation

The canonical authority fixture family is now generated from registered-profile
flight evidence and mutually bound job/status/result identities. Qualified
execution reuses one recomputed flight across seeded regional-ground trials,
forwards cancellation through the physical solver, and publishes no partial or
unbound result. The real authenticated loopback success path is exercised.

Python and TypeScript presentation models preserve exact identity, provenance,
progress, cancellation, typed failure, and complete result evidence. PyQt6 is
observer-only and React has no handlers; at that earlier fixture slice both
controls were disabled from the false capability. The newer headless admission
supersedes only the service state; visible integration, protected evidence, and
downstream parity remain mandatory.
Consolidated evidence passes 2,004 Python/PyQt/shared-simulation and 858 React
tests plus real-loopback, static, security, build, fixture, documentation,
manifest, and diff gates. The optional Rust parity test is skipped because the
local interpreter has no `swing_core` wheel, so compiled parity remains open.

## 2026-08-11 #4369 hosted MyPy 1.13 correction

The exact PR #4372 head
`e91ef8dcde8cdd8e6545ffc0ea7cb755058ec2fb` reached the hosted pinned MyPy
1.13 delta gate after Ruff and formatting passed. Its only error was a
redundant cast applied after an exact-bool cancellation callback check. The
cast and unused import are removed; callback validation, typed failure
behavior, false production capability, and every physics/release limitation
remain unchanged. The pinned MyPy 1.13 profile and 13 focused submitter tests
pass locally; protected rerun evidence remains required.

## 2026-08-11 #4369 composed authority continuation

The current continuation composes strict status parity, fail-closed production
preflight, authenticated PyQt submission over the real loopback process, exact
flight-profile recomputation, and an abortable React execution controller. The
canonical job remains unqualified because its synthetic declared flight hashes
do not match recomputation. Production therefore exposes no runner, true
capability, visible execution control, successful ground physics, or release
claim.

Complete local evidence passes 1,148 Python/PyQt and 854 React tests, real
process retesting, static typing/lint/security gates, the production build, and
all campaign documentation and structural gates. The next agent must preserve
the false capability until a profile-produced canonical job is pinned and the
complete flight-through-ground result is qualified; protected CI and ancestor
integration remain open.

## 2026-08-11 local #4369 PyQt real-loopback qualification

The real PyQt transport/submitter now has process-level evidence against an
actual loopback Uvicorn/FastAPI authority. Tests prove authenticated canonical
job lifecycle behavior, typed preflight failure, cooperative cancellation,
bounded close, token non-exposure, process ownership, and the current
false-capability factory returning no adapter.

A validated application-factory runtime seam supports this integration without
changing the default server factory. The only injected runners are fail-closed
preflight and cancellation-only test doubles; no flight/ground physics succeeds
or is claimed, and there is no production registration or visible control.
## 2026-08-11 local #4369 versioned flight execution profile

From exact local composed head
`7e4069e891d8b4bde3f1d712b5b47897359a414e`, the application layer now owns
one exact versioned flight profile for `waterloo_penner` /
`tools-core/1.0.0`. The registry strictly validates the three settings and
binds them to launch-relative planar-contact Waterloo/Penner recomputation,
base sampling, deterministic decimation, and terminal retention.

The evidence contract exposes only stable qualification reasons and computed
digests; the result-returning boundary fails unless both declared hashes match.
The canonical job's declared synthetic hashes do not match deterministic
recomputation, so production preflight remains zero-of-total failed and never
enters regional-ground physics. No runner injection, capability promotion,
visible execution control, persistence, or release is included. Keep
#4369/#4273/#4267 open until a profile-produced fixture, pinned runtime,
ground runner, and matched clients are qualified.

TDD evidence is green for 20 focused registry/preflight tests and 147 composed
authority, execution-contract, manifest, and shared flight-pipeline tests.
Ruff, Black, focused MyPy, Bandit, manifest validation, and structural checks
are clean. No GitHub write occurred.
## 2026-08-11 local #4369 React execution controller

From exact local composed head
`7e4069e891d8b4bde3f1d712b5b47897359a414e`, the React hooks layer adds a
UI-neutral controller around `RegionalGroundAuthorityClient`. It parses the
exact capability and job before submit, enforces one active job, polls serially,
retains exact progress and server-owned typed failures, calls the existing
POST-cancel client, and retrieves only a succeeded job's strictly bound complete
result.

Every lifecycle request carries an abort signal. Run-generation and operation
guards prevent stale publication after reset, cancellation, unmount, or request
replacement, including React StrictMode's development effect cycle. At that
earlier controller slice production capability was false-only and the admission
override was test-only. The newer headless admission removes that override but
adds no visible UI, TypeScript physics, persistence, or release claim; keep
#4369/#4273/#4267 open.

TDD RED captured the absent hook and the StrictMode remount defect. Evidence
passes 31 focused controller/client/capability tests, strict TypeScript,
zero-warning ESLint, the 214-module build, all eight manifest tests, and
module/minimum-test governance. This child is local-only and performs no
GitHub write.

## 2026-08-11 local #4369 canonical authority status wire

The public transport-neutral
`rate_of_closure.application.regional_ground_authority_status` contract owns
the six lifecycle states, stable failure codes/stages, immutable snapshots,
exact wire mapping, and bounded duplicate-safe canonical JSON parse/serialize
helpers. Every status is bound to the exact execution job ID, job digest, and
trial total; progress, result availability, and failure semantics fail closed.

`web_authority.jobs` imports these types, while React proves semantic and
canonical-byte parity against the Python-produced shared fixture. No physics,
UI, persistence, capability promotion, or execution claim is added. Production
capability and all execution controls remain false.
## 2026-08-11 local #4369 production-runner preflight qualification

From exact published #4372 head
`3571952c2344ca23ffa65121c606faab1b735a23`, production-runner qualification
now rejects every v1 regional-ground job before physical execution. Known
flight registry IDs receive `flight_profile_unregistered`; unknown IDs receive
`flight_model_unknown`. The existing generic numeric settings mapping, model
version string, and declared flight digests do not specify a reproducible
mapping to solver inputs and surface-event semantics.

Cancellation is evaluated before preflight. Callback defects and preflight
rejection become typed complete-only batch failures with exact zero-of-total
counts and chained internal causes; the authority manager exposes only its
generic stable failure stage and no result. No production execution profile,
factory runner, capability promotion, visible control, or model invocation is
added. A future profile must bind and test exact model/version/settings,
construction, integration/surface behavior, and digest recomputation. Keep
#4369/#4273/#4267 open.

Evidence passes 7 focused preflight tests, 98 composed authority, contract,
variation, and manifest tests, and 28 underlying flight/regional-ground
pipeline tests. Ruff, Black, focused MyPy, Bandit, and manifest validation are
clean. The serial complete Rate suite reached the 10-minute local command cap
without a reported failure; root will run the nonredundant full composed gate.
No GitHub write occurred.

## 2026-08-11 local #4369 PyQt authenticated loopback submitter

From exact published PR #4372 head
`3571952c2344ca23ffa65121c606faab1b735a23`, the widget-free application port
adds an injectable client for canonical submit/status/POST-cancel/result routes.
It uses the owned loopback runtime authority, strict shared status and result
contracts, bounded timeout/backoff, one active-job guard, cooperative cleanup,
typed non-secret failures, late-result suppression, and bounded close.

The production factory remains fail-closed because authority capability is
false; no submitter is registered with the existing PyQt controller. There is
no new widget, physical execution, persistence, qualification, protected
evidence, or release claim. The atomic implementation commit is `SELF` and
exact local gate evidence follows.

## 2026-08-11 #4369 authority terminal-count binding

The authority manager rejects cancellation/failure terminals whose total does
not match the submitted job or whose completed count regresses observed
progress. It retains the prior count and emits a typed validation failure only.

## 2026-08-11 #4369 result-digest typing stability

The result digest uses an explicit string local at the imported-helper
boundary so both isolated skipped-import and full PR-delta MyPy 1.13 roots are
clean. Runtime serialization and canonical evidence remain unchanged.

## 2026-08-11 local #4369 PyQt job-submission port

From exact published #4372 head `990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`,
the PyQt6 layer now provides a QWidget-independent QThread worker/controller
port. A dependency-injected submitter receives the strict qualified job plus
the existing typed progress/cancellation hooks. Qt signals expose only exact
progress, typed cancellation/failure, or a complete result that has passed the
existing expected-job identity, trial, order, and series checks.

The controller rejects concurrency and stale signals and offers cooperative
cancel and bounded shutdown. There is deliberately no production submitter,
physics invocation, visible control, capability promotion, or browser route;
future adapters must not enable execution until a qualified authority is bound.
Headless evidence passes 7 focused controller tests, 79 composed contract tests,
and the complete 1,068-test Rate suite, plus focused lint/type and governance
gates. One existing empty-legend warning remains outside this slice.
## 2026-08-11 local #4369 authenticated authority job lifecycle

The local `codex/4369-authority-api` child starts exactly from published PR
#4372 head `990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`. It adds a bounded
thread-safe manager with one active job, oldest-first terminal retention,
typed progress/status/failure, cooperative cancellation forwarding, and
complete exact job-bound result-only publication. The authenticated FastAPI
boundary now exposes submit, status, cancel, and result routes with no-store
responses, streaming 1 MiB input enforcement, strict JSON/content handling,
and generic non-secret errors.

The default isolated authority has no runner. Submit therefore fails closed
and the capability document remains false even when tests inject a runner.
Cancellation can only interrupt where the current application/physical runner
polls; it cannot forcibly stop Python code already inside an uncooperative
call. No physical invocation, client integration, persistence/recovery,
compiled physics, protected evidence, or release is claimed. The atomic local
implementation commit is `SELF`; no push or GitHub write occurs. TDD RED
captured the absent manager; 21 focused, 88 related contract, and all 1,076
Rate of Closure Python/PyQt tests pass with Ruff/format, MyPy, Black,
changed-file Bandit, manifest, placeholder, module-budget, and diff gates.
## 2026-08-11 local #4369 React authority client contracts

From exact published #4372 head
`990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`, the React model layer now reserves
strict same-origin submit/status/POST-cancel/result REST contracts around the
existing canonical execution job and job-bound complete result. The bounded
status parser validates exact job identity and digest, completed/total
progress, the six authority lifecycle states, result availability, and the
nullable stable failure code/stage before publication. Auth, unknown-job,
unavailable, malformed-error, and abort outcomes publish no synthetic status.

The capability hook is serial, abortable, timer-clean, and stale-response safe.
Its four execution-control flags remain disabled under the only capability the
current Python authority can produce: unavailable. A separately composed child
provides matching Python routes, but no qualified production runner. There is
no executor invocation, browser physics, visible control wiring, result
storage, or release claim in this slice. Keep #4369/#4273/#4267 open.

Complete local regression evidence passes 1,061 Python/PyQt and 841 React
tests, production build, static checks, manifest governance, and module/test
budgets. Existing Hypothesis collection, polynomial empty-legend, Node
local-storage, and Vite chunk notices remain non-blocking and unchanged.

## 2026-08-11 local #4369 job-bound execution result envelope

The bounded Python/React
`rate-of-closure/regional-ground-execution-result/v1` envelope binds exact
job/input identities and canonical dataset bytes. Expected-job matching also
binds result ID, declared trial count, zero-based row order, and every series
ID. Parsed evidence without its source job is integrity-checked but is not
authenticated or proof that the declared physics ran.

The Python-produced shared golden preserves complete and censored rows with
typed nulls. Full local gates passed 1,048 Python/PyQt and 818 React tests plus
build and static/governance gates. No executor, partial publication,
UI/backend/storage, compiled physics, or downstream parity is added.
Hosted MyPy 1.13 remediation removed a redundant result-digest cast without
changing the wire contract or canonical evidence.

## 2026-08-11 #4369 physical and launch-origin job qualification

From exact PR #4370 head `0a485958bd6ed46dce18e65fd3e3cd1fa797502a`,
the execution job now embeds exact callback-free regional options, every
`SkidRollSettings` field, executor revision, source plan, and a separately
digested launch-origin execution plan. A pure qualifier translates the base,
every overlay, and axis origin together and rebinds provenance to the source
plan, launch, transfer surface, ball radius, and ball setup. Matched Python and
React validators recompute this evidence.

V1 retains only implemented serial fail-fast `max_trials` and rejects false
parallelism, timeout, and configurable fail-fast claims. Local gates passed
243 Python regressions, 35 focused Python tests, all 804 React tests, static
analysis, production build, manifest, and module budgets. Physics invocation,
result binding, in-flight cancel, controllers, protected integration, and
release remain open.

## 2026-08-11 local #4369 typed validator failure boundary

Stacked from exact published PR #4370 head
`0a485958bd6ed46dce18e65fd3e3cd1fa797502a`, the complete-only regional
variation runner now converts injected outcome-validator exceptions into
`GroundRegionalVariationFailed` with the stable `validation` stage. The
terminal reports only accepted trials, preserves the original exception as
`__cause__`, and publishes no rows or dataset. Successful canonical output is
byte-identical. No authority, physics, worker, or UI execution is added.

## 2026-08-11 #4369 authenticated browser-authority capability boundary

The local `codex/4369-ground-authority-capability` child starts from exact
published prerequisite PR #4370 head
`0a485958bd6ed46dce18e65fd3e3cd1fa797502a`. It adds an isolated,
loopback-only FastAPI/Uvicorn process, an ephemeral bearer token passed only
through the child and Vite dev-server environments, a same-origin Vite proxy
that injects that token server-side, and strict Python/TypeScript
`rate-of-closure/regional-ground-authority-capability/v1` contracts. The
launcher owns and reaps the authority process. The browser converts unreachable,
unauthenticated, malformed, oversized, or unqualified evidence into explicit
non-executable capability states without exposing exception text or silently
falling back to TypeScript physics.

This slice is deliberately fail-closed: the only authority endpoint is the
authenticated capability query, and it advertises
`regional_ground_execution=false`. It adds no job submission, result polling,
cancellation endpoint, qualified execution profile, Python model invocation,
or Run-button enablement. Issue #4369 remains open until those contracts,
matched PyQt6/React controllers, process isolation limits, job-bound result
evidence, and protected integration are complete.

Focused evidence is green: seven Python authority/launcher tests, six React
capability/proxy tests, strict TypeScript, zero-warning ESLint, Ruff/format,
focused MyPy, and a live isolated-process readiness/authentication/shutdown
probe. The shared `node_modules` directory used for local React verification
is an untracked junction and is not publication content.
Hosted Bandit B310 remediation replaced generic URL opening in the readiness
probe with an explicit fixed-host `HTTPConnection`; capability behavior and
the loopback-only boundary are unchanged.

## 2026-08-11 #4369 qualification audit after prerequisite composition

`915c80f38` composes the job, batch-control, and result-import prerequisites,
but must remain fail-closed. Its golden job contains synthetic flight digests;
the contract omits physical skid/roll settings and executor revision; the
runner does not implement every declared orchestration option; current surface
editors require an explicit launch-origin translation for teed shots; and the
scalar result is not yet cryptographically bound to its job. The next authority
slice must close these gaps in Python, forward cancellation into bounce and
regional skid/roll, publish a job-bound complete-result envelope, and provide
one capability handshake consumed by matched PyQt and React controllers.

## 2026-08-11 local #4369 complete-only variation execution controls

This continuation is stacked on exact local execution-job contract commit
`a5a1b99bfa6cb6400bc18b13139d7893471824f4` in
`codex/4369-regional-ground-execution-job`. It extends only the UI-neutral
Python seeded regional-ground batch seam.

Application callers may supply frozen `GroundRegionalVariationHooks` with a
typed immutable completed/total progress callback and cooperative cancellation
predicate. Cancellation is checked before and immediately after each unchanged
physical executor call and after progress delivery. Pre-cancel executes no
trial; cancellation raised during a call rejects that in-flight outcome.

`GroundRegionalVariationCancelled` and `GroundRegionalVariationFailed` are
terminal signals with exact accepted/total counts. Failure identifies one of
four stable stages: cancellation callback, executor, progress callback, or
publication. No terminal object carries trial outcomes, rows, or a dataset.
The generic complete-batch helper retains all intermediate outcomes privately
and invokes the scalar-ensemble publisher exactly once only after every trial
is accepted. A broken callback cannot mutate results or cause a partial
publication.

Successful execution preserves the prior canonical output bytes at SHA-256
`671e5fd6c59aa1c068f2a3bd608ff7ef58c585b7ee4897ca49ef4ae73743f6a0`, as well
as seed streams, trial indexes, request IDs, sampled plans, and provenance
digests. Existing exact outcome-contract failures remain fail-fast DbC errors;
ordinary executor exceptions become typed terminal failures.

Focused execution-job plus variation coverage passes 47 tests. Ruff/format,
focused MyPy, relevant cross-suite physics/variation tests, manifest and docs
governance, structural budgets, assertion/minimum-test checks, and diff checks
are required before commit `SELF`. No UI, browser physics, worker/thread,
execution-job binding, or physical executor change is included. Keep
#4369/#4273/#4267 open for those integrations and protected release.

## 2026-08-11 local #4369 regional-ground execution-job contract

The unpublished `codex/4369-regional-ground-execution-job` branch starts from
exact PR #4368 head `7d2d155b35f2ae55842de120864c4a343a5ebcb6`.
It adds the first UI-neutral prerequisite for real seeded regional-ground
execution: a strict 1 MiB
`rate-of-closure/regional-ground-execution-job/v1` envelope implemented with
Python/TypeScript parity and one shared canonical golden fixture.

The immutable job binds the exact SI constant-wind launch and ball setup,
flight model identity plus bounded numeric settings, independently canonical
trajectory and result SHA-256 identities, the complete existing
flight-to-ground transfer surface/calibration/provenance/settings authority,
capture threshold, bounded trial/parallelism/timeout/fail-fast options, and the
existing seeded regional-ground variation request. Canonical input and complete
job digests are recomputed on every import. The parser rejects duplicate or
extra fields, wrong versions, nonfinite/cross-runtime-unsafe/Boolean numbers,
surrogates, malformed digests, oversize text, mismatched trial counts, model
identity drift, and any regional base surface not exactly equal to the
launch-relative transfer surface.

The contract reuses the existing canonical numeric JSON, strict JSON,
ball-setup, transfer, surface, regional-plan, and seeded-request authorities.
It does not duplicate physics, invoke a solver, invent browser execution,
persist results, or prove that the supplied precomputed flight digests were
produced by the declared model. Version 1 accepts the current resolved
constant-wind launch contract; time/space-varying wind requires a separately
qualified scenario wire contract. Keep #4369/#4273/#4267 open for executor
binding, cancellation/result evidence, matched UI invocation, wind-scenario,
compiled/downstream parity, protected publication, and release.

TDD RED captured the absent Python and TypeScript modules. Focused Python and
React parity suites, Ruff, TypeScript, ESLint, campaign-manifest validation,
documentation governance, and repository structural gates are the required
local evidence. The implementation, shared fixture, SPEC, campaign manifest,
and all canonical handoffs commit together as `SELF`; no push or GitHub write
occurred.
## 2026-08-11 local #4369 regional scalar-result import prerequisite

The unpublished `codex/4369-regional-result-parser` child starts exactly from
published PR #4368 head `7d2d155b35f2ae55842de120864c4a343a5ebcb6`.
React now has a strict bounded import-only adapter for the two Python-owned
regional `scalar-ensemble/v1` result variants. It reuses the shared ensemble
contract and regional evidence types, preserves exact metadata, digests,
definition taxonomy, ordered identities, cohorts, and censored typed nulls,
and rejects duplicate/extra/version/nonfinite/unsafe/Boolean/surrogate/
oversize/fatal-UTF-8 inputs plus forged row, series, cohort, and evidence
identity. Both runtimes assert the same Python-produced four-cohort fixture.

Limits are 8 MiB encoded JSON and 100,000 rows, with both declared and actual
file size checked. This parser does not run browser physics or establish a Run
claim; it adds no result workspace, persistence, overlay variation,
solver/capability or wind integration, compiled/downstream parity, protected
evidence, or release. Focused React/Python, full React, TypeScript, ESLint,
Vite, Ruff, manifest, and docs gates are recorded in the implementation
evidence. The implementation and all governance files commit as `SELF`; no
push or GitHub write occurred.

## 2026-08-11 local #4273 contextual regional-ground request File controls

The unpublished `codex/4273-ground-variation-file-controls` child starts from
exact ready PR #4367 head `0968a4ced5644aa8e2673ca278d261eeb92c31f8`.
It turns that prerequisite's typed App-owned request port into matched,
contextual File commands without introducing another request or physics model.

React parses and serializes the exact Python-owned v1 combined envelope with
strict JSON, nested validation, and the same 1 MiB UTF-8 bound. A
Python-produced golden payload is asserted in both runtimes. In Variation and
Ground Surfaces only, accessible Open and Save As commands import into or
snapshot the App owner. Invalid imports preserve all prior state and expose an
alert. Downloads disclose that the browser owns destination, overwrite, and
atomicity semantics.

PyQt6 exposes the same stable commands only in the two relevant modules.
Native Open fully validates before applying both editors; Save validates before
showing the chooser and uses the existing atomic writer. Cancellation changes
nothing. Imported exact evidence remains authoritative until either editor
changes. The untouched illustrative regional draft cannot be saved until it is
explicitly validated, and oversized wire-valid run counts fail before editor
mutation.

This slice executes no physics and adds no illustrative fallback. All 782 React
tests in 123 files, the 87-test focused Python/PyQt selection, and a 25-test
post-fixture follow-up pass; type-check, ESLint, Vite build, Ruff, MyPy, and
repository policy gates are green. The code, SPEC, manifest, and all canonical
handoffs commit together as `SELF`; no push
or GitHub write occurred. Keep #4273/#4267 open for pipeline invocation,
overlay variation, solver/capability and wind integration, compiled/downstream
parity, protected review, publication, and release.

## 2026-08-11 local #4273 React request-workspace ownership

The unpublished `codex/4273-ground-variation-file-ui` child starts from exact
PR #4366 head `8dfb1189c13f0fce99901e1ffbba152d813f9006`. React previously owned the
variation and regional-surface inputs inside separate panels that unmount on
navigation. The new App-owned reducer/hook retains the complete request-editor
state and exact imported regional evidence; both panels are controlled.

`RegionalGroundVariationRequestTs` and its typed port provide a future
File-command seam. Snapshot uses only the current plan and current regional
draft/import evidence. Complete apply validates all fields and derives the
editor draft before dispatch, so invalid input cannot partially replace state.
The disclosed illustrative regional draft is not composable until the user
explicitly edits it or imports qualified evidence.
The web registry exposes the two existing ground-material keys, while the
scalar browser runner rejects them with a visible unsupported-path message.
No physics or persistence behavior is added.

RED captured the missing owner and tab-reset behavior. The full React suite is
green at 763 tests in 121 files. TypeScript, ESLint, Vite production build,
campaign manifest validation and its eight tests, documentation governance,
blocking-quality policy, scoped module-size, and diff gates pass; the Vite
main-chunk warning is inherited.

File controls, strict combined-schema TypeScript serialization, browser
upload/download, PyQt native actions, pipeline execution, protected review,
publication, and downstream parity remain open. Keep #4273/#4267 open. No push
or GitHub write occurred.

## 2026-08-11 local #4273 seeded-request persistence

The unpublished `codex/4273-ground-variation-persistence` child starts from
exact PR #4365 docs head `27d2a68d3738d61307af9235f3f97f7bd400e0f3`.
The new application-layer contract composes existing authorities into one
strict v1 seeded-study request envelope: exact `VariationPlan`, exact regional
plan, result/source/series identifiers, and the bounded row cap. Canonical
numeric JSON is compact, deterministic, cross-runtime safe, and suitable for a
browser download; native persistence reuses the existing bounded UTF-8 reader
and atomic writer.

Import rejects duplicate or unknown fields, unsupported outer or nested
versions, nonfinite/unsafe/Boolean numbers, surrogate text, malformed
identifiers and caps, invalid nested contracts, and payloads above 1 MiB. It
explicitly registers the Rate ground variables only when parsing a request.
Successful import returns the existing immutable request and executes no
physics. Native cancellation is a no-op; failed replacement retains the prior
file and removes its temporary file.

RED captured the absent module. Twenty-two focused tests, 82 composition tests,
and 545 relevant Rate/shared flight-ground-variation tests pass; the broad set
has six expected missing-Rust-wheel skips and one environment warning. Ruff,
import-skipping MyPy, Bandit, campaign manifest and its eight tests,
documentation, blocking-quality, minimum-test, module-size, changed-test
assertion, placeholder, structural, and diff gates pass.

No UI/editor, browser filesystem behavior, workspace embedding, overlay
variation, solver/capability invocation, wind, compiled runtime, downstream
parity, protected review, publication, or release is included. Keep #4273 and
#4267 open. This branch has not been pushed.

## 2026-08-11 PR #4365 seeded regional-ground material variation

Ready PR [#4365](https://github.com/D-sorganization/Tools/pull/4365) is stacked
on exact PR #4364 head `f13f0908dd2a553cf4d114afd31bb474d1b967c7`;
its independently reviewed implementation is
`8c9c9512c61bac6f958ae7c7c0fe58e8f70525bf`.
`regional_ground_variation` samples only base normal restitution and rolling
resistance with the existing `VariationPlan`/`sample_inputs` engine, creates an
immutable plan/provenance-bound trial, and calls an injected exact
`FlightRegionalGroundPipelineResult | FlightGroundTransferError` executor. It
then augments the existing bounded scalar-ensemble projection so each qualified
or typed-null outcome remains aligned with its sampled inputs.

Registration is explicit and idempotent through the shared registry extension
seam, avoiding import-time global state. Validation fails before executor entry
for invalid keys, base mismatch, missing/nonfinite/Boolean/out-of-range bounds,
nonfinite scale/sample, invalid exact records, and row overflow. Exact pipeline
results must retain the sampled regional plan and canonical digest.

Twelve focused tests pass. The 43-test focused-plus-registry selection and the
506-test Rate-adapter/shared-flight/ground/variation selection are green; the
latter has six expected Rust-wheel skips and one environment warning. A live
pipeline test confirms greater sampled rolling resistance shortens qualified
total distance. Ruff and import-skipping MyPy pass. Remaining policy evidence
also passes: Bandit, campaign manifest and its eight tests, documentation,
blocking-quality, minimum-test, default module-size, changed-test assertion,
the new module's 397-line budget, placeholder, and diff checks. A stricter
whole-directory 400-line scan reports only inherited `plot_data.py` at 433
lines.

No UI, persistence, region-overlay variation, solver/capability invocation,
wind physics, target/playback changes, compiled runtime, downstream parity,
protected release evidence is included. Keep #4273 and #4267 open; publication
of this bounded contract does not close either issue.

## 2026-08-11 PR #4364 post-ground spatial-target projection

Ready-for-review PR [#4364](https://github.com/D-sorganization/Tools/pull/4364)
is stacked on exact PR #4363 head
`ec50fdf059f91ca9e4664da891398af218e1ba65`. Independently reviewed target
implementation commit `b480f17f11b86a57326622168e4c748efc77aaf3`
adds the UI-neutral `regional_ground_target_projection` boundary without
modifying the inherited playback production code. The adapter accepts only an exact
`FlightRegionalGroundPipelineResult | FlightGroundTransferError` and exact
`SpatialTarget`. It reuses #4361's promoted complete-rest qualifier and exact
evidence attributes instead of duplicating endpoint eligibility.

Only regional `COMPLETE` plus ground `COMPLETE/REST` with a summary produces
an endpoint, hold, or miss. Ground v1's sole `GroundFrame.TARGET` is recorded
explicitly as x-downrange/y-up/z-right. Final x/z pass through unchanged; the
terminal ball-center y is replaced exactly once by the target's declared
course-surface elevation before delegating geometry and signed long/high/right
residuals to `SpatialTarget.miss`. App- and flight-authored target points
therefore give the same canonical result. Aerial targets return
`AERIAL_REQUIRES_FLIGHT_TRAJECTORY` and are never flattened.

Transfer failures, every non-settled bounce reason, regional cancellation,
failure or partial execution, `LEFT_SURFACE`/non-rest termination, missing
summaries, and all censored outcomes retain null target numerics with exact
availability, phase, reason, frame, model, and digest attributes. The bounded
ordered `ScalarEnsembleDataset` projection exposes hold, miss distance, and
signed downrange/elevation/lateral values with deterministic row identity and
source provenance.

RED captured the missing module. Sixteen new focused tests plus all seven
parent study-adapter tests pass; the complete Rate/flight/ground selection is
green for 1,315 tests with 14 environment-only Hypothesis collection warnings
and one inherited polynomial-generator legend warning.
Strict MyPy, focused Ruff check/format, Bandit, campaign-manifest validation
and its eight tests, documentation governance, blocking-quality,
minimum-test, changed-Python, 400-line module-size, changed-test assertion,
placeholder, and diff checks are green. Fresh protected current-head checks,
dependency order, and ordinary merge gates remain. The PR adds no editor/UI,
persistence, solver/capability invocation, aerial trajectory evaluation,
compiled runtime, new physics, or geometry. Keep #4192, #4273, and #4267 open.


## 2026-08-11 PR #4363 matched ground playback

Ready-for-review PR [#4363](https://github.com/D-sorganization/Tools/pull/4363)
is based exactly on published PR #4361 head
`81de044075a4f72c6da8fedb972437df79a06ab8`; its independently reviewed
implementation commit is `7f7d4b01d83d914ae5684715dc20c69388cf799f`.
It hand-integrates only the reviewed matched playback slice: strict Python and
TypeScript absolute-time timelines, matched additive PyQt6/React workspaces,
and explicit import adapters for standalone results and validated regional
execution envelopes. The regional adapters return the already-validated nested
ground result and never calculate physics. Existing `Ground Surfaces`, saved
navigation behavior, and help remain available.

Controls provide exact step, phase jump, play, pause, restart, loop, granular
speed, locked-scale 3D orbit/zoom/reset, summary status, warnings, calibration,
provenance, and accessible event/transition/trajectory evidence. Cross-phase
interpolation holds the lower exact record rather than fabricating a state.
For 100,000-point inputs, per-frame selection is binary, visual materialization
is capped at 2,048 landmark-aware points, and evidence tables disclose their
256-row window while retaining the full validated result.

RED first failed on the absent timeline/UI. Local qualification passes all
1,125 Rate/shared-ground Python tests and all 119 React files / 754 tests.
Ruff check/format, scoped Black, strict MyPy on the five new Python production
modules, Bandit, ESLint, TypeScript type-check, production build, campaign
manifest, documentation governance, the 400-line new-module budget, and diff
checks are green. Fresh protected current-head checks, dependency order, and
ordinary merge gates remain required.

Keep #4274/#4267 open for terrain meshes and changing normals, direct editor
handoff, comparison, persistence, rendered visual QA, camera presets/tracking,
downstream UpstreamDrift/four-surface parity, and protected release.
## 2026-08-11 PR #4361 qualified regional-ground study adapter

Ready-for-review PR [#4361](https://github.com/D-sorganization/Tools/pull/4361)
starts from exact published PR #4360 head
`74f1ceafd87f952a76917dc868baa6414f856144`. Its independently reviewed
implementation commit is `d71c43fdd729b35e1abe5573f41ed60201698608`.
A read-only audit of current
flight metric, target, scalar-ensemble, capability, regional readback, and
ground-result contracts plus the historical `ground-study-scalar-adapter`,
`ground-study-result-adapter`, and `ground-study-projection` worktrees retained
only the complete-rest qualification invariant and scalar taxonomy. The stale
parallel study model and its numeric censored totals were not copied.

The UI-neutral Rate adapter reuses `to_ground_model_result`,
`FlightMetricInputs`, and `ScalarEnsembleDataset`. Complete-rest evidence may
populate canonical total, roll, final-offline, and bounce-count values plus
distinct bounce-air/skid/surface-path/final-downrange detail. Carry remains
separate. Partial/left-surface, every non-settled bounce reason, regional
cancelled/failed, missing-summary, and typed transfer-error outcomes retain
null numerics with exact typed status/reason/model/digest attributes. Applying
unqualified evidence clears stale ground metric inputs.

Seven focused tests and 1,299 Rate/flight/ground tests pass. Ruff, strict
MyPy, pinned Bandit, manifest validation plus 8 manifest tests, documentation
governance, blocking-quality, minimum-test, default module-size, and diff gates
pass. Inherited main-relative assertion and 400-line findings do not include
this 328-line module or its assertion-bearing test.

Solver/capability invocation, variation UI, wind strategy, persistence,
TypeScript/compiled and four-surface parity, protected CI/review, publication,
release, and #4273/#4267 completion remain open.

## 2026-08-11 PR #4360 flight-through-regional-ground pipeline

Ready-for-review PR [#4360](https://github.com/D-sorganization/Tools/pull/4360)
on `feat/4271-flight-regional-ground-pipeline` starts from exact published
Tools #4359 head `e53c6fb1bd273292c02085ee5d0a2b5497820871`; its reviewed implementation
commit is `090e835477d1f19614f37f978a1b8a0e2f50ae21`. Audit established that the
regional envelope is semantically exact only after `SETTLED_TO_SKID`, not for
bounce time/event limits or no-recontact.

The shared flight facade now composes exact flight output through the existing
flight-to-bounce and regional-ground authorities. Preflight rejects type,
capture, and launch-relative plan/base mismatches before bounce physics. The
strict bounded versioned in-memory result preserves exact request, bounce, and
plan identities/digests/provenance; it requires regional evidence only for a
settled bounce and retains every other bounce termination unchanged. The child
also centralizes canonical regional-plan hashing without changing physics.

RED captured the missing module/result/exports, GREEN passed 17 focused tests,
and REFACTOR passed 39 pipeline/public/regional tests. The complete 377-test
flight-plus-ground suite is green, as are Ruff check/format, scoped Black,
protected changed-file and import-following MyPy, Bandit, placeholder/diff,
documentation, blocking-quality, minimum-test, test-assertion, changed-Python,
both LOC, campaign-manifest, and 11 manifest/layout gates. Explicit casts at
dynamic wire-parser boundaries close protected skipped-import MyPy without a
runtime or canonical-byte change. Standalone Black keeps one inherited
preference in `test_contract_api.py`; authoritative Ruff is green and its
delta is only the required public API additions.

The ready PR remains `not_released` pending protected checks and review. No
wire schema/migration, PyQt6/React, TypeScript/Rust/WASM, persistence,
playback, calibration, target/solver/variation integration, or downstream
release is included. Keep #4271, #4273, and #4267 open.

## 2026-08-11 PR #4359 shared Python flight-to-bounce composition

Ready-for-review PR [#4359](https://github.com/D-sorganization/Tools/pull/4359)
on `feat/4270-flight-bounce-execution` is based on exact clean published Tools
#4357 head `c492b52f9f7615c5bc38e780965167cc8f64327c`; its reviewed implementation
commit is `869b626e2d3ebd4097ae76b8fc9720cda6696947`.
The shared flight facade now exposes `execute_repeated_bounce_from_flight`,
which accepts exact flight, launch, and transfer contracts, validates callback
and capture inputs before transfer, and delegates without copied physics to the
existing transfer builder, repeated-bounce request, and request executor.
Typed transfer failures and request/result identity evidence remain intact.

RED-GREEN testing captured the missing module/export. Independent follow-up
coverage now proves exact transfer-error message, field, and reason propagation
plus zero executor calls for no-contact, grazing, and missing-angular-state
paths. Seventeen focused tests and the complete 365-test flight-plus-ground
suite pass. Ruff check/format,
scoped Black, protected and import-following MyPy, Bandit, placeholder/diff
checks, documentation and blocking-quality governance, minimum-test and
test-assertion contracts, changed-Python/module-size policies, campaign
manifest validation, and 11 manifest/layout tests are green. Standalone Black
retains one inherited advisory in `test_contract_api.py`; authoritative Ruff
is green and the file's only delta is the public API entry. The committed
changed-file size gate is green with zero violations across four changed
Python files.

The ready PR is not yet reviewed, protected, integrated, or released. It adds
no PyQt6/React controls, TypeScript/Rust/WASM physics,
persistence, playback, camera behavior, regional-material chaining, skid/roll
completion, or total-distance result. Keep #4270 and #4267 open.

## 2026-08-11 PR #4357 repeated-bounce request execution binding

Ready-for-review PR [#4357](https://github.com/D-sorganization/Tools/pull/4357)
on `feat/4270-repeated-bounce-execution` starts exactly from published #4356
head `2387430fc78baa92ba122c7ad008a498118bf62d` and is published at
implementation head `cf54d3528a71fd429ad19f53f04e4a1a84495097`.
It adds one UI-neutral Python executor that requires the
exact repeated-bounce request and callable-or-`None` cancellation, consumes the
request-bound capture threshold through fixed-version settings, invokes the
existing Python physics authority, and returns the existing identity-validated
request/result pair. It changes no wire schema, physics law, TypeScript
runtime, or PyQt6/React surface.

TDD recorded the expected missing-public-executor failure. Qualification is
green for 28 focused contract tests, all 189 ground tests, 11 campaign
manifest/layout tests, Ruff check/format, Black, protected changed-file MyPy,
Bandit, placeholder/diff checks, documentation and blocking-quality
governance, minimum-test/test-assertion contracts, changed-Python policy,
module-size policy, and campaign-manifest validation. A non-authoritative
import-following MyPy probe reports three inherited redundant casts in
unchanged ground modules; the protected `--follow-imports=skip` profile is
green for the new production module.

This remains `not_released`; protected checks and review are pending. UI request construction and
invocation, persistence, playback, TypeScript/compiled physics,
regional-material chaining, measured terrain calibration, downstream parity,
protected exact-head evidence, review, approval, release, and issue/epic
completion remain open.

All exact open dependencies from #4203 through #4357 are now ready for review
without base or history changes. No current-head check was failing at the
release reconciliation snapshot, but protected queues and all approvals remain
open. The manifest records the reconciled parent heads and #4357 without
claiming protected completion or release.

## 2026-08-11 PR #4356 published current-parent propagation

Published ready-for-review PR #4356 keeps `feat/4270-repeated-bounce-request-wire`
over `feat/4271-repeated-bounce-wire`. Exact current child
`23897eac03e8a3edf4a37855f0ba05e8c2527986` is the first parent and exact
published #4355 head `a04d14e9308990e676e8c90ddb1d80e368dd1387` is
the second parent of normal no-ff merge
`345c329e6b6e3fc7a8fc981abf65795f356b94cf`. The child's strict cross-runtime repeated-bounce request
envelope, canonical ground-request and joint-execution-input digests, exact
request/result identity pairing, shared golden corpus, adversarial
capture-speed digest follow-up, and live-PR handoff remain intact alongside the
complete #4355 result-wire and cancellation evidence, both typed-Boolean
protected-MyPy repairs, and all regional/ground ancestry.

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
Node local-storage flag, and 528.82 kB Vite chunk. Exact heads #4351 through
#4356 are now ready for review without base or history changes. The first
protected #4356 checkpoint had one successful quality check, four skipped
checks, twelve queued checks, no failure, and no review. UI request
construction, executor invocation, persistence, playback, measured
calibration, compiled and downstream parity, protected completion, approval,
dependency integration, release, and issue completion remain open.

## 2026-08-11 PR #4355 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-repeated-bounce-wire` over
`feat/4271-regional-trajectory-export`. Exact current child
`b67af52226fa6334dd3570cf650aebeaf81912fc` is the first parent and exact
published #4354 head `97925e4803f4fbd72d576eb1c11c47f8e61b0b66`
is the second parent. The child's complete strict cross-runtime repeated-bounce
result-wire contract, canonical golden corpus, phase/chronology/energy
invariants, and pre-contact cancellation follow-up remain intact alongside
regional trajectory inspection/export, both typed-Boolean protected-MyPy
repairs, and the complete regional/ground ancestry.

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
local-storage flag, and 528.82 kB Vite chunk. The candidate is not published
or released. Request construction, executor invocation, persistence,
playback, measured calibration, compiled and downstream parity, protected
exact-head evidence, review, approval, dependency integration, release, and
issue completion remain open.

## 2026-08-11 PR #4354 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-regional-trajectory-export` over
`feat/4271-regional-event-inspection`. Exact current child
`99b0739bdc3ece814ed6039e6ba31f7ac38c0227` is the first parent and exact
published #4353 head `e0433adbc3c82272745d098867f261462a790d08`
is the second parent. The child's matched bounded PyQt6/React raw-trajectory
inspection and canonical semantic-lossless evidence export remain intact
alongside inherited ground-event and regional-transition ledger inspection,
the complete qualified result projection, the explicit Boolean local required
by protected delta-MyPy, embedded-plan execution/provenance and request-I/O
boundaries, and complete regional physics ancestry.

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
legend, Node local-storage flag, and 528.82 kB Vite chunk. The candidate is not
published or released. Input construction, UI executor invocation,
interpolation/playback, calibration workflows, compiled regional physics,
downstream parity, protected exact-head evidence, review, approval, dependency
integration, release, and issue completion remain open.

## 2026-08-11 PR #4353 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-regional-event-inspection` over
`feat/4271-regional-result-readback`. Exact current child
`7fc00f43561c31923b74563bc2bf6caf89bbc9eb` is the first parent and exact
published #4352 head `12fc80798d2a15b44c0215688ffb031dd99cbdd1`
is the second parent. The child's matched bounded PyQt6/React inspection of
validated ground-event and regional-transition ledgers remains intact
alongside the complete qualified result projection, the explicit Boolean local
required by protected delta-MyPy, embedded-plan execution/provenance and
request-I/O boundaries, complete regional physics ancestry, the capability-
only extended finite-float serializer, and the default ground safe-number
boundary.

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
flag, and 526.79 kB Vite chunk. The candidate is not published or released.
Trajectory-sample inspection, lossless export, UI executor invocation,
playback, calibration workflows, compiled regional physics, downstream
parity, protected exact-head evidence, review, approval, dependency
integration, release, and issue completion remain open.

## 2026-08-11 PR #4352 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-regional-result-readback` over
`feat/4271-regional-execution-ui`. Exact current child
`10fdac4860035fd5c845a621752e93688e2e674e` is the first parent and exact
published #4351 head `4024c8a1ad2d3871c6b06ef6369250a873789c39`
is the second parent. The child's complete matched PyQt6/React qualified result
projection remains intact alongside bounded evidence import/readback, the
explicit Boolean local required by protected delta-MyPy, embedded-plan
execution/provenance and request-I/O boundaries, complete regional physics
ancestry, the capability-only extended finite-float serializer, and the
default ground safe-number boundary.

Local qualification passes all `1,057` combined Rate/shared-ground Python
tests, all `113` React files / `697` tests, the complete Cargo workspace, `76`
focused Python result/readback/execution/I/O/capability tests, and `37` focused
React tests. Pinned Ruff 0.14.10 check/format passes three child-delta Python
files; isolated-import strict MyPy passes both child production modules and the
coherent 35-module ground profile passes with inherited imports skipped and
only the parent's documented `redundant-cast` code disabled; Bandit passes both
production files. TypeScript, zero-warning ESLint, the 202-module Vite build,
Rust format and warning-denied clippy, both LOC gates, campaign/manifest tests,
docs/tool-manifest/blocking-gate/assertion/minimum-test governance, child-
feature and inherited Boolean-local byte checks, marker scans, and diff checks
pass. Existing Hypothesis ignored-cache, polynomial-generator empty-legend,
Node local-storage option, and 523.34 kB Vite chunk warnings remain non-failing.

The candidate is not published or released. UI executor invocation,
trajectory/event tables, playback, calibration workflows, compiled regional
physics, downstream parity, protected exact-head evidence, review, approval,
dependency integration, release, and issue completion remain open.

## 2026-08-11 PR #4351 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-regional-execution-ui` over
`feat/4271-regional-execution-binding`. Exact current child
`351a3051e9093c6b80cabf0f1db04aeeb15abfac` is the first parent and exact
published #4350 head `98f86990e9225903fbe84cd1f267ed38ef0a15d8`
is the second parent. The child's matched bounded PyQt6/React execution-
evidence import and readback, including the explicit Boolean local required by
protected delta-MyPy, remain intact alongside the embedded-plan execution and
provenance contract, request I/O, complete regional physics ancestry,
capability-only extended finite-float serializer, and default ground safe-
number boundary.

Local qualification passes all `1,056` combined Rate/shared-ground Python
tests, all `113` React files / `696` tests, the complete Cargo workspace, `75`
focused Python evidence/readback/execution/I/O/capability tests, and `36`
focused React tests. Pinned Ruff 0.14.10 check/format passes six child-delta
Python files; isolated-import strict MyPy passes all five child production
modules and preserves the Boolean-local repair; the coherent 35-module ground
profile passes with inherited imports skipped and only the parent's documented
`redundant-cast` code disabled; and Bandit passes those five files. TypeScript,
zero-warning ESLint, the 202-module Vite build, Rust format
and warning-denied clippy, both LOC gates, campaign/manifest tests,
docs/tool-manifest/blocking-gate/assertion/minimum-test governance,
child-feature byte checks, marker scans, and diff checks pass. Existing
Hypothesis ignored-cache, polynomial-generator empty-legend, Node local-storage
option, and 521.54 kB Vite chunk warnings remain non-failing.

The candidate is not published or released. UI executor invocation, playback,
compiled regional physics, downstream parity, protected exact-head evidence,
review, approval, dependency integration, release, and issue completion remain
open.

## 2026-08-11 PR #4350 current-parent propagation candidate

This local no-publish merge keeps `feat/4271-regional-execution-binding` over
`feat/4274-regional-plan-io`. Exact current child
`dfb4b97481f187ff3594eceb08c427f650aca4e3` is the first parent and exact
published #4342 head `de66a851aa5dded680279cf9a2b25a5094966593`
is the second parent. The child's embedded-plan execution/provenance envelope,
executor authority, transition binding, cross-runtime fixtures, and frozen
base-result boundary remain intact alongside current request I/O, matched
editors, complete regional physics ancestry, the capability-only extended
finite-float serializer, and the default ground safe-number boundary.

Local qualification passes all `1,052` combined Rate/shared-ground Python
tests, all `111` React files / `692` tests, the complete Cargo workspace, `71`
focused Python execution/I/O/capability tests, and `36` focused React tests.
Pinned Ruff 0.14.10 check/format passes seven child-delta Python files;
isolated-import strict MyPy passes the four execution modules and the coherent
35-module ground profile passes with only the parent's documented
`redundant-cast` code disabled. Bandit passes five child production files.
TypeScript, zero-warning ESLint, the 199-module Vite build, Rust
format/warning-denied clippy, both LOC gates, campaign/manifest tests,
docs/tool-manifest/blocking-gate/assertion/minimum-test governance,
child-feature byte checks, marker scans, and diff checks pass.

The first CPU-contended Python run produced `1,051` passes and one Hypothesis
input-generation `too_slow` health check; the property passed alone and all
`1,052` tests passed in the single uncontended rerun. The candidate is not
published or released. Execution UI/playback, compiled regional physics,
downstream parity, protected exact-head evidence, review, approval, dependency
integration, release, and issue completion remain open.

## 2026-08-11 PR #4342 current-parent propagation candidate

This local no-publish merge keeps `feat/4274-regional-plan-io` over
`feat/4274-regional-surface-ui`. Exact current child
`c1f47f2ef68b3db102da5416aaac17a40f675207` is the first parent and exact
reviewed local #4339 candidate `db335937afc4b587d235eb705e315f577519c5e6`
is the second parent. Child-owned canonical request import/export, bounded UTF-8,
native atomic save, browser-qualified download, tests, and limitations remain
intact while inheriting current editor, wire, regional physics, and complete
ground ancestry.

The default shared canonical encoder continues to reject floats and integers
outside JavaScript's safe range. Only the capability-observation facade selects
a separately named extended finite-float policy. It shares the same recursive
encoder, retains safe-range integer checks, emits exact exponent-free `1e20`
and `1e21` tokens matching TypeScript, and rejects non-finite values.

Local qualification passes all `909` Rate-of-Closure Python tests, all `110`
React files / `686` tests, the complete Cargo workspace, `47` focused Python
compatibility/regional-I/O tests, and `12` focused React capability tests.
Pinned Ruff 0.14.10 check/format passes `17` changed Python files; pinned MyPy
1.13 and Bandit pass `12` changed production files. TypeScript, zero-warning
ESLint, the 199-module Vite build, Rust format/warning-denied clippy, 400- and
500-LOC gates, manifest/docs/blocking-gate/assertion/minimum-test governance,
marker scans, and diff checks pass. One untouched manual-delivery UI test timed
out in the first concurrent full run, then passed alone and in the single
complete rerun.

The candidate is not published or released. Execution/playback, result
interchange, measured calibration, persistence, changing geometry or velocity,
TypeScript/compiled regional physics, downstream parity, protected exact-head
evidence, review, approval, dependency integration, and release remain open.

## 2026-08-11 PR #4339 current-parent propagation candidate

This local no-publish merge keeps `feat/4274-regional-surface-ui` over
`feat/4271-regional-wire-contract`. Exact current child
`d21741e312b849a63f73cabf351a15d9de80fb94` is the first parent and exact
published #4335 head `8f933ed8dcb29e55ece4ec6bb1e60813f6794d57`
is the second parent. Child-owned matched PyQt6/React editors, validation
invalidation, engineering hints, strict readback, and limitations remain intact
alongside the parent's regional wire/resolver/physics and complete ground
ancestry. The parent-extracted navigation-state contract remains canonical and
includes the child-owned `regional_surfaces` module in default/migration order.

Local qualification passes all `891` Rate-of-Closure Python tests, all `110`
React files / `678` tests, `177` focused regional/ground/navigation Python
tests, `14` focused React editor/navigation/wire tests, and all `137`
`tools-core` Rust tests. TypeScript, zero-warning ESLint, the 198-module Vite
build, Rust format and warning-denied clippy, pinned Ruff 0.14.10 across seven
PR-delta Python files, pinned MyPy 1.13 across six production files, Bandit
medium/high, both changed-file size gates, manifest/docs/assertion/minimum-test
governance, child-feature byte checks, marker scans, and diff checks pass.
Existing Hypothesis ignored-cache and Node local-storage option warnings remain
non-failing.

The candidate is not published or released. Execution/playback, result
interchange, measured calibration, persistence, changing geometry or velocity,
TypeScript/compiled regional physics, downstream parity, protected exact-head
evidence, review, approval, dependency integration, and release remain open.


## 2026-08-11 PR #4351 delta-MyPy boundary repair candidate

Exact PR #4351 head `fe463b5503a8c7b599a329da18bb690d008871cd`
is runtime-correct but fails the protected changed-file MyPy profile: with
`--follow-imports=skip`, the imported atomic writer is `Any`, so returning its
call directly violates the adapter's declared `bool` contract. An explicitly
typed local records that contract without a cast, remaining clean whether or
not the helper is included as another MyPy root. No persistence, validation,
wire, UI, or physics behavior changes.

This local candidate is not protected or published. It must be reviewed and
propagated in order through #4352, #4353, and #4354 before those descendants
can rely on the repair.


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


## 2026-08-11 regional execution evidence hardening

The review follow-up to `696a3ff8f124bebf6dc22ae0d584cf35f6d92843`
removes synthetic execution evidence and closes cross-runtime validation gaps.
`ground-regional-execution-result/v1` now embeds and hashes the exact plan,
binds plan/source/base identities, enforces the executor authority while
leaving source revision variable, and proves each transition is a real
ordered crossing between the declared plan regions and surfaces. Python and
TypeScript share adversarial safe-number/text/vector acceptance cases and
executor-produced representable/cancelled/failed fixtures.
Null-result cancellation/failure envelopes reject nonempty transition ledgers
because no embedded result exists to substantiate those events.

The pre-existing capability-observation test using `1e20` remains outside this
slice: exact parent `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`
and this child both raise `ValueError: canonical JSON number exceeds
cross-runtime safe range` in
`test_stable_wire_uses_canonical_numeric_tokens_for_every_float`.

## 2026-08-11 regional ground execution binding

`feat/4271-regional-execution-binding` is an isolated child of exact PR #4342
head `8e1c7ccd99a7c4886c5fb9ccc7e4d94a6d7e3833`. It adds one
UI-neutral Python executor that accepts exact ground request/prefix/plan values
plus bounded settings/cancellation, creates its resolver only from the plan,
and delegates unchanged physics to `simulate_skid_roll` and
`compose_ground_result`.

The separate strict `ground-regional-execution-result/v1` envelope preserves
frozen ground-result v1 bytes for complete/partial output and uses typed
null-result cancellation/failure states where v1 cannot represent an honest
result. Canonical request/plan digests, plan/executor provenance, model IDs,
and ordered from/to regional transition evidence are cross-runtime through a
shared fixture and a TypeScript parser/serializer. UI execution/playback,
compiled regional physics, downstream parity, protected evidence, and issue
completion remain open.



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

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-11

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

## 2026-08-11 Regional execution evidence readback

Local unpublished branch `feat/4271-regional-execution-ui` starts at exact PR
#4350 head `dfb4b97481f187ff3594eceb08c427f650aca4e3`. Both standalone PyQt6 and
React regional-plan surfaces now import bounded strict Python-produced
`ground-regional-execution-result/v1`, require exact equality with the current
validated plan, preserve accepted evidence after import errors, and clear
stale evidence after plan edits. Status, termination/failure, model, skid,
roll, total, transitions, and executor provenance are visible. React remains
readback-only and runs no physics.

Local gates pass: 207 expanded Python ground/plan/PyQt/layout tests, 111 React
files / 690 tests, strict MyPy, Ruff/format, TypeScript, zero-warning ESLint,
production build, manifest and eight manifest tests, docs governance,
structural budgets, and diff checks. The inherited 500 kB build advisory
remains. Independent review is still required before any GitHub write. Do not
claim #4267/#4271 complete; UI executor inputs/invocation, playback, measured
calibration, compiled regional physics, downstream parity, protected evidence,
and release remain open.

## 2026-08-11 Complete regional result readback

Local unpublished `feat/4271-regional-result-readback` is an exact child of PR
#4351 head `fe463b5503a8c7b599a329da18bb690d008871cd`. Its matched PyQt6 and React
evidence presenters now expose every qualified result summary field, final
position, ground time, completion, bounce count, model/surface authority,
calibration evidence, ordered phases, warnings, executor provenance, and
limitations. Null-result states remain unavailable rather than fabricated;
partial endpoints retain their warning.

This remains import/readback only. Complete local gates and independent review
before any GitHub write. Executor invocation, trajectory/event tables, export,
playback, calibration workflows, compiled parity, downstream parity, protected
evidence, release, and #4267/#4271 completion remain open.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 691 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 202-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The inherited 500 kB build advisory
remains.

## 2026-08-11 Regional execution ledger inspection

Local unpublished `feat/4271-regional-event-inspection` is an exact child of PR
#4352 head `10fdac4860035fd5c845a621752e93688e2e674e`. Matched PyQt6 and React
surfaces now inspect validated ground-event and regional-transition rows with
explicit SI units, frames, before/after velocity and spin vectors, and bound
from/to region and surface identities. Both retain the full accepted envelope,
cap rendered ledgers at 256 rows, and disclose truncation. Null-result ledgers
stay empty and partial endpoint warnings stay visible.

This remains import/readback only. Trajectory-sample inspection, lossless
export, executor invocation, playback, calibration workflows, compiled parity,
downstream parity, protected evidence, release, and #4267/#4271 completion
remain open. Complete local gates and independent review before any GitHub
write.

Exact local gates are green: 208 expanded Python ground/plan/PyQt/layout tests,
111 React files / 692 tests, strict MyPy, Ruff/format, TypeScript type-check,
zero-warning ESLint, the 203-module production build, campaign-manifest
validation plus eight manifest tests, documentation governance, module-size
budget, placeholder scan, and diff checks. The inherited 500 kB build advisory
remains.

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
