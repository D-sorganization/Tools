# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Last updated: 2026-08-11.

## 2026-08-12 #4385 Windows authority-state security

Published draft PR #4392 carries branch `codex/4385-windows-state-security`
against `codex/4380-playwright-production-browser`. Implementation commit
`48197ad25` was reconciled by normal parent propagation with exact PR #4391
head `0de3de8a41c018aec03dead8371a1f3ec6e1912f`; the resulting published
pre-handoff head is `dde43534babf385530a95b2ff6ce3477f73ac9b3`. It must follow #4391 in
ordinary dependency order; do not rebase, force-push, retarget, rewrite the
shared browser branch, or bypass protection. Human review is approved, but
fresh exact-head protected CI and ordinary non-admin merge behavior remain
mandatory.

The Windows authority store now requires a named path on fixed local NTFS with
persistent ACL and named-stream support. A process-lifetime native lease opens
every ancestor without delete sharing, rejects reparse points, and pins volume
and file identities. The dedicated root and every database, WAL, SHM, journal,
and lock artifact use a protected DACL with exactly full-control allow ACEs for
the current token user, SYSTEM, and Builtin Administrators. Existing broad
DACLs migrate in place without replacing the directory or file; batch failure
rolls changes back in reverse order and reports a distinct rollback-incomplete
code if restoration cannot be proven. New roots are created with the current
token user as owner; an existing owner mismatch is rejected rather than
requiring elevated owner-changing authority.

| Epic                                                                          | Status (one line)                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 — Swing–Impact–Ball-Flight Simulation Platform                          | Phases 0-6 implemented on branch `feat/impact-simulation-platform`, consolidated into PR **#4119** (open, auto-merge armed, awaiting review). Phase 7 (WASM web parity swap, Pages CI) still open.                  |
| #4120 — Investigation & Variation Suite (plotting/viewers/Monte Carlo/help)   | V1-V4 implemented, stacked on #4119, consolidated into PR **#4124** (open, draft-for-review, no auto-merge yet — targets `feat/investigation-suite`, itself stacked on #4119).                                      |
| #4125 — Realistic Clubs/Kinetics/Putting/Public Release Mgmt/Showcase Styling | H1-H7 implemented, stacked on #4124, consolidated into PR **#4129** (open, draft-for-review, targets `feat/course-showcase`, stacked on #4124). H5 (public release-management repo) is cross-repo, not yet started. |
| #4130 — Impact-Interval Club Dynamics (contact-interval rigid-body model)     | Foundation epic only (F1 formulation doc not yet started); no PR yet. Next major physics wave after #4125 lands.                                                                                                    |

The separate shared Club Builder epic #4146 is active. Its first dependency
slice, #4147, lives on `feat/4147-club-builder-core` and establishes the
UI-independent assembly mass/CG/inertia, frame, length-datum, and persistence
contracts that the later shaft, CAD, export, fitting, and UI issues consume.
The boundary rejects UNC/non-fixed/non-NTFS storage, overlong or reserved
components, alternate-data-stream syntax and planted named streams, junctions
and symlinks, hard-linked files, unexpected root entries, type changes, and
out-of-root paths. Stable typed diagnostics never include the sensitive path.
SQLite temporary storage is memory-backed, and no-delete handles remain live
for the root and durable artifacts; transient sidecar handles are released in a
bounded order before SQLite shutdown.

State retention is deliberate: normal shutdown, restart, upgrade, and package
uninstall do not delete the per-user authority root. Terminal records remain
subject to the existing oldest-first job bound. This slice adds no automatic or
in-app destructive purge; explicit user removal is out of band and is safe only
after every Rate of Closure authority process has stopped. The protection does
not claim resistance to the same user, an elevated Administrator or SYSTEM,
offline disk access, or a malicious process able to inject into this process.
Installer-owned removal remains part of the still-open frozen-distribution
work.

A dedicated protected workflow targets only
`[self-hosted, Windows, X64, d-sorg-windows-security]` for Python 3.11 and
3.12. It performs credential-free exact-head checkout, requires the symlink
adversarial case rather than accepting a skip, runs the native contracts,
builds exact React assets and a wheel, then qualifies the clean installed wheel
from an unrelated Unicode working directory. No credential, native state,
database, browser profile, or raw diagnostic is uploaded. Do not route this
gate to the shared ControlTower/MATLAB runner; absence of the dedicated
restricted runner is an explicit release blocker, not permission to weaken the
label contract.

Local evidence on Windows is green: the final focused native,
store/API/loopback/companion, and workflow suite passed 86 tests with four
expected local symlink-privilege skips and the expected POSIX-mode skip. Ruff,
format, YAML lint, and focused MyPy pass after excluding only the repository's
pre-existing transitive unreachable-code diagnostic. The fully provisioned
Rate of Closure suite reached 1,344 passes and seven expected skips; its one
companion timeout passed serially and its sole deterministic tooltip failure
reproduces unchanged on parent PR #4391. Fresh wheel installs on Python 3.11
and 3.12 both proved installed-module isolation, exact private ACLs, schema v1,
completed-result restart recovery, interrupted-job no-replay, and zero
unrelated-CWD pollution. Wheel metadata and both installs prove
`pywin32>=311` is a Windows base dependency; the existing
`rate-of-closure-web` extra supplies the separate SciPy web runtime boundary.

The publication handoff records PR #4392 and reconciles the previously missing
#4376/#4388/#4390/#4391 carriers in the campaign manifest. Independent review
found no remaining P1/P2 code findings and classified this slice as code-ready,
not ship-ready: no registered runner carries the restricted
`d-sorg-windows-security` label, so both protected Windows matrix jobs remain
an explicit external release blocker. The final documentation head requires
fresh checks and must not inherit evidence from the pre-handoff SHA. The system
volume was nearly full during local qualification, so all disposable build and
test environments were isolated on `D:`; this is local infrastructure context,
not product evidence.

## 2026-08-12 #4380 production-browser qualification

Acceptance completion extension: the release matrix now exercises the complete
combined-request workflow through Python-authoritative preparation, explicit
identity confirmation, one submit, polling, canonical job/result downloads,
reload/import, retained-result recovery, cooperative cancellation, and prepared
job staleness without automatic resubmission. A forbidden `Worker` constructor
proves that this ground execution path does not substitute browser physics.

Adversarial browser qualification now covers malformed capability data, a
missing declared entry script, corrupt persisted workspace/layer preferences,
private-authority replacement with both token and port rotation, and full public
gateway loss. The native harness scans bounded public HTML/capability responses
while retaining the secret identity out of browser state; combined with the
same-origin credential-free request audit, this proves the bearer token and
private child port are absent from public responses and requests. Intentional
cancellation and gateway-loss transport failures are separately bounded while
all successful paths retain zero console, page, or network failures.

Current working-tree evidence atop `de673971bfae83a9d673bba4859def5322635af9`:
TypeScript and zero-warning ESLint pass; 15 native harness contracts pass; and
all 36 deterministic Playwright scenarios pass across Chromium, Firefox, and
WebKit with configuration-owned zero retries. The publication commit and its
final handoff-recording child still require exact-revision local gates and fresh
protected qualification before ordinary merge.

Branch `codex/4380-playwright-production-browser` was created at
`0821557d80c366133e3de5af54d5ad82a01b14b0` as an exact child of Tools
PR #4390. Corrected parent head
`c3ecfd48910aa5aafb89962a256333690e8e72c5` is now propagated normally into
this branch. Human review is approved, but parent order, fresh protected CI,
and ordinary non-admin merge behavior remain mandatory. Published child PR:
#4391. Its exact pre-publication browser-qualified head is
`5de71c74d2de9e7105d486b60c48e4ed6569e8fd`; protected CI must qualify the
final handoff-recording head independently.

This slice adds deterministic Playwright qualification against the exact
revision-built production surfaces in Chromium, Firefox, and WebKit. The
static-inspection harness exercises the deliberately host-owned nested-path and
fragment navigation contract. The packaged same-origin companion is stricter:
only `/` and `/index.html` are document-shell routes, declared hashed assets
are exact routes, and arbitrary nested application paths remain rejected rather
than silently becoming an SPA fallback.

Browser assertions cover bootstrap mode and revision, primary surface
rendering, no unexpected page/console failures, the network-silent
static-inspection boundary, and the same-origin companion capability/lifecycle
workflow. Security observation is browser-visible only: requests, DOM,
runtime metadata, storage, and qualification results must not disclose the
authority bearer token or child port, and browser code performs no physics.
This does not claim protection from same-user native malware.

The browser gate exposed and fixes two release-path defects: the companion CSP
now admits the bundle's local `data:` font payloads without relaxing script or
connection policy, and Vitest explicitly owns only unit-test paths so it cannot
mistake Playwright specifications for unit suites.

Local qualification passes TypeScript type-check and zero-warning ESLint, all
922 React/Vitest tests, six deterministic release-artifact checks with one
expected Windows symlink skip, 60 focused Python companion/harness contracts,
all nine three-engine smoke scenarios, and all three three-engine hard-loss
lifecycle scenarios. The generated release manifest matched the qualified head.

Exact-head hosted run `31595498982` built the production bundle and passed all
13 harness tests before app-contract collection exposed Starlette 1.6's split
TestClient dependency. The isolated browser job now pins `httpx2==2.10.0`.
Hosted quality run `31595499033` also correctly classified the native harness
as changed test support; its exact path is now documented in the fixture-only
assertion allowlist while behavioral assertions remain in the separate harness
test module. These are qualification-environment fixes, not product expansion.

Final-revision local qualification then exposed a WebKit-only lifecycle race:
the authority could be stopped while the application's initial capability
request was still in flight, producing a real 502 before successful recovery.
The browser contract now observes the initial 200 response before inducing
hard loss. Runtime diagnostics retain bounded error meaning while redacting
origins and long token-like values, with cross-engine coverage.

The protected distribution workflow gains an independent 30-minute
**Production browser qualification** job. It fetches the exact public head
without credentials, uses Python 3.11 and Node 22, performs `npm ci`, builds
with the exact `ROC_RELEASE_REVISION`, installs the packaged web extra and
pinned Pytest plugins, installs all three browser engines, and runs the separate
smoke and lifecycle scripts with configuration-owned one-worker/zero-retry
behavior. Only structured Playwright JSON qualification records are uploaded;
traces, videos, screenshots, and raw browser profiles are not release evidence.

Forced parent-process termination and descendant-tree cleanup are not qualified
by this slice. Windows authority-state ACL/reparse privacy, frozen web/PyQt
artifacts, installers, signing, SBOM/attestation, calibrated or compiled
physics, downstream parity, protected release, and #4377/#4380 completion all
remain open.

## 2026-08-12 #4379 same-origin source production companion

Branch `codex/4379-same-origin-companion` starts exactly from Tools PR #4388
head `a35b259fd6a6ad57815544d228d73a806bb8d84e` and must target
`codex/4378-static-inspection-runtime`. Human review is approved; protected
dependency order and ordinary non-admin merging remain mandatory.
Published child PR: #4390. Its exact pre-publication artifact-qualified head is
`822e90c914baeb73338d037df89c5281811b9f7f`; protected CI must qualify the
final handoff-recording head independently.

Production `launch_web.py` and the `rate-of-closure-web` entry point now run a
Python-only foreground companion over an exact manifest-qualified packaged
bundle. Node/Vite remains only in `launch_web_dev.py`. One derived in-memory
`local_companion` index publishes schema, exact revision, and the fixed relative
API root; package data stays unchanged and token, child port, state path, PID,
and environment data remain private.

Gateway and authority bind their own ephemeral IPv4 `127.0.0.1` listeners. The
authority child owns its listener before reporting the selected port through a
private bounded pipe, removing the former reserve/release token-disclosure
race. Token and port are absent from argv and sensitive runtime fields from
repr.

The gateway admits only capability, preparation, submit, status, cancel, and
result method/path contracts. It rejects queries, encoding/traversal ambiguity,
unknown routes, browser credentials, forwarding/CORS headers, unsupported
media/encoding, duplicate headers, and oversized or mismatched bodies. Exact
Host is mandatory; POST additionally requires exact Origin and same-origin
fetch metadata. It reconstructs bounded authority requests and browser
responses, never relaying upstream errors or headers. Every authority response
must satisfy the selected route's exact status set and duplicate-safe domain
decoder before canonical JSON publication. Synchronous authority I/O runs off
the ASGI event loop, and uncommon methods plus protocol truncation retain the
same sanitized security envelope. Shell/API are `no-store`,
hashed assets are immutable-cacheable, and security headers are explicit.

A single-flight supervisor replaces an already-dead authority before a later
explicit request using a fresh token/port and the same durable state root. It
never retries after dispatch or replays physics. Normal close stops admission,
releases the gateway listener, and reaps the live child. Partial startup closes
every acquired listener/supervisor and releases the durable lock; forced server
exit remains bounded and retryable. The child-port report deadline tolerates a
busy 14-worker host while readiness remains independently authenticated.

Current local evidence is 101 focused companion/authority/status tests, 1,313
complete Rate-of-Closure Python passes with two expected Windows/POSIX skips,
and all 138 React files / 922 tests plus six Node release-contract passes and
one expected Windows symlink skip. Changed Ruff/format, focused MyPy, high
severity Bandit, YAML/TOML, campaign-manifest, module-budget, policy, and diff
gates pass. The exact-revision clean-wheel gate and protected CI remain required
at the final published head. Its installed-artifact smoke materializes the
`importlib.metadata` entry-point selection as a tuple for Python 3.11-3.13
compatibility before asserting the exact advertised console script.

Ordinary parent propagation incorporates corrected #4388 head
`a35b259fd6a6ad57815544d228d73a806bb8d84e`. The companion workflow preserves
its broader package-side contract suite while adding the pinned benchmark and
xdist plugins required by the repository's declared Pytest arguments.

Hosted exact-head run `31592370252` then reached the full 95-test companion
contract suite and identified two isolated-runner assumptions: an explicit
`PYTHONPATH=src` polluted the child-environment preservation assertion, and the
async lifespan regression required the Pytest asyncio plugin. The package-side
step now exercises the installed project without the source-path override and
installs pinned `pytest-asyncio==1.3.0`. This changes qualification plumbing
only, not the companion, authority, physics, or browser contract.

Playwright, forced parent-process tree cleanup, Windows ACL/reparse privacy,
frozen packages, installers, signing/SBOM/attestation, protected release,
compiled/calibrated physics, and downstream parity remain open under #4377 and
#4380-#4385. Same-origin checks do not authenticate same-user native malware.

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
matching TypeScript detail/media-type enforcement. Older no-runner statements
are historical slice evidence, not the current default-server state.

The default loopback server now atomically binds the qualified production
runner to an exact true/true service capability. Python and TypeScript reject
split capability flags or mismatched reasons; FastAPI rejects capability/runner
split-brain at construction; readiness parses the authenticated bounded exact
document; and application shutdown cooperatively cancels and joins the owned
worker. React production admission now trusts only that strict qualified
capability and no longer uses the lifecycle-test bypass.

The full combined suite exposed an order-dependent registry assertion after
Rate tests registered the documented ground-variable extensions in the same
xdist worker. The shared contract now pins the five built-in launch variables
as an ordered prefix, preserving extension behavior and deterministic testing.

Independent review reproduced and closed two shutdown races: submission can no
longer start a worker after concurrent close, and exceptional FastAPI lifespan
exit always closes the manager. Direct Python construction and React response
handling now share exact reason, detail-length, media-type, and body bounds.

Complete local qualification passes 2,014 Python/PyQt/shared-simulation tests
with one optional Rust-wheel parity skip and 860 React tests across 132 files.
Pinned MyPy 1.13 and Ruff 0.14.10, Ruff format, high-severity Bandit,
TypeScript, zero-warning ESLint, the 214-module production build,
deterministic fixture generation, campaign-manifest validation, repository
governance, and diff-hygiene checks also pass.

This is headless admission, not visible Run integration. Neither live client
constructs a complete regional-ground execution job from current editor state,
the matched presentation components remain unmounted/disabled, static hosting
has no authority, and results are in-memory only. Next use strict imported-job
Run/Cancel/save surfaces, then a Python-authoritative current-editor job
preparation boundary. Keep direct editor execution, packaged-helper claims,
persistence/recovery, compiled/downstream parity, release, #4369, #4273, and
#4267 open.

## 2026-08-11 #4369 qualified fixture, runner, and matched presentation

Starting from corrected published PR #4372 head
`ff1310c09c066a32e57b50e4daee4da5a40d7bf3`, the canonical execution job now
uses deterministic registered-profile flight digests and a checked generator
rebuilds its dependent status/result identities. The qualified runner reuses
one flight solution across seeded regional trials, forwards cancellation into
physics, retains typed transfer/executor/publication outcomes, and publishes
only a complete job-bound result. An actual authenticated loopback process now
proves the successful submit/status/result path.

Matched Python/TypeScript presentation models plus observer-only PyQt6 and
handler-free React views expose exact job/model/provenance/digest and immutable
progress/failure/result evidence with visibly disabled controls. At that
earlier fixture slice the default server registered no runner; the newer
headless-admission section above supersedes that service state without adding a
runnable UI, protected release, persistence, compiled-runtime, or downstream
claim.
The consolidated gate passes 2,004 Python/PyQt/shared-simulation tests and 858
React tests plus the focused real-loopback, static, security, build,
fixture-generation, documentation, manifest, and diff checks. One Rust parity
test is skipped because this interpreter has no `swing_core` wheel.

## 2026-08-11 #4369 hosted MyPy 1.13 correction

PR #4372 head `e91ef8dcde8cdd8e6545ffc0ea7cb755058ec2fb` passed hosted
checkout, dependency installation, Ruff, and formatting, then failed only the
pinned MyPy 1.13 delta gate because the already exact-bool-validated
cancellation result was redundantly cast. The cast and now-unused import are
removed without changing runtime behavior. The exact local pinned MyPy 1.13
profile and 13-test submitter suite now pass; protected CI, the canonical
fixture qualification, false production capability, and the open ancestor
stack remain gates.

## 2026-08-11 #4369 composed authority continuation ready for PR #4372

The exact continuation after published head `3571952c2344ca23ffa65121c606faab1b735a23`
now composes the canonical Python/React status wire, typed fail-closed production
preflight, authenticated PyQt loopback submitter, actual-process integration,
strict Waterloo/Penner execution-profile qualification, and UI-neutral React
execution controller. Production remains unavailable because the canonical
job's declared synthetic flight digests do not match deterministic profile
recomputation; no ground physics, capability promotion, or visible Run control
is claimed.

The consolidated gate passes 1,148 Python/PyQt tests and 854 React tests, plus
Ruff, changed-file formatting, focused MyPy, changed-file high-severity Bandit,
strict TypeScript, zero-warning ESLint, the 214-module production build,
real-loopback retest, docs governance, module budget, minimum-test contract,
campaign manifest, and diff hygiene. One inherited polynomial empty-legend
warning and existing Node/Vite advisories remain. Implementation, tests, SPEC,
manifest, and all canonical handoffs are ready for one guarded fast-forward
push; protected CI and the open ancestor stack remain release gates.

## 2026-08-11 local #4369 PyQt real-loopback qualification

From exact composed head `f7342cae7296410f8cfd262fd9877363beb5dc63`,
process-level tests start the actual loopback Uvicorn/FastAPI authority and run
the PyQt submitter's real HTTP transport against it. They cover wrong-bearer
rejection without token exposure, canonical submit and job-bound status,
idempotent POST cancel, unavailable result, typed production-preflight failure,
bounded client close/join, false-capability non-construction, and authority
process reaping.

The runtime accepts a strictly bounded `module.path:function` factory seam for
process integration only; the default environment-token production factory is
unchanged. Injected test runners either fail in production preflight before any
physics or wait solely for cooperative cancellation. No successful physics,
capability promotion, controls, persistence, protected evidence, downstream
parity, or release is claimed. Atomic commit and exact gate evidence follow.
## 2026-08-11 local #4369 versioned flight execution profile

This isolated child starts from exact local composed head
`7e4069e891d8b4bde3f1d712b5b47897359a414e`. The new application registry
maps only `waterloo_penner` + `tools-core/1.0.0` to a strict bounded
`max_time_s`/`step_s`/whole-number `sample_every` schema. Its explicit v1
recomputation contract uses default Waterloo/Penner coefficients, adaptive
RK45, the launch-relative transfer plane, base dense-output samples,
deterministic decimation, and terminal retention.

Qualification returns stable typed evidence for absent identity, invalid
schema, recomputation failure, either digest mismatch, or exact success. The
qualified boundary releases the physical flight result only when both job
digests match. The current canonical fixture recomputes deterministically but
its synthetic declared digests differ, so runner preflight reports
`flight_evidence_mismatch`, completes zero trials, invokes no ground physics,
and publishes no result. At that earlier profile slice production injected no
runner and all client execution controls were disabled. A canonical fixture was
produced by the exact registered profile under a pinned numerical runtime
before the physical runner can progress. Keep #4369/#4273/#4267 open.

TDD RED first proved the registry module absent. Evidence passes 20 focused
registry/preflight tests and 147 composed authority, job/result, manifest, and
flight/transfer/pipeline tests. Ruff, Black, focused MyPy, Bandit, JSON and
manifest validation, diff hygiene, and structural limits are clean. No GitHub
operation occurred.
## 2026-08-11 local #4369 React execution-controller prerequisite

This isolated `codex/4369-authority-react-controller-v1` child starts from the
exact local composed head `7e4069e891d8b4bde3f1d712b5b47897359a414e`.
It adds a UI-neutral React hook over the existing strict authority client. The
hook validates exact capability/job input before submission, owns at most one
active job, polls status serially, preserves exact progress and typed terminal
failure, delegates cancellation through the client's POST route, and retrieves
only a succeeded job's complete job-bound result.

All requests are abortable. Reset, cancellation, unmount, operation IDs, and
run generations prevent obsolete status or result publication; a dedicated
regression covers React StrictMode's development effect probe. At that earlier
controller slice production admission was false and the named admission
override was unit-test-only. The newer headless admission removes the override
without adding a visible control, TypeScript physics, qualified
runner, persistence, downstream parity, or issue completion is claimed.

TDD RED first captured the missing hook, then a separate RED exposed the
StrictMode cleanup/remount defect. Evidence passes 31 focused controller and
adjacent authority-contract tests, strict TypeScript, zero-warning ESLint, the
214-module production build, manifest validation and all eight manifest tests,
and module/minimum-test governance. Code, tests, SPEC, manifest, and all three
handoffs commit together; no push or GitHub write belongs to this child.

## 2026-08-11 local #4369 canonical authority status wire

From exact published PR #4372 head
`3571952c2344ca23ffa65121c606faab1b735a23`, the transport-neutral
`rate_of_closure.application.regional_ground_authority_status` module owns the
six lifecycle states, stable failure codes/stages, exact wire records, and
duplicate-safe 4,096-byte JSON parser/serializer. The authority manager now
imports these objects instead of maintaining a server-only projection.

The Python-produced golden covers every state and failure stage and both
failure codes. Python and React reject extra/duplicate/mistyped/non-finite/
unsafe fields, impossible progress/result/failure semantics, and mismatch to
the exact source job; React reserializes every golden case byte-for-byte. This
adds no physics, UI, transport, persistence, capability promotion, or execution
claim. All production execution controls remain disabled.
## 2026-08-11 local #4369 production-runner preflight qualification

This isolated continuation starts from exact published PR #4372 head
`3571952c2344ca23ffa65121c606faab1b735a23`. The v1 job's generic numeric
`flight.settings` mapping has no authoritative mapping to the existing flight
solver, and the golden fixture's `sample_every` setting has no production
consumer. Its model version and embedded flight digests likewise do not define
a recomputable execution profile. Invoking flight-through-ground physics would
therefore fabricate semantics.

The new production-runner boundary fails closed before physics with distinct
typed reasons for an unknown model and a recognized model lacking a registered
versioned execution profile. Cancellation wins before preflight; callback
defects and profile rejection preserve typed terminal stages, exact zero-of-N
counts, cause chaining, and complete-only authority publication. No profile is
registered, no runner is injected into the production factory, capability
remains false, and no UI or release claim is promoted. The next physical slice
must first define and qualify the exact model/version/settings/solver/surface
mapping and recompute both declared flight digests. Keep #4369/#4273/#4267
open.

TDD RED first captured the absent runner module. Evidence is green for 7
focused runner/preflight tests, 98 composed authority/job/result/variation and
manifest tests, and 28 underlying flight/regional-ground pipeline tests. Ruff,
Black, focused MyPy, Bandit, JSON/manifest validation, and the eight manifest
tests are clean. A serial full Rate suite exceeded the 10-minute local command
ceiling without reporting a failure; root owns the nonredundant full composed
gate. No GitHub operation occurred.

## 2026-08-11 local #4369 PyQt authenticated loopback submitter

From exact published PR #4372 head
`3571952c2344ca23ffa65121c606faab1b735a23`, the UI-neutral application layer
adds a dependency-injected submitter for the existing PyQt QThread controller.
It POSTs canonical execution-job bytes through the runtime-owned fixed-loopback
bearer transport, validates canonical status snapshots against the exact job,
polls with bounded timeout/backoff, POSTs cooperative cancellation once, and
retrieves only a complete result that passes expected-job validation.

Callback, transport, status, timeout, result, and shutdown failures publish
only existing typed terminals. After acceptance, client-side failures make one
bounded best-effort cancellation request without masking the original terminal;
cancelled, obsolete, and late-success responses cannot publish a result. Raw
transport exception and token text is excluded from the client error surface.

The construction factory returns `None` under the current false capability, so
production registers no submitter. This adds no widgets, visible controls,
physical runner or model invocation, persistence, protected carrier,
scientific/physics claim, downstream parity, or release. Code, tests, SPEC,
manifest, and all handoffs commit together as `SELF`; exact gates follow.

## 2026-08-11 #4369 authority terminal-count binding

Authority cancellation and failure terminals must match the exact submitted
job total and cannot regress already observed progress. Mismatches now retain
the prior completed count and publish only a typed validation failure; no
result or misleading cancellation state escapes.

## 2026-08-11 #4369 result-digest root-set stability

The composed PyQt continuation exposed a skipped-import MyPy root-set
dependency at the result-digest helper boundary. An explicit `str` local now
keeps both the isolated consumer-module and complete 14-file PR-delta MyPy 1.13
profiles clean without changing runtime bytes or canonical evidence.

## 2026-08-11 local #4369 widget-free PyQt submission controller

This exact-parent continuation starts from published PR #4372 head
`990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`. It adds a widget-free PyQt6
`RegionalGroundExecutionWorker` and owning controller around an injected
`RegionalGroundExecutionSubmitter` protocol. The controller accepts only one
strict qualified `RegionalGroundExecutionJob`, runs the injected authority on a
QThread, forwards immutable typed progress/cancellation/failure records through
queued Qt signals, and supports cooperative cancel plus bounded shutdown.

Success is emitted only after an exact `RegionalGroundExecutionResult` passes
full expected-job binding. Wrong result types or identities become typed
validation failures; ordinary adapter exceptions become typed executor
failures with their cause chained; inconsistent terminal totals and stale
queued signals fail closed. No partial dataset is exposed.

The injected submitter remains intentionally absent in production. This slice
does not invoke flight or ground physics, advertise authority availability, add
visible Run/Cancel controls, add a browser endpoint, or claim execution. Keep
#4369/#4273/#4267 open for the qualified authority, matched clients, protected
integration, downstream parity, and release.

TDD RED first proved the worker/controller module absent. Seven focused
QThread/controller tests, 79 job/result/qualification/variation regressions,
and all 1,068 Rate Python/PyQt tests pass; the full suite retains one unrelated
polynomial-generator empty-legend warning. Ruff, Black, focused MyPy, manifest
validation and eight manifest tests, docs governance, and structural gates are
green. No GitHub write occurred.
## 2026-08-11 local #4369 bounded authority job API

From exact published PR #4372 head
`990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`, branch
`codex/4369-authority-api` adds a one-active-job in-memory manager and
authenticated submit/status/cancel/result routes to the existing loopback
FastAPI authority. Submission streams and caps the exact job body at 1 MiB,
rejects encoded or non-JSON content, and retains only a bounded oldest-first
set of terminal records. Status and failures are typed and publish no raw
exception or token text. Results appear only after exact job-bound result
validation; cancellation and every failure leave result unavailable.

Cancellation is forwarded through the existing variation hooks and any late
return after cancellation is discarded. Production still constructs no
runner, rejects submission with `execution_unavailable`, and advertises
`regional_ground_execution=false`; injected runners are a test seam, not a
capability claim. No physical job invocation, client Run/Cancel controller,
persistence, restart recovery, compiled runtime, protected carrier, or release
is included. Keep #4369/#4273/#4267 open.

TDD RED first captured the missing manager module. Green evidence is 21 focused
manager/API tests, 88 job/result/qualification/variation/manifest regressions,
and all 1,076 Rate of Closure Python/PyQt tests. Ruff, Ruff format, focused
MyPy, Black check, changed-file Bandit, manifest JSON/eight tests, placeholder,
module-budget, and diff gates pass. Code, tests, SPEC, manifest, and all
handoffs commit together as `SELF`; no push or GitHub write occurs.
## 2026-08-11 local #4369 React authority client contracts

The unpublished `codex/4369-authority-react-client-v1` child starts exactly
from published PR #4372 head
`990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`. It adds strict same-origin React
client contracts for future canonical submit, job-bound status, POST cancel,
and complete-result retrieval routes. Status/result parsing is bounded,
duplicate-safe, identity-bound to the exact validated job, and rejects
impossible terminal/progress semantics. Invalid jobs fail before network I/O.
Status matches the composed authority API's six exact lifecycle states,
completed/total progress, result-availability rule, and nullable stable
failure code/stage. Typed failures distinguish authentication, unknown jobs,
execution unavailability, known API errors, malformed errors, and aborts.

`useRegionalGroundAuthority` polls the capability endpoint serially, forwards
an `AbortSignal`, clears its timer, aborts active work on cleanup, and suppresses
obsolete effect responses. All submit/status/cancel/result control flags remain
false because the current Python-owned v1 capability accepts only
`regional_ground_execution=false`. A separately composed child supplies the
matching Python routes, but no qualified production runner. This client child
adds no Python endpoint, model execution, TypeScript physics, visible Run
control, persistence, or downstream parity.

Complete local gates pass 1,061 Python/PyQt tests and 841 React tests across
130 files, the 214-module production build, strict TypeScript, zero-warning
ESLint, release-manifest validation/tests, and module/minimum-test budgets.
Pytest retains 14 Hypothesis collection notices and one unrelated polynomial
empty-legend warning; Node retains its local-storage notices and Vite its
existing main-chunk advisory. No GitHub write or push belongs to this child.

## 2026-08-11 local #4369 job-bound execution result envelope

From exact published PR #4370 head
`0a485958bd6ed46dce18e65fd3e3cd1fa797502a`, the strict bounded Python/React
`rate-of-closure/regional-ground-execution-result/v1` envelope carries exact
job/input identities, embeds complete `scalar-ensemble/v1`, and recomputes its
canonical dataset SHA-256. Explicit expected-job matching additionally binds
dataset result ID, trial count, zero-based order, and every series ID. Without
the originating job it proves internal integrity, not authenticity.

Full local gates passed 1,048 Python/PyQt and 818 React tests plus build,
Ruff, MyPy, TypeScript, ESLint, manifest, and module budgets. The slice adds no
executor, partial publication, UI/backend/storage, compiled physics, or
downstream parity. Keep #4369/#4273/#4267 open.
Hosted MyPy 1.13 remediation removed a redundant result-digest cast; the
runtime contract and canonical evidence are unchanged.

## 2026-08-11 #4369 execution qualification child

Local branch `codex/4369-execution-qualification` starts exactly from
published PR #4370 head `0a485958bd6ed46dce18e65fd3e3cd1fa797502a`.
It binds exact callback-free regional options, all skid/roll settings, executor
revision, source plan, and a separately hashed launch-origin plan. The base,
every overlay, and axis origin receive one identical tee/ball-center
translation; provenance and digests are recomputed in Python and TypeScript.

V1 now truthfully contains only `max_trials`, rejecting unsupported
parallelism, timeout, and configurable fail-fast fields. The teed-driver golden
remains serialization evidence; there is no physics invocation or Run path.
Local evidence passed 243 Python regressions, 35 focused Python tests, all 804
React tests, MyPy, Ruff, TypeScript, ESLint, production build, manifest, and
module gates. Result binding, in-flight cancellation, controllers, protected
integration, and release remain open.

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

The composed local head `915c80f38` is a contract/control/parser prerequisite,
not a qualified executable authority. A real Waterloo/Penner recomputation did
not match the execution-job golden fixture's synthetic flight trajectory/result
digests, so that fixture remains serialization evidence only. The job also does
not yet bind the physical skid/roll settings or executor revision, implement
every declared orchestration option, or bind a completed scalar ensemble back
to the job/input digests. Current editors express regional surfaces at zero
height, while a teed launch requires an explicit launch-origin translation of
the base and every overlay with provenance rebinding. Keep Run disabled until
an exact Python model/profile registry, coordinate qualification adapter,
complete-result envelope, cancellable authority, loopback host, and matched
PyQt/React controllers prove these invariants end to end.

## 2026-08-11 local #4369 regional-ground batch progress and cancellation

The unpublished `codex/4369-regional-ground-execution-job` continuation starts
from exact local execution-job contract commit
`a5a1b99bfa6cb6400bc18b13139d7893471824f4`. It extends only the existing
Python `run_regional_ground_variation()` application seam; it does not bind the
new job envelope to launch/flight/ground execution and changes no physics.

`GroundRegionalVariationHooks` exposes an immutable exact `(completed, total)`
progress record and a zero-argument cooperative cancellation check. The runner
polls immediately before and after each injected executor call and after each
progress notification. A cancellation detected during an executor call drops
that in-flight outcome. Typed terminal exceptions record accepted completed and
total counts; failures also record the exact cancellation-callback, executor,
progress-callback, or publication stage plus cause type/message. They never
carry a partial dataset or row collection.

The complete-batch helper retains outcomes privately and calls its publisher
only after all trials are accepted. Callback defects and publisher failures
therefore terminate without externally visible partial rows. The physics
executor signature and outcome validator are unchanged, and the original
successful scalar-ensemble JSON remains byte-identical at SHA-256
`671e5fd6c59aa1c068f2a3bd608ff7ef58c585b7ee4897ca49ef4ae73743f6a0`.
Sampling, trial ordering, request IDs, and input digests remain owned by the
existing deterministic path.

TDD RED captured the absent terminal types and hooks. Focused execution-job,
variation, and control coverage passes 47 tests. The implementation is split
into contract, complete-batch execution, and dataset-projection modules below
400 lines. Ruff/format, focused MyPy, relevant ground/flight/variation
regressions, campaign manifest, documentation, structural, assertion, and diff
gates are required before local commit `SELF`; no push or GitHub write occurs.

Keep #4369/#4273/#4267 open. This slice supplies no worker/thread, UI Run/Cancel
controls, progress throttling, execution-job-to-physics binding, browser
executor, partial scalar schema, result workspace/import, variable-wind wire,
compiled/downstream parity, protected evidence, publication, or release.

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
It adds a React/TypeScript import-only boundary for the Python-owned study and
variation `scalar-ensemble/v1` results. Strict parsing preserves exact schema,
provenance/model/input digests, definitions, units, categories, stages, series
identity, trial order, cohorts, and typed-null censored outputs; it never
coerces unavailable evidence to zero. The parser rejects duplicate and extra
fields, unsupported versions, nonfinite or unsafe numbers, Boolean numerics,
Unicode surrogates, forged identities/evidence, unknown cohorts, oversize
documents, and malformed UTF-8. A Python-produced complete/partial/failed/
unavailable golden fixture is consumed by both runtimes.

The boundary is capped at 8 MiB encoded JSON and 100,000 rows. Browser file
metadata is checked before reading and the actual buffer is checked again.
This is not a browser physics implementation or Run claim and adds no result
workspace, persistence, overlays, solver/capability or wind integration,
compiled/downstream parity, or release evidence. Focused React and Python,
full React, TypeScript, ESLint, Vite, Ruff, manifest, and documentation gates
are recorded in the commit evidence. Code, SPEC, manifest, and all handoffs
commit together as `SELF`; no push or GitHub write occurred.

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

Publication audit note: the workflow-pinned Ruff 0.14.10 formatter normalized
only `test_regional_ground_variation_request_io.py` after the implementation
commit. This is a test-layout-only follow-up with no material handoff, runtime,
wire, UI, or physics change; the focused gates were rerun before publication.

Hosted-current-head audit then found one MyPy-only mixin boundary: the file
controller requires a `QWidget` parent while the reusable mixin cannot prove its
eventual `QMainWindow` base statically. The follow-up now performs an explicit
TYPE_CHECKING-only `QWidget` cast at that construction boundary. The runtime
object and behavior are unchanged; the exact hosted MyPy profile and focused
file-control tests are required before treating the repair as green.

## 2026-08-11 local #4273 React request-workspace ownership

The unpublished `codex/4273-ground-variation-file-ui` branch starts from exact
PR #4366 head `8dfb1189c13f0fce99901e1ffbba152d813f9006`. Its read-only integration audit
found that PyQt already has a main-window owner for both editors, while React
kept the variation plan and regional draft/import evidence in mutually
exclusive panel-local state. Switching tabs destroyed one side of the request,
so binding File commands at that boundary would have been misleading.

This bounded prerequisite adds one App-owned reducer and hook. The variation
plan, analysis-execution selection, regional draft, and exact imported request
now survive panel unmount/remount; both panels are controlled adapters. A typed
request port composes only current state and transactionally applies a complete
request after validating the two ground keys, global bounded noise, plan/base
equality, identities, caps, and editor-qualified regional evidence. Invalid
apply leaves every prior field unchanged, and the disclosed illustrative
regional draft cannot be composed until explicitly edited or imported. The
toolstrip receives the reserved
typed port but exposes no new command in this slice.

React now names the existing Python-owned ground restitution and rolling-
resistance keys so applied plans remain inspectable. The scalar browser runner
rejects those inputs visibly instead of routing them into unrelated flight
physics. No physics, file dialog, upload/download, PyQt, browser-persistence,
or filesystem semantics changed. TDD captured the missing owner and navigation
reset. The complete React suite passes 763 tests across 121 files; TypeScript,
ESLint, and the production Vite build pass. The campaign manifest and its eight
tests, documentation governance, blocking-quality policy, module-size budget,
and diff checks also pass. Vite retains its inherited main-chunk warning.

Keep #4273/#4267 open. The next bounded child may add strict browser
serialization and contextual File controls plus the PyQt native controller,
using this owner rather than an imperative registry or illustrative fallback.
No push or GitHub write occurred from this branch.

## 2026-08-11 local #4273 seeded-request persistence

The unpublished `codex/4273-ground-variation-persistence` branch starts from
exact published PR #4365 documentation head
`27d2a68d3738d61307af9235f3f97f7bd400e0f3`. A read-only persistence audit
found a clean composition seam: the existing immutable seeded request,
`VariationPlan` and regional-plan object serializers, shared canonical numeric
JSON, duplicate-key-rejecting ground parser, bounded UTF-8 snapshot reader, and
native atomic text writer. No alternate storage or physics contract was needed.

The new UI-neutral v1 envelope persists the exact variation plan, exact
regional material plan, result/source/series identifiers, and row cap. It has a
1 MiB UTF-8 bound and deterministic compact canonical text suitable for the
existing browser-download model. Import requires exact fields and current
schema versions before delegating nested objects to their existing parsers.
Duplicate fields, nonfinite or cross-runtime-unsafe numbers, Boolean numeric
substitutes, surrogate text, malformed identifiers/caps, oversized documents,
and invalid nested plans fail closed. Import registers the two Rate extension
variables explicitly; merely importing the module still does not mutate the
shared registry.

Native reads take one sentinel-bounded strict UTF-8 snapshot. Writes serialize
and validate before reusing the existing flush/fsync/atomic-replace seam;
cancellation is a no-op and a failed replacement preserves the last-known-good
file. No UI, browser filesystem claim, or physics execution is added.

RED captured the missing persistence module. Twenty-two focused tests and 82
composition tests pass. The relevant Rate adapter/file plus complete shared
flight/ground/variation selection passes 545 tests with six expected
missing-Rust-wheel skips and one environment-only Hypothesis warning. Ruff,
import-skipping MyPy, Bandit, campaign-manifest validation and its eight tests,
documentation governance, blocking-quality, minimum-test, module-size,
changed-test assertion, placeholder, structural, and diff gates are green.

This is not a #4273/#4267 completion claim. UI/editor integration, workspace
embedding, regional-overlay variation, solver/capability consumption, wind,
compiled/downstream parity, protected review, publication, and release remain
open. No branch was pushed and no GitHub state was changed.

## 2026-08-11 PR #4365 seeded regional-ground material variation

Ready PR [#4365](https://github.com/D-sorganization/Tools/pull/4365) is stacked
on exact Tools PR #4364 head
`f13f0908dd2a553cf4d114afd31bb474d1b967c7`; independently reviewed
implementation commit `8c9c9512c61bac6f958ae7c7c0fe58e8f70525bf` follows.
It adds one UI-neutral runner that samples only base-surface normal restitution
and rolling resistance through the existing `VariationPlan`/`sample_inputs`
authority, rebinds an immutable regional request for every trial, and delegates
all flight, impact, bounce, and regional skid/roll physics to an injected exact
pipeline executor. The adapter adds no alternate physics or UI path.

The two adapter variables use the shared registry's explicit extension seam;
importing the module does not mutate the process-global registry. Deterministic
trial IDs and SHA-256 input provenance bind seed, plan, trial order, sampled
values, and exact base-plan identity. Sampled values are aligned with qualified
scalar metrics, while transfer failures and censored outcomes retain those
inputs and typed null output metrics. Requests fail before execution for
unsupported or missing keys, mismatched bases, implicit/nonfinite/Boolean/out-
of-range bounds, nonfinite scales or samples, invalid exact types, and row-cap
overflow.

RED captured the missing module. Twelve focused tests pass, including byte-for-
byte seed replay, subset-stable streams, changed-seed divergence, immutable
identity/provenance/order, failure nulls, fail-before-execution boundaries, and
a real-pipeline check that increased rolling resistance reduces qualified total
distance. The focused adapter plus pinned-registry suite passes 43 tests; the
complete relevant Rate adapters and shared flight/ground/variation selection
passes 506 tests with six expected missing-Rust-wheel skips and one environment
warning. Ruff, import-skipping MyPy, Bandit, campaign-manifest validation and
its eight tests, documentation governance, blocking-quality, minimum-test,
default module-size, changed-test assertion, 397-line new-module, placeholder,
and diff gates are green. The stricter full variation-directory 400-line scan
reports only the inherited 433-line `plot_data.py`, not this candidate.

This is deliberately not a #4273 or #4267 completion claim. Regional overlay
variation, UI/editor integration, persistence, wind coupling, solver/capability
consumption, compiled/runtime parity, downstream UpstreamDrift parity,
protected CI and release remain open. Publication does not promote this bounded
slice to issue or epic completion.

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
starts from exact published PR #4361 head
`81de044075a4f72c6da8fedb972437df79a06ab8`; its independently reviewed
implementation commit is `7f7d4b01d83d914ae5684715dc20c69388cf799f`.
The bounded hand integration adds matched additive `Ground Playback` views to
PyQt6 and React while preserving `Ground Surfaces`, its navigation state, and
current help. No later historical 72-file chain was merged or cherry-picked.

Both clients accept a strict standalone ground result or an explicitly
validated regional-execution envelope, reuse its nested result, and reject
null, cancelled, failed, empty-trajectory, or missing-summary evidence. The
adapters delegate to existing parsers and do not invoke or duplicate physics.
The shared absolute-time policy steps and jumps exactly and holds the lower
sample across phase boundaries. Play, pause, restart, looping, granular speed,
locked-scale 3D orbit/zoom/reset, honest carry/first-contact and observed-end
language, and accessible summary/evidence inspection are matched.

Playback hardening uses binary per-frame lookup, a deterministic 2,048-point
landmark-aware visual path, and a disclosed 256-row evidence window while
retaining the validated full result. RED first captured the absent timeline and
UI. Local qualification passes all 1,125 Rate/shared-ground Python tests and
all 119 React files / 754 tests. Ruff check/format, scoped Black, strict MyPy
on all five new Python production modules, Bandit, ESLint, TypeScript
type-check, production build, campaign-manifest validation, documentation
governance, the 400-line new-module budget, and diff checks are green. Fresh
protected current-head checks, dependency order, and ordinary merge gates
remain required.

Keep #4274 and #4267 open. Terrain mesh/changing-normal rendering, direct
editor-to-playback handoff, comparison, persistence, rendered visual QA,
camera presets/tracking, downstream UpstreamDrift/four-surface parity, and
protected CI/review/release remain explicit follow-on work.
## 2026-08-11 PR #4361 qualified regional-ground study adapter

Ready-for-review PR [#4361](https://github.com/D-sorganization/Tools/pull/4361)
starts from exact published PR #4360 head
`74f1ceafd87f952a76917dc868baa6414f856144`. Its independently reviewed
implementation commit is `d71c43fdd729b35e1abe5573f41ed60201698608`.
A read-only audit of current
flight metric, target, scalar-ensemble, capability, regional readback, and
ground-result contracts plus the historical `ground-study-scalar-adapter`,
`ground-study-result-adapter`, and `ground-study-projection` worktrees found
one reusable invariant: final ground study values are qualified only after
regional `COMPLETE` and ground `COMPLETE/REST` with a summary. The stale
parallel study model and its numeric censored totals were not copied.

The new UI-neutral Rate adapter reuses `to_ground_model_result`,
`FlightMetricInputs`, and `ScalarEnsembleDataset`. Complete-rest evidence may
populate canonical total distance, roll, final offline, and bounce count plus
distinct bounce-air/skid/surface-path/final-downrange study detail. Carry stays
separate. Partial and left-surface endpoints, all non-settled bounce outcomes,
regional cancellation/failure, missing summaries, and typed transfer failures
produce null numerics with exact typed cohort/reason/status/model/digest
attributes. Applying unqualified evidence clears stale ground inputs, so
censored total distance cannot become an optimizer final-rest objective.

RED captured the missing adapter. GREEN passes 7 focused tests, including a
positive numeric time-limited summary that is deliberately nulled, complete
left-surface censorship, every non-settled bounce reason, regional failure and
cancellation, transfer failure, and forged missing-summary defense. The full
Rate/flight/ground selection passes 1,299 tests. Ruff check/format, strict
MyPy, pinned Bandit, campaign-manifest validation and 8 manifest tests,
documentation governance, blocking-quality policy, minimum-test contract,
default module-size budget, and diff checks are green. The assertion and
400-line scans report only inherited stack files when compared with main; the
new test has behavioral assertions and the new production module is 328 lines.

This PR adds no UI, solver/capability invocation, wind strategy,
persistence, TypeScript or compiled runtime, four-surface parity, protected
CI/review, publication, or release. Keep #4273 and #4267 open.

## 2026-08-11 PR #4360 flight-through-regional-ground pipeline

Ready-for-review PR [#4360](https://github.com/D-sorganization/Tools/pull/4360)
on `feat/4271-flight-regional-ground-pipeline` starts from exact published PR
#4359 head `e53c6fb1bd273292c02085ee5d0a2b5497820871`. Its reviewed implementation
commit is `090e835477d1f19614f37f978a1b8a0e2f50ae21`. Audit found that the existing
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

The ready PR is not yet protected or reviewed and remains `not_released`. It
adds no new wire schema or migration and no PyQt6/React,
TypeScript/Rust/WASM, persistence, playback, calibration,
target/solver/variation, or downstream integration. Keep #4271, #4273, and
#4267 open.

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
