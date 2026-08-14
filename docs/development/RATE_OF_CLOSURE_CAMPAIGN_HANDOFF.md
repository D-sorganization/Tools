# Rate of Closure Ball-Flight Campaign Handoff

## 2026-08-12 #4385 Windows authority-state security
## 2026-08-10 #4143 durable Playwright artifact contract

- A child of exact PR #4325 head
  `f0f0dee074d29e3f12bf5f566c38d0080e43c756` adds the missing committed,
  reproducible web visual scenario without modifying the parent or production
  code.
- One Chromium test clears local persistence, opens Simulation/Strike, records
  the Driver default Tee at 38.1 mm, changes to explicit Ground, reruns, and
  records zero effective height with no representative tee. Semantic UI state,
  completed execution, zero console/page errors, nonblank images, and distinct
  image digests all gate the test.
- The local-fleet, path-filtered `Rate of Closure Visual Evidence` workflow builds first,
  captures at 1600 x 1200, and uploads PNGs, a v1 size/digest manifest, traces,
  and the HTML report for 14 days. There are no committed pixel baselines.
  Issue #4143 still waits on protected child CI/review, its normal dependency
  chain, and release to `main`.
- Local gates: deterministic `npm ci`; 526 passing Vitest tests; one passing
  Playwright scenario; TypeScript, ESLint, Vite build, workflow routing and
  pinning, YAML parsing, and whitespace checks all pass.

## 2026-08-10 #4143 child receives repaired launch-registry parent

- Ready PR `#4325` stays on `feat/4143-tee-parity-fixture`, based on
  `feat/4181-launch-monitor-registry`.
- Exact parent `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76` is incorporated by a
  normal merge commit. There is no production/test-code conflict and no
  rebase, retarget, force-push, or parent rewrite.
- Preserve the shared parity fixture and deterministic web/PyQt evidence.
  Fresh exact-head CI, review, dependency order, and release to `main` remain
  required before #4143 can close.

## 2026-08-10 #4143 Python/React golden ball-setup parity

- The bounded `feat/4143-tee-parity-fixture` branch starts at exact PR #4203
  head `31cbc007d4c85b5479b7cd0fb0969124eab2af67`, preserving its draft state,
  base, and stack order.
- A single `ball_setup_golden_v1.json` fixture declares schema/version, metre
  units, the ground-plane-to-ball-bottom reference, ball radius, Driver/Tee and
  iron/Ground defaults, explicit club-default overrides, Ground zero effective
  height, center/serialization geometry, invalid finite-domain cases, and a
  legacy simulation-run migration.
- Python and React independently consume every case through their public
  configuration/persistence boundaries. Verification is 18 passing Python
  tests, 24 passing React tests, and green TypeScript, ESLint, Vite production
  build, Ruff check, and Ruff format.
- Recorded visual evidence is stored under
  `C:\Users\diete\AppData\Local\Temp\rate-4143-visual-evidence-8050eeba`.
  Playwright captured the 1600 x 1200 default Driver/Tee and rerun
  explicit-Ground React states after semantic control/diagram and zero-error
  checks. A hidden 1400 x 900 PyQt harness captured the same states after
  canonical center, editor, and tee-artist assertions. The browser manifest
  SHA-256 is `43df78e04b47e1b3209ff7a574718f90847ccda6dde5afd863d43191a950ccf7`;
  the PyQt manifest SHA-256 is
  `07822495dbcfa7568615ccb2728481210c28963614434c80f6997210c325a6f9`.
  PNGs remain external evidence rather than oversized repository binaries.
- #4143 remains open for protected CI/review and release to `main`. The strict
  campaign release manifest does not exist in this exact #4203 history; it was
  added later on a divergent branch and is not backported by this bounded
  slice.

## 2026-08-10 Second propagation into launch-monitor registry

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

- Independent post-admission review verifies deterministic submit-versus-close
  and exceptional-lifespan regression coverage, exact Python reason/detail
  typing, and matching TypeScript detail/media-type enforcement. Earlier
  no-runner statements are historical evidence, not current service state.
- Default local authority construction now atomically couples the qualified
  regional-ground runner with a strict ready capability; impossible flag,
  reason, or manager/runner combinations fail closed.
- Readiness validates the authenticated exact capability body, not status alone.
  The manager cooperatively cancels and joins its one owned worker on shutdown.
- React production admission accepts only the qualified v1 evidence and removes
  its test-only bypass. No TypeScript physics or automatic execution is added.
- The shared launch-registry test now pins built-ins as an ordered prefix, so
  documented Rate extensions do not create xdist worker-order dependence.
- Independent review reproduced and closed submit-versus-close and exceptional
  lifespan-exit races. Python and React now share exact reason, detail-length,
  media-type, and bounded-body capability validation.
- Complete local qualification passes 2,014 Python/PyQt/shared-simulation
  tests with one optional Rust-wheel parity skip and 860 React tests across
  132 files. Pinned MyPy 1.13/Ruff 0.14.10, Ruff format, high-severity Bandit,
  TypeScript, zero-warning ESLint, the 214-module production build,
  deterministic fixtures, campaign-manifest, governance, and diff checks pass.
- Visible controls remain unmounted/disabled until a strict imported job can be
  selected, confirmed, run/cancelled, and explicitly saved. Direct editor
  execution requires a later Python-authoritative job preparation boundary.
- Static hosting, persistent recovery, packaged PyQt helper behavior, compiled
  runtime parity, downstream parity, protected integration, and release remain
  open. Do not close #4369, #4273, or #4267 from this headless service slice.

## 2026-08-11 #4369 qualified fixture, runner, and matched presentation

- Exact published source head before composition:
  `ff1310c09c066a32e57b50e4daee4da5a40d7bf3`.
- The checked fixture generator derives flight digests from the registered
  profile and rebuilds all job/status/result identities deterministically.
- Qualified jobs reuse one flight solve across seeded regional trials and
  preserve cooperative cancellation, typed-null transfer rejection, typed
  terminal defects, and complete-only job-bound publication.
- Matched PyQt6/React presentation shows exact identity/provenance/progress/
  failure/result evidence but offers no executing handler. At that earlier
  fixture slice the default runner was unregistered and capability was false;
  visible Run/Cancel remains disabled after the newer headless admission.
- Consolidated evidence: 2,004 Python/PyQt/shared-simulation tests and 858 React
  tests, real authenticated loopback success, deterministic fixture `--check`,
  Ruff/format, MyPy, Bandit, TypeScript, zero-warning ESLint, 214-module build,
  docs, manifest, and diff gates. The optional Rust parity test remains skipped
  because the local interpreter has no `swing_core` wheel.

## 2026-08-11 #4369 hosted MyPy 1.13 correction

- Exact failing PR #4372 head:
  `e91ef8dcde8cdd8e6545ffc0ea7cb755058ec2fb`.
- Hosted quality-gate job `93999818041` passed checkout, dependency install,
  Ruff, and formatting, then reported one pinned MyPy 1.13 `redundant-cast`
  finding in the loopback submitter's exact-bool cancellation path.
- The redundant cast and unused import are removed with no runtime or
  capability change. The matching pinned MyPy 1.13 profile and 13-test
  submitter suite pass locally; protected CI and ancestor integration remain
  open.

## 2026-08-11 #4369 composed authority continuation ready for PR #4372

- Exact published parent head: `3571952c2344ca23ffa65121c606faab1b735a23`.
- The composed continuation adds canonical cross-runtime status, typed
  preflight, the authenticated PyQt client and real-process tests, a strict
  digest-qualified Waterloo/Penner execution profile, and an abortable
  one-active-job React controller.
- The fixture's synthetic flight hashes failed deterministic recomputation at
  that earlier continuation head, so production injected no runner and
  capability was false. No visible execution control is promoted now.
- Consolidated evidence: 1,148 Python/PyQt and 854 React tests, real loopback,
  Ruff/format, focused MyPy, changed-file high-severity Bandit, TypeScript,
  zero-warning ESLint, 214-module build, docs/manifest/structural governance,
  and diff checks. Protected CI and the full ancestor stack remain mandatory.

## 2026-08-11 local #4369 PyQt real-loopback qualification

- Exact source parent: composed head
  `f7342cae7296410f8cfd262fd9877363beb5dc63`.
- Actual loopback Uvicorn/FastAPI processes validate the PyQt submitter's real
  bearer HTTP path, canonical submit/status/cancel/result lifecycle, typed
  production-preflight terminal, bounded close/join, and process reaping.
- Wrong authentication exposes neither the ephemeral token nor model data. The
  production false capability still constructs no submitter.
- A bounded import-factory seam is the only runtime change; the default factory
  and environment-only token path are unchanged. Test runners reject before
  physics or wait only for cancellation, so no successful model or physics,
  UI, persistence, protected, downstream, or release claim is added.
- Code, tests, SPEC, manifest, and all handoffs commit atomically; exact focused,
  composed, static, and governance evidence follows.
## 2026-08-11 local #4369 versioned flight execution profile

- Exact source parent: local composed head
  `7e4069e891d8b4bde3f1d712b5b47897359a414e`.
- One strict registry identity maps Waterloo/Penner `tools-core/1.0.0` to the
  exact bounded `max_time_s`, `step_s`, and whole-number `sample_every`
  settings schema.
- The v1 recomputation authority binds default model coefficients, adaptive
  RK45, launch-relative planar transfer-surface contact, base sampling,
  deterministic decimation, and terminal retention. Only exact trajectory and
  result digest matches can release the recomputed flight.
- The canonical fixture deterministically fails trajectory/result hash
  qualification because its declared flight evidence is synthetic. Preflight
  reports a typed non-sensitive mismatch at zero trials; ground physics is not
  invoked and no result is published.
- Production runner injection, capability, visible client controls,
  persistence, ground execution, protected integration, and downstream parity
  remain open. Replace the fixture only with profile-produced evidence under a
  pinned numerical runtime. Keep #4369/#4273/#4267 open.
- TDD RED captured the missing registry. Evidence passes 20 focused
  registry/preflight tests, 147 composed authority/contract/manifest/flight
  tests, Ruff, Black, focused MyPy, Bandit, JSON/manifest validation, and
  structural gates. No GitHub write occurred.
## 2026-08-11 local #4369 React execution-controller prerequisite

- Exact source parent: local composed head
  `7e4069e891d8b4bde3f1d712b5b47897359a414e`; isolated branch
  `codex/4369-authority-react-controller-v1`.
- A UI-neutral hook validates the exact capability and execution job, admits at
  most one active job, polls status one request at a time, exposes exact
  completed/total progress and typed terminal failure, delegates POST cancel,
  and requests a result only after succeeded status.
- Result publication reuses the strict client's complete expected-job binding.
  Abort signals, operation IDs, and run generations suppress late publication
  after cancellation, reset, unmount, or a superseding request. React
  StrictMode remount behavior has a dedicated regression.
- At that earlier controller slice production admission was false and a named
  admission override was confined to lifecycle tests. The newer headless
  admission removes the override without adding visible controls, a qualified
  UI physical runner, TypeScript physics, persistence, protected carrier,
  downstream parity, or issue completion.
- TDD RED captured both the missing controller and the StrictMode lifecycle
  bug. Evidence passes 31 focused React authority tests, strict TypeScript,
  zero-warning ESLint, the 214-module production build, manifest validation and
  all eight manifest tests, and module/minimum-test governance. Implementation
  and all durable handoffs are atomic; no push or GitHub write occurs.

## 2026-08-11 local #4369 canonical authority status wire

- Exact parent: published PR #4372 head
  `3571952c2344ca23ffa65121c606faab1b735a23`.
- The transport-neutral application contract owns strict Python construction,
  parsing, source-job validation, and canonical serialization for all six
  status states and every stable failure code/stage under 4,096 bytes.
- The server manager consumes that contract. A Python-recreated shared golden
  is parsed and canonically serialized byte-for-byte by React, with adversarial
  duplicate, extra, typed, non-finite, unsafe-number, state-semantic, and
  mismatched-job coverage in both runtimes.
- This slice adds no physics, UI, persistence, transport adapter, capability
  promotion, or execution claim. Keep #4369/#4273/#4267 open.
## 2026-08-11 local #4369 production-runner preflight qualification

- Exact source parent: published PR #4372 head
  `3571952c2344ca23ffa65121c606faab1b735a23`.
- Audit found no versioned authority mapping the job's generic numeric flight
  settings, model version, and declared flight digests to exact solver and
  surface-event semantics. The fixture's `sample_every` key is not consumed by
  production flight code.
- A typed runner preflight now distinguishes unknown model IDs from recognized
  models lacking a qualified profile. It checks cancellation first and maps
  callback defects or rejection to exact complete-only terminal failures.
- Tests prove that preflight invokes neither flight nor regional-ground
  physics; the manager publishes only a generic `preflight` failure with zero
  completed trials and no result.
- At that earlier preflight slice no profile was registered, the production
  factory injected no runner, and capability was unavailable. The later
  headless service qualification does not promote a client control or release.
  The next physical UI slice must preserve the exact versioned input,
  solver/surface, and digest-recomputation contract. Keep #4369/#4273/#4267
  open.
- TDD RED captured the missing module. Evidence passes 7 focused runner tests,
  98 composed authority/contract/variation/manifest tests, 28 physical
  pipeline regressions, Ruff, Black, focused MyPy, Bandit, and manifest
  validation. The serial full Rate suite hit the 10-minute local command cap
  without reporting a failure; root owns the nonredundant complete gate. No
  GitHub write occurred.

## 2026-08-11 local #4369 PyQt authenticated loopback submitter

- Exact source parent: published PR #4372 head
  `3571952c2344ca23ffa65121c606faab1b735a23`.
- The widget-free application adapter sends canonical jobs through the existing
  authenticated loopback runtime, validates the shared job-bound status wire,
  polls with bounded timeout/backoff, forwards cooperative POST cancellation,
  and retrieves only complete expected-job-bound results.
- Client failures after acceptance attempt one bounded best-effort cancel;
  callback/transport/status/result failures remain typed and non-secret, and
  cancellation, shutdown, stale status, or late success cannot publish data.
- At that earlier submitter slice the false capability returned no adapter.
  The visible PyQt workflow still registers no submitter; no controls,
  persistence,
  scientific claim, protected carrier, downstream parity, or release is added.
- Code, tests, SPEC, manifest, and all handoffs commit atomically as `SELF`;
  exact local gate evidence follows.

## 2026-08-11 #4369 authority terminal-count binding

- Injected cancellation/failure terminals must match the submitted job total
  and cannot regress observed progress. Mismatches retain the prior count and
  become typed validation failures with no result publication.

## 2026-08-11 #4369 result-digest typing stability

- The result digest now crosses its helper boundary through an explicit string
  local, keeping both isolated skipped-import and complete PR-delta MyPy 1.13
  root sets clean without changing runtime bytes or canonical evidence.

## 2026-08-11 local #4369 PyQt worker/controller prerequisite

- Exact source parent: published PR #4372 head
  `990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`.
- A widget-free QThread worker/controller accepts the exact qualified job and a
  dependency-injected submitter. It forwards typed progress, cancellation, and
  failure signals and prevents overlapping work or stale publication.
- Success requires one complete execution-result envelope that passes the
  existing expected-job identity, trial-count/order, and series binding.
  Invalid or unbound results become typed validation failures; no partial rows
  or dataset are exposed.
- No physical submitter is registered, no visible control is wired, and the
  authority capability remains unavailable. Browser submission, matched client
  controls, protected integration, and release remain open.
- TDD RED captured the absent port. Evidence passes 7 focused QThread tests, 79
  job/result/qualification/variation regressions, all 1,068 Rate Python/PyQt
  tests, Ruff, Black, focused MyPy, manifest validation/tests, docs governance,
  and structural gates. The full suite's one empty-legend warning predates and
  is unrelated to this slice.
## 2026-08-11 local #4369 authority job manager and endpoints

- Exact source parent: published PR #4372 head
  `990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`.
- One thread-safe in-memory manager owns at most one active job and a bounded
  oldest-first terminal/result set.
- Authenticated submit/status/cancel/result endpoints reuse the exact job and
  result contracts, disable caching, stream-cap requests at 1 MiB, and reject
  encoded, mistyped, duplicate, malformed, and oversized bodies.
- Cancellation is forwarded through existing variation hooks and prevents a
  late result from being published. Typed failure evidence omits raw exception
  and token text; only a complete result revalidated against its job is exposed.
- Production has no injected runner, returns `execution_unavailable`, and
  retains `regional_ground_execution=false`. This is an orchestration boundary,
  not proof of physical execution or a browser Run feature.
- TDD RED captured the absent manager. Green evidence includes 21 focused
  manager/API tests, 88 related contract regressions, all 1,076 Rate of Closure
  Python/PyQt tests, Ruff/format, focused MyPy, Black, changed-file Bandit,
  manifest JSON/eight tests, placeholder, module-budget, and diff gates.
- Code, tests, SPEC, manifest, and all handoffs commit together as `SELF` with
  no push or GitHub write. Physical job invocation, full in-flight interruption,
  client controllers, persistence/recovery, compiled/downstream parity,
  protected evidence, and release remain open.
## 2026-08-11 local #4369 React authority client contracts

- Exact parent: published PR #4372 head
  `990b2a156e4a939dbd1bd0c874895dc4f3fd53e7`; local branch
  `codex/4369-authority-react-client-v1`.
- React reserves strict same-origin canonical submit, job-bound status,
  POST-cancel, and complete-result retrieval contracts. Responses are
  byte-bounded, duplicate-safe, exact-shape validated, and bound to the source
  job before state/result publication; invalid jobs produce no network I/O.
- Status matches the authority API exactly: queued, running,
  cancel-requested, succeeded, failed, and cancelled states; completed/total
  progress; result availability; and a nullable stable failure code/stage.
  Auth, unknown-job, unavailable, malformed-server-error, and abort behavior
  are explicit and tested without publishing synthetic status.
- Capability polling is one request at a time. Cleanup clears scheduled polls,
  aborts active fetches, and prevents an earlier effect from publishing stale
  capability state after its query changes or the component unmounts.
- At that earlier client slice all submit/status/cancel/result flags were
  disabled while the Python-owned capability was false. The newer qualified
  headless API still is not visible Run integration and adds no browser
  physics, visible controls, persistence,
  downstream parity, protected evidence, or issue completion.
- Complete local evidence passes 1,061 Python/PyQt and 841 React tests across
  130 files, the 214-module production build, strict TypeScript, zero-warning
  ESLint, manifest validation/tests, and module/minimum-test budgets. Existing
  Hypothesis collection, polynomial empty-legend, Node local-storage, and Vite
  chunk notices remain. No push or GitHub mutation is authorized for this
  child.

## 2026-08-11 local #4369 job-bound result envelope

- Exact source parent: published PR #4370 head
  `0a485958bd6ed46dce18e65fd3e3cd1fa797502a`.
- The bounded Python/React execution-result v1 envelope carries job/input
  identities, complete scalar evidence, and recomputed canonical dataset SHA.
- Expected-job matching binds result ID, trial count, zero-based order, and
  every series ID. SHA-256 provides integrity/provenance, not authentication.
- Full local evidence passed 1,048 Python/PyQt and 818 React tests, production
  build, static analysis, manifest, and module gates.
- No executor, partial publication, UI/backend/storage, compiled physics, or
  downstream parity. Keep #4369/#4273/#4267 open.
- Hosted MyPy 1.13 remediation removed a redundant result-digest cast without
  changing runtime behavior or canonical evidence.

## 2026-08-11 #4369 execution qualification continuation

- Exact source parent: published PR #4370 head
  `0a485958bd6ed46dce18e65fd3e3cd1fa797502a`.
- The job binds exact callback-free regional options, all skid/roll settings,
  executor revision, source plan, and launch-origin execution plan.
- Teed-driver tests prove the base, every overlay, and axis origin receive one
  translation and that provenance/digests are recomputed in both runtimes.
- V1 exposes only implemented `max_trials`; unsupported parallelism, timeout,
  and configurable fail-fast fields are rejected.
- Local evidence passed 243 Python regressions, 35 focused Python tests, all
  804 React tests, static analysis, production build, manifest, and module
  gates. No physics invocation, result binding, controller, or release claim.

## 2026-08-11 local #4369 validator-failure hardening

- Exact parent: published PR #4370 head
  `0a485958bd6ed46dce18e65fd3e3cd1fa797502a`.
- Outcome-validator exceptions now terminate through the same typed,
  complete-only boundary as executor and callback defects, using stable stage
  `validation` and explicit exception chaining.
- Counts include only accepted trials; no partial rows or dataset escape, and
  the successful-output SHA-256 regression remains authoritative.
- Scope excludes authority, physics, job binding, backends, workers, browsers,
  and matched UI. Keep #4369/#4273/#4267 open.

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

- Exact local composition head `915c80f38` contains the execution-job contract,
  complete-only batch progress/cancellation, and strict React result import.
- Do not execute the shared job golden as scientific evidence: recomputed
  Waterloo/Penner trajectory/result digests differ from its synthetic values.
- Before enabling Run, bind exact skid/roll settings and executor revision,
  implement or reject every orchestration option, translate base and overlays
  into launch-origin coordinates for teed shots, bind result bytes to job/input
  digests, and expose one cancellable Python authority to both clients.
- The intended delivery is direct `QThread` use in PyQt and an authenticated
  loopback FastAPI sidecar plus same-origin Vite proxy for React. Static web
  builds remain explicitly unavailable and retain strict result import only.

## 2026-08-11 local #4369 batch progress/cancellation prerequisite

- This continuation starts from exact local execution-job contract commit
  `a5a1b99bfa6cb6400bc18b13139d7893471824f4` on
  `codex/4369-regional-ground-execution-job` and changes only the Python seeded
  regional-ground variation execution boundary.
- Frozen hooks provide exact accepted completed/total progress and a cooperative
  cancellation check. The batch polls immediately before and after each
  unchanged injected executor call and after progress delivery. Pre-cancel runs
  no trial; in-flight cancellation rejects the current outcome.
- Typed cancellation and failure terminals carry counts but no partial rows or
  dataset. Failures preserve a stable cancellation-callback, executor,
  progress-callback, or publication stage and cause metadata. The complete
  publisher is invoked once only after all outcomes are accepted, so callback,
  executor, or aggregation defects cannot leak a partial scalar ensemble.
- Successful output stays byte-identical at SHA-256
  `671e5fd6c59aa1c068f2a3bd608ff7ef58c585b7ee4897ca49ef4ae73743f6a0`;
  deterministic sampling, trial order/identity, and provenance are unchanged.
  The production modules remain below 400 lines.
- TDD RED captured the missing controls. The execution-job, base variation, and
  control suites pass 47 tests; focused type/lint plus relevant cross-suite and
  governance gates are required before local commit `SELF`. No push or GitHub
  write occurs.
- This is not an execution-feature completion: job-to-executor binding,
  worker/thread and matched Run/Cancel UI, browser-capable qualified physics,
  result import/workspace integration, variable wind, compiled/downstream
  parity, protected evidence, publication, and release remain open. Keep
  #4369/#4273/#4267 open.

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

- The unpublished `codex/4369-regional-result-parser` child starts exactly
  from published PR #4368 head
  `7d2d155b35f2ae55842de120864c4a343a5ebcb6`.
- React strictly imports the Python-owned regional study and variation
  `scalar-ensemble/v1` results under an 8 MiB encoded-wire and 100,000-row
  bound. It preserves schema, provenance/model/input digests, definitions,
  units, categories, stages, series identity, ordered trials, cohorts, and
  typed-null censored outcomes; unavailable values are never converted to
  zero. Declared browser file size is checked before read and the returned
  buffer is independently bounded and fatally decoded as UTF-8.
- Both runtimes assert a Python-produced complete/partial/failed/unavailable
  fixture. Duplicate/extra/version/nonfinite/unsafe/Boolean/surrogate and
  forged identity/evidence inputs fail closed, including duplicate definition
  or value keys, unknown cohorts, and inconsistent row/series identities.
- This is parser/import only: no browser physics, Run claim, result workspace,
  persistence, regional overlays, solver/capability or wind integration,
  compiled/downstream parity, protected evidence, or release. Focused and full
  test/static/governance evidence is recorded with the implementation. Code,
  SPEC, manifest, and all handoffs commit together as `SELF`; no push or
  GitHub write occurred.

## 2026-08-11 local #4273 contextual regional-ground request File controls

- The unpublished `codex/4273-ground-variation-file-controls` child starts
  from exact ready PR #4367 head
  `0968a4ced5644aa8e2673ca278d261eeb92c31f8` and binds contextual File
  controls to its App-owned combined request port.
- React strictly parses and canonically serializes the Python-owned v1
  envelope under the same 1 MiB UTF-8 bound. Both runtimes assert a
  Python-produced golden payload. Accessible import/download commands appear
  only in Variation and Ground Surfaces, apply transactionally, report errors
  visibly, and disclose browser-owned destination/overwrite/atomicity behavior.
- PyQt6 uses the same stable command IDs and the existing application parser,
  bounded reader, and atomic writer. Open validates before applying either
  editor; Save validates before opening its chooser; cancellation is a no-op.
  Exact imported evidence is retained until an owning editor changes. The
  illustrative draft is not silently persisted and an unsupported run count
  cannot partially mutate the editors.
- No physics or illustrative fallback is added. All 782 React tests in 123
  files, the 87-test focused Python/PyQt selection, and a 25-test post-fixture
  follow-up pass; TypeScript, ESLint, Vite, Ruff, MyPy, and policy gates are
  green. The implementation, SPEC, manifest, and all handoffs commit together
  as `SELF`; no push or GitHub write
  occurred. Keep #4273/#4267 open for pipeline invocation, overlays,
  solver/capability and wind integration, compiled/downstream parity,
  protected evidence, publication, and release.

## 2026-08-11 local #4273 React request-workspace ownership

- The unpublished `codex/4273-ground-variation-file-ui` child starts from exact
  PR #4366 head `8dfb1189c13f0fce99901e1ffbba152d813f9006`. Audit showed a clean PyQt
  main-window composition seam but no React owner above the mutually exclusive
  Variation and Ground Surfaces panels; navigation discarded one editor.
- A UI-neutral reducer and App-owned hook now retain the physical variation
  plan, analysis policy, regional draft, and exact imported request across
  navigation. Both panels are controlled. A reserved typed toolstrip port
  snapshots only current state and validates a complete replacement before one
  atomic reducer transition, so invalid requests cannot partially apply. The
  disclosed illustrative draft remains ineligible until edited or imported.
- The two existing Rate ground input keys are inspectable in the web registry.
  Browser scalar execution fails closed with a visible unsupported-path status;
  no regional physics was reimplemented or approximated.
- RED captured the missing owner and navigation reset. All 763 React tests in
  121 files pass, as do TypeScript, ESLint, the production Vite build, campaign
  manifest validation plus eight tests, docs governance, blocking-quality
  policy, the scoped 400-line module budget, and diff checks. Vite reports only
  its inherited main-chunk size warning.
- This local slice adds no File controls, request upload/download adapter,
  native dialog, PyQt wiring, persistent browser workspace, or protected
  evidence. Keep #4273/#4267 open and stack those remaining clients only on
  this state owner. No branch was pushed and no GitHub state changed.

## 2026-08-11 local #4273 seeded-request persistence

- The unpublished `codex/4273-ground-variation-persistence` branch starts from
  exact published PR #4365 docs head
  `27d2a68d3738d61307af9235f3f97f7bd400e0f3`. Audit confirmed that the
  immutable seeded request, existing nested serializers, canonical safe-number
  JSON, strict duplicate-key parser, bounded reader, and atomic writer form a
  complete persistence seam without a parallel schema or storage mechanism.
- A UI-neutral v1 envelope persists the exact variation and regional plans,
  result/source/series identities, and row cap as deterministic compact JSON.
  The 1 MiB UTF-8 contract is portable to existing browser-download behavior;
  native reads and writes reuse sentinel-bounded snapshots and atomic replace.
- Exact field and current-version checks precede existing nested parsers.
  Duplicate keys, unsafe/nonfinite/Boolean numbers, surrogate text, malformed
  identities/caps, invalid nested contracts, and oversize payloads fail closed.
  Parsing explicitly registers the Rate ground variables and never runs
  physics.
- RED captured the absent module. Twenty-two focused, 82 composition, and 545
  relevant Rate/shared tests pass. The broad run has six expected missing-Rust-
  wheel skips and one environment-only warning. Ruff, import-skipping MyPy,
  Bandit, campaign manifest and eight tests, documentation, blocking-quality,
  minimum-test, module-size, changed-test assertion, placeholder, structural,
  and diff gates pass.
- This candidate is local and unpushed. UI/editor wiring, workspace embedding,
  browser filesystem claims, overlay variation, solver/capability use, wind,
  compiled/downstream parity, protected review, publication, and release remain
  open; #4273/#4267 are not complete.

## 2026-08-11 PR #4365 seeded regional-ground material variation

- Ready PR [#4365](https://github.com/D-sorganization/Tools/pull/4365) starts
  from exact Tools PR #4364 head
  `f13f0908dd2a553cf4d114afd31bb474d1b967c7`; independently reviewed
  implementation `8c9c9512c61bac6f958ae7c7c0fe58e8f70525bf` follows.
  It adds a UI-neutral, bounded seeded runner for only the base regional-plan
  normal-restitution and rolling-resistance values. Sampling stays in the
  existing `VariationPlan`/`sample_inputs` authority; physics stays in an
  injected exact flight-through-regional-ground executor.
- Each immutable trial carries stable seed/trial identity, sampled values, a
  canonical input digest, and a rebound regional request/provenance record.
  Qualified complete-rest outputs and all censored/transfer-failure typed-null
  outcomes remain row-aligned with their sampled inputs in the existing scalar
  ensemble contract. Explicit registry registration avoids import-time global
  mutation.
- DbC rejects unsupported/missing keys, mismatched base values, missing,
  nonfinite, Boolean, or out-of-range bounds, forged nonfinite scales or
  samples, invalid exact outcomes or plan identities, and row-cap overflow
  before executor use where applicable.
- RED captured the absent module. Twelve focused tests, 43 focused-plus-
  registry tests, and 506 relevant Rate-adapter/shared-flight/ground/variation
  tests pass. The broad set has six expected missing-Rust-wheel skips and one
  environment warning. A real-pipeline test observes shorter qualified total
  distance at higher rolling resistance. Ruff, import-skipping MyPy, Bandit,
  campaign-manifest validation plus eight tests, documentation,
  blocking-quality, minimum-test, default module-size, changed-test assertion,
  the candidate's 397-line budget, placeholder, and diff gates pass. A strict
  whole-directory 400-line scan reports only inherited 433-line
  `plot_data.py`.
- This published bounded candidate does not close #4273/#4267. Region
  overlay variation, UI/persistence, solver/capability use, wind coupling,
  compiled and downstream parity, protected CI/review, publication, and release
  remain open, as do protected current-head checks and dependency integration.

## 2026-08-11 PR #4364 post-ground spatial-target projection

- Ready-for-review PR [#4364](https://github.com/D-sorganization/Tools/pull/4364)
  is stacked on exact PR #4363 head
  `ec50fdf059f91ca9e4664da891398af218e1ba65`; independently reviewed target
  implementation commit `b480f17f11b86a57326622168e4c748efc77aaf3`
  leaves inherited playback production code untouched. Its UI-neutral adapter accepts only exact
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
  changed-test assertion, placeholder, and diff gates pass. Fresh protected
  current-head checks, dependency order, and ordinary merge gates remain; no
  editor/UI, persistence, solver/capability,
  aerial trajectory evaluation, compiled runtime, geometry, or physics is
  claimed. Keep #4192, #4273, and #4267 open.

## 2026-08-11 PR #4363 matched ground playback

- Ready-for-review PR [#4363](https://github.com/D-sorganization/Tools/pull/4363)
  starts from exact published PR #4361 head
  `81de044075a4f72c6da8fedb972437df79a06ab8`; independently reviewed
  implementation commit `7f7d4b01d83d914ae5684715dc20c69388cf799f`
  hand-integrates only the initial playback slice.
- Additive matched PyQt6/React workspaces import strict standalone results or
  explicit validated regional envelopes. They reuse the nested result, reject
  null/cancelled/failed/empty/missing-summary evidence, and never run physics.
- The phase-safe absolute-time policy holds the lower sample at discontinuous
  boundaries and gives exact step/jump/play/pause/restart/loop/speed behavior.
  Locked physical scale, orbit/zoom/reset, honest endpoint labels, and
  accessible summary, warning, calibration, provenance, and evidence tables
  are shared product behavior.
- Large results use binary frame lookup, at most 2,048 landmark-aware visual
  points, and disclosed 256-row tables while retaining the full validated
  result. Local qualification passes all 1,125 Rate/shared-ground Python tests
  and all 119 React files / 754 tests. Ruff/Black, strict scoped MyPy, Bandit,
  ESLint, TypeScript type-check, production build, manifest, documentation,
  400-line new-module budget, and diff gates are green. Fresh protected
  current-head checks, dependency order, and ordinary merge gates remain.
- Keep #4274/#4267 open for terrain meshes/changing normals, direct editor
  handoff, comparison, persistence, rendered visual QA, camera presets and
  tracking, downstream parity, and protected CI/review/release.
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
Status verified 2026-08-10. This isolated integration is published as draft
## 2026-08-11 Capability input specification in whole workspaces

- Draft PR [#4348](https://github.com/D-sorganization/Tools/pull/4348)
  publishes this bounded child from independently approved implementation head
  `5730e74752ffb84ab3560bed6318b7d97b6e627d`, preserving base
  `feat/4144-workspace-variation-study`. Protected current-head CI, review,
  parent landing, integration, and release remain required.

- The remaining no-publish finding against local head `68692bbcb` is repaired
  locally. Interactive projection requires the exact ordered `ball_speed`
  (`m/s`), `launch_angle` (`deg`), and `launch_direction` (`deg`) basis with a
  3-by-3 correlation matrix, one club, and one spin default. Alternate units,
  covariance, reordered parameters, and unsupported shapes fail closed before
  projection, UI apply, or File/Open mutation; no conversion/rescaling occurs.
- Earlier findings are also repaired: full-document authority plus editable overlays preserve accepted
  evidence and advanced policy; unsupported interactive documents fail closed.
  Native identity/generation gates reject stale success, and both parsers cap
  numeric wire magnitude at `1e300` through the native File/Open error path.

- Branch `feat/4197-workspace-capability-request` starts from exact draft
  PR #4343 head `4ff103d9a6ef886099c180da560e8458d5e20b49`; it does not modify
  or publish the parent branch.
- Explorer-session v5 embeds the established cross-runtime
  `capability-optimization-workflow/v1` request. PyQt6 and React round-trip the
  editable profile/club, capability ranges/distributions, objective, target,
  fixed-spin evaluator assumptions, integration policy, budgets, and seed.
- Full parsing precedes application. Both clients retain the full validated
  document while controls overlay only represented fields. Legacy v1-v4 files
  require an explicit current capability fallback. Native apply is rollback-safe,
  and both clients
  invalidate computed results when restored inputs replace the live request.
- Results, observation ensembles, runtime objects, inferred identity, and
  optimizer-execution claims remain excluded. Wind-aware optimizer inputs are
  not yet user-editable in this workflow and therefore are not fabricated.
  #4197/#4225, UpstreamDrift parity, protected CI/review, integration, and
  release remain open.
- Local qualification passes 71 focused Python workflow/workspace/File/PyQt/
  manifest tests and 70 focused React contract/File/UI tests; pinned MyPy, Ruff,
  TypeScript, zero-warning ESLint, the 211-module production build, 11
  campaign-manifest tests, docs governance, and manifest-layout validation
  also pass.
## 2026-08-11 Camera child quality-gate repair

- The Ruff-only repair was published normally as exact #4349 head
  `317d2b0c16c9516ef2cac028e77b25c6f13aced4`. Fresh protected CI passed lint
  and format, then exposed strict MyPy errors in camera-preference enum parsing
  and a redundant compositor cast. The current repair adds explicit string
  narrowing plus adversarial non-string tests, removes the redundant cast,
  binds the camera callback through a typed partial, and makes the Qt putting
  speed's float boundary explicit. It does not change valid wire values or
  runtime behavior. Independent review, normal fast-forward publication, and
  fresh exact-head CI remain open.
- The first independent audit found only the SPEC header still declared
  1.14.35 despite the 1.14.36 changelog. Both version fields are now aligned;
  no source, test, schema, or runtime artifact changes in that correction.

## 2026-08-11 Camera-preference workspace child

- Draft PR [#4349](https://github.com/D-sorganization/Tools/pull/4349)
  publishes this bounded child from independently approved exact head
  `f2d3be771a9ba1d17f5e8942484b3fb49c236527`, preserving its #4343 base and
  composed #4331/#4303 histories. Fresh protected CI/review, rendered platform
  qualification, dependency landing, and release remain required.

- `feat/4218-camera-preference-persistence` begins at exact published #4343
  head `4ff103d9a6ef886099c180da560e8458d5e20b49`, then normally
  composes exact #4331 `e07e2a66a894c93b50c1ded308fc8902f2ff6c24`
  and #4303 `98cf35994488dd6f3d66916415bbc9f8e7c8bf3f` in order.
- `camera-preferences/v1` owns three stable viewport keys and exactly five
  deliberate fields: preset, face-on side, bounded zoom, tracking, and Auto
  Fit. It excludes the moving target and manual suspension.
- View-workspace v2, QSettings, localStorage, and whole-workspace File adapters
  share the contract. V1 migration uses #4303 defaults; malformed/future
  documents reject before mutation. Five native/browser camera adapters are
  connected and animation-frame tracking cannot trigger persistence.
- Protected CI/review, rendered cross-platform qualification, UpstreamDrift
  consumers, ordered integration, and #4218 completion remain open.

## 2026-08-11 variation-study workspace protected publication

- Branch `feat/4144-workspace-variation-study` is published normally as draft
  PR [#4343](https://github.com/D-sorganization/Tools/pull/4343), targeting
  exact PR #4340 branch `feat/4136-workspace-torque-profiles` at parent head
  `26105f668de260d75a99f450726348570db7ff89`.
- Independently reviewed implementation head
  `73041194a7cfd8cae14cd1739b806617af933648` contains explorer-session v4,
  deterministic file-operation ordering, complete swing-output registry
  parity, and valid saved-focus controls. Publication documentation commits
  change no runtime code.
- Protected CI/review, #4142/#4144, dependency ordering, UpstreamDrift
  qualification, integration, and release remain open.

Status verified 2026-08-11. This isolated integration is published as draft

## 2026-08-11 Variation-study workspace child

- Worktree `Tools-worktrees/workspace-variation-study` and branch
  `feat/4144-workspace-variation-study` start at exact published PR #4340 head
  `26105f668de260d75a99f450726348570db7ff89`.
- Explorer-session v4 persists canonical user-authored variation inputs,
  distributions/ranges/groups, trial count, deterministic seed,
  simultaneous/individual/both analysis policy, and a strict selected-output
  focus on both Tools clients.
- Simulation is the sole persisted Ground/Tee authority. Duplicate ball setup,
  Tee Height under Ground support, unknown/duplicate/empty output selections,
  ambiguous legacy plans, and partial native application all fail closed.
- Independent review makes asynchronous browser Open recheck the latest dirty
  state and legacy fallbacks before applying, and derives the complete
  17-field swing output focus from the executor rather than a stale duplicate
  registry. The focus is persisted policy; complete results remain available.
- Native focus controls preserve one selected output, matching React and
  keeping every File operation inside the validated contract. Browser file
  reads capture their selected type and a monotonic operation ID; stale
  out-of-order reads cannot overwrite newer choices or change parser mode.
  Confirmed New/Close operations invalidate pending reads; cancelled resets do
  not discard the user's pending Open.
- This slice stores no results, identity, optimizer outputs, or flight-run
  outputs; it does not qualify UpstreamDrift or close #4142/#4144/#4218.
  Protected review/CI and dependency-ordered release remain open. Post-review
  gates pass 21 focused Python and 48 focused React tests, pinned MyPy 1.13,
  Ruff, TypeScript, zero-warning ESLint, the 210-module production build, 11
  campaign manifest tests, changed-file/module-size, docs, manifest-layout,
  changed-Python, JSON, and diff validation. A broader PyQt pair was stopped
  without failure output after two contention timeouts; the three directly
  affected native workflows passed serially.

## 2026-08-10 Torque-profile workspace child

- Worktree `Tools-worktrees/workspace-torque-profiles` and branch
  `feat/4136-workspace-torque-profiles` start at exact published PR #4336 head
  `6e9dd85a3c5f43d37cf8a704db0555bdad734e7e`.
- Explorer-session v3 references the root canonical torque-profile library
  through a strict selection payload shared by Python and TypeScript. It stores
  active stable identity, passive/prescribed run configuration, canonical joint
  locks, and profile-source provenance without duplicating or inventing profile
  data.
- PyQt6 and React map the real editor library and selection. Complete validation
  precedes atomic application; canonical schema/SI-unit/c0-first order/joint/
  fit/source rules remain authoritative. Legacy v1/v2 migration requires an
  explicit current fallback and rejects conflicting root profiles.
- Remaining work includes optimizer, variation-run and flight-run persistence,
  installed UpstreamDrift parity, protected CI/review, and ordered release.
  Local qualification passes 34 focused Python tests, 43 focused React tests,
  pinned MyPy 1.13, Ruff check/format, TypeScript, zero-warning ESLint, the
  206-module production build, the 11-test campaign-manifest suite,
  module-size, docs, manifest-layout, changed-Python, changed-test assertion,
  JSON, and diff gates. Protected release evidence remains open.

## 2026-08-10 Ball setup and spatial target workspace child

- Worktree `Tools-worktrees/workspace-ball-target` and branch
  `feat/4225-ball-target-session-mappers` start at exact draft PR #4333 head
  `bd7da1e6d42557d5e8782b8f4f64fc4ed183e5ce`.
- Explorer-session v2 embeds the same strict simulation-setup v1 contract in
  Python and TypeScript. It includes Ground/Tee mode, SI tee height, derived
  ball centre, club-default/override provenance, and the complete canonical
  spatial-target label/kind/frames/position/source/tolerance document.
- PyQt6 maps the real Simulation tab state; React lifts ball state to the app
  model beside the shared target. Nested corruption is rejected before any
  live mutation. Legacy explorer-session v1 requires an explicit preserve-
  current fallback and never fabricates missing physical values.
- Remaining #4225 work is torque-editor, optimizer, variation-run, flight-run,
  installed UpstreamDrift parity, protected CI/review, and ordered release.
  Post-refactor qualification passes 8 focused native tests, 27 focused React
  tests, pinned MyPy 1.13, Ruff check/format, TypeScript, zero-warning ESLint,
  the 203-module production build, the 11-test campaign-manifest suite,
  module-size, docs, manifest-layout, changed-Python, and diff gates. The
  broader suites were stopped without failure output after several minutes on
  the overloaded workstation; they are not claimed as validation.

## 2026-08-10 Live workspace File-adapter continuation

- Worktree `Tools-worktrees/workspace-file-adapters` and branch
  `feat/4225-workspace-file-adapters` start at exact draft PR #4330 head
  `d8176bb5863a35725199bb8357a5f000f9bdd3ba`.
- PyQt6 and React now connect New, Open, Save As, strict compositor-layout
  Import/Export, and Close to real validated state. Native adds atomic Save and
  persisted Open Recent. Browser Save/Recent remain honestly disabled because
  the browser surface cannot safely overwrite or retain a filesystem path.
- The whole-workspace live slice is impact scenario, club, units, primary
  navigation, and compositor. Parsing finishes before mutation; invalid or
  cancelled reads preserve state, dirty destructive commands confirm first,
  native writes are atomic, and unsupported torque/variation payloads fail
  closed.
- Remaining work is explicit: add strict mappers for simulation-local ball,
  target, torque-editor, optimizer, variation-run, and flight-run state; qualify
  installed UpstreamDrift consumers; pass protected CI/review and merge in
  dependency order. This slice does not close #4218 or #4225.
- Local evidence: 921/921 Rate-of-Closure Python tests; focused MyPy, Ruff, and
  Black; React TypeScript, zero-warning ESLint, 116 files / 693 tests, and the
  201-module production build; plus baseline-aware module-size, docs,
  linked-debt, changed-Python policy, changed-test assertion, and diff gates.
  The legacy full-tree 500-LOC scanner has no loaded grandfather baseline in
  this checkout and reports 232 pre-existing files; all newly added modules
  remain below 500 lines.
## 2026-08-11 PR #4331 current-parent propagation repair

- Exact live child `c7bccbccc6cda0c9b938b2862ed660cebdcb7597` is retained
  first and exact current PR #4330 parent
  `304a069b1777dcf8cf107de26caa3b9fbe96dbb3` is incorporated second through a
  normal merge on `feat/4284-orthographic-axis-polish`.
- The failed hosted format gate was an ancestry defect: stale merge base
  `d8176bb5863a35725199bb8357a5f000f9bdd3ba` exposed the parent's formatting
  commit and six worktree pointers as child-local changes. With current parent
  ancestry, the effective child delta contains only nine intended camera-polish
  documentation, adapter, and regression files.
- The merge is content-clean and camera behavior is unchanged. No rebase,
  retarget, force-push, parent rewrite, CI retry, or GitHub write was used.
  Independent review, release-owner publication, and protected exact-head CI
- Root `SPEC.md` is synchronized at version `1.14.30` with the exact per-preset
  depth-axis mapping, full Matplotlib artist-suppression boundary, preservation
  of visible engineering axes/native one-sided ticks, and restoration on
  isometric/manual orbit. This governance repair changes no runtime file.
- Fresh merged-tree evidence is 71 Python/PyQt camera, compositor, layout,
  main-window, and manifest tests; exact-delta Ruff/format, pinned MyPy 1.13,
  Bandit, documentation, changed-code, module-size, minimum-test, assertion,
  manifest-layout, Spec Check, version, whitespace, and diff gates. React
  passes 114 files / 686
  tests, TypeScript, zero-warning ESLint, the 199-module production build, and
  four serial Playwright camera cases across desktop and constrained 2x-DPR
  projects. `npm ci` audited 337 packages with zero vulnerabilities.

## 2026-08-10 Repaired compositor-parent propagation into persistence child

- Continuation branch `feat/4225-multiview-persistence` now normally
  incorporates exact repaired compositor parent
  `0e3054e6a7fa0e3e38e1312b4132bbd1e4336fb2`.
- Keyboard/persistence production and test code did not conflict; only the four
  additive handoff/spec files required reconciliation.
- No rebase, retarget, force-push, parent rewrite, or history rewrite was used.
  Fresh local verification, protected exact-head CI, and review remain.
- The pinned-MyPy delta requires an explicit typed current-workspace validation
  local; parsing, validation, migration, and serialization behavior is unchanged.

## 2026-08-10 Repaired legend-parent propagation into PR #4327

- Draft PR `#4327` keeps branch `feat/4225-multiview-compositor` and base
  `fix/4224-default-legend-layout-local`.
- Exact repaired legend parent `531a851dc125e83ad86abe1601651e163f5f866d`
  is incorporated through a normal merge.
- Multi-view production/test code did not conflict; only the four additive
  handoff/spec files required reconciliation.
- No rebase, retarget, force-push, parent rewrite, or history rewrite was used.
  Fresh local verification, protected exact-head CI, and review remain.

## 2026-08-10 Issue #4225 multi-view keyboard/export acceptance slice

Worktree `Tools-worktrees/issue-4225-multiview-persistence` and branch
`feat/4225-multiview-persistence` begin at exact draft PR #4327 head
`e975f66bdcfc5a32f9688b8c2c6e34fe1b53ce6e`. The parent slice replaces the
disabled/direct-route placeholders with three real distinct viewport hosts in
PyQt6 and React. Impact, Swing, and Flight can be selected directly or composed
as single, horizontal, vertical, and grid layouts. The active run and playback
time are synchronized, flight time is mapped relative to impact, and each host
keeps independent camera/overlay ownership. React Flight also retains the
canonical spatial-target editor. React's established Strike, Swing, Kinetics,
and Flight displays remain reachable beside Multi View, while direct toolstrip
commands return to Multi View and select a real host.

Both clients persist the same version-1 workspace shape and defensively migrate
legacy layouts, corrupt values, unsupported future IDs, and missing active
views to deterministic known-view layouts. Cardinality is strict across both
clients: one host is Single, two use a valid split, and three are Grid. Valid
per-slot legends survive recovery/transitions. The live transport owns saved
play/loop/rate and settled time; PyQt6 debounces active-frame writes. Native
controls now include hover guidance and constrained multi-view grids expose
scroll navigation instead of clipping real plots.

This continuation closes the two remaining local acceptance-proof gaps. React
quick-view tabs implement roving focus and Arrow Left/Right, Home, and End;
native controls use an explicit Layout -> Impact -> Swing -> Flight tab order.
Both clients are tested manipulating view membership entirely by keyboard.
Strict version-1 import/export boundaries validate before mutation, reject
future formats, preserve playback and legend state, and the native import
survives QSettings reconstruction. The canonical view document is now proven
inside the whole-app workspace v2 envelope too. This exposed and repaired its
pre-existing nested-array double-freeze parser defect. Local evidence is 921
Python/PyQt Rate tests, 114 React files / 686 tests, focused MyPy,
Ruff/format, TypeScript, zero-warning ESLint, production build, module-size,
changed-policy, assertion, whitespace, and diff gates; 337 npm packages audit
with zero vulnerabilities. File-menu commands remain
disabled because no file picker or full live-session mapper is claimed here.
Keep #4225 and #4218 open for protected CI/review, dependency-ordered
integration, and UpstreamDrift consumer parity.
## 2026-08-10 Repaired mobile-parent propagation into PR #4324

- Draft PR `#4324` keeps branch `fix/4224-default-legend-layout-local` and
  base `fix/rate-mobile-tools-menu`.
- Exact repaired mobile parent `16a1167c31126238163297983862004afc5001d9`
  is incorporated through a normal merge.
- Legend/layout production/test code did not conflict; only the four additive
  handoff/spec files required reconciliation.
- No rebase, retarget, force-push, parent rewrite, or history rewrite was used.
  Fresh local verification, protected exact-head CI, and review remain.

## 2026-08-10 Issue #4224 non-obscuring legend rail slice

Immutable implementation evidence
`6c65a69624007912d45615fbe59314924c5107dc` plus real-canvas follow-up
`83b4baa3be7424777db4dd50883b7a9e45c8ca91` on an isolated branch from exact
PR #4301 head `5c8efcbe5fcd6f993ef947a85e39852d268780a6` advances the default-legend part
of #4224 without changing any existing worktree or remote branch. PyQt6
Swing/Flight scenes use a figure-owned Outside Right legend and compute the
reserved axes gutter from its rendered width. Retained figure legends are
removed before each scene rebuild, real canvas resize triggers legend-only
reflow without camera/playback advancement, and inside or hidden choices
cannot leave a stale outside artist. The minimum 360 x 280
regression proves the legend remains inside the figure and outside the axes;
accessible names cover both controls.

React plot cards now share a pure layout contract between the data rectangle
and legend drawing. At 520 px, the plot ends at x=330 and the legend begins at
x=350. The exact focused PyQt6/manifest suite is 69 passed; changed Python Ruff/format and
pinned MyPy pass. The installed React evidence passes the focused one-file /
four-test regression, the complete 111-file / 674-test Vitest suite,
TypeScript, scoped zero-warning ESLint, and the 196-module production build;
`npm ci` audited 337 packages with zero vulnerabilities. This is a bounded
non-obscuring-default slice, not completion of #4224: workspace
persistence/migration, exported-layout proof,
complete rendered QA, protected CI/review, and dependency-ordered integration
remain open. PR #4303 is a separate camera-default child and is not claimed as
ancestry of this local branch.
## 2026-08-11 PR #4303 current-parent propagation repair

- Preserve exact live child `2e07bec58b8a759c9db36ea7afb26a1c835434f5`
  first and normally merge exact current PR #4301 parent
  `c653f9ff9193d6cdb8e11a13ad0001707e468a42` second without changing PR #4303's
  branch or base.
- The former merge base `05713bcdd8f9889dcdcbaa5bdbaeab139d599b64`
  was stale, causing GitHub conflicts and exposing parent formatting as
  child-local. The merge keeps the current parent for one unrelated shared
  flight-contract test and reconciles four additive current-state documents;
  camera production code merges cleanly.
- Preserve the shared Python/TypeScript moving-subject default: animated PyQt6
  and React clubhead/flight viewports start at 2x zoom with bounded tracking
  and Auto Fit enabled, static viewports remain neutral, and controls remain
  independently user-overridable. Physics and geometry are unchanged.
- No rebase, retarget, force-push, parent rewrite, publication, GitHub write,
  or CI retry is used. Independent review and protected exact-head CI remain
  release gates; #4300 and epic #4218 stay open.
- Fresh merged-tree evidence is 49 focused Python/PyQt camera, layout, and
  simulation tests plus 14 campaign/launcher-manifest tests; exact-delta
  Ruff/format on six Python files; pinned MyPy 1.13 and Bandit on four
  production files; documentation, changed-code, module-size, minimum-test,
  assertion, manifest-layout, whitespace, and diff gates. React passes 111
  files / 673 tests, TypeScript, zero-warning ESLint, the 195-module production
  build, and six serial Playwright camera/toolstrip cases across desktop and
  constrained 2x-DPR projects.

## 2026-08-10 Repaired camera-parent propagation into PR #4301

- Draft PR `#4301` keeps branch `fix/rate-mobile-tools-menu` and base
  `feat/4284-camera-snap-tracking`.
- Exact repaired camera parent `104503aac9779b195d46d38e8ed32611ffc8dfd7`
  is incorporated through a normal merge.
- Mobile-toolstrip production/test code did not conflict; only the four
  additive handoff/spec files required reconciliation.
- No rebase, retarget, force-push, parent rewrite, or history rewrite was used.
  Fresh local verification, protected exact-head CI, and review remain.

## 2026-08-10 PR #4301 four-surface parent propagation

Draft PR #4301 retains base `feat/4284-camera-snap-tracking`. Its normal
two-parent merge keeps original constrained-toolstrip child
`05713bcdd8f9889dcdcbaa5bdbaeab139d599b64` first and exact, independently
reviewed #4299 head `142631a90c008942bad99745e279748a7eda2ffa`
second. No branch is rebased, retargeted, rewritten, or force-pushed. The
File/View/Tools popovers keep their shared 16 px viewport-gutter clamp,
unchanged desktop anchor, bounded mobile width, and native keyboard and
accessibility semantics while inheriting the four-surface inventory, repaired
flight-to-ground stack, and complete camera/playback controls.

Fresh combined-tree evidence is 1,589 Python/PyQt/shared-swing tests with one
explicit unavailable-wheel skip; 111 React files / 673 tests; TypeScript,
zero-warning ESLint, the 195-module build, and six desktop/constrained 2x-DPR
browser cases; all 137 `tools-core` tests plus format and warning-denied
Clippy; and exact-delta Ruff/format, pinned MyPy, Bandit, deterministic-
authority, assertion, minimum-test, documentation, manifest-layout, size,
conflict-marker, and diff gates. Independent staged-tree review found no
findings; protected current-head CI is still open. This propagation does not
complete #4300,
#4284, #4264, #4260,
or their parent epics; native rendered QA and installed-consumer conformance
remain release gates.

## 2026-08-10 PR #4299 camera/ground-stack propagation

Draft PR #4299 keeps base `feat/4199-wind-workflow` and normally merges the
original four-surface child head
`dca40c6c0168df3aa0cd0de0e5ae0ff109715b6a` first with independently
reviewed #4298 head `57942e64744a199e4fd7d604fe2eeb9faddd062a`
second. No branch is rebased, retargeted, rewritten, or force-pushed. The
result retains `four-surface-capability/v1`, its declared-scope generator,
schema, canonical inventory, and exact evidence paths while inheriting the
complete camera-control and repaired flight-to-ground stack.

The declared inventory still covers 15 structured campaign programs, 18
unique linked active release specifications, and six curated capability
records across model, control, output, view, persistence, and export
categories. Every record classifies Tools PyQt6, Tools React, UpstreamDrift
PyQt6, and UpstreamDrift React explicitly. Both UpstreamDrift cells remain
unsupported unless an immutable installed consumer pin and repository-bound
conformance evidence exist; unstructured narrative features remain outside
the governed boundary until promoted to a structured authority.

Local integration evidence is 1,589 Python/PyQt/shared-swing tests with one
explicit unavailable-wheel skip; 110 React files / 670 tests; TypeScript,
zero-warning ESLint, and the 194-module production build; four Playwright
camera cases across desktop and constrained 2x-DPR viewports; and all 137
`tools-core` tests plus formatting and warning-denied Clippy. The exact hosted
delta also passes Ruff/format on 52 Python files, pinned MyPy 1.13 on 36
production files, Bandit on 34 source files, both deterministic authorities,
and documentation, changed-code, source-size, assertion, manifest-layout,
conflict-marker, and diff gates.

Independent exact-tree review found no findings. This propagation is not issue
or epic completion: protected current-head CI, installed-consumer evidence,
four-surface conformance, native rendered QA, and dependency-ordered release

## 2026-08-10 Native orthographic-axis presentation polish

Native Windows review of the current camera carrier confirmed Face On, Down
the Line, Overhead, Reset, Track, Auto Fit, Re-center, replay, pause, and loop,
then identified one presentation defect: the depth-axis labels and ticks
collapsed into the screen plane and overlapped plot titles or visible axes in
the exact orthographic presets.

The shared PyQt camera adapter now applies one explicit display-axis contract
after each Simulation and Flight render. Face On hides only display x/right,
Down the Line hides only display y/target, and Overhead hides only display
z/up. Because Axes3D can retain cached artists, the adapter suppresses the
depth axis container, label, line, pane, and tick artists together. The two
in-plane physical axes remain labelled. Isometric/reset and manual orbit
restore all axis artists from each fresh render, preventing stale visibility
across modes without forcing both tick-label sides visible. A
parameterized headless GUI regression proves all three mappings and both
restoration paths in both viewports. Physics, camera angles, limits, tracking,
zoom, geometry, and React behavior are unchanged.

Exact implementation head `c6f7122d8fb63eacaf94fc0f295c2e470f80fce8`
passes 11 focused camera tests, the 33-test camera/layout/workspace integration
set, Ruff, pinned MyPy on the three changed production modules, and diff
validation. Native Windows inspection at 1282 x 752 confirms Face On, Down the
Line, and Overhead retain only the two in-plane labelled axes with no duplicate
ticks; reset/isometric restores the complete 3D frame. Protected exact-head CI,
review, parent-first integration, and installed UpstreamDrift parity remain
required before this follow-up is released.

## 2026-08-10 Current-parent propagation into camera PR #4298

- Draft PR `#4298` keeps branch `feat/4284-camera-snap-tracking` and base
  `feat/4199-wind-workflow`.
- Exact current parent head `1e82f15026786ea0b08f78f4c001590ddce9ff39`
  is incorporated through a normal merge.
- Camera production/test code did not conflict. Only the four additive
  current-state handoff/spec files required reconciliation.
- No rebase, retarget, force-push, parent rewrite, or history rewrite was used.
  Fresh local verification, protected exact-head CI, and review remain required.

## 2026-08-10 Repaired wind-scalar parent propagation into PR #4282

- Draft PR `#4282` keeps branch `feat/4199-wind-workflow` and base
  `feat/4199-wind-scalar-adapter`.
- Exact repaired parent head `d6fb04e07c2a625412e9208b07103acdc42c621b`
  is incorporated through a normal merge after its quality gate passed.
- No wind-workflow production or test code conflicted. No rebase, retarget,
  force-push, parent rewrite, draft-state change, or merge was used.
- Twenty-five focused tests plus documentation governance, changed-file size,
  and whitespace checks pass locally. Fresh protected CI and review must pass
  before this exact head can propagate normally into PR `#4285`.

## 2026-08-10 PR #4298 exact hosted-mypy repair

Exact head `a51e49e4d2e7f5b1985c802f8290ea7649e7927e` passed Ruff and
formatting, then protected quality-gate job `93503197807` failed at pinned
MyPy 1.13 with 18 integration-only errors in the inherited flight-to-ground
adapter. The hosted delta checks every changed production file from the
preserved PR base in one skipped-import invocation; that exposed compatibility
`StrEnum` members as `str` and generator-built NumPy tuples as variable-length
tuples.

The repair constructs exact typed enum members through their public
constructors and builds explicit three-component tuples. Runtime values, wire
bytes, coordinate transforms, physics, and camera behavior are unchanged. The
exact hosted command now passes all 33 production files; 79 focused ground,
transfer, and flight-physics tests plus Ruff/format pass. Fresh protected
current-head CI is required after the normal fast-forward follow-up; do not
retry the obsolete failed head. This repair does not complete #4284, #4269, or
their parent epics.

## 2026-08-10 PR #4288 exact repaired-ground propagation
## 2026-08-10 issue #4274 workspace-v2 adversarial review repair

Independent review of local workspace-v2 commit
`b9579c0b16aaf6188f216acbca0b75828c2a5fe6` found two publication blockers.
An integer too large for `float` escaped strict Python parsing as
`OverflowError`, bypassing PyQt's transactional import-error boundary; Python
and TypeScript also disagreed on whether public limit overrides could raise the
fixed 11 MiB, 100,000-point-per-result, and 200,000-point-combined contract
caps. The follow-up normalizes numeric conversion overflow to `ValueError` and
rejects every override above the same hard caps in both runtimes while still
allowing stricter caller limits.

RED-first regressions cover v1 and v2 parsing, active PyQt playback retention,
and all three cross-runtime hard caps. The complete Rate suite passes 913 tests
and the React suite passes 111 files / 690 tests; focused Ruff/format,
TypeScript, zero-warning ESLint, and Prettier gates pass. The canonical golden
bytes and physics remain unchanged. This is a review repair, not completion of
#4274 or #4267; protected publication, approval, parent landing, and the
existing downstream boundaries remain open.

## 2026-08-10 issue #4274 comparison workspace v2

Local branch `feat/4274-ground-playback-comparison-workspace-v2` starts from
exact raw-evidence parent `9be08f9a67336aae4ac1f6add68c190d42e67f10`.
PyQt6 and React now save one strict
`rate-of-closure-ground-playback-workspace/v2` document containing the primary
`flight-to-ground-result/v1`, an always-present nullable comparison envelope
with independent visibility, union-window playback state, and orbit view.
The stable shared comparison type is `GroundPlaybackComparisonState`.

Strict v2-only parsing is separate from discoverable version dispatch. V1
documents migrate one way to normalized v2, visibly disclose migration, and
subsequent saves remain v2. Exact-field recursive validation, duplicate-key
rejection, finite bounds, 100,000 points per result, 200,000 combined points,
and an 11 MiB UTF-8 bound apply before parse and after serialization. Python
and TypeScript pin the same LF-terminated 9,055-byte golden document at SHA-256
`28b94af16a05315a9d1067bda894a3817dce5849a9562e1ffef7d0d8caecd654`.

Import is transactional on both clients. Invalid input preserves every
last-good primary, comparison, visibility, playback, camera, and running state;
valid input commits paused. PyQt restores comparison visibility with signals
blocked, avoids intermediate seeks/callbacks, and commits union time last.
React memoizes result timelines so comparison errors or rerenders do not reset
time/camera, and portable time uses the union window.

RED-first contract and UI failures were observed before implementation.
Focused evidence passes 34 Python/PyQt tests and eight React panel tests;
complete suites pass 910 Rate tests and 111 React files / 689 tests. TypeScript,
zero-warning ESLint, the 204-module production build, Ruff, Black, pinned MyPy
1.13, Prettier, Python 3.10 compilation, Bandit with the inherited unchanged
low-severity B101 assertion excluded, minimum-test, module-size,
documentation, and conflict-marker gates pass. Maximum-ledger paging/lazy
mounting, camera/Playwright/native visual evidence, terrain editing/meshes,
ensembles, compiled runtimes, UpstreamDrift parity, protected publication,
issue acceptance, and epic closure remain open; keep #4274/#4267 open.

## 2026-08-10 PR #4318 raw comparison evidence publication

The independently reviewed raw comparison-evidence continuation is published
as ready-for-review [PR #4318](https://github.com/D-sorganization/Tools/pull/4318)
on the preserved `feat/4274-ground-playback-comparison` base. Exact reviewed
implementation `b89d642441e72c84481293c1f7bf6de03e933feb` descends directly
from exact PR #4317 head `b55bfa3d710a4ff8fabd4ebc7ec31cddad37cee4`.
Independent review found and verified the RED-first repair of a late React
comparison-import race after strict-result or workspace primary replacement.
The GitHub App identity and fast-forward remote state were verified before the
normal push; no branch was rewritten or retargeted. SPEC 1.14.44 records
publication. Protected exact-head CI, approval, parent landing, issue
acceptance, and epic closure remain required; keep #4274/#4267 open.

## 2026-08-10 issue #4274 matched raw comparison evidence

Local branch `feat/4274-ground-playback-comparison-evidence` starts from exact
published comparison parent
`b55bfa3d710a4ff8fabd4ebc7ec31cddad37cee4`. PyQt6 and React now show
separately labelled primary/comparison trajectory and event ledgers. Trajectory
rows contain exact absolute and result-relative times plus phase; event rows
contain exact event time and identity. Both retain positions, linear
velocities, and angular velocities from each result without pairing rows or
implying correspondence between different sample grids or event sequences, and
they remain available when only the dashed graphical overlay is hidden.

Dedicated comparison trajectory/event CSV actions reuse the existing canonical
full-ledger serializers, so frame fields, numeric normalization, deterministic
ordering, and LF endings stay matched across clients. PyQt comparison exports
now reuse a shared `QSaveFile` atomic writer instead of direct replacement.
Invalid comparison import retains the prior tables and exports; successful
primary replacement clears them together and invalidates any in-flight
comparison import that began against the previously displayed primary. React
raw rendering and toolbar controls were extracted from the panel, and every
touched production module remains below 400 lines.

RED-first failures were observed on both surfaces before implementation.
Focused evidence passes 14 PyQt tests and eight React tests; complete suites
pass 900 Rate tests and 110 React files / 685 tests. Ruff check/format, pinned MyPy
1.13, Python 3.10 compilation, TypeScript, zero-warning ESLint, Prettier, the
203-module production build, Bandit, minimum-test, module-size, documentation,
and conflict-marker governance gates pass. SPEC 1.14.43 records the slice.
Independent exact-commit review is clean and PR #4318 is published ready for
review. Protected CI, approval, parent landing, and issue acceptance remain required.
Workspace-v2 comparison persistence, maximum-ledger paging or lazy mounting,
camera/Playwright/native visual verification, terrain editors, ensembles,
compiled runtimes, UpstreamDrift parity, and epic closure remain open; keep
#4274/#4267 open.

## 2026-08-10 PR #4317 comparison playback publication

The independently reviewed comparison continuation is published as ready-for-
review [PR #4317](https://github.com/D-sorganization/Tools/pull/4317) on the
preserved `feat/4274-ground-playback-persistence` base. Exact reviewed
implementation head `ab0d07c0b60b2034259444a8fb68253fa24ddac7` has clean
ancestry through merge `395c48f4142520b0f0a1b41479d02ac29c27abcf`, whose
parents remain original comparison commit `5ceb806961e76c3699934fafcd4aba96c06bbd20`
first and exact PR #4316 head
`2c56294ecda0204886508946239c7ca5b50b8b14` second. The GitHub App identity
was verified before the normal push; no branch was rewritten or retargeted.
SPEC 1.14.42 records publication. Protected exact-head CI, approval, parent
landing, issue acceptance, and epic closure remain required; keep #4274 and
#4267 open.

## 2026-08-10 issue #4274 comparison-session review repair

Independent review of the normally propagated comparison continuation found
four release-blocking contract defects. The repaired PyQt6/React behavior now
keeps the union absolute-time session loaded when comparison artists are
hidden, so visibility no longer changes scrubber limits, stepping, or the
current observation. PyQt workspace v1 remains intentionally primary-only:
export clamps only the serialized time to the primary timeline and does not
mutate the live comparison time. File-dialog failures explicitly disclose when
the last valid comparison remains loaded.

Direct comparison deltas are normalized once through the shared eleven-decimal
cross-runtime numeric policy. Python JSON, CSV, tables, and React exports
therefore agree on canonical values such as `0.2` and `0`, without binary
floating-point noise. Paired calibration evidence now includes ID, kind,
source, and canonical confidence on both clients (twelve total identity,
status, provenance, and calibration rows).

Regression-first evidence is green: 22 focused Python/PyQt tests, the complete
Rate suite (898 tests), 12 focused React tests, and the complete React suite
(110 files / 683 tests) cover canonical deltas/CSV, comparison-only workspace export,
visibility-only overlay toggling, complete calibration evidence, and retained
file-error disclosure. Ruff check/format, pinned MyPy 1.13, TypeScript,
zero-warning ESLint, and the 200-module production build pass. SPEC 1.14.41
records the review repair. Fresh complete suites, independent exact-head
re-review, guarded publication, protected CI, approval, parent landing, issue
acceptance, and epic closure remain open. Keep #4274 and #4267 open.

## 2026-08-10 issue #4274 matched comparison playback

The local continuation on `feat/4274-ground-playback-comparison` adds matched
PyQt6/React comparison playback from exact local parent
`0ef91e84b6d49551723ba0fbfb8eb1bf7b1ebfa2`. A second strict
`flight-to-ground-result/v1` is parsed and fully validated before it can
replace the last-good comparison; failure leaves both the primary and prior
comparison intact. A successful primary replacement deliberately clears a
stale comparison only after the new primary commits.

Both clients use one absolute-time window spanning the two observed results.
Each result is phase-safely interpolated only inside its own samples; outside
its observed interval the marker is clamped and explicitly labelled as waiting
for first contact or held at its qualified/observed end. One locked physical
metre scale contains both paths. Primary and comparison use solid versus dashed
trajectories, circular versus diamond event/ball markers, an accessible
show/hide control, paired identity/status/provenance, and a complete fourteen-
row direct scalar table. Deterministic JSON retains both exact result records
and the table; deterministic CSV exports the same direct
`comparison_minus_primary` deltas. No causal, inferential, or extrapolated
physics claim is made.

Focused evidence passes 22 Python/PyQt contract, GUI, and tooltip tests; the
complete Rate suite passes 898 tests. React passes 17 focused tests and the
complete suite (110 files / 683 tests), TypeScript, zero-warning ESLint, and a
200-module production build. Ruff check/format, MyPy, CPython 3.10 compilation,
module/changed-file budgets, documentation, minimum-test, assertion, secrets,
marker, and diff gates pass. A normal two-parent merge now has original
comparison commit `5ceb806961e76c3699934fafcd4aba96c06bbd20` first and
exact live parent PR #4316 head
`2c56294ecda0204886508946239c7ca5b50b8b14` second. The base remains
`feat/4274-ground-playback-persistence`; no branch was rewritten or
retargeted. SPEC 1.14.39 records the feature and 1.14.40 records this
propagation. Independent exact-merge review and guarded publication remain
open. Comparison persistence in workspace v1, comparison trajectory/
event evidence tables, ensembles/statistical inference, terrain meshes and
editors, inverse solving, compiled runtimes, UpstreamDrift parity, protected
CI, approval, parent landing, issue acceptance, and epic closure remain open.
Keep issue #4274 and epic #4267 open.

The exact pinned MyPy 1.13 propagation gate additionally reconciles skipped-
import and fully resolved inference: explicit typed time/serializer bindings
avoid both `Any` returns and redundant casts, while distinct metric and
provenance row variables prevent incompatible loop-variable reuse. These are
type-only repairs with no runtime coercion or scientific behavior change.

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

## 2026-08-10 PR #4323 exact hosted-MyPy repair

The repair is published on ready PR #4323 at exact current head
`3957f013eeadd448ffa381f12d65b6a076abe21b`, a guarded normal fast-forward
from prior head `b8101e070ea59fd9b336b960c2c7a0648bf5fb3f`. Base
`feat/4275-ground-tilted-conformance` is unchanged. No retarget, merge, force
operation, parent rewrite, or existing-worktree edit occurred.

The failure is reproduced from hosted run `31429284874`, job `93588443824`:
Python 3.12 plus pinned MyPy 1.13, `MYPYPATH=src:src/python/src`, and
`--follow-imports=skip` reported eight `no-any-return` errors in the three-file
production delta. The imports skipped by that profile caused otherwise typed
`Vector3`, `SurfaceRun.result`, and rest-predicate expressions to appear as
`Any`. Explicit `typing.cast` boundaries plus one DRY result helper resolve all
eight errors without runtime conversion or numerical, event, termination, or
wire-contract changes. The exact hosted command, all 247 ground tests, 42
focused skid/passivity/conformance tests, Ruff/format, campaign-manifest and
eight manifest tests, documentation-governance, changed-Python-policy, and
diff gates are green locally.

This repair does not extend scientific qualification or close #4275/#4267.
Fresh exact-head hosted CI, approval, parent integration, and release remain
unclaimed.

## 2026-08-10 issue #4275 mirrored-frame and seeded-property conformance

Branch `feat/4275-ground-mirrored-property` is published as ready PR #4323
from exact current ready PR #4322 head
`8b065dd299acc7cab39321b0e2d7f34ca64f159b`. It preserves base
`feat/4275-ground-tilted-conformance` and the protected stack; no retarget,
merge, force operation, or parent rewrite occurred. The implementation is
exact commit `08d631d7169019aee9067f3739051a50d88b9554`; initial
evidence/handoff head `74a23c21bb20f13bf608f463915b00d2d53d5a7f` and this
publication follow-up change no implementation evidence.

The shared corpus grows from six to seven cases with the analytically mirrored
incline `n=[0,sqrt(0.99),-0.1]`. The reflection applies the correct distinction
between polar position/velocity vectors and the angular-velocity
pseudovector. All four runtime consumers pass the resulting contact-plane,
no-slip, path, terminal vector, event, and time-limit oracle.

The companion fixed-seed 20-case sweep exercises Python and the installed PyO3
authority with nonzero x components and both signs of z tilt while varying
bounded ball, surface, material, launch, and spin properties. The RED sweep
found that the Python default unbounded planar domain selected world +x even
when it was not tangent. The repair derives a stable intrinsic tangent by
projecting the least-aligned Cartesian axis; explicit finite-domain axes and
bounds are not rewritten.

Local validation passes the complete 247-test Python ground package, four
native Rust corpus tests over seven cases, fresh CPython 3.13 PyO3 corpus and
seeded exact-parity harnesses, freshly rebuilt Node/WASM corpus execution,
pinned MyPy 1.13, Ruff, Prettier, and diff checks. The seven-case raw corpus
SHA-256 is
`c1c363a8ee79b12ab2b7d9c69677e71ab8ab30ba5288c275fff8ddcd4e683465`.

This remains `partial_implementation`. Keep #4275/#4267 open for broader
property-based coverage in every compiled surface, uncertainty and performance
qualification, calibrated/evolving terrain, UI/3D rendering, and downstream
release. Local parity is not hosted CI, protected approval, integration, or
release evidence; PR #4323 remains gated on all of them and on PR #4322.

## 2026-08-10 issue #4275 tilted-plane conformance and passivity

Branch `feat/4275-ground-tilted-conformance` is published as ready PR #4322. It
begins at exact ready PR #4321 head
`7efbf4796c2d0f4e41ce776a60ab4db5cb5dd74e` and preserves base
`feat/4275-ground-conformance-corpus`. Its implementation/evidence publication
head was `a0c8e49a40badc3ce96193e031d2a9dec557d143`; this documentation-only
follow-up changes no implementation evidence. It extends the single shared corpus from
five horizontal cases to six cases with one analytically tractable incline:
`n=[0,sqrt(0.99),0.1]`, initial pure roll, zero rolling resistance, and a
four-second gravity-driven suffix. Whitelisted checks add the center-to-plane
constraint and pin exact event/status semantics, no-slip capture, path, and
terminal position/velocity/spin for Python, native Rust, PyO3, and WASM.

The initial RED run exposed a real fail-closed false positive. Reconstructing
mechanical energy only from repeatedly quantized endpoints produced about
`3.2e-9 J` of apparent creation on an otherwise analytic passive trajectory.
The repair does not widen a global tolerance. Each Python and Rust integration
segment now evaluates and rejects its physical gravity/contact/kinetic balance
before the canonical 11-decimal endpoint snap, preventing prior dissipation
from masking a later defect. Canonical snaps have accumulated fixed-component
error bounds, rolling projection is slip-bounded, and an endpoint outside that
budget fails. The reproducible public endpoint ledger is unchanged. Masking
and unexplained-endpoint regressions now pass in both languages. Final local
GREEN passes 238 Python ground tests; 191/206/203 default/Python/WASM Rust
tests; 19 focused Python conformance/passivity tests; four native corpus tests
over all six cases; a fresh installed CPython 3.13 PyO3 wheel; and rebuilt
Node/WASM. Strict lint, type, format, policy, and documentation gates pass, and
independent adversarial review is `READY`.

The reviewed implementation is exact commit
`5d333a4448d6484f8c98e78c9878cb83b40aa522`; the raw six-case corpus SHA-256 is
`502dae7cacb346e55a0624b5758efce1baf123065a45571cd3aaf2ee0045bb76`.
This is immutable local evidence. PR #4322 was initially open, ready, and
mergeable, with protected jobs queued/in progress and no review decision.
Green hosted checks, approval, integration, and release are not claimed.

The broader runtime matrix then exposed a separate resistance-cusp defect on a
translating incline. A frozen resistance direction could cross through zero
relative speed and create energy. Python and Rust now bound non-collinear
closing roll steps; when resistance can balance slope drive, zero relative
speed is held while the plane carries the ball. This does not emit an absolute
rest event, and contact-force work remains explicit. A sub-tolerance residual
is projected to exact co-motion through the existing bounded slip, velocity,
spin, and energy checks before the hold. Dedicated Python and Rust tilted/
moving regressions cover the repair. The independent slip tolerance gates
pre-projection contact slip; the velocity tolerance and its radius-scaled
angular equivalent gate the holding correction. A stationary projected stop
returns `REST` in the same solver step, with one zero-motion interval used only
at the handoff boundary to satisfy the strict increasing-time wire contract.

Await ordinary protected CI/review and parent-stack integration. This is still
`partial_implementation`;
#4275/#4267 remain open for mirrored and
randomized tilted frames, broader properties, performance, calibration,
terrain/material evolution, deformation, interfaces, visualization, and
downstream release.

## 2026-08-10 issue #4275 scientific conformance corpus

Branch `feat/4275-ground-conformance-corpus` is a normal child of exact ready
PR #4320 head `64506a54d546021f3c16fbe0b627f35057ec6dd1`; preserve PR base
`feat/4275-ground-compiled-reference-runtime`. It adds a single versioned
`ground-reference-conformance/v1` authority artifact plus small consumer
harnesses for Python, direct Rust, a real installed PyO3 wheel, and rebuilt
Node/WASM. Production physics is unchanged. The five cases independently pin
linear contact localization, Newton restitution, passive stationary impact,
the solid-sphere Coulomb skid-to-roll limit, constant rolling-resistance stop,
proper active -90-degree rotation about +y, and moving-surface relative-motion invariance.
Every numeric oracle carries a unit, derivation narrative, and an applicable
bounded tolerance. The established full-result golden remains byte-identity
evidence; the new corpus intentionally tests scientific observables instead of
copying complete implementation output.

Focused RED/GREEN evidence passes eight Python corpus tests, four direct native
Rust tests, a unique CPython 3.13 wheel install/run, and a newly built
WASM release package/run. The implementation commit cannot self-name its final
SHA. Before publishing, create a documentation/evidence child that binds the
exact implementation parent, raw corpus SHA-256, complete test matrices,
independent review, PR number/head/base, and protected-CI state in the strict
manifest and all handoffs.

The independently reviewed implementation is exact commit
`9df3928a1ef32d81db2e568884ca24d8c576d49a`; corpus SHA-256 is
`f7fda73e45c5c64951a9934ba126cd9edbde7f7f85843a69612f86b8ec518310`.
Final local gates pass 227 Python ground tests, 184/199/196
default/Python/WASM Rust tests, eight focused Python and four native corpus
tests, a real installed CPython 3.13 PyO3 wheel, rebuilt Node/WASM, strict
Clippy, MyPy, Ruff, formatting, manifest plus eight tests, docs governance,
structural budgets, and independent READY review. This is immutable local
evidence only; no carrier PR, hosted/protected result, approval, integration,
or release is claimed.

This remains `partial_implementation`. #4275/#4267 still require tilted-frame
and property breadth, ensemble/determinism/performance qualification,
asynchronous WASM cancellation, calibration/uncertainty, changing terrain and
materials, deformation/torsional damping/roll-to-skid, matched clients and 3D
rendering, downstream exact-pin integration, and ordinary protected release.

## 2026-08-10 issue #4275 compiled ground-reference runtime

Implementation commit `50682f251d5e9c0424ba633d1ce5be7fa1379a3c` on
`feat/4275-ground-compiled-reference-runtime` begins at exact PR #4312 head
`e3f1d7dd7eecaecfed1253b7fe72577c9ed6989d` and is intended to
target `feat/4275-ground-result-wire-parity`. The bounded continuation ports
the canonical rigid-sphere reference execution into `tools-core`: interpolated
sphere-plane contact, passive restitution/Coulomb impulse, repeated ballistic
bounce and capture, frozen-direction Coulomb skid, exact skid-to-roll
transition, pure roll, rolling resistance, and qualified rest. The native,
PyO3, and WASM paths use one strict execution contract and one canonical result
boundary, including typed phase/reason/request-fingerprint errors and bounded
cancellation. Unsupported resolvers and serialized callbacks fail closed.

The v1 execution schema is unchanged. Independent preflight budgets cap
scheduled endpoint-inclusive output at 200,001 points, declared surface-loop
work at 1,000,000 steps, events at 10,000, and the complete trajectory at
210,003 points including unscheduled phase/event/terminal evidence. Output
density and integration work are not compared. Excess declarations fail before
callbacks, allocation, or physics with resource-specific reasons; an admitted
small `max_steps` still reaches the existing runtime `step_limit`. Dynamic
point/event guards and per-sample cancellation checks preserve those bounds
through execution.

Independent review identified three defects before publication. First, adding
a small interval to a large valid absolute time could produce the same `f64`
and trap a catch-up loop. One bounded integer schedule now operates on elapsed
time across bounce and surface phases; absolute time is applied only to emitted
wire evidence. Second, direct native calls now normalize the typed request
exactly once and reuse that authority for fingerprint, preflight, and physics;
the JSON boundary reuses its normalized parser result. Third, PyO3 releases the
GIL across the physics run and reacquires it only at cancellation polls without
weakening exception or boolean-result handling. RED/GREEN tests cover a large-
epoch bouncing case, a large-epoch immediate-capture surface case, a
sub-canonical direct mutation, and real-wheel cancellation from a second
Python thread.

A fourth independent-review defect was then reproduced at an absolute contact
epoch of `9e15 s`: elapsed integration remained bounded, but sub-ULP output
intervals collapsed distinct impact, surface, and termination evidence to one
wire timestamp. The runtime now preflights the endpoint-inclusive requested
grid's first and terminal-adjacent canonical projections before callbacks or
physics. Non-advancing grids fail with typed Bounce `time_resolution` rather
than a late composition failure or a malformed successful result. Bounce and
surface append guards additionally reject any unexpected positive elapsed
advance that maps at or before the prior wire timestamp; intentional
same-elapsed phase transitions remain replaceable. RED/GREEN tests cover both
bounce and immediate-capture failure paths at `9e15 s` and monotonic successful
execution for both at a representable large epoch.
An additional callback-zero regression proves that an individually valid epoch
and duration whose sum exceeds the canonical safe-number range returns the
same typed failure instead of panicking.

A final review pass removed infallible canonicalization of derived physics
and evidence. Unsafe derived states, timestamps, events, ledgers, summaries,
or final recursively inspected JSON numbers return typed owning-phase
`NumericalFailure`/`numeric_range` across native, PyO3, and WASM. Immediate
capture with `max_events=1` returns a coherent `Partial`/`EventLimit` result at
the unchanged terminal state, while rebound remains a typed Bounce
`event_limit` failure. Exact overflow payloads and monotonic `1e12 s`
bounce/capture success are pinned on all three execution surfaces.

The established full-pipeline golden result is byte-identical at SHA-256
`23f567f125ec9631e2a7638dfa217b78891883fc4e5092bea3b1f21fb063e8af`.
Twenty seeded moving-surface/material cases plus an immediate-capture edge
case reproduce Python exactly over the common resolver-free horizontal-plane
scope. A separate native test proves
the compiled static-plane implementation accepts a tilted plane; the Python
default domain cannot form that same resolver-free case because it fixes its
tangent axis and origin. Complete default/Python/WASM `tools-core` suites pass
180/195/192 tests, all 219 Python ground tests pass, and fresh CPython 3.13 and
Node/WASM builds pass golden, default-control, 100-run determinism,
cancellation, callback-exception, typed wire-resolution, numeric-range,
resource-cap, event-limit, and representability checks. Formatting and strict default
Clippy pass; feature all-target linting passes with eight explicit inherited
unrelated allowances. New production modules are all below 400 lines, and the
principal runtime test is exactly 500 lines. Eight manifest tests and docs
governance pass. The strict campaign authority binds local evidence to exact
implementation commit `50682f251d5e9c0424ba633d1ce5be7fa1379a3c`
through the existing `commit_sha` contract; no dirty-tree evidence type was
added. Independent final review is READY. No hosted check, durable benchmark
artifact, or performance-budget pass is claimed.

The fresh `wasm-pack` release build succeeds but reports a packaging notice:
the nested crate directory has no local license file although the repository
root tracks `LICENSE`. This work verifies the generated Node runtime and makes
no npm-publication claim; the package metadata must be resolved before such a
distribution.

No push, PR, protected-CI, protected GitHub review, or merge is claimed here. The
runtime is deliberately restricted to one immutable planar profile, standard
gravity, and the v1 model identities. Changing normals/material regions,
terrain deformation, torsional damping, roll-to-skid, production calibration,
ensembles, UI, UpstreamDrift consumers, and asynchronous WASM cancellation
remain excluded. Keep #4275 and #4267 open pending independent review,
protected gates, normal stack integration, and downstream parity.

## 2026-08-10 PR #4312 corrected-reference propagation

Draft PR #4312 remains on `feat/4275-ground-result-wire-parity`, targeting the
unchanged `feat/4275-ground-reference-execution` base. Exact corrected #4309
parent `f4ca3f801f60c1c3042d4ed1a6100fdd7cfebd4b` is incorporated by the normal
two-parent merge containing this handoff. The child retains strict typed Rust
`flight-to-ground-result/v1` parsing, raw and normalized semantic validation,
recursive duplicate-key and unsafe-number rejection, trajectory/event/status/
summary/geometry coherence, deterministic canonical JSON, lowercase digest
emission, and real PyO3/WASM validation boundaries while acquiring the
corrected reference-execution and scalar-study ancestry. No branch was
rebased, retargeted, rewritten, or force-pushed.

This is not compiled ground-solver or epic completion. The bindings validate
evidence but do not run bounce/skid/roll/rest physics. UI, ensembles,
production calibration, changing terrain/material regions, UpstreamDrift
consumers, and four-surface parity remain open. Keep #4275 and #4267 open;
protected CI, independent review, dependency-order collapse, and consumer
delivery remain required.

Merged-tree validation is `238` focused ground/scalar tests on CPython 3.11.9
and real CPython 3.10.20; the broader Rate of Closure/swing/flight/ground/
import-alias selection reports `1,404` passed and seven documented optional-
Rust-wheel skips. Complete default/Python/WASM `tools-core` suites pass
`144`/`159`/`156` tests, including seven focused result-wire tests. Cargo
formatting, strict default/focused Clippy, and feature Clippy with eight
enumerated inherited unrelated allowances pass. Fresh CPython 3.13 and
Node-targeted WASM artifacts prove uppercase-to-lowercase digest emission and
malformed rejection. The relevant TypeScript transfer/contract suite passes
`20` tests; the 189-module Vite build and zero-warning ESLint pass. Pinned MyPy
1.13 passes 51 production modules. Manifest validation plus eight contracts,
documentation governance, module and protected changed-only file budgets,
14-file scoped marker scan, and diff checks pass. Hosted evidence must be
re-established on the new exact merge head.

## 2026-08-10 issue #4275 uppercase result-digest parity repair

Independent review marked local implementation
`b802f041e1a348e365b98e77f969961b8cd11133` not ready because Rust rejected
uppercase `provenance.input_sha256` text that the Python and TypeScript
contracts accept and canonicalize to lowercase. The repair admits exactly 64
ASCII hexadecimal characters in either case during raw semantic validation,
lowercases the digest during result normalization, and revalidates before
canonical emission. Wrong-length and non-hex values remain rejected.

The repaired focused result-wire suite has 7 tests; complete `tools-core`
counts are 144 default, 159 Python-feature, and 156 WASM-feature tests. Direct
binding regressions and freshly rebuilt real CPython 3.13/PyO3 and wasm-pack
Node artifacts prove uppercase input emits lowercase and malformed evidence is
still rejected. Cargo formatting and focused/Python/WASM Clippy gates pass.
This review repair changes only result-wire case normalization and does not add
compiled ground physics or alter the open #4275/#4267 delivery boundaries.

## 2026-08-10 issue #4275 Rust result-wire validation parity

The local `feat/4275-ground-result-wire-parity` branch is based exactly on PR
#4309 carrier `51492c3ddc8b15b1358434da9b29f600261c918a`. Its bounded
`tools-core` continuation implements the exact typed
`flight-to-ground-result/v1` wire record, semantic state-machine validation,
and deterministic canonical JSON. The boundary rejects unknown or duplicate
keys, unsafe integers, invalid raw numeric/text/hash evidence, trajectory and
event ordering errors, summary/geometry drift, and contradictory
status/termination/payload combinations. Validation runs before and after
canonical number normalization, preventing an invalid raw value from being
rounded into the accepted domain.

The same validation/canonicalization contract is exported through PyO3 as
`validate_flight_to_ground_result_v1` and through wasm-bindgen as
`validateFlightToGroundResultV1`. A real CPython 3.13 wheel and a real
wasm-pack Node artifact both preserve the full shared
`ground_reference_pipeline_golden_v1.json` result and its canonical SHA-256.
Local evidence is 6 focused adversarial result-wire tests, 143 default
`tools-core` tests, 157 Python-feature tests, 154 WASM-feature tests, 219
Python ground tests, and 19 TypeScript ground-contract/transfer tests. Cargo
formatting, focused strict Clippy, feature all-target Clippy with explicit
inherited allowances, docs governance, source-size, and diff checks pass; each
new production module is at most 258 lines.

This is result-wire parity, not compiled ground-solver parity: the bindings do
not execute bounce/skid/roll physics. PyQt6/React workflows, ensembles,
production calibration, UpstreamDrift consumers, protected CI, review,
publication, integration, and closure of #4275/#4267 remain outstanding.
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
The inherited ground descendant passes 1,703 Python tests with two optional `build123d`
skips, 643 React tests across 105 files plus type-check/lint/build, 12 Rust tests,
and 77 real-CPython-3.10 ground and compatibility tests. Ruff/format pass 78
changed Python files; pinned mypy and Bandit pass 52 changed production files.
Manifest, docs, minimum-test, assertions, 500-LOC, changed-file secrets, Python
3.10 compilation, and diff checks are clean. Hosted CI and review still apply
only to the new exact head. That repaired ground head is incorporated into
#4288, and exact #4288 is incorporated into #4298 by the normal merge
containing this handoff. Current #4298 CI and review are now the next ancestry
gates.

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
This is ancestry reconciliation, not new landing physics. Qualified bounce,
skid, roll, terrain profiles, total distance, UI execution, Rust/WASM parity,
protected CI, review, and release remain open. The exact repaired #4285 head is
incorporated into #4288, and that corrected descendant is incorporated into
#4298 by the normal merge containing this handoff.

## 2026-08-10 PR #4298 exact repaired flight-ground propagation

Draft PR #4298 keeps base `feat/4199-wind-workflow`. The normal merge containing
this handoff keeps original camera child
`9ffd8d280c77977a41e93bd0caef9678d1c231b6` first and incorporates exact
repaired #4288 head `108a841b1378c992defd3c7b7ee263d41a6c8b24`
second. Exact #4288 contains repaired #4285
`e5bcbd1096d3be1f621a805c9d9f3fd321e375a5` and repaired #4282
`686016196a2f895058b8a566dff103a0fd32cd10`. Neither branch was rebased,
retargeted, rewritten, or force-pushed. The camera, tracking, and playback
behavior is preserved while the child inherits the qualified flight-to-ground
transfer plus deterministic workspace, variation, scalar-wind, capability,
campaign-release, hosted-mypy, and import-identity corrections. The only code
conflict was the controls import seam; semantic resolution retains both the
child `CameraCommandId` and parent `ImpactLayerControls` dependencies.

SPEC 1.14.22 records this merge monotonically above repaired parent 1.14.21.
The exact composed tree passes 1,738 Python tests with two explicit optional
`build123d` skips, including the installed `tools_core` flight parity path;
110 React files / 670 tests; all 137 `tools-core` Rust tests; and four
Playwright camera/playback cases across desktop and constrained 2x-DPR
Chromium. TypeScript, zero-warning ESLint, the 194-module production build,
Ruff check/format across 61 changed Python files, pinned mypy 1.13 and Bandit
across 43 changed production files, warning-denied `tools-core` clippy, Rust
format, campaign-manifest validation, documentation governance, module and
500-LOC budgets, conflict-marker checks, and staged/working diff checks are
clean. The focused control seam passes 12 PyQt camera and impact-layer tests.
Protected current-head CI, review, native rendered review, camera-state
persistence, UpstreamDrift parity, and protected release remain separate gates.

## 2026-08-09 Camera snap/tracking continuation

Draft PR #4298 publishes branch `feat/4284-camera-snap-tracking` with tested
camera evidence through immutable commit
`2095e748ddca2d7036bbd49a731528f5634daff9`. The current local propagation
normally merges exact #4282 head
`5f77af4add23547a21cc3fabce98ae9ad4260427` into exact camera parent
`42753a576f42d4c43c35fd786d0748e1d03672c5`; PR #4298 keeps base
`feat/4199-wind-workflow`. No branch was rebased, retargeted, force-pushed, or
published by this propagation. The two-parent merge itself contains this record, so
its future SHA is intentionally not self-recorded. It implements the canonical
camera contract for all five Tools 3D adapters (PyQt Simulation and Flight;
React Club, Impact, and Flight) with exact snap orientations, bounded opt-in
subject tracking, safe zoom preservation, Auto Fit, manual suspension, and
Recenter. UpstreamDrift parity is not started, and protected release remains
open.
Evidence commit `2095e748` adds solver-owned previous/next frame controls to
React ball-flight playback through the existing validated timeline boundary.
It passes 39 focused Python/PyQt camera tests and the full 107-file / 650-test
React suite. Playwright passes a bounded
play/pause/restart/loop/speed/frame-step/zoom/snap/tracking matrix and
responsive backing-store assertions in desktop Chromium and a 520 x 900,
2x-DPR Chromium project (4 browser tests total). TypeScript, zero-warning
ESLint, the 193-module Vite build, Ruff, targeted mypy, campaign validation,
and diff checks also pass. Native-font/manual visual review, hosted CI/review,
protected release, preference persistence, and UpstreamDrift parity remain due.

The prior documentation-only successor records the already-published camera
evidence commit. The campaign manifest now names carrier
`evidence_commit_sha` rather than pretending a commit can contain its own
future PR-head SHA. Legacy `head_sha` input remains migration-compatible,
while new schema output uses the truthful field name. This local restack
records its exact two parents without attempting to self-record its merge SHA.

The composed local merge candidate passes 65 focused camera, PyQt6,
compatibility, and campaign-manifest tests on Python 3.13; all 15 compatibility
contracts also pass on real CPython 3.10.20. The complete React suite remains
107 files / 650 tests, all four Playwright desktop/constrained-DPR cases pass,
and TypeScript, zero-warning ESLint, and the 193-module production build pass.
Canonical Ruff check/format passes all 28 changed Python files; pinned Python
3.12/mypy 1.13 passes 20 changed production modules; manifest/schema,
documentation-governance, and staged/working-tree diff checks pass.

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

SPEC 1.14.12 records this propagation monotonically above the corrected
parent's 1.14.11, 1.14.10, and 1.14.9 entries. Protected CI, review,
publication, installed-package evidence, downstream UpstreamDrift parity, and
the remaining scientific/accessibility/performance release gates remain
separate. This local merge is not a protected release.

Focused evidence is 89 ground-contract, compatibility, scalar-adapter, and
responsive-wind tests on Python 3.11 plus the same 89 on real CPython 3.10.20.
Ruff check/format passes 34 focused Python files, pinned mypy 1.13 passes 23
production modules, and ground modules/functions remain within 400/50-line
budgets without placeholders. The inherited campaign manifest validates and
all nine manifest/parity contracts pass. Documentation governance, ancestry,
SPEC-order, and final diff assertions are required in the same local merge.

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

## 2026-08-10 PR #4280 workspace timestamp propagation

Exact parent `05383d333b6fd87eaf5e37305476f50b505c2c2e` is incorporated into
`feat/4144-variation-export-continuation` through the normal merge containing
this handoff. PR #4280 keeps base `feat/4218-toolstrip-workspace`; neither
branch was rebased, retargeted, force-pushed, or rewritten. The reconciled tree
retains all variation export/accessibility behavior and adds the strict
cross-version workspace timestamp parser.

SPEC 1.14.10 remains the parent compatibility entry and SPEC 1.14.11 becomes
the child variation entry. The reconciled tree passes `778` Rate tests, `27`
real-Python-3.10.20 compatibility tests, `1 file / 8` focused React tests,
TypeScript, focused zero-warning ESLint, Ruff, format, and pinned mypy 1.13.
Documentation, size, and diff gates must remain clean in the merge commit.
Protected CI, review, and later propagation remain open.

## 2026-08-09 PR #4280 variation-export propagation

Draft #4280 remains based on `feat/4218-toolstrip-workspace` and includes exact
corrected parent `3f67ed466fefc8991db9c4409f921f25e1c37142` through a normal
merge. The child retains complete selected-scatter-axis CSV export parity,
typed unavailable rows, PyQt accessible raw tables, and the focused
table/scatter/matrix split.

## 2026-08-09 Capability results stabilization

The isolated `feat/4201-capability-results-diagnostics` continuation is based
on exact wind-workflow carrier head `18fe89201`. It closes two audit findings
without changing the optimizer or physics: both clients expose every ranked
diagnostic and parameter unit, both provide result CSV plus versioned result
JSON, and React scalar scatters now carry numeric scales and cohort legends.
Raw observation exports remain lossless and distinct. This is feature-stack
implementation evidence only; #4197/#4201 remain open through integration,
hosted CI, review, downstream parity, and release to `main`.

Local evidence: 813 Rate Python/PyQt tests and 104 React files / 628 tests,
Ruff check/format, targeted mypy for the new export and tab modules, TypeScript,
zero-warning ESLint, and the 188-module Vite production build passed. The
pre-existing mypy 1.19.1/Python 3.13 internal serialization assertion on
`capability_results.py` reproduces unchanged at carrier `18fe89201`.

## 2026-08-09 Web-release stability continuation

Local branch `feat/4201-web-release-stability` is based exactly on campaign
carrier `18fe89201`. It fixes direct-path execution of
`src/rate_of_closure/launch_web.py`, adds a subprocess delegation smoke
contract, and reconciles the supported release boundary to the files that
exist: static Vite web output plus PyQt6/PyInstaller desktop output. Rate has
no `src-tauri` project, so its stale Tauri scripts, CLI dependency, lockfile
entries, and current documentation claims were removed instead of inventing an
unqualified wrapper.

The load-sensitive `FlightExplorerPanel` assertion now explicitly settles the
real lazy Wind Strategy import inside React `act`; it does not extend a timeout
or replace the child component with a mock. This is test synchronization only,
not a UI or physics change. The branch must remain local until reviewed and
folded into the established campaign carrier; no PR, push, or merge belongs to
this handoff slice.

Exact local evidence: 813 Rate Python/PyQt tests passed with 15 existing
warnings; 9 launcher/registration tests passed; 102 React files / 624 tests
passed; and five sequential focused Flight Explorer runs passed 25/25 tests.
TypeScript type-check, zero-warning ESLint, the 187-module static build,
focused Ruff check/format, targeted mypy, package installation/audit, and diff
checks pass. `tests/test_dry_compliance.py` retains two unrelated baseline
failures for the Movement Optimizer and Optimizer GUI PyQt launchers.

## 2026-08-09 Machine-readable campaign authority

`docs/release/rate_of_closure_campaign.v1.json` is now the canonical current
state for the primary Rate, impact, flight, variation, club-builder, wedge,
toolstrip, design-quality, parity, and ground programs. Its normalized carrier
and evidence tables replace status inference from the chronological entries
below. `scripts/rate_campaign_manifest.py` supplies strict Pydantic validation
and a generated JSON Schema; it rejects missing programs, unresolved evidence,
placeholders, malformed SHAs, and contradictory release claims.

The authority records the current campaign as **not released**. Capability PRs
#4294, #4289, and #4283 are merged into feature parent #4282, but those were not
protected `main` merges. Current carrier head `18fe89201` contains the hermetic
`swing_core` parity-lane correction and its focused workflow regression; a
fresh hosted parity run, top-down stack propagation, #4133 reconciliation,
#4119 current-main conflict resolution, protected checks, installed-package
evidence, and UpstreamDrift parity are still required.

Maintain the JSON authority and the current-state handoffs in the same commit
whenever a carrier SHA, test result, limitation, supported surface, or release
stage changes. Historical detail below remains useful provenance but must not
override a contradictory validated manifest record.

The four reviewed stabilization slices are composed on implementation head
`2c1a77baa`: strict workflow parsing/signed editing, complete capability
diagnostics and result exports, static-web entrypoint/package truth, and this
manifest authority. Combined local evidence is 828 Rate Python/PyQt tests and
104 React files / 642 tests, TypeScript, zero-warning ESLint, the 188-module
Vite build, Ruff, targeted mypy, deterministic manifest/schema validation, and
nine manifest/parity contracts. This remains local evidence; #4282 still needs
a normal push, exact-head hosted CI, review, and dependency-ordered release.

The first exact-head hosted quality gate (`31340032608`) passed checkout,
dependency installation, Ruff, and formatting, then failed mypy 1.13 because
`--follow-imports=skip` exposed Pydantic and Qt boundary returns as `Any`. The
loader now casts the validated model and the playback adapter converts elapsed
milliseconds to a concrete `float`; these are static typing fixes, not schema
or runtime behavior changes. The exact Python 3.12/mypy 1.13 command passes all
54 delta files; Ruff, 62 focused regression tests, and eight campaign-manifest
tests pass locally.

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
## 2026-08-11 #4261 calculation-runtime manifest contract slice

Draft PR #4344 publishes the next four-surface authority slice on
`feat/4261-runtime-manifest-contract` at exact head
`c06cefb5c541fe7d87b54cd36269917d92d837a6`, from exact remote
`fix/rate-mobile-tools-menu` head
`c653f9ff9193d6cdb8e11a13ad0001707e468a42`. The child incorporated the
parent's advance from `16a1167c...` by an ordinary conflict-free merge and
retains the same base branch. It defines the strict immutable
`calculation-runtime-manifest/v1` contract in Python and TypeScript and shares a
canonical parity fixture. The record names the exact product surface, package,
build, Tools SHA, and complete impact/flight/ground status ledger. Available
domains require model/version, implementation authority, backend, integrator,
request/result schemas, frame, units, and unit-explicit finite options;
unavailable domains require a reason and reject all identities/options.

The contract rejects unknown fields and schema/surface values, non-SHA builds,
placeholder evidence, domain/status contradictions, duplicate option/evidence
IDs, non-finite values, unsafe integers, invalid unit semantics, unpaired
surrogate text, and duplicate JSON fields. Python and TypeScript emit the same stable
11-decimal canonical bytes and expose pure factories that accept explicit
evidence rather than reading ambient Git or installed-package state.

Adding the nested active specification expanded the deterministic declared
scope from 18 to 19 specifications. Its four-surface cells remain explicitly
unsupported because the protected inventory pin predates this unpublished
child and no live surface attachment or consumer conformance evidence exists.

This is deliberately not wired into live simulation results, workspaces,
exports, regional ground execution, or UpstreamDrift. It does not close #4261
or #4260. Future attachment and provider-resolution work must be independent,
dependency-ordered slices after active assembly and workspace edits settle.

### 2026-08-11 independent-review hardening

The contract follow-up after `15b951e40` repairs the independent-review
blockers without expanding delivery scope. Python and TypeScript manifest
options share the safe-magnitude domain, while the general Python canonical
encoder retains its established large-float behavior for capability-observation
exports. Both runtimes enforce SemVer 2 numeric identifiers and one
deterministic substantive-reason grammar: the same explicit Unicode White_Space
boundary set, Unicode-scalar length, ASCII explanatory-word threshold, and
normalized sentinel rejection. TypeScript now distinguishes valid UTF-16 pairs
from unpaired surrogates, matching Python and preserving valid non-BMP text.
RED-first shared-fixture tests cover `1e16`, `1e20`, both safe boundaries,
leading-zero versions, astral text, all declared boundary whitespace, unpaired
surrogates, `x`, `n/a`, and bare/whitespace `unavailable`; capability tests
restore the existing `1e21` wire behavior. The declared-scope prose accurately
includes recursive `docs/specs/**/*.md`. This remains a draft contract child
with all four spec cells unsupported at the older protected inventory pin.

### 2026-08-11 stable-ID placeholder boundary repair

Independent publish review found that regex word boundaries treated `_` as a
word character, allowing placeholder tokens such as `todo_build` to pass even
though `_` is a valid stable-ID separator. Python and TypeScript now recognize
placeholders as complete ASCII-alphanumeric-delimited tokens. The shared parity
fixture exercises every token next to `.`, `_`, `/`, and `-` in both directions
and preserves legitimate longer substrings such as `todolist`. The same bounded
repair removes the pinned-MyPy redundant reason cast. Draft PR #4344 remains a
contract-only child and does not expand the #4261 completion claim.
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
