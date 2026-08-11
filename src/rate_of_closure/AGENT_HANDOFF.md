# AGENT_HANDOFF — rate_of_closure

> Update with every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-11.

## 2026-08-11 Explorer-session v4 variation-study mapper

The child `feat/4144-workspace-variation-study` is based on exact published PR
#4340 head `26105f668de260d75a99f450726348570db7ff89`. Its Python and
TypeScript contracts persist one canonical authored variation plan plus a
strict selection payload for all-together/individual/both execution and a
non-empty, mode-valid output focus. The canonical plan already owns inputs,
distributions, ranges, groups, trial count, seed, and flight model.

Both clients map actual live controls. The simulation setup remains the sole
ball-support authority; workspace plans cannot duplicate it, and Tee Height is
rejected under Ground support. Legacy v1-v3 sessions require an explicit,
nonconflicting variation fallback. Full parsing precedes mutation and native
application rolls the supported workspace back on failure.

Independent review now guarantees that asynchronous browser Open applies only
after rechecking the latest dirty state and parsing against the latest legacy
fallbacks. React's selectable swing outputs derive directly from the complete
17-field swing executor contract, closing omissions for spin loft,
face-to-path, and spin-axis tilt. Both clients explicitly call the selector a
saved output focus; it never filters or truncates canonical run/export data.

This does not persist results, identity, optimizer outputs, or flight-run
results and does not close #4142/#4144/#4218. UpstreamDrift consumers,
protected CI/review, and ordered release remain open. Post-review qualification
passes 20 focused Python workspace/PyQt tests, 43 focused React
workspace/variation tests, pinned MyPy 1.13, Ruff check/format, TypeScript,
zero-warning ESLint, the 210-module production build, the 11-test
campaign-manifest suite, changed-file and module-size budgets, docs,
manifest-layout, changed-Python, JSON, and diff gates. A broader PyQt pair was
stopped without failure output after two 120-second workstation-contention
timeouts; the three directly affected native workflows passed serially.

## 2026-08-10 Explorer-session v3 torque-profile mapper

The child `feat/4136-workspace-torque-profiles` is based on exact published PR
#4336 head `6e9dd85a3c5f43d37cf8a704db0555bdad734e7e`. PyQt6 and React now
persist their live canonical torque-profile library, active stable profile ID,
passive/prescribed run selection, canonical joint locks, and profile-source
provenance through explorer-session v3 and the existing workspace root library.

The adapter delegates coefficient, SI-unit, coefficient-order, identity,
source, timestamp, time-domain, and fit-evidence validation to the existing
shared profile contracts. Selection membership and provenance are checked
before UI mutation. Legacy v1/v2 sessions require an explicit current torque
fallback and reject a conflicting embedded library instead of guessing.

This does not close #4136/#4220/#4218. Optimizer and other run payloads,
UpstreamDrift consumers, protected CI/review, and ordered release remain open.
Local qualification passes 34 focused Python tests and 43 focused React tests,
pinned MyPy 1.13, Ruff check/format, TypeScript, zero-warning ESLint, the
206-module production build, the 11-test campaign-manifest suite, module-size,
docs, manifest-layout, changed-Python, changed-test assertion, JSON, and diff
gates. The exact local child commit is recorded after commit creation.

## 2026-08-10 Explorer-session v2 ball/target mapper

The child `feat/4225-ball-target-session-mappers` is based on exact draft PR
#4333 head `bd7da1e6d42557d5e8782b8f4f64fc4ed183e5ce`. The Python and
TypeScript File adapters now embed one strict versioned simulation subpayload.
It round-trips Ground/Tee support, SI tee height, the derived ball-centre
invariant, club-default/explicit-override provenance, and the canonical spatial
target's label, kind, app/source frame, position, elevation/ground source, and
complete tolerance geometry.

PyQt6 captures/applies the actual Simulation tab controls; React lifts ball
setup/provenance into app-owned state beside its already shared spatial target.
Both parse the whole document before touching live state. Club-default claims
must match the saved club and geometry. Explorer-session v1 requires an
explicit current-state fallback, preserves those values, and reclassifies a
cross-club default as an explicit override rather than inventing a value.

This does not close #4143/#4225/#4218. Torque/optimizer/run payloads,
UpstreamDrift consumers, protected CI/review, and release remain open.
Post-refactor evidence is 8 focused native tests and 27 focused React tests,
pinned MyPy 1.13, Ruff check/format, TypeScript, zero-warning ESLint, the
203-module production build, the 11-test campaign-manifest suite, module-size,
docs, manifest-layout, changed-Python, and diff gates. The broader suites were
stopped without failure output after several minutes on the overloaded
workstation and are not claimed as validation.

## 2026-08-10 Production File-command adapters

Branch `feat/4225-workspace-file-adapters` is isolated from exact draft PR
#4330 head `d8176bb5863a35725199bb8357a5f000f9bdd3ba`. The existing strict
`rate_of_closure.workspace/2` envelope and `rate_of_closure.view_workspace/1`
document now drive production PyQt6 and React File operations. New/Open/Save
As/view Import/view Export/Close are available on both; native also owns atomic
Save and persisted Recent. Browser Save/Recent expose platform-specific disabled
reasons.

The live whole-session mapper covers the impact scenario, club, units, primary
module presentation, and compositor. Complete validation precedes mutation;
dirty New/Open/Close are protected, failed native application restores the
prior supported state, and unsupported torque-profile/variation-plan payloads
are rejected. Native writes replace atomically and Recent changes only after a
successful save/open.

Do not infer full workspace coverage: ball setup, target, torque editor,
optimizer, variation runs, flight runs, and other simulation-tab-local state
still require strict domain adapters. Installed-consumer parity, protected CI,
review, dependency integration, and #4218/#4225 completion remain open.

Local qualification passes 921 Rate-of-Closure Python tests, focused MyPy,
Ruff, and Black, React TypeScript and zero-warning ESLint, all 116 Vitest files
/ 693 tests, the 201-module Vite build, and baseline-aware module-size, docs,
linked-debt, changed-Python policy, changed-test assertion, and diff checks.
The legacy full-tree 500-LOC script loads no grandfather baseline in this
checkout and therefore reports 232 pre-existing files; all new modules in this
slice are below 500 lines.

## 2026-08-10 Repaired compositor-parent propagation into persistence child

Continuation branch `feat/4225-multiview-persistence` now normally incorporates
exact repaired compositor parent `0e3054e6a7fa0e3e38e1312b4132bbd1e4336fb2`.
Keyboard/persistence production and test code did not conflict; only the four
additive handoff/spec files required reconciliation. No rebase, retarget,
force-push, parent rewrite, or history rewrite was used. Fresh local
verification and protected exact-head CI remain required. The pinned-MyPy
delta additionally requires an explicit typed current-workspace validation
local; runtime parsing, validation, migration, and serialization are unchanged.

## 2026-08-10 Repaired legend-parent propagation into PR #4327

Draft PR `#4327` remains on `feat/4225-multiview-compositor`, based on
`fix/4224-default-legend-layout-local`, and now normally incorporates exact
repaired legend parent `531a851dc125e83ad86abe1601651e163f5f866d`.
Multi-view production/test code did not conflict; only the four additive
handoff/spec files required reconciliation. No rebase, retarget, force-push,
parent rewrite, or history rewrite was used. Fresh verification and protected
CI remain.

## 2026-08-10 Issue #4225 multi-view keyboard/export acceptance slice

The continuation `feat/4225-multiview-persistence` branch is based on exact
draft PR #4327 head `e975f66bdcfc5a32f9688b8c2c6e34fe1b53ce6e`. PyQt6 owns persistent,
distinct `StrikeView`, synchronized `SimulationView`, and `FlightView`
instances in a real compositor tab; enabled View commands select those hosts.
React provides matching Impact/Swing/Flight hosts, quick-view tabs, visibility
toggles, responsive layouts, the canonical target editor in Flight, and direct
toolstrip routing. Visible hosts share run identity and time (flight receives
impact-relative time), while camera and overlay ownership remains local to
each host. The established Strike, Swing, Kinetics, and Flight displays remain
reachable beside Multi View; direct toolstrip commands return to Multi View
and select the requested real host.

The shared `rate_of_closure.view_workspace/1` wire shape is honored on both
clients. Persistence safely migrates legacy `views` documents and drops unknown
future IDs with deterministic active-view and strict cardinality fallback:
one host is Single, two hosts use a valid split, and three hosts are Grid.
Valid legend placement survives recovery and transitions. Playback persistence
comes from the real transport, stores settled time plus play/loop/rate, and is
debounced in PyQt6 so animation frames do not become settings writes. New
native controls have hover guidance, and a resizable scroll viewport keeps
minimum-size plots reachable in constrained grids.

Keyboard behavior is now explicit and regression-pinned: React quick-view tabs
implement the standard roving-tab pattern with Arrow Left/Right, Home, and End;
Qt exposes a deterministic Layout -> Impact -> Swing -> Flight focus chain.
Tests create and reduce layouts using focus and keyboard activation alone.
Strict version-1 export/import functions exist on both client boundaries,
validate fully before applying state, reject future formats without partial
mutation, preserve playback and legend state, and persist a native import
through QSettings recreation. The whole-workspace Python envelope now
round-trips the canonical view document rather than placeholder data.

That realistic nested slot list exposed and fixed a pre-existing
`VersionedPayload.from_json_dict` double-freeze defect. Exact local gates are
921 Python/PyQt Rate tests, 114 React files / 686 tests, focused MyPy,
Ruff/format, TypeScript, zero-warning ESLint, production Vite build,
module-size, changed-policy, assertion, whitespace, and diff checks. The web
dependency audit reports 337 packages and zero vulnerabilities.

This still is not full #4225 completion: the File commands remain disabled
until the whole application-session adapter and chooser workflow lands, and
protected CI/review, normal stack integration, and UpstreamDrift parity remain
required. Solver UI was not changed.

## 2026-08-10 Parent compositor rendered-QA evidence

Evidence is the complete 919-test Python/PyQt Rate of Closure suite, focused
type checks and clean Ruff/format; and 114 React files / 684 tests with
TypeScript, zero-warning ESLint, and a production build. Browser control at
1280 x 720 and 760 x 800 verifies balanced distinct hosts, responsive stacking,
legacy displays, and direct commands. Isolated PyQt6 control at 1282 x 752
verifies Single to Split Horizontal to Grid normalization, distinct real plots,
and navigable overflow. This is not full #4225 completion: complete focus and
keyboard layout manipulation, export round-trip evidence, protected CI/review,
normal stack integration, and UpstreamDrift parity remain required. Solver UI
was not changed.
## 2026-08-10 Repaired mobile-parent propagation into PR #4324

Draft PR `#4324` remains on `fix/4224-default-legend-layout-local`, based on
`fix/rate-mobile-tools-menu`, and now normally incorporates exact repaired
mobile parent `16a1167c31126238163297983862004afc5001d9`. Legend/layout
production/test code did not conflict; only the four additive handoff/spec
files required reconciliation. No rebase, retarget, force-push, parent rewrite,
or history rewrite was used. Fresh local verification and protected CI remain.

## 2026-08-10 Issue #4224 non-obscuring legend rail slice

Immutable implementation evidence is
`6c65a69624007912d45615fbe59314924c5107dc` plus real-canvas follow-up
`83b4baa3be7424777db4dd50883b7a9e45c8ca91`, based on exact PR #4301 head
`5c8efcbe5fcd6f993ef947a85e39852d268780a6`. The PyQt6 3D simulation default
now places its legend at figure scope inside a measured right rail, shrinks the
axes to the legend's rendered left boundary, removes retained legends before
redraw, and performs legend-only reflow on real canvas resize without a full
scene redraw. This keeps the visible legend outside the
axes at the 360 x 280 minimum while preserving inside and hidden choices.
The checkbox and position selector expose explicit accessible names.

React plot cards use one pure `resolvePlotLayout` contract for both plot margins
and legend coordinates. The 520 px regression locks a 20 px separation between
the plot edge and outside legend. Exact evidence is 69 focused Python/PyQt/manifest
tests, clean changed-file Ruff/format and pinned MyPy, plus the installed React
focused regression (one file / four tests), full 111-file / 674-test Vitest
suite, TypeScript, scoped zero-warning ESLint, and 196-module production build.
`npm ci` audited 337 packages with zero vulnerabilities. Native rendered QA,
workspace persistence/export, protected CI, review, and normal dependency-
ordered integration remain open, so #4224 and epic #4218 stay open.
## 2026-08-10 Repaired camera-parent propagation into PR #4301

Draft PR `#4301` remains on `fix/rate-mobile-tools-menu`, based on
`feat/4284-camera-snap-tracking`, and now normally incorporates exact repaired
camera parent `104503aac9779b195d46d38e8ed32611ffc8dfd7`. Mobile-toolstrip
production/test code did not conflict; only the four additive handoff/spec
files required reconciliation. No rebase, retarget, force-push, parent rewrite,
or history rewrite was used. Fresh local verification and protected CI remain.

## 2026-08-10 PR #4301 four-surface parent propagation

Draft PR #4301 preserves base `feat/4284-camera-snap-tracking` and normally
merges original responsive-toolstrip child
`05713bcdd8f9889dcdcbaa5bdbaeab139d599b64` first with exact, independently
reviewed #4299 parent `142631a90c008942bad99745e279748a7eda2ffa`
second. The branch is not rebased, retargeted, rewritten, or force-pushed.
File, View, and Tools continue to share one horizontal collision clamp with a
16 px constrained-screen gutter, unchanged desktop anchoring, and native
`<details>/<summary>` keyboard/accessibility behavior. The child now inherits
the declared four-surface authority, repaired flight-to-ground stack, and all
camera/playback controls carried by #4299.

Fresh combined-tree evidence is 1,589 Python/PyQt/shared-swing tests with one
explicit unavailable-wheel skip; 111 React files / 673 tests; TypeScript,
zero-warning ESLint, a 195-module build, and six desktop/constrained 2x-DPR
Playwright cases; all 137 `tools-core` tests with format and warning-denied
Clippy; and exact-delta Ruff/format (four files), pinned MyPy 1.13 (three
production files), Bandit (two source files), deterministic-authority,
assertion, minimum-test, documentation, manifest, size, conflict-marker, and
diff gates. Independent staged-tree review found no findings; protected
current-head CI remains required before publication. The local
propagation is not completion evidence for #4300, #4284, #4264, #4260, or any
parent epic; native rendered QA and installed-consumer conformance stay open.

## 2026-08-10 PR #4299 camera/ground-stack propagation

Draft PR #4299 keeps base `feat/4199-wind-workflow` and normally merges the
original four-surface child head
`dca40c6c0168df3aa0cd0de0e5ae0ff109715b6a` first with independently
reviewed #4298 head `57942e64744a199e4fd7d604fe2eeb9faddd062a`
second. No branch is rebased, retargeted, rewritten, or force-pushed. The
result retains `four-surface-capability/v1`, its declared-scope generator,
schema, canonical inventory, and exact evidence paths while inheriting the
complete camera-control and repaired flight-to-ground stack.

The governed boundary covers 15 structured campaign programs, 18 unique
linked active release specifications, and six curated capability records.
Every record classifies all four product surfaces explicitly. UpstreamDrift
support remains unavailable without an immutable installed Tools pin and
repository-bound conformance evidence, and narrative-only features remain
outside this declared scope until promoted to a structured authority.

Local integration evidence is 1,589 Python/PyQt/shared-swing tests with one
explicit unavailable-wheel skip; 110 React files / 670 tests; TypeScript,
zero-warning ESLint, the 194-module production build, and four desktop/high-DPI
Playwright camera cases; all 137 `tools-core` tests plus formatting and
warning-denied Clippy; and the exact hosted Ruff/format, pinned MyPy, Bandit,
deterministic-authority, documentation, changed-code, size, assertion,
manifest-layout, conflict-marker, and diff gates.

Independent exact-tree review found no findings. Protected current-head CI,
installed-consumer evidence, native rendered QA, and dependency-ordered
release remain open. Neither #4264 nor #4260 is complete.
## 2026-08-10 Current-parent propagation into camera PR #4298

Draft PR `#4298` retains branch `feat/4284-camera-snap-tracking` and base
`feat/4199-wind-workflow`. It now normally incorporates exact current parent
head `1e82f15026786ea0b08f78f4c001590ddce9ff39`; camera production/test code
did not conflict, and only the four additive handoff/spec files required
reconciliation. No rebase, retarget, force-push, parent rewrite, or history
rewrite was used. Fresh local verification, protected CI, and review remain.

## 2026-08-10 Repaired scalar-adapter propagation into wind workflow

Draft PR `#4282` remains on `feat/4199-wind-workflow`, based on
`feat/4199-wind-scalar-adapter`. It now normally incorporates repaired parent
head `d6fb04e07c2a625412e9208b07103acdc42c621b` after that head's quality gate
passed. The merge had no wind-workflow production/test conflict and used no
rebase, retarget, force-push, or history rewrite. Twenty-five focused tests and
the documentation, size, and whitespace gates pass locally; protected CI,
review, and later-stack propagation remain outstanding.

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

## Issue #4284 local implementation

Draft PR #4298 publishes branch `feat/4284-camera-snap-tracking` with tested
camera evidence through immutable commit
`2095e748ddca2d7036bbd49a731528f5634daff9`. The normal merge containing this
handoff keeps original camera child
`9ffd8d280c77977a41e93bd0caef9678d1c231b6` first and incorporates exact
repaired #4288 head `108a841b1378c992defd3c7b7ee263d41a6c8b24`
second; the PR base remains `feat/4199-wind-workflow`. Exact #4288 contains
repaired #4285 `e5bcbd1096d3be1f621a805c9d9f3fd321e375a5` and repaired #4282
`686016196a2f895058b8a566dff103a0fd32cd10`. No branch was rebased,
retargeted, rewritten, or force-pushed. The camera child has a shared Python/TypeScript
camera contract and adapters for PyQt6 Simulation/Flight and React Club,
Impact, and Flight 3D viewports. Canonical snap directions use x downrange,
y up, z right; face-on side is explicit. Tracking is opt-in, bounded, isolated
per viewport, preserves safe zoom, suspends after manual orbit, and resumes on
Recenter. Camera state is deliberately not persisted in this slice.

The published evidence adds adjacent solver-sample frame stepping to
React flight playback and Playwright coverage spanning play/pause/restart,
loop, speed, frame steps, wheel zoom, snap views, tracking suspension, and
Recenter. It also verifies control containment and canvas backing resolution
at 520 x 900 and 2x DPR. Do not report released: required follow-up is native
rendered review, hosted CI/review, normal stack integration, and UpstreamDrift
PyQt6/React consumer parity. See
`docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md` and issue #4284.

Evidence commit `2095e748` passes 39 focused Python/PyQt camera tests, the full
107-file / 650-test React suite, four Playwright browser tests across desktop
and constrained 2x-DPR Chromium, Ruff format/check, targeted mypy, TypeScript,
zero-warning ESLint, the 193-module Vite build, campaign validation, and diff
checks. Headless desktop and 700 px camera-control renders were inspected
without overlap; the local offscreen Qt font directory is unavailable, so
native-font/browser rendered review remains explicitly open. Browser
automation does not close the native visual or downstream parity gates.

The prior documentation-only successor records the already-published camera
evidence commit. `evidence_commit_sha` is immutable evidence, not an impossible
self-reference. This local merge records its exact two parents while omitting
its own future SHA from the commit it creates.

The exact composed tree passes 1,738 Python tests with two explicit optional
`build123d` skips, including the installed `tools_core` flight parity path;
110 React files / 670 tests; all 137 `tools-core` Rust tests; and four
Playwright camera/playback cases across desktop and constrained 2x-DPR
Chromium. TypeScript, zero-warning ESLint, the 194-module Vite production
build, Ruff check/format across 61 changed Python files, pinned mypy 1.13 and
Bandit across 43 changed production files, warning-denied `tools-core` clippy,
Rust format, campaign-manifest validation, documentation governance, module
and 500-LOC budgets, conflict-marker checks, and staged/working diff checks are
clean. The focused child/parent control seam passes 12 PyQt camera and impact
layer tests. Protected current-head CI, review, native rendered review,
UpstreamDrift parity, camera persistence, and protected release remain open.

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

The preceding exact-head Python 3.12 CI lane loaded embedded
`src.shared.python.swing_sim` tests and canonical `shared.python.swing_sim`
imports as distinct package trees, producing ground/impact collection errors.
The shared alias registry now coalesces that package root. A subprocess identity
contract failed before the fix and now passes together with both affected
public-API suites. The file-size job was cancelled in checkout before its
budget step and is not a code failure.

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
ground head is incorporated into #4288, and exact #4288 is incorporated into
#4298 by the normal merge containing this handoff. Current #4298 CI and review
are now the next ancestry gates.

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

Those feature-parent merges were not protected releases. Their corrected
#4282 carrier is now incorporated into #4285 through normal ancestry; preserve
dependency order and use normal merges only.

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
