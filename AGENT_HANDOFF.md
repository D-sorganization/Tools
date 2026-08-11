# AGENT_HANDOFF — Tools

> Update this file in every implementation commit and every push to `main`.
> Current-state only; history lives in git. Last updated: 2026-08-11.

## 2026-08-11 PR #4331 current-parent propagation repair

Branch `feat/4284-orthographic-axis-polish` now normally incorporates exact
current PR #4330 parent `304a069b1777dcf8cf107de26caa3b9fbe96dbb3`
after live PR #4331 head `c7bccbccc6cda0c9b938b2862ed660cebdcb7597`.
The failed hosted quality gate compared the child with stale merge base
`d8176bb5863a35725199bb8357a5f000f9bdd3ba`, so the parent's broad formatting
commit and six worktree pointers incorrectly appeared in the child delta. The
normal two-parent merge has no content conflict and reduces the effective PR
delta to the nine intended camera-polish documentation, adapter, and regression
files. Camera behavior is unchanged. No rebase, retarget, force-push, parent
rewrite, or CI retry was used. This local candidate still requires independent
review, publication by the release owner, and protected exact-head CI.

The repository specification is now synchronized at version `1.14.30`. It
defines the per-preset depth-axis mapping, complete artist suppression,
visible engineering-axis preservation, native one-sided ticks, and full-axis
restoration without claiming any camera-state or simulation-behavior change.

Fresh local evidence on the merged tree is 71 Python/PyQt camera, compositor,
main-window, layout, campaign-manifest, and launcher-manifest tests; Ruff lint
and format on the exact Python delta; pinned MyPy 1.13 on three production
adapters; Bandit; documentation, changed-code, module-size, minimum-test,
assertion, manifest-layout, Spec Check, version, whitespace, and diff gates.
React passes all 114
files / 686 tests, TypeScript, zero-warning ESLint, the 199-module production
build, and four serial Playwright camera cases across desktop and constrained
2x-DPR projects. `npm ci` audited 337 packages with zero vulnerabilities.

## 2026-08-10 Repaired compositor-parent propagation into persistence child

Continuation branch `feat/4225-multiview-persistence` now normally incorporates
exact repaired compositor parent `0e3054e6a7fa0e3e38e1312b4132bbd1e4336fb2`.
Keyboard/persistence production and test code did not conflict; only the four
additive handoff/spec files required reconciliation. No rebase, retarget,
force-push, parent rewrite, or history rewrite was used. Fresh focused/local
verification, protected exact-head CI, and review remain required. The exact
pinned-MyPy delta also required an explicit typed current-workspace validation
local; this changes no parsing, validation, migration, or serialization behavior.

## 2026-08-10 Repaired legend-parent propagation into PR #4327

Draft PR `#4327` retains branch `feat/4225-multiview-compositor` and base
`fix/4224-default-legend-layout-local`. It now normally incorporates exact
repaired legend parent `531a851dc125e83ad86abe1601651e163f5f866d`.
Multi-view production/test code did not conflict; only the four additive
handoff/spec files required reconciliation. No rebase, retarget, force-push,
parent rewrite, or history rewrite was used. Fresh focused/local gates and
protected exact-head CI remain required.

## 2026-08-10 Issue #4225 multi-view keyboard/export acceptance slice

Continuation branch `feat/4225-multiview-persistence` starts at exact draft PR
#4327 head `e975f66bdcfc5a32f9688b8c2c6e34fe1b53ce6e`; no existing branch was
rewritten.
PyQt6 and React now expose real, distinct Impact, Swing, and Flight viewport
hosts through enabled UI-neutral commands and single, horizontal, vertical, or
grid layouts. One simulation run and playback clock drive the visible hosts,
while each real view retains its own camera and overlay state. React preserves
the established Strike, Swing, Kinetics, and Flight displays beside the new
Multi View display, including the canonical spatial-target workflow. Direct
toolstrip commands select a real compositor host without making the legacy
displays unreachable.

Independent review tightened the shared contract: one visible host always
normalizes to Single, two to a valid split, and three to Grid; corrupt saved
split-plus-three documents recover to Grid instead of blocking launch. Valid
per-slot legend placement survives recovery and view transitions. Real
playback state now owns the serialized workspace without writing settings on
every animation frame: React persists settled time plus play/loop/rate changes,
and PyQt6 debounces active playback writes. PyQt6 adds hover guidance for every
new control and a resizable scrolling viewport so minimum-size real plots stay
navigable instead of being clipped in a desktop grid.

The remaining local acceptance gaps are now executable contracts. React uses a
roving tab stop with Arrow Left/Right, Home, and End selection; native Qt uses
an explicit Layout -> Impact -> Swing -> Flight tab chain with stable object
names. Both tests manipulate Single/Split/Grid membership using only focus and
keyboard activation. Both clients also expose strict version-1 compositor
export/import boundaries. Imports validate completely before mutation, reject
future versions, preserve shared playback and per-host legend state, and a
native imported document survives a fresh QSettings reconstruction. The real
version-1 view document is also embedded in and recovered from the whole-app
workspace envelope instead of using placeholder layout data.

The realistic nested slot list exposed and repaired a pre-existing
`VersionedPayload.from_json_dict` double-freeze defect; imports now validate and
freeze JSON exactly once. Local evidence is the complete 921-test Python/PyQt
Rate suite and 114-file / 686-test React suite, focused MyPy, Ruff/format,
TypeScript, zero-warning ESLint, the production Vite build, module-size,
changed-policy, assertion, whitespace, and diff gates. `npm ci` audited 337
packages with zero vulnerabilities.

The File menu remains truthfully disabled because this slice adds reusable
adapters and round-trip proof, not a file-picker/application-session workflow.
#4225 and epic #4218 remain open for protected CI/review, dependency-ordered
integration, and UpstreamDrift consumer parity. Do not push or open a PR until
the root release owner reviews this slice.

## 2026-08-10 Parent compositor rendered-QA evidence

Current local evidence is the complete 919-test Python/PyQt Rate of Closure
suite, Ruff/format and focused type checks; plus the complete 114-file /
684-test React suite, TypeScript, zero-warning ESLint, and the production Vite
build. Browser QA at 1280 x 720 and 760 x 800 proves responsive non-overflowing
hosts, distinct balanced grid content, legacy-display reachability, and direct
command routing. Isolated native QA at 1282 x 752 proves persisted Single,
two-host Split Horizontal, three-host Grid, distinct real plots, and explicit
overflow navigation. This slice still does not close #4225 or epic #4218:
complete keyboard focus/layout manipulation, workspace export proof, protected
CI and review, stack integration, and UpstreamDrift parity remain open. Do not
push or open a PR until the root release owner reviews this slice.
## 2026-08-10 Repaired mobile-parent propagation into PR #4324

Draft PR `#4324` retains branch `fix/4224-default-legend-layout-local` and base
`fix/rate-mobile-tools-menu`. It now normally incorporates exact repaired
mobile parent `16a1167c31126238163297983862004afc5001d9`. Legend/layout
production/test code did not conflict; only the four additive handoff/spec
files required reconciliation. No rebase, retarget, force-push, parent rewrite,
or history rewrite was used. Fresh focused/local gates and protected exact-head
CI remain required before descendant propagation.

## 2026-08-10 Issue #4224 non-obscuring legend rail slice

Immutable implementation evidence is
`6c65a69624007912d45615fbe59314924c5107dc` plus real-canvas follow-up
`83b4baa3be7424777db4dd50883b7a9e45c8ca91` on isolated branch
`fix/4224-default-legend-layout-local`, starting at exact draft PR #4301 head
`5c8efcbe5fcd6f993ef947a85e39852d268780a6`.
PyQt6 Swing/Flight scenes now render the default Outside Right legend as a
figure-owned artist in a measured reserved gutter rather than as an axes-owned
artist inside the plot surface. The renderer removes retained figure legends
before every redraw, recomputes the axes boundary from the rendered legend
width, and performs legend-only reflow after real canvas resize without
advancing camera/playback state or rebuilding the scene. Inside placements and Hidden remain
independent and clear the outside artist. Legend visibility and position also
have explicit accessible names.

React managed plots now derive plot margins and legend origins from one pure
`resolvePlotLayout` contract. At the constrained 520 px reference width the
outside plot edge is 330 px and the legend begins at 350 px, preserving a 20 px
gap without duplicated geometry constants in the canvas renderer. The focused
PyQt6 camera/plot/simulation/wedge/manifest suite passes 69 tests; Ruff check/format and
pinned MyPy 1.13 pass the changed Python production files. The installed React
toolchain passes the focused 520 px regression (one file / four tests), the
complete 111-file / 674-test Vitest suite, TypeScript, scoped zero-warning
ESLint, and the 196-module production build; `npm ci` audited 337 packages with
zero vulnerabilities. Rendered native inspection, persistence/export work,
protected CI, review, and parent integration remain open. This bounded slice
does not close #4224, #4218, #4300, #4303, or their dependency stack.
## 2026-08-10 Repaired camera-parent propagation into PR #4301

Draft PR `#4301` retains branch `fix/rate-mobile-tools-menu` and base
`feat/4284-camera-snap-tracking`. It now normally incorporates exact repaired
camera parent `104503aac9779b195d46d38e8ed32611ffc8dfd7`. Mobile-toolstrip
production/test code did not conflict; only the four additive handoff/spec
files required reconciliation. No rebase, retarget, force-push, parent rewrite,
or history rewrite was used. Fresh focused/local gates and protected exact-head
CI remain required before further descendant propagation.

## 2026-08-10 PR #4301 four-surface parent propagation

Draft PR #4301 keeps base `feat/4284-camera-snap-tracking`. This normal
two-parent merge keeps original constrained-toolstrip child
`05713bcdd8f9889dcdcbaa5bdbaeab139d599b64` first and incorporates exact,
independently reviewed #4299 head
`142631a90c008942bad99745e279748a7eda2ffa` second. No branch is rebased,
retargeted, rewritten, or force-pushed. The child retains the shared
File/View/Tools collision clamp, a 16 px viewport gutter, unchanged desktop
anchoring, and native `<details>/<summary>` keyboard and accessibility
semantics while inheriting the declared four-surface inventory, complete
camera controls, and repaired flight-to-ground ancestry.

The composed tree passes 1,589 Python/PyQt/shared-swing tests with one explicit
unavailable-wheel skip; 111 React files / 673 tests; TypeScript, zero-warning
ESLint, and the 195-module production build; six desktop/constrained 2x-DPR
Playwright camera/toolstrip cases; and all 137 `tools-core` tests plus format
and warning-denied Clippy. The exact base delta also passes Ruff/format on four
Python files, pinned MyPy 1.13 on three production files, Bandit on two source
files, deterministic authorities, behavioral-assertion, minimum-test,
documentation, manifest-layout, module-size, conflict-marker, and diff gates.
Independent staged-tree review found no findings; protected current-head CI
remains required.
This propagation does not complete #4300, #4284, #4264, #4260, or their
parent epics; native rendered QA and installed-consumer evidence remain open.

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
remain open.
## 2026-08-10 Current-parent propagation into PR #4298

Draft PR `#4298` remains on `feat/4284-camera-snap-tracking`, based on
`feat/4199-wind-workflow`. It now normally incorporates exact current parent
head `1e82f15026786ea0b08f78f4c001590ddce9ff39`; camera production and test
code did not conflict. Only the four current-state handoff/spec files required
additive reconciliation. No rebase, retarget, force-push, parent rewrite, or
history rewrite was used. Fresh focused/local gates and protected exact-head
CI are required before this descendant can advance.

## 2026-08-10 Repaired wind-scalar parent propagation

Draft PR `#4282` retains branch `feat/4199-wind-workflow` and base
`feat/4199-wind-scalar-adapter`. Exact repaired parent head
`d6fb04e07c2a625412e9208b07103acdc42c621b` is incorporated through a normal
merge commit after its quality gate passed. No wind-workflow production or test
code conflicted, and no rebase, retarget, force-push, or history rewrite was
used. The current-state handoff remains authoritative. Verification covers 25
focused tests plus documentation governance, changed-file size, and whitespace
checks; fresh protected CI, review, and downstream propagation remain required.

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

## Issue #4284 camera continuation

Draft PR #4298 publishes branch `feat/4284-camera-snap-tracking` with tested
camera evidence through immutable commit
`2095e748ddca2d7036bbd49a731528f5634daff9`. The normal merge containing this
handoff keeps original camera child
`9ffd8d280c77977a41e93bd0caef9678d1c231b6` first and incorporates exact
repaired #4288 head `108a841b1378c992defd3c7b7ee263d41a6c8b24`
second; the PR base remains `feat/4199-wind-workflow`. Exact #4288 contains
repaired #4285 `e5bcbd1096d3be1f621a805c9d9f3fd321e375a5` and repaired #4282
`686016196a2f895058b8a566dff103a0fd32cd10`. No branch was rebased,
retargeted, rewritten, or force-pushed. The camera child implements the
shared, UI-neutral camera command contract in Tools PyQt6 and React swing,
impact, and flight 3D views: exact Face On/Down the Line/Overhead/Isometric
snaps, opt-in bounded subject tracking, zoom-preserving Auto Fit, predictable
manual suspension, and one-action Recenter. The published evidence also adds
solver-sample frame stepping plus real-browser Playwright coverage for a
bounded playback/camera interaction matrix and a 520 x 900, 2x-DPR viewport.
UpstreamDrift consumers, native rendered cross-platform review, hosted CI,
review, and protected release remain open; do not close #4284 on local evidence.

Evidence commit `2095e748` passes 39 focused Python/PyQt camera tests, the full
107-file / 650-test React suite, four Playwright tests across desktop and
constrained 2x-DPR Chromium, TypeScript, zero-warning ESLint, the 193-module
production build, Ruff format/check, targeted mypy, campaign validation, and
diff checks. Headless desktop and 700 px camera-bar renders show no control
overlap; this Qt runtime lacks usable fonts, so native-font visual review
remains an integration gate. Browser automation is not a substitute for that
manual native review.

The prior documentation-only successor records the already-published camera
evidence commit. The campaign contract uses `evidence_commit_sha`, not a
self-referential current-head field. This local merge records its exact two
parents; its own future SHA is intentionally absent from the commit it creates.

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

Authorities are `docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md`,
`src/rate_of_closure/application/camera_commands.py`, and the cross-runtime
golden fixture under `web/src/model/__fixtures__/`. Every implementation commit
must keep this file and `src/rate_of_closure/AGENT_HANDOFF.md` current.

## PR #4288 exact repaired-ground propagation

Draft PR #4288 remains on `feat/4269-flight-ground-transfer` with unchanged
base `feat/4268-ground-contract`. Original child
`247215422a6d4b677552955b4923bc609a553259` normally incorporates exact repaired
#4285 parent `e5bcbd1096d3be1f621a805c9d9f3fd321e375a5` second in the
merge containing this handoff. The child preserves its signed terminal state,
physical sphere/terrain contact brackets, strict provenance, and qualified
Python/TypeScript/Rust/PyO3/WASM transfer while inheriting deterministic
workspace timestamps, canonical `swing_sim` import identity, the hosted-mypy
manifest repair, and the complete variation, wind, capability, and campaign
ancestry. No branch was rebased, retargeted, rewritten, or force-pushed.

This propagation adds no bounce, skid, roll, terrain response, total distance,
or UI execution. Protected CI, independent review, and dependency-ordered
collapse remain separate release gates.

Exact composed-tree verification is 1,080 Python tests passed with six explicit
optional installed-`tools_core` wheel skips, 107 React files / 662 tests,
26 direct Rust transfer/wire tests, TypeScript, zero-warning ESLint, and the
189-module production build. Ruff check/format passes all 28 changed Python
files; pinned mypy 1.13 passes all 21 changed production files. Campaign
manifest validation, documentation governance, and diff checks are clean. The
missing local wheel is not accelerated installed-package evidence and remains
an explicit release boundary; the direct Rust suites executed rather than
being relabeled as wheel parity.

## 2026-08-10 Exact repaired #4282 propagation into PR #4285

Draft PR #4285 remains on `feat/4268-ground-contract` with unchanged base
`feat/4197-capability-observer`. The normal merge containing this handoff keeps
original child `788aa547651a3685a363ea401824a5d81477bafb` first and incorporates
exact repaired #4282 carrier `686016196a2f895058b8a566dff103a0fd32cd10`
second. That carrier contains merged capability PR #4283 commit
`c1827bbdc50a6e11cc475db2636b4e47a4c15416`, exact observer head
`9bbb98e16e435a0d4c74153b909f2ebfefbbce7a`, and the hosted-mypy manifest
repair following predecessor `aa6eeffb0395f7ed7954f2315b1c625cada552d8`.
No branch was rebased, retargeted, rewritten, or force-pushed.

The child retains its strict, UI-neutral flight-to-ground schemas, canonical
fixture, migrations, legacy-result adapter, explicit dependency, and pinned
typing repairs while inheriting the deterministic Python 3.10-3.12 UTC parser
and the corrected variation, scalar, wind, capability, and campaign-release
ancestry. This merge does not claim bounce, skid, roll, terrain profiles,
total distance, UI execution, or Rust/WASM delivery. Protected CI, review, and
normal descendant propagation remain separate release gates.

Exact-head CI on the preceding child head exposed an actionable collection
defect after installing the package: embedded `src.shared.python.swing_sim`
tests and canonical `shared.python.swing_sim` imports could load distinct
package trees, making the ground and impact subpackages unavailable by import
order. The shared alias registry now coalesces `swing_sim`; an isolated RED/GREEN
identity contract and both affected public-API suites pin the correction. The
separate file-size job was cancelled during checkout and did not execute its
budget check, so it remains infrastructure evidence rather than a source
failure.

The inherited ground descendant passes 1,703 Python tests with two explicit optional
`build123d` skips, 643 React tests across 105 files plus type-check,
zero-warning lint and production build, 12 Rust tests, and 77 ground plus
compatibility tests on real CPython 3.10. Ruff/format cover 78 changed Python
files; pinned mypy and Bandit cover 52 changed production files. Campaign
manifest, documentation, minimum-test, changed-assertion, 500-LOC, changed-file
secrets, Python 3.10 compilation, and diff gates are clean. Protected
current-head CI and required review remain separate release gates. Its exact
repaired head is incorporated into #4288, and exact #4288 is incorporated into
#4298 by the normal merge containing this handoff. Current #4298 CI and review
are now the next ancestry gates.

The composed pinned-mypy profile additionally exposed integration-only `Any`
returns at the Pydantic manifest and scalar plotting boundaries. The repaired
#4282 parent owns the manifest annotation; scalar extraction normalizes the
already-validated value with `float`. Bandit also replaced an optimized-away
command-state assertion with an explicit invariant error. The alias finder
retains its intentional optional-probe catch-and-continue boundary under a
narrow explanatory `B112` annotation.

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

The historical capability optimization checkout is
`C:\Users\diete\Repositories\Tools-worktrees\capability-optimization-ui` on
branch `feat/4197-capability-optimization-ui`. Its functionality is now part of
the corrected #4282 ancestry incorporated above. Preserve the protected stack;
do not retarget, rebase, force-push, or merge descendants out of order.

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

The #4285 ground-contract carrier incorporates #4282 exact head
`5f77af4add23547a21cc3fabce98ae9ad4260427` by normal ancestry. Keep downstream
ground PRs stacked behind the resulting exact #4285 head.

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
