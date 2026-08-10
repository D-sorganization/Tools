# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-10

## 2026-08-10 Second workspace-parent propagation into PR #4280

- Draft PR `#4280` retains branch `feat/4144-variation-export-continuation`
  and base `feat/4218-toolstrip-workspace`. Exact repaired parent head
  `61b7f48b5aeb7d57246b4963da3df086e79cbe15` is incorporated through a
  normal merge commit without rebase, retarget, force-push, or history rewrite.
- There is no feature-code conflict. The variation/export implementation stays
  intact while both branches' append-only handoff/SPEC evidence is retained.
- Parent quality-gate success is not child release evidence. Fresh exact-head
  protected CI, review, and all earlier dependency gates remain open.
- Reconciled-child evidence is 25 focused D-plane/impact tests plus docs
  governance, changed-file size budget, and whitespace checks.

## 2026-08-10 Exact workspace-parent propagation into PR #4280

Draft PR `#4280` remains on `feat/4144-variation-export-continuation` with
base `feat/4218-toolstrip-workspace`. The normal merge starts from original
child head `f90836e342efc8be624739802375af2876d11e5f` and incorporates exact
published parent head `6717e9e09d507dbc24bedb36177f1cdf0b4fd90b` as
the second parent. No rebase, retarget, force-push, parent rewrite, or draft-
state change is permitted.

The source merge is conflict-free: the child's selected-scatter CSV export,
typed unavailable values, accessible raw tables, linked selection, and focused
PyQt/React visualization modules remain intact alongside the parent's complete
workspace/toolstrip, playback, plot, navigation, Python 3.10, and module-budget
repairs. SPEC 1.14.12 records this combined child release above the unique
parent 1.14.11 entry. Staged review found and the continuation corrected one
release-blocking rerun defect: replacing a study after selecting a later trial
now clears linked selection atomically. Every PyQt public setter validates
against the current trial count, while React clears on result identity change
and shares only bounded selections, so a smaller rerun cannot crash or leave
all new points dimmed.

Verification is green: 37 focused variation GUI/export tests and the complete
Rate/shared-swing/golf-club matrix passes 1,528 tests with two explicit
optional build123d skips; all
546 React tests pass across 90 files, as do TypeScript, ESLint, and the Vite
production build. Exact-parent Ruff, Ruff-format, Black, pinned MyPy 1.13 for
all four changed Python production modules, Bandit, the four-file 500-line gate,
docs, minimum-test, changed-test assertions, detect-secrets fingerprints, and
diff checks pass. CPython 3.10.20 compiles all changed Python files, and 30
compatibility/date-boundary regressions pass. There is no Rust delta from the
exact parent, whose 12-test `swing-core` evidence remains applicable.
Independent staged re-review found no actionable findings after verifying the
atomic result-identity boundary and validated-value widget updates. Protected
exact-head CI and required repository review remain release gates after
guarded publication.

## 2026-08-10 PR #4280 workspace timestamp propagation

Draft PR #4280 remains on `feat/4144-variation-export-continuation` with base
`feat/4218-toolstrip-workspace`. Exact corrected parent
`05383d333b6fd87eaf5e37305476f50b505c2c2e` is incorporated through the normal
merge containing this handoff, without rebasing, retargeting, force-pushing,
or rewriting either branch. The child retains its selected-scatter CSV export,
typed unavailable values, accessible raw tables, and focused visualization
modules while inheriting the deterministic Python 3.10-3.12 UTC parser.

The monotonic specification assigns the parent repair version 1.14.10 and the
variation child version 1.14.11. The reconciled tree passes all `778` Rate
tests, the real-CPython-3.10.20 compatibility suite (`27 passed`), the focused
React variation suite (`1 file / 8 tests`), TypeScript, zero-warning focused
ESLint, Ruff, format, and pinned mypy 1.13. Documentation, size, and diff gates
must remain clean in the merge commit. Protected CI, review, and propagation
to #4281 and later descendants remain separate release gates.

## 2026-08-10 Second parent propagation into PR #4279

- Draft PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
  `feat/4181-launch-monitor-registry`; exact repaired parent head
  `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76` is incorporated through a
  normal merge commit with no rebase, retarget, or history rewrite.
- The trees have no feature-code conflict. Reconciliation retains both
  branches' append-only handoff/SPEC evidence, while the inherited D-plane
  ndarray typing repair remains numerically and behaviorally neutral.
- Parent quality-gate success does not authorize this child. Fresh exact-head
  protected CI, review, and all dependency gates remain open.
- Reconciled-child evidence is 25 focused D-plane/impact tests plus docs
  governance, changed-file size budget, and whitespace checks.

## 2026-08-10 Exact parent propagation into PR #4279

Draft PR `#4279` remains on `feat/4218-toolstrip-workspace` with base
`feat/4181-launch-monitor-registry`. This merge starts from original child
head `05383d333b6fd87eaf5e37305476f50b505c2c2e` and incorporates exact
published parent head `31cbc007d4c85b5479b7cd0fb0969124eab2af67` through a
normal two-parent merge; no rebase, retarget, force-push, or draft-state change
is permitted.

The semantic reconciliation preserves the child's File/View/Tools workspace,
module visibility, granular playback, trail, plot, and launcher integration.
It also keeps the parent's focused `ImpactLayerControls`, plotting catalog,
triple-pendulum, and primary-navigation modules. Workspace navigation now
aliases the parent's canonical stable-ID/settings constants, while retaining
the child's visibility and required-module state. Impact-layer automation and
the rendered controls share the parent's single checkbox mapping.

Combined verification is green: 1,339 Python tests pass with six explicit
optional build123d/Rust-wheel skips; 545 React tests pass across 89 files, and
TypeScript, ESLint, and the Vite production build pass. The post-format semantic
rerun is 40 passing navigation/simulation GUI tests. Ruff, Ruff-format, Black,
pinned MyPy 1.13 across all 18 changed production modules, Bandit, the exact
18-file 500-line gate, docs, minimum-test, changed-test assertions,
detect-secrets fingerprints, and staged/unstaged diff checks pass. CPython
3.10.20 compiles every changed Python file and its dependency-free navigation
state round-trip passes; 30 compatibility/date-boundary regressions also pass.
There is no Rust delta from the exact parent, whose 12-test `swing-core`
evidence remains applicable. Independent staged review found no actionable
findings after 76 additional focused PyQt/navigation/workspace tests; protected
exact-head CI and required repository review remain release gates after an
ordinary guarded push.

## 2026-08-10 PR #4279 fractional-timestamp Python 3.10 repair

Exact-head CI on descendant PR #4281 exposed one remaining workspace parser
difference: CPython 3.10 accepts only three or six fractional-second digits in
`datetime.fromisoformat`, while newer supported interpreters accept the
one-digit value already used by the workspace ordering contract. The earliest
owner is PR #4279. Its shared workspace validator now enforces one anchored UTC
grammar and parses zero through six fractional digits consistently, rejecting
greater precision instead of silently truncating it on newer Python versions.
UTC-only validation, serialized values, schema fields, and chronological
comparison are unchanged.

Evidence is `778 passed` for the full Rate suite and `45 passed` for the
compatibility plus complete workspace document suites on the local supported
interpreter, plus `27 passed` for the source-level compatibility suite on real
CPython 3.10.20. The latter has only expected warnings for plugins absent from
that intentionally minimal runtime. Ruff, formatting, pinned mypy 1.13,
documentation governance, and the 400-line budget pass. Exact remote-head
identity and normal publication/descendant propagation remain release gates.

## 2026-08-09 PR #4280 parent propagation and SPEC restoration

Draft PR #4280 remains on `feat/4144-variation-export-continuation` with base
`feat/4218-toolstrip-workspace`. Exact corrected parent
`3f67ed466fefc8991db9c4409f921f25e1c37142` is incorporated by the normal
merge containing this handoff; no branch was rebased, retargeted, force-pushed,
or published by this continuation. The result retains the complete workspace
and Python 3.10 compatibility history together with #4280's independently
owned variation scatter/export changes.

The child exports every selected scatter axis to CSV with stable trial index,
typed outcome, and explicit unavailable values in both PyQt6 and React. PyQt6
also exposes the selected raw rows in bounded accessible tables, with shared
table population and focused scatter/matrix modules. SPEC 1.14.11 now records
this child source delta separately from parent versions 1.14.10 and earlier.

Current evidence is `46 passed` on Python 3.11 and `19 passed` on real CPython
3.10.20 with PyQt6 present. React passes `1 file / 8 tests`, TypeScript, and
focused zero-warning ESLint. Ruff check/format passes five child Python files;
pinned mypy 1.13 passes the four child production modules. Documentation
governance, ancestry/SPEC assertions, and final diff checks pass locally.
Protected CI, review, publication, and descendant propagation remain separate
release gates.

## 2026-08-09 PR #4279 Python 3.10 compatibility completion

Draft PR #4279 remains on `feat/4218-toolstrip-workspace` with unchanged base
`feat/4181-launch-monitor-registry`. Exact corrected parent
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` was incorporated by the normal
local merge `a340fabefa443d47325c5538f342683b38c01ade`; no branch was rebased,
retargeted, force-pushed, or published by this continuation.

The child-owned command registry and view-workspace enums now obtain
`StrEnum` from `shared.python.compatibility` at runtime while retaining the
native type behind `TYPE_CHECKING`. Workspace timestamp validation likewise
uses the shared `UTC` value. This removes all three Python 3.11-only imports
introduced by #4279 without changing command IDs, enum values, timestamp
serialization, workspace schemas, or UI behavior. The merged regression
guards nine string-enum modules and both UTC modules by inspecting the actual
runtime import branch, then executes the three child workspace modules.

Evidence is `126 passed` on Python 3.11, plus `14 passed` and a successful
10-module dotted-import probe on real CPython 3.10.20. Ruff check/format passes
the 11 production modules and compatibility test; pinned mypy 1.13 passes all
11 production modules with the changed-file CI settings. Documentation
governance and final diff checks must pass in this same local commit. Protected
CI, review, publication, and descendant propagation remain separate gates.

## 2026-08-09 PR #4279 launch-registry propagation

Draft PR #4279 remains on `feat/4218-toolstrip-workspace` with unchanged base
`feat/4181-launch-monitor-registry`. Exact parent head
`08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` is incorporated through the normal
merge commit containing this handoff; neither branch was rebased, retargeted,
force-pushed, or pushed by this continuation. Source changes applied cleanly;
the overlapping handoff and specification histories were reconciled
monotonically. The result preserves the parent's package-relative facade and
Python 3.10 enum repairs plus the child's workspace, toolstrip, playback, plot,
and module-navigation implementation.

Focused validation is `126 passed` across both facade contract modules and all
Python/PyQt test files changed by #4279. React workspace validation is `8 files
/ 32 tests passed`. Ruff check and format pass for all 28 relevant Python
files. CI-pinned mypy 1.13 passes the 18 changed production files plus the two
facade contract tests. The type gate exposed one real child-boundary defect:
`legend_visible()` returned an untyped Qt value; it now converts that value at
the widget boundary with `bool(...)`. The affected simulation GUI rerun is `29
passed`; documentation governance and staged/unstaged diff checks also pass.

## 2026-08-10 Second Parent Repair Propagation (#4202 -> #4203)

- Draft child PR `#4203` retains base `feat/4189-dplane` and receives exact
  repaired parent head `7d8d2f06dc797021d01939691e58f8425b652b33`
  through a normal merge commit; no base, parent history, or draft state is
  rewritten.
- The propagated parent repair adds explicit NumPy ndarray result boundaries
  to the two D-plane helpers that failed the hosted pinned MyPy gate. Numerical
  behavior, frames, schemas, and UI behavior remain unchanged.
- The exact parent quality gate is green. Protected child checks, review, and
  all earlier dependency gates remain open; this propagation does not
  authorize a merge or close #4189.
- Child-tree verification after conflict reconciliation is 25 focused D-plane,
  impact-contract, impact-kinematics, and impact-scene tests; docs governance,
  changed-file size budget, and whitespace checks also pass. The Windows
  unpinned MyPy 1.15 environment cannot parse the installed Python-3.12-only
  NumPy stub syntax under this branch's Python 3.11 target, and WSL currently
  fails to start with `E_FAIL`; neither attempt is reported as a passing gate.

## 2026-08-10 Parent Repair Propagation (#4202 → #4203)

- Draft child PR `#4203` remains on `feat/4181-launch-monitor-registry`, based
  on `feat/4189-dplane`; neither branch nor PR base is rewritten.
- Original child head `08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11` is
  normally merged with exact parent head
  `b443fdbed7064c5db0320106013c8413e3e24356`, in that parent order.
- The semantic reconciliation preserves #4203's responsive
  `SimulationViewControlsMixin` architecture while delegating persisted
  D-plane checkbox state to the parent's focused `ImpactLayerControls`
  helper. The existing `_impact_layer_checks` automation seam aliases the
  helper's single checkbox mapping, so no duplicate UI state is introduced.
- The inherited Python 3.10 compatibility repairs, frame-explicit D-plane
  geometry, launch-monitor registry, responsive layout, and exports remain
  additive. `simulation_view.py` and its controls mixin both remain below the
  protected 500-line limit.
- The exact original child also carried three ungrandfathered module-budget
  blockers before this propagation: `simulation/sources.py` (540 LOC),
  `plotting/catalog.py` (533 LOC), and `ui/pyqt6/main_window.py` (528 LOC).
  Behavior-preserving extractions now isolate triple-pendulum dynamics,
  plotting metadata, and versioned primary-navigation persistence. The legacy
  source, catalog, and main-window imports are identity-preserving re-exports;
  the split modules are 282/282, 459/98, and 494/85 lines respectively.
- Focused reconciliation and extraction evidence is green: 36 PyQt
  simulation/layout tests, 38 plotting/navigation tests, and 21 simulation
  source/export tests. Final combined-stack evidence is 1,249 passing Python
  tests with six explicit optional-dependency/Rust-wheel skips; 521 passing
  React tests plus TypeScript, ESLint, and Vite production gates; 12
  `swing-core` tests; real CPython 3.10 compilation/import checks; scoped
  Ruff/Black/pinned MyPy; docs, minimum-test, assertion, changed-file size,
  detect-secrets, and diff gates. A full-tree audit separately retains two
  untouched non-candidate size findings (`kinetics.py`, 646 LOC;
  `torque_profile_panel.py`, 612 LOC). Independent staged review found no
  actionable findings after 95 additional focused tests. Exact-head protected
  CI and required repository review remain release gates.

## 2026-08-09 PR #4203 Python 3.10 string-enum compatibility

Current-head child CI exposed a collection boundary that was already present
in PR #4203's parent surface: seven Rate/shared swing modules imported
`enum.StrEnum`, which does not exist on Python 3.10. Runtime imports now use
the repository's existing `shared.python.compatibility.StrEnum`; type checking
retains the native stdlib symbol behind `TYPE_CHECKING` so pinned mypy 1.13
keeps enum-member types rather than weakening them to strings. No enum values,
serialized contracts, physics, or UI behavior changed.

This repair is published at exact #4203 head
`ab7de5a47977417e02926c3fbc7476002e82b690`. Evidence:
64 focused convention, D-plane, manual-delivery, flight-result, inverse,
impact-family, capability, and compatibility tests pass; Ruff and format pass;
pinned mypy 1.13 passes all eight changed Python files; and a real CPython
3.10.20 probe verifies the shared fallback and all seven runtime import paths.
Propagate the new parent normally through #4279, #4280, #4281, and #4282.

The subsequent full Rate scan also found one parent-owned direct
`datetime.UTC` import in the torque-profile controller. It now uses the same
shared Python 3.10 compatibility module. The focused torque-profile UI suite
and the real-3.10 source/runtime probe are required before publishing this
follow-up; serialized timestamps remain UTC and the workspace schema is
unchanged.

## 2026-08-09 PR #4203 Linux collection repair

Draft PR #4203 remains on `feat/4181-launch-monitor-registry`, based on
`feat/4189-dplane`; no branch was rebased, retargeted, force-pushed, or merged
on GitHub. Exact-head CI run `31199764932` passed the quality gate but all
three Python lanes failed while collecting the in-package flight and solver
facade contract tests. Pytest loaded those tests through its `src.shared...`
package namespace, while their absolute dotted aliases requested the editable
`shared...` namespace; Python then reported that `flight`/`solver` could not be
imported from `src.shared.python.swing_sim` before any assertion ran.

The bounded repair uses package-relative facade imports in those two tests, so
collection and the public API assertions stay in one namespace. It does not
change production code or widen either facade. Verify both contract modules
with `--import-mode=importlib`, Ruff/format, and pinned mypy before a normal
push. The run's Rust `-lpython3.11` link error is missing runner toolchain
state, not simulation evidence; do not modify the model to hide it. After the
new #4203 head passes, propagate it through #4279, #4280, #4281, and #4282 in
normal stack order.

Local evidence is now `12 passed` on Windows and `12 passed` under WSL Python
3.11 with importlib collection. Ruff check/format and pinned mypy 1.13 pass
for both changed test modules. The frozen-dataclass assertion casts only its
introspection target to `Any`, matching the later carrier boundary while
retaining the runtime assertion. The minimal WSL environment reports only
unknown-option warnings for intentionally omitted optional pytest plugins.

## 2026-08-10 PR #4202 D-plane ndarray typing repair

- Draft PR `#4202` remains on `feat/4189-dplane`, based on
  `feat/4162-wedge-impact-visualization`; the verified published repair base is
  `b443fdbed7064c5db0320106013c8413e3e24356`.
- CI Standard run `31384810375`, job `93442745760`, exposed two exact pinned
  MyPy 1.13 `no-any-return` findings in the private D-plane ndarray helpers.
  Explicit local ndarray result boundaries now preserve those helpers' return
  contracts without changing validation, arithmetic, frames, schemas, or any
  public API.
- TDD evidence: the exact two-error MyPy failure was reproduced before the
  repair; the same command is green afterward. Twenty-four focused D-plane,
  impact-contract, impact-kinematics, and impact-scene tests pass, together
  with seven metadata/pre-push contract tests, scoped Ruff, Ruff format, Black,
  docs governance, minimum-test, module-size, changed-file-size, and diff
  checks. An exploratory CI-workflow contract slice retains three unrelated
  failures for later toolcache/environment steps absent from this older branch;
  no workflow file is changed by this repair.
- This is a bounded quality-gate repair, not completion of D-plane issue
  `#4189`. Protected current-head CI, dependency order, and required review
  remain release gates; no push, retarget, or draft-state change is included.

## 2026-08-10 Parent Repair Propagation (#4179 → #4202)

- Child PR `#4202` remains on `feat/4189-dplane`, based on
  `feat/4162-wedge-impact-visualization`; neither branch nor PR base is
  rewritten.
- Original child head `b4abec03bccfbdd87ddf91427159c5c2332c21dd` is
  normally merged with exact parent head
  `6704a3e541a3e74c28b4a284530d1a21269dd340`, in that parent order.
- The Python 3.10 UTC repair and source-wide AST guard remain additive to the
  frame-explicit 3D D-plane geometry, desktop/web overlays, and export
  contracts.
- The persisted D-plane layer controls are extracted into a focused helper,
  restoring `simulation_view.py` to the protected 500-line module budget
  without changing its compatibility seam or behavior.
- Combined-stack verification is green: 93 focused and 825 scoped Python tests
  with two optional `build123d` skips; 360 React tests plus TypeScript, ESLint,
  and Vite production gates; real CPython 3.10.20 compilation/UTC; Ruff/Black;
  focused pinned MyPy 1.13; docs, minimum-test, file-size, detect-secrets, and
  diff checks. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited broad MyPy
  baseline remains 17 Qt/NumPy typing findings in 11 untouched files.
  Protected CI and required review remain release gates.

## 2026-08-10 Parent Repair Propagation (#4178 → #4179)

- Child PR `#4179` remains on `feat/4162-wedge-impact-visualization`, based on
  `feat/4166-wedge-turf-physics`; neither branch nor PR base is rewritten.
- Original child head `0eb804e70887c788421332369e42792411aff55a` is
  normally merged with exact parent head
  `bfa83aedc88ead380babc73a699377d98b971006`, in that parent order.
- The Python 3.10 UTC repair and source-wide AST guard remain additive to the
  exact-event, locked-scale wedge impact visualization contracts.
- Combined-stack verification is green: 58 focused and 739 scoped Python tests
  with two optional `build123d` skips; 347 React tests plus TypeScript, ESLint,
  and Vite production gates; real CPython 3.10.20 compilation/UTC; Ruff/Black;
  focused pinned MyPy 1.13; docs, minimum-test, file-size, detect-secrets, and
  diff checks. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited broad MyPy
  baseline remains 17 Qt/NumPy typing findings in 11 untouched files.
  Protected CI and required review remain release gates.

## 2026-08-10 Parent Repair Propagation (#4174 → #4178)

- Child PR `#4178` remains on `feat/4166-wedge-turf-physics`, based on
  `feat/4161-wedge-ground-clearance`; neither branch nor PR base is rewritten.
- Original child head `aaae3f73e17dbfaad5cca1dc6f49559b3aebe9d5` is
  normally merged with exact parent head
  `9ea93e92563280ec34bca682ad44d7409edd7a02`, in that parent order.
- The Python 3.10 UTC repair and AST guard remain additive to the validated
  turf-contact contracts and their explicit scientific/calibration boundary.
- Combined-stack verification is green: 56 focused and 732 scoped Python tests
  with two optional `build123d` skips; real CPython 3.10.20 compilation/UTC;
  Ruff/Black; focused pinned MyPy 1.13; docs, minimum-test, file-size,
  detect-secrets, and diff checks. With no TypeScript or Rust delta, the exact
  parent evidence of 345 React tests/all web gates and 12 `swing-core` tests is
  unchanged. The inherited broad MyPy baseline remains 17 Qt/NumPy typing
  findings in 11 untouched files. Protected CI and review remain release gates.

## 2026-08-10 Parent Repair Propagation (#4173 → #4174)

- Child PR `#4174` remains on `feat/4161-wedge-ground-clearance`, based on
  `feat/4163-impact-inspector`; neither branch nor PR base is rewritten.
- Original child head `880a6465fc872cf3d6650283db154ddc41793a31` is
  normally merged with exact parent head
  `9ddaff3b6bca542fd7a2befc7d7b0ae53910a60a`, in that parent order.
- The inherited Python 3.10 UTC repair and source-wide AST guard remain
  additive to the swept wedge ground-clearance contracts.
- Combined-stack verification is green: 56 focused Python tests; 703 scoped
  Python tests with two optional `build123d` skips; 345 React tests plus
  TypeScript, ESLint, and Vite production gates; 12 `swing-core` tests; real
  CPython 3.10.20 compilation and UTC checks; Ruff/Black; focused pinned MyPy
  1.13; docs, minimum-test, file-size, detect-secrets, and diff checks. The
  inherited broad MyPy baseline is 17 Qt/NumPy typing findings in 11 untouched
  files and remains outside this propagation scope. Protected CI and review
  remain release gates.

## 2026-08-10 Parent Repair Propagation (#4167 → #4173)

- Child PR `#4173` remains on `feat/4163-impact-inspector`, based on
  `feat/4144-variation-visualizations`; neither branch nor PR base is rewritten.
- The original child head `3c43955aaeb3964ff8c3ef2748d626baae518b76`
  is being merged normally with exact parent head
  `22b66b560652b78de84141344c4ddd9a92a83b26`, in that parent order.
- The inherited repair replaces Python 3.11-only `datetime.UTC` use in Rate
  torque-profile persistence with `shared.python.compatibility.UTC` and adds a
  Rate-source AST regression guard for direct, aliased, and module-attribute
  access.
- Combined local verification is green: 63 focused Python tests, all 562 Rate
  tests, all 334 React tests, TypeScript type-check, ESLint, Vite production
  build, 12 `swing-core` Rust tests, real CPython 3.10.20 compile/UTC checks,
  Ruff check/format, Black, focused pinned MyPy 1.13, docs governance,
  minimum-test, file-size, detect-secrets, and diff checks. A broader MyPy
  sweep still reports 17 pre-existing Qt/NumPy typing errors in 11 untouched
  files. Protected CI and required review remain release gates; queued,
  cancelled, missing-toolcache, and dependency-download jobs are not green
  evidence.
- Every implementation commit must update this file, the Rate handoff, the
  campaign handoff, and `SPEC.md`, or explicitly record why there is no
  material handoff change.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, being grown from a single
closure-rate calculator into a full swing → impact → ball-flight simulation
platform under **Repository_Management#1390** (this handoff rollout) and a
stack of golf-simulation epics:

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

Active infrastructure repair: #4155 hardens the Rust/PyO3 job against
incomplete setup-python cache entries whose interpreter works but whose
declared link library is missing. The repair is isolated on
`fix/4155-rust-libpython-cache` and does not change simulation code.

See `src/rate_of_closure/AGENT_HANDOFF.md` for the detailed stack breakdown
and architecture pointers for this tool specifically.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — repo-wide conventions, CI gate list, cross-repo dependency
   rules (Tools is a leaf dependency; UpstreamDrift and Gasification_Model
   consume it).
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — living specification; §12 Change Log requires a dated row for
   every PR touching `src/` (enforced by `spec-check.yml`, see gates below).
4. `src/rate_of_closure/AGENT_HANDOFF.md`, `src/pendulum_simulator/AGENT_HANDOFF.md`,
   `src/rotation_converter/AGENT_HANDOFF.md` — per-tool handoff docs.
5. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for adding a handoff doc to a
   new tool.

## In-Flight Branches (what stacks on what)

```
main
 └─ feat/impact-simulation-platform   (PR #4119, epic #4103, auto-merge armed)
     └─ feat/investigation-suite      (PR #4124, epic #4120, stacked on #4119)
         └─ feat/course-showcase      (PR #4129, epic #4125, stacked on #4124)
docs/agent-handoff-1390               (this branch, off origin/main, Repository_Management#1390)
```

Other active non-golf branches worth knowing about: `fix/file-size-budget-bounded-checkout`
(#4096, CI checkout-scope fix), `agent/scada-phase-a-foundation` (#4091, SCADA
epic #4085), several `scada/pr*` branches (SCADA epics #4085-#4089), and a
handful of Bolt/Palette/Sentinel micro-PRs (#4070-#4102) unrelated to the
golf-sim stack.

## Gate Commands (repo-wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m contract                     # API contract tests (downstream-facing)
python3 -m pytest -m integration --timeout=60     # cross-repo integration
```

SPEC freshness (CI job `spec-freshness` in `.github/workflows/spec-check.yml`):
any PR touching `src/**`, `tests/**`, `config/**`, `pyproject.toml`,
`Cargo.toml`, `package.json`, or `requirements.txt` must also modify
`SPEC.md` in the same PR, or carry the `spec-exempt` label. Runs on the
`d-sorg-fleet` self-hosted runner.

## Do-Not List

- Do not modify public function signatures in `src/shared/python/**` without
  opening coordinated migration issues in UpstreamDrift and Gasification_Model
  (see `CLAUDE.md` Cross-Repo Dependencies).
- Do not import across package boundaries (e.g. `signal_processing_studio`
  importing from `sidekick.process_calculators`) — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not regenerate the sidekick API baseline (`tests/sidekick_api_baseline.json`)
  without coordinating a breaking-change migration.
- Do not merge #4124 or #4129 ahead of their base (#4119, #4124 respectively)
  — they are stacked and will conflict/duplicate SPEC.md sections if merged
  out of order.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web`
  yet — Phase 7 of #4103 owns this; today Pages hosting elsewhere in the repo
  (e.g. `unit_converter`) is done via manual branch-folder publish, not CI.

## Short-Term Roadmap (ordered)

1. Land PR #4119 (base platform) — currently the long pole; everything else
   stacks on it.
2. Get #4124 out of draft-for-review and merge into `feat/investigation-suite`
   → cascades onto #4119.
3. Get #4129 out of draft-for-review and merge into `feat/course-showcase`
   → cascades onto #4124.
4. Start #4130 Phase F1 (formulation document) once #4125's stack is in.
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy for
   `rate_of_closure/web`.
6. #4125 H5: stand up the public release-management repo (cross-repo, not
   started).
