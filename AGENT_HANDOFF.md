# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-11

## 2026-08-11 reviewed variation parent propagation into wind PR #4281

- Exact remote wind child `247046d55afcad3e6cd4f8029f854856c427f59c`
  normally merges exact reviewed variation parent
  `3337945699966b63cb5cd8e52d7c3b194315e911` in child-first order. PR base,
  stack topology, and both histories remain intact; no rebase, reset, retarget,
  force-push, or parent rewrite is used.
- Python/TypeScript scalar-ensemble and wind-strategy plot-adapter behavior is
  unchanged while inheriting the parent's variation/export UI, kinetics split,
  Ground/Tee parity, workspace/toolstrip, and Ruff-normalized repairs. The sole
  feature-code conflict was obsolete monolithic kinetics and is resolved to the
  validated parent `pendulum.sample(...)` facade.
- This reconciliation is local only. Independent review, fresh exact-head CI,
  unresolved-thread checks, dependency order, and ordinary publication remain
  required before merge or release.

## 2026-08-11 reviewed workspace parent propagation into variation PR #4280

- Exact remote variation child
  `668ba96746f79f7a12e8092161bd610054197f58` normally merges exact reviewed
  workspace/toolstrip parent `ccd0e026c580c93038fdf5c59d5d452a85ba27a0`
  in child-first order. PR base, stack order, and both histories remain intact;
  no rebase, reset, retarget, force-push, or parent rewrite is used.
- The variation/export feature remains unchanged while inheriting the current
  kinetics split, Ground/Tee parity contracts, protected Ruff normalization,
  and complete workspace/toolstrip behavior. The sole feature-code conflict
  was obsolete monolithic kinetics source and is resolved to the validated
  parent façade. Seven duplicate child automation edits fail protected Ruff
  0.14.10 formatting and are normalized to the exact reviewed parent blobs;
  the automation commit remains reachable in the child-first history.
- This reconciliation is local only. Independent review, fresh exact-head
  protected CI, unresolved-thread checks, dependency order, and ordinary
  publication remain required before merge or release.

## 2026-08-11 remote automation reconciliation for workspace PR #4279

- Exact local workspace head `0b22c401a26c31441a599d8d9b39de123706e7ea`
  ordinarily merges exact remote head
  `61fe2d556a5413e525d958612ccfd57e65b8d5a2`, preserving both histories and
  the existing PR base `feat/4181-launch-monitor-registry` without rebase,
  reset, retarget, force-push, or parent rewrite.
- The remote commit is a broad formatting-only automation sweep. Fifteen of
  its 23 paths, including six pre-existing `.codex-worktrees` gitlinks, were
  already byte-identical in the current parent. Its seven unique formatting
  edits did not match protected Ruff 0.14.10 output and are normalized back to
  the pinned form. Its sole content conflict was obsolete pre-split kinetics
  code; the current `pendulum.sample(...)` façade implementation remains
  authoritative, preserving the parent split and runtime behavior.
- Workspace/toolstrip, module visibility, navigation, playback, independent
  plots, physics, frames, units, schemas, and public contracts remain intact.
  This history-preserving local reconciliation still requires independent
  review, exact-head protected CI, unresolved-thread checks, and dependency
  gates before publication or merge.

## 2026-08-11 hosted MyPy repair propagation into workspace PR #4279

- Exact workspace child `7806a16f58e1c6999d32f0127a187fbb21f839a1`
  normally merges exact published parent
  `3796b49e40b677fbac4e05739f8be49f905df2cb`; PR base, stack order, and
  both histories remain unchanged.
- The inherited production delta is limited to four static
  `numpy.ndarray` casts in the kinetics façade, series, and dynamics modules.
  Workspace/toolstrip behavior and runtime arrays, physics, units, frames,
  public identity, and UI behavior are unchanged.
- This merge is local only. Fresh exact-head protected CI, required review,
  unresolved-thread checks, and dependency gates remain release blockers.

## 2026-08-11 latest #4203 propagation into workspace PR #4279

- Exact parent `0216a547aa79727091a2939b96e779e8ddbd7304` is normally merged into
  child `61b7f48b5aeb7d57246b4963da3df086e79cbe15` without changing PR base,
  stack order, or either history.
- No feature-code conflict exists. The workspace/toolstrip, visibility,
  navigation, playback, and plot controls remain intact while the child
  inherits the parent's pinned formatting and identity-preserving kinetics
  size-budget repair.
- This merge is local only. Fresh exact-head protected CI, review, unresolved
  threads, and dependency gates remain required before publication or merge.

## 2026-08-11 #4203 hosted MyPy kinetics repair

- Exact head `0216a547aa79727091a2939b96e779e8ddbd7304` failed CI Standard run
  `31477542889`, job `93734652129`, at runtime merge ref `aede309`: NumPy's
  typed API exposed four `no-any-return` findings in the newly extracted
  kinetics modules.
- The repair narrows only the results of `numpy.linalg.norm`,
  `numpy.concatenate`, and the app-frame matrix projection to the already
  declared `numpy.ndarray` return contract. Explicit casts do not allocate,
  convert, or change any array, physics, units, frames, public identity, or UI.
- RED evidence is the exact four hosted diagnostics at
  `_kinetics_series.py:121/131`, `_kinetics_dynamics.py:194`, and
  `kinetics.py:61`. GREEN evidence requires the complete PR-base changed-source
  MyPy profile, not a three-file-only run, plus focused/full Rate regression and
  normal protected CI after publication.
- Local GREEN evidence is 102/102 complete-delta MyPy source files, 141/141
  Ruff/format files, 101 Bandit source files with no medium/high finding, 28/28
  focused tests, and 701/701 full Rate tests. Size, documentation, minimum-test,
  and diff gates also pass.
- This commit is local only. PR base, stack order, protected review, and the
  paused #4279 parent propagation remain unchanged.

## 2026-08-11 #4203 kinetics size-budget repair

- Exact head `572bf525dd1ded26cbc3fbb4f228d1f6ca16e118` passes the PR-base
  changed-file scan because `kinetics.py` is byte-identical to the inherited
  parent, but the stricter `HEAD~1` scan selects the Ruff-formatted file and
  fails it at 646 LOC against the ungrandfathered 500-LOC limit.
- The behavior-preserving LoD/DRY split leaves
  `simulation/kinetics.py` as the stable 222-LOC public façade, moves pure
  dynamics to `_kinetics_dynamics.py` (205 LOC), and moves the immutable
  series/DbC contract to `_kinetics_series.py` (131 LOC). Public constants,
  class/function objects, and the private `_reaction_forces` compatibility
  seam are identity-pinned by a RED-first contract test.
- Physics, units, frames, numerical parity fixture, PyQt presentation, and
  release-stack order are unchanged. This implementation is committed locally
  only; no push, PR mutation, or release claim is authorized by this handoff.
- Verification includes 28 focused kinetics/presentation/PyQt tests and the
  complete 701-test Rate-of-Closure Python suite, all passing locally.

## 2026-08-11 #4203 pinned-Ruff formatting repair

- No material handoff change: this commit only applies the repository-pinned
  Ruff 0.14.10 formatter to the eight Python files named by current-head CI
  Standard run `31468208320`, job `93705508050`.
- Physics, application behavior, public contracts, schemas, UI layout, stack
  bases, and dependency order are unchanged. Fresh protected CI and review at
  the resulting head remain required; queued runner jobs are not green
  evidence.

## 2026-08-10 #4143 child receives repaired #4203 parent

- Ready PR `#4325` retains branch `feat/4143-tee-parity-fixture` and base
  `feat/4181-launch-monitor-registry`; repaired parent head
  `12dd76a8dbcc106c4683f2f2e53076f8dc6f1b76` is incorporated through a
  normal merge commit without rebasing, retargeting, or rewriting history.
- The merge tree has no production/test-code conflict. Both branches'
  append-only handoff/SPEC evidence is retained, including the shared parity
  fixture and deterministic web/PyQt visual evidence.
- Fresh exact-head protected CI and review remain required. #4143 stays open
  until the dependency line lands on `main`.

## 2026-08-10 #4143 Python/React Golden Ball-Setup Parity

- Branch `feat/4143-tee-parity-fixture` starts from exact draft PR #4203 head
  `31cbc007d4c85b5479b7cd0fb0969124eab2af67`; it does not rewrite or retarget
  the release stack.
- `ball_setup_golden_v1.json` is the single Python/React source of truth for
  schema `rate_of_closure.ball_setup_golden` version 1, SI metre units, and
  reference `ground_plane_to_ball_bottom`. It pins Driver/Tee and iron/Ground
  defaults, explicit overrides, Ground's zero effective tee height, derived
  center geometry, serialization, negative and non-finite rejection, and
  legacy-run migration to Ground.
- New consumer tests plus the existing tee suites pass: 18 Python tests and 24
  React tests. Web TypeScript, ESLint, and production Vite build gates pass;
  scoped Ruff check and format pass. This is test/fixture-only and does not
  change production physics or UI behavior.
- Deterministic 1600 x 1200 Playwright captures cover the default Driver/Tee
  and rerun explicit-Ground React states with checked/disabled controls,
  present/absent tee geometry, and zero console/page errors. A hidden 1400 x
  900 PyQt harness records the same states with canonical center and tee-artist
  assertions; a headless regression keeps its temporary PNGs nonblank and
  structurally distinct without pixel-perfect baselines. Artifact manifests
  and SHA-256 digests are under
  `C:\Users\diete\AppData\Local\Temp\rate-4143-visual-evidence-8050eeba`.
- Issue #4143 remains open for exact-head protected CI/review and release to
  `main`. The strict campaign release manifest is not present on #4203's exact
  history (it was introduced later on a divergent campaign branch), so this
  bounded child records that limitation instead of reconstructing it.

## 2026-08-10 Second variation-parent propagation into PR #4281

- Draft PR `#4281` retains branch `feat/4199-wind-scalar-adapter` and base
  `feat/4144-variation-export-continuation`. Exact repaired parent head
  `b90e5021a59e2081415b51ef29fbed06377bc201` is incorporated through a
  normal merge commit without rebase, retarget, force-push, or history rewrite.
- There is no scalar-adapter production/test conflict. Both append-only
  handoff/SPEC histories are retained and the adapter contract is unchanged.
- Parent quality success is not child release evidence. Fresh exact-head CI,
  review, and all earlier dependencies remain open.
- Reconciled-child evidence is 25 focused D-plane/impact tests plus docs
  governance, changed-file size budget, and whitespace checks.

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

## 2026-08-10 Exact variation-parent propagation into PR #4281

Draft PR `#4281` remains on `feat/4199-wind-scalar-adapter` with base
`feat/4144-variation-export-continuation`. The normal merge starts from
original child `cf52529b1e68479321bb93b1be3d59c77f782008` and incorporates exact
published parent `8bcc49fc4e16e5e43be0b7f0f03c3017d5b79d0c` as its second parent. Do
not rebase, retarget, force-push, rewrite either parent, or change draft state.

The child's matching Python/TypeScript `scalar-ensemble/v1` contract and wind
strategy plot adapter remain additive with the parent's selected-scatter
exports, accessible raw tables, result-local linked selection, complete
workspace/toolstrip, playback, navigation, compatibility, and module-budget
repairs. Both adapters now enforce the same deterministic wind-scenario
boundary: exact per-trial provenance, zero shear and turbulence, seed zero, and
no gusts. SPEC 1.14.13 is the unique combined child entry above parent 1.14.12.

Verification is green across 27 focused Python contract/adapter/selection
tests, the complete Rate/shared-swing/golf-club matrix (`1,549 passed`, two
explicit optional build123d skips), and all 556 React tests across 92 files.
TypeScript, zero-warning ESLint, Vite production build, Ruff, Ruff-format,
pinned MyPy 1.13, Bandit, the exact-parent two-file 500-line budget, docs,
minimum-test, assertions, detect-secrets, diff, and CPython 3.10 compilation
pass. Independent re-review found no actionable findings after checking the
exact provenance derivation and all five deterministic-scenario regressions;
protected current-head CI, unresolved threads, and required repository review
remain release gates.

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

## 2026-08-09 PR #4281 scalar-ensemble parent propagation

Draft PR #4281 remains on `feat/4199-wind-scalar-adapter` with unchanged base
`feat/4144-variation-export-continuation`. Exact corrected #4280 parent
`38ed58cab96842a3007e76a855db83ee2452b8fd` is incorporated by the normal
merge containing this handoff; no branch was rebased, retargeted, force-pushed,
or published by this continuation. The result preserves the complete parent
workspace, Python 3.10 compatibility, and variation-export history together
with #4281's independently owned scalar-ensemble and wind-strategy adapters.

The child supplies matching UI-neutral Python/TypeScript
`scalar-ensemble/v1` contracts with stable RFC3986 composite row identity,
immutable nullable rows, and overall/per-cohort availability. Its wind adapter
retains every actual and perfect-information status and exposes true/estimated
wind, launch/aim, target, landing, miss, cost, and information delta. Impact
variables remain explicitly unavailable because the analysis starts at
prescribed launch. SPEC 1.14.12 records this child source delta separately
from parent versions 1.14.11 and earlier.

Current focused evidence is `49 passed` on Python 3.11 and `49 passed` on real
CPython 3.10.20 with PyQt6/scientific dependencies present. React passes `2
files / 10 tests`, TypeScript, and focused zero-warning ESLint. Ruff
check/format passes the four child Python files; pinned mypy 1.13 passes both
production modules and the corrected wind-adapter test. Documentation
governance, ancestry/SPEC assertions, and final diff checks are required in
this same local merge. Protected CI, review, publication, and descendant
propagation remain separate release gates. The background runner, progress and
cancellation, PyQt6/React strategy/scatter workflow, persistence, and export
wiring remain incomplete and must not be represented as delivered.

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
