# AGENT_HANDOFF — rate_of_closure

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-12

## 2026-08-12 Shared Morris screening foundation (#4142 R13.2-R13.4)

The reusable owner `src/shared/python/swing_sim/variation/` now includes a
deterministic, bounded Morris elementary-effects design and analysis contract.
It carries source variable/spec/locus/unit and downstream target
unit/frame/point/time metadata; reports `mu`, `mu*`, `mu*` standard error,
`sigma`, method assumptions, design provenance, adequacy, and complete typed
denominators. Canonical evaluated-hit, evaluated-no-impact, and
numerical-failure wire values are accepted: finite no-impact state metrics can
contribute, but absent impact/shot outputs never become fabricated values.

This branch normally includes exact parent
`feat/4144-variation-export-continuation@7fb5d7f489db49742b7bc82ef009570ad2502456`.
No Rate UI or existing #4280 export logic changes in this slice. A deterministic
JSON-safe report serializer and committed React-consumable golden fixture pin
the cross-runtime contract. The model execution adapter, PyQt6/React views, and
UpstreamDrift integration remain explicit follow-up work. Interpret Morris
`sigma` only as a screening indicator for nonlinearity and/or interaction, not
as separated interaction variance or causality.

## 2026-08-11 Current workspace parent propagation (#4279 → #4280)

PR `#4280` remains on `feat/4144-variation-export-continuation` with base
`feat/4218-toolstrip-workspace`. Exact clean child head
`9b45bd5beca38370c1d541f8c488ef0edad08517` is merged normally, child first,
with exact parent `983805d799b76e5e1ad1dbdc7a5ab28957d805c8`. Variation
scatter CSV parity, typed unavailable outcomes, bounded accessible trial tables,
linked selection, and all-trial arc analysis remain unchanged alongside the
workspace/toolstrip contracts. This remains a pre-manifest stack: the later
strict campaign release manifest artifact/checker/test exists on neither side
and is not recreated. Both histories are retained under a monotonic `1.16.20`
through `1.16.0` sequence. Pinned Ruff `0.14.10` check/format is green across
all five changed Python files; 87 focused Python and 25 focused React tests
pass. React type-check, lint, and production build plus documentation,
minimum-test, changed-file-size, module-size, SPEC-version, and diff gates are
green. Protected checks and parent-first release order remain mandatory.

## 2026-08-11 #4280 receives exact reconciled #4279 parent

PR `#4280` retains branch `feat/4144-variation-export-continuation` and base
`feat/4218-toolstrip-workspace`. A normal merge combines exact published child
`e6c7460a01082631565fb9ed48aa32538bd7772c` with exact published parent
`89af587c8f4141680bb923fc4295e261829f5c75`. All implementation and test paths
merge automatically; only the two append-only handoffs conflict textually, and
both histories are retained. Variation-export behavior is unchanged while the
parent contributes its current workspace, kinetics, solver, layout, and Qt
typing repairs. `SPEC.md` advances monotonically and uniquely to `1.14.20`; the
later campaign manifest is absent from both sides of this pre-manifest stack.
Pinned Ruff 0.14.10 check/format and diff hygiene pass, as do 91 focused
variation/workspace Python tests, 232 inherited workspace/kinetics/impact
Python tests, and 42 React tests across 11 files. The merge is local only;
publication, protected exact-head CI, independent review, unresolved-thread
checks, dependency order, and release remain open.

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
## 2026-08-11 Current registry parent propagation (#4203 → #4279)

PR `#4279` remains on `feat/4218-toolstrip-workspace` with base
`feat/4181-launch-monitor-registry`. Exact clean child head
`89af587c8f4141680bb923fc4295e261829f5c75` is merged normally, child first,
with exact parent `1e29c6e52169de5d984144af29664c0419b51a21`. Workspace
documents, application commands, module visibility/order, primary navigation,
deterministic playback, and independent plots remain unchanged. This remains a
pre-manifest stack: the later strict campaign release manifest
artifact/checker/test exists on neither side and is not recreated. Both
histories are retained under a monotonic `1.15.12` through `1.15.0` sequence.
Pinned Ruff `0.14.10` check/format is green across all 27 changed Python files;
142 focused Python and 32 focused React tests pass. React type-check, lint, and
production build plus documentation, minimum-test, changed-file-size,
module-size, SPEC-version, and diff gates are green. Protected checks and
parent-first release order remain mandatory.

## 2026-08-11 Workspace child receives reconciled #4203 parent

PR #4279 retains `feat/4218-toolstrip-workspace` and base
`feat/4181-launch-monitor-registry`. A normal merge combines exact published
child `efbca84095b617b4018732f7802c2da3f0525387` with exact parent
`9ce2c70f11a15420f0ba2d3b4fef6726b6eacefa`. All implementation merges
automatically; only the two append-only handoffs require reconciliation and
both histories remain. Workspace/toolstrip, navigation, playback, and plot
behavior are unchanged while the parent format repair and split kinetics
ancestry are inherited. Pinned Ruff 0.14.10 passes all five inherited files;
142 focused workspace/plot Python tests, 125 inherited kinetics/impact/registry
tests, and 32 focused React tests pass. Diff checks are clean. Publication,
protected CI, review, unresolved threads, and dependency order remain open.

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
## 2026-08-11 Current D-plane parent propagation (#4202 → #4203)

PR `#4203` remains on `feat/4181-launch-monitor-registry` with base
`feat/4189-dplane`. Exact clean child head
`9ce2c70f11a15420f0ba2d3b4fef6726b6eacefa` is merged normally with exact
parent `9f83cd379ce8ae2805aa4a5608b5645a529f9c3c`. Launch-monitor registries,
analytics, cross-runtime fixture, D-plane ndarray repair, split typed kinetics
façade, and pinned Ruff `0.14.10` files remain unchanged. The strict campaign
release manifest is still absent from this exact history and is not recreated.
Both handoff histories and the parent's post-base SPEC records remain additive
under new monotonic `1.14.x` revisions. Pinned Ruff `0.14.10` check/format is
green across 18 registry, analytics, D-plane, delivery, and kinetics files; 79
focused Python and 31 focused React tests pass. Documentation, minimum-test,
changed-file-size, module-size, SPEC-version, and diff gates are also green.
Protected checks and parent-first release order remain mandatory.

## 2026-08-11 Exact #4202 format repair reconciliation

PR #4203 remains on `feat/4181-launch-monitor-registry` with base
`feat/4189-dplane`. Exact published child
`7abce9ad767fe8311da66a1e5998b892ea3ca9de` is normally merged with exact
parent `ba4aa35cc384d51ed3aa52eb532a67e960669c27`. The append-only handoffs
retain both histories. At the already documented split kinetics seam, the
typed `pendulum.sample(...)` facade call remains authoritative while the
parent's formatted app-frame geometry comment is preserved; the obsolete
`source.inner.sample(...)` monolith is not reintroduced. Physics, numerical
values, frames, and contracts are unchanged. Pinned Ruff 0.14.10 check/format
passes the five inherited Python files. All 81 focused
kinetics/impact/PyQt/layout tests and 44 launch-registry/D-plane/delivery/API
tests pass; diff checks are clean. This merge remains local-only pending normal
publication, protected CI, and review.

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

## 2026-08-10 Variation child receives second repaired workspace parent

PR `#4280` retains branch `feat/4144-variation-export-continuation` and base
`feat/4218-toolstrip-workspace`. It normally merges exact repaired parent head
`61b7f48b5aeb7d57246b4963da3df086e79cbe15` without feature-code conflict or
history rewrite. Variation/export behavior remains unchanged; fresh child CI,
review, and dependency order remain required.
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
## 2026-08-11 pinned-Ruff parent propagation (#4179 → #4202)

Exact clean child head `ba4aa35cc384d51ed3aa52eb532a67e960669c27`
is merged normally with exact parent head
`7e5dfecf569b39dbbf8cc2101c7426cbc53a2771`, while keeping the configured
`feat/4162-wedge-impact-visualization` base. The D-plane ndarray typing repair,
frame-explicit geometry, pinned Ruff `0.14.10` files, and all Rate/wedge/turf
handoff and specification histories remain additive. No runtime or presentation
contract changes. Pinned Ruff `0.14.10` check/format verification and 129
focused D-plane, impact, solver, kinetics, PyQt, and layout tests are green.
Documentation, minimum-test, SPEC-version, and diff gates are also green.
Protected checks and parent-first release order remain mandatory.

## 2026-08-11 PR #4202 pinned-Ruff format repair

CI Standard run `31483390692`, job `93753191911`, checked exact published head
`f3363aa88868f6a5c7e9ccfc682a9eca014e86c1` with the workflow-pinned Ruff
`0.14.10` formatter and identified five changed files requiring mechanical
formatting. They are now formatted with that exact version. There is no
material handoff or runtime behavior change: physics, frames, DbC validation,
public contracts, schemas, tests, and user-visible behavior are unchanged.
The workflow-mirrored scoped Ruff check and format check, `git diff --check`,
and 71 focused impact, kinetics, PyQt, and layout tests are green. Protected
checks and parent-first release order remain mandatory.

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
## 2026-08-11 pinned-Ruff parent propagation (#4178 → #4179)

Exact clean child head `ea7acebf033379d6beefd70eb51027ebd3d01be7`
is merged normally with exact parent head
`188f491ccc88a335ad36afdd66b52289e2e24808`, while keeping the configured
`feat/4166-wedge-turf-physics` base. Parent Ruff `0.14.10` formatting and all
Rate/wedge/turf/visualization handoff and specification histories remain
additive. No runtime or presentation contract changes. Pinned Ruff `0.14.10`
check/format verification and 130 focused impact-scene, solver, kinetics,
PyQt/layout, wedge-clearance, and turf-model tests are green. Documentation,
minimum-test, SPEC-version, and diff gates are also green. Protected checks and
parent-first release order remain mandatory.

## 2026-08-11 PR #4179 pinned-Ruff format repair

Exact published head `ec73b63a748347b42686758d4738c0fd2fd09332`
failed its current CI Standard quality gate because five changed Python files
did not match Ruff `0.14.10`. They are now mechanically formatted with the
workflow-pinned version. There is no material handoff or behavior change:
impact visualization, wedge/turf physics, frames, DbC validation, public
contracts, schemas, tests, and UI behavior remain unchanged. Protected checks
and parent-first release order still apply.

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
## 2026-08-11 pinned-Ruff parent propagation (#4174 → #4178)

Exact clean child head `ca567fe7d3fa48b1900ad3098045f4200cfe86a7`
is merged normally with exact parent head
`3e1b44cf42f4c0838149e0bc8e88ce4cb79b72b0`, while keeping the configured
`feat/4161-wedge-ground-clearance` base. Parent Ruff `0.14.10` formatting and
all Rate/wedge/turf handoff and specification histories remain additive. No
runtime contract changes. Workflow-pinned Ruff check/format, 127 focused turf/
wedge/impact/kinetics/PyQt/layout tests, and documentation, minimum-test,
SPEC-version, and diff gates are green. Protected checks and parent-first
release order remain mandatory.

## 2026-08-11 pinned-Ruff parent propagation (#4173 → #4174)

Exact child head `01ecf9a7b1922d1609fb99093226799a0b564704` is
merged normally with exact parent `#4173` head
`bd48852d303db6281ed5891d4a271d99e76a94e6`, while keeping the configured
`feat/4163-impact-inspector` base. Parent Ruff `0.14.10` formatting and all
prior handoff/spec history remain additive. No Rate or wedge runtime contract
changes. Workflow-pinned Ruff check/format, 98 focused impact/kinetics/wedge/
PyQt/layout tests, and repository documentation, minimum-test, SPEC-version,
and diff gates are green. Protected checks and parent-first release order
remain mandatory.
## 2026-08-11 PR #4178 pinned-Ruff format repair

Exact published head `b8822401f4522e867d6b160125953981a39a770d`
failed its current CI Standard quality gate because five changed Python files
did not match Ruff `0.14.10`. They are now mechanically formatted with the
workflow-pinned version. There is no material handoff or behavior change:
turf/impact physics, frames, calibration boundaries, DbC validation, public
contracts, schemas, tests, and UI behavior remain unchanged. Protected checks
and parent-first release order still apply. Workflow-mirrored Ruff,
`git diff --check`, and 71 focused impact, kinetics, PyQt, and layout tests are
green.

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

## 2026-08-11 PR #4174 pinned-Ruff format repair

Exact published head `525696e0c1080616eb5055e2cb1c93565f98672e`
failed CI Standard run `31485402975`, job `93759519460`, because five changed
Python files did not match Ruff `0.14.10`. They are now mechanically formatted
with the workflow-pinned version. There is no material handoff or behavior
change: physics, frames, DbC validation, public contracts, schemas, tests, and
UI behavior remain unchanged. Workflow-mirrored Ruff, `git diff --check`, and
71 focused impact, kinetics, PyQt, and layout tests are green. Protected checks
and parent-first release order still apply.

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

## 2026-08-11 pinned-Ruff repair

Exact published head `3c19aaa9d3e812e4659053735a2955d62a080d34`
inherits the five-file Ruff `0.14.10` format mismatch reported on its immediate
child. The files are now mechanically formatted with that pinned version. No
material handoff or runtime behavior changes: variation, physics, frames, DbC
validation, public contracts, schemas, tests, and UI behavior remain intact.

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
