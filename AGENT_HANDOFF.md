# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-13

## 2026-08-13 Integrated dispersion and localized-locus/browser stack (#4142)

Version 1.16.64 normally merges approved dispersion head
`71634bf7393c8343a53f9acaa9f4db76cb4ac8db` as first parent with published
localized-locus/browser head `393f80e8e6b7ebcc7207136aa8a7aa47899a6eda`
as second parent. Both append-only histories, implementations, workflows, and
feature tests remain present without rebase or rewrite. The one unexpected
non-document conflict retained the locus persistence cases and the newer
dispersion analysis block; two stale assertions in the locus-split analysis
file now follow the approved metric-generic accessible names.

Integrated local evidence is 338 combined dispersion/PyQt/shared variation
tests, all 841 React tests, seven workflow/runner-policy tests, and the exact
23-source Python 3.12/Mypy 1.13 cumulative delta. Ruff/format, TypeScript,
ESLint, Vite production build, documentation governance, and the 500-line
changed-file gate pass, as do all five production-Worker Chromium checks.

Protected publication remains open, as do plot-definition import UI, full
ellipsoid meshes, WebKit/Firefox and assistive-technology coverage, approved
visual baselines, React localized execution/results/export, Rust parity,
complete persistence, and the remaining #4142 scope.

## 2026-08-13 Plot-definition compatibility/static closure (#4142 R12.1/R12.2)

The strict v2 contract now preserves one historically emitted v1 form without
weakening current applicability: scalar-scatter and distribution-matrix v1
documents may carry the exact `APP_FRAME_ID`, which migration normalizes to
null; any other legacy frame still fails closed. Authentic Python and
TypeScript v1 fixtures pin both acceptance and rejection. Python dictionary
serialization now emits `variable_keys` as a JSON list, so the returned
document round-trips directly through the strict reader.

PyQt dispersion export kwargs use a precise `TypedDict`, closing the pinned
Python 3.12/Mypy 1.13 changed-source gate. Migration logic and the plot-
definition contract tests were split into focused modules; every affected
production and test module remains below 400 lines. Evidence is 1,163 Rate
Python/PyQt and 804 React tests, including focused 70-case Python and 58-case
TypeScript contract suites, plus Ruff and exact hosted-toolchain Mypy. SPEC is
1.16.63. Import UI, ellipsoid meshes, cross-browser E2E, protected publication,
and #4142 completion remain open.

## 2026-08-12 Plot-definition complete-domain hardening (#4142 R12.1/R12.2)

Plot definitions now use one explicit applicability matrix on both runtimes.
Scalar scatter accepts its x/y keys and selected-trial state; distribution
matrix accepts only its variable-key list; geometric plots accept only their
declared point/frame/unit/alignment, dispersion, filter, and applicable camera
state. Every inapplicable field must be null, and geometric variable keys are
therefore impossible. Geometric definitions require the exact current
`APP_FRAME_ID`, not an arbitrary non-empty frame label.

All persisted IDs and variable keys reject C0/C1/DEL controls in addition to
whitespace instability. Python direct constructors normalize supported finite
`Real`/`Integral` values, including NumPy and `Fraction` cases, to built-in
JSON-safe float/int values; strict readers still reject non-JSON object-domain
numerics. PyQt and React exporters no longer attach a coordinate frame to
non-geometric plots, and v1 migration rejects contradictory legacy state.
Evidence is 1,160/1,160 Rate Python/PyQt tests, 802/802 React tests, and focused
Python/TypeScript contract tests, plus Ruff/format, scoped MyPy, TypeScript,
ESLint, and the production web build. SPEC is 1.16.62. This does not add
plot-definition import UI,
ellipsoid meshes, cross-browser E2E, protected publication, or #4142 closure.

## 2026-08-12 Dispersion plot-definition closure (#4142 R12.1/R12.2)

Python and React now enforce the same complete plot-definition domain at both
construction and write time. Exact plot kinds and stable trimmed identifiers,
genuine non-Boolean trial indices, finite camera angles/zoom, bounded pitch and
phase, canonical outcomes, geometric SI/frame-alignment declarations, and
source/band relationships fail closed before serialization. Python revalidates
then uses `allow_nan=False`; React reparses the typed object before
`JSON.stringify`, preventing JavaScript from silently converting NaN or
infinity to null. Exact v1 migration outputs satisfy these stronger v2
invariants.

React timeline copy now states the true persistence boundary: adequacy counts
and ranked intervals are calculated results from the loaded ensemble, while
only their selection criteria persist in a plot definition. Local evidence is
1,138/1,138 Rate Python/PyQt tests, 786/786 React tests, the production web
build, Ruff, MyPy, TypeScript, ESLint, and secret scanning. SPEC is 1.16.61.
This correction does not add a plot-definition import UI, confidence-ellipsoid
mesh, cross-browser E2E, protected publication, or epic-completion claim.

## 2026-08-12 Dispersion consumer review hardening (#4142 R12.1/R12.2)

Point-specific PyQt quiet intervals now dense-rank only within the selected
modeled point, matching React even when the shared criteria originally names
multiple points. A multi-point Python/TypeScript golden regression prevents
other points from shifting the displayed ranks. React replaces the bounded
erfc approximation with a regularized-gamma, bracketed chi-square inversion
validated against SciPy from the declared `1e-12` lower confidence boundary to
a near-one upper-tail case, including confidence radius and unit-covariance
volume.

Plot-definition readers now accept exact v2 documents or strictly migrate exact
v1 documents. V1 geometric plots become RMS-radius/m definitions, preserve a
positive legacy threshold, use 0.005 m when the legacy threshold is null, and
default to zero minimum duration and one minimum sample; non-geometric fields
remain null. Unknown, omitted, nonfinite, and coercively typed fields fail
closed. New PyQt dispersion controls have explicit accessible names and label
buddies. SPEC is 1.16.60; no ellipsoid mesh, cross-browser E2E, publication, or
epic-completion claim is added.

## 2026-08-12 Dispersion-metric visualization consumers (#4142 R12.1/R12.2)

PyQt6 and React variation geometry now select the shared RMS radius, largest
principal sigma, or Gaussian confidence-ellipsoid volume authority. Controls
persist metric, SI threshold/unit, applicable confidence, minimum duration, and
minimum samples in plot-definition schema v2 while presenting length and volume
as readable mm and mm³. Both surfaces report estimable, rank-deficient,
insufficient, invalid, and unavailable sample counts plus dense-ranked quiet
intervals. Confidence is enabled only for volume and is explicitly described as
Gaussian position content from plug-in sample covariance, not uncertainty in
the population mean.

React mirrors the Python authority against one Python-owned golden fixture and
fails closed on unequal time grids or nonfinite coordinates instead of
truncating. The existing sparse yellow glyphs remain labeled as 2σ principal-
axis indicators; this slice does not claim a rendered confidence-ellipsoid mesh,
cross-browser E2E coverage, protected publication, or epic completion.

## 2026-08-12 PR #4414 hosted MyPy hardening (#4142, 1.16.63)

The localized-locus UI now narrows a nullable variable key before querying the
stable variable-to-joint mapping and returns the locus editor's declared
Boolean result directly instead of applying a redundant type cast. These are
type-boundary corrections only; variable selection, locus visibility, joint
identity, authoring precision, and runtime behavior are unchanged.

Local evidence is the exact PR-base set of 15 changed source files under pinned
MyPy 1.13.0 with redundant-cast warnings enabled, plus seven focused PyQt locus
tests, Ruff, format, documentation governance, diff, and changed-file size
checks. Protected current-head CI and ordinary publication remain open.

## 2026-08-12 Integrated localized locus and Playwright browser stack (#4142)

Normal merge version 1.16.62 preserves localized-locus head
`05d9d9bba22940b738d1d3d447ca5ab95642511d` as first parent and published
browser head `8bcd055f5711c122ec5332b8da8c41d6a974dfcb` as second parent. The merge
retains both implementations byte-for-byte; only the four durable handoff/spec
documents are reconciled. Strict localized execution/authoring, Python/React
wire parity, and the 400-line policy coexist with the trust-separated real
production-Worker Playwright gates.

The browser history retains hosted-only PR CI, main-push-only trusted CI,
immutable action pins, real hashed-Worker progress/cancellation/rerun checks,
and responsive desktop/narrow Chromium coverage. This remains an R14.5
foundation, not complete certification: protected runner evidence, WebKit,
Firefox, assistive-technology automation, PyQt E2E, approved visual baselines,
React localized results/export, Rust parity, complete raw persistence, protected
publication, and epic completion remain open.

## 2026-08-12 Integrated localized torque and Playwright stack (#4142, 1.16.59)

This branch now preserves the complete histories of localized-torque head
`10524cc2151c7b60c4a097939b29202158aff012` and reviewed Playwright head
`6df0ed09388ba36630c5fc6be7a31a334a4b6243` in a normal two-parent merge.
The localized Python execution/validation contracts and the Rate-web production
Worker/browser gates coexist without changing either scientific or browser
contract. Publication, protected CI, remaining localized UI/Rust/persistence
work, and full R14.5 visual/browser certification remain open.
Integrated local evidence is 171/171 localized changed-test cases, 18/18
Playwright workflow/security tests, and 5/5 real Chromium tests, plus scoped
Ruff, Ruff format, documentation governance, workflow validation, and diff
hygiene.

## 2026-08-12 Real-browser variation Worker foundation (#4142 R14.5)

Local branch `codex/4142-rate-web-playwright` starts from exact integrated
commit `11a699155588d3d948990c5f08b72c5cc8d2c746`. The Rate web package pins
Playwright Test 1.62.1 in its own manifest/lock and owns a deterministic
Chromium configuration across two path-filtered workflows. Every PR runs only
on ephemeral `ubuntu-latest`; the PR YAML has no fleet/self-hosted reference.
The separate trusted workflow runs only for pushes to `main`, checks out the
push event commit, and has neither a PR nor manual-dispatch ref seam. Checkout,
Node setup, and artifact upload actions are pinned to full immutable SHAs.
Artifact names include the workflow run and attempt IDs.

The gate builds and serves the Vite production bundle, then uses role/label
locators against the actual bundled module Worker. It observes strict
intermediate and terminal progress during a seeded 24-run study and proves a
deterministic rerun. Cancellation of a 500-run swing/OAT job observes actual
Worker termination before two identical seeded reruns, proving the cancelled
generation cannot publish a partial or late result. Navigation terminates
active work on primary-tab unmount, and every case rejects browser page errors.
Blocking service workers in the
test context does not replace or disable the dedicated module Worker; every
lifecycle case observes the hashed production Worker chunk. Desktop 1440x1000
and narrow 390x844 checks enforce zero document-level horizontal overflow and
attach full-page screenshots to the retained Playwright report.

This is a narrow R14.5 foundation, not R14.5 completion or complete visual
certification. It covers bundled Chromium only; screenshots are review
artifacts, not cross-platform golden baselines. WebKit, Firefox, assistive-
technology automation, PyQt interaction, protected runner evidence, and a
CI-authority visual baseline remain open. Local evidence is 5/5 Playwright and
743/743 Vitest tests, plus TypeScript, ESLint, and the Vite production build.

## 2026-08-12 Localized torque identity and 400-line policy closure (#4142)

Python and React plan readers now reject coercive identity fields. Discriminator
text must be a real string; spec/group/point/member IDs must also be nonempty,
trimmed, C0/C1-control-free stable strings, and ID collections must be real
arrays with unique entries. Numeric, scalar-string, control-bearing, and
duplicate stand-ins fail before plan construction.

PyQt worker lifecycle, registry mode policy, PyQt GUI test concerns, and React
analysis tests now live in focused modules. Every cumulative changed Python/TS/
TSX source or test is <=400 lines; official 500-line and explicit 400-line gates
pass. Evidence: 190 focused Python/PyQt/core tests, 780 full React tests, TS
type/lint/build, Ruff/format, 15-file MyPy, docs-governance, and diff/size checks.
React localized execution/results/export, Rust parity, raw persistence, visual
E2E, protected publication, and epic completion remain open.

## 2026-08-12 Localized torque authoring review hardening (#4142)

Independent review corrections are complete. Focused PyQt editor helpers own
locus controls and Variation-tab row operations, reducing the changed
`variation_tab.py` and `variation_rows.py` modules to 482 and 292 lines under
the official 500-line gate. Locus endpoints retain per-field exact authority:
editing only one endpoint preserves the other endpoint's full imported value.

React v2 decoding now rejects coercive numeric wire values before construction.
Schema versions, scales, bounds, base values, windows, run counts, seeds, and
correlation entries require strict finite numbers; integer fields additionally
require integers. Evidence: 173 focused Python/PyQt/core tests, 763 full React
tests, TypeScript type/lint/build, Ruff/format, changed-source MyPy, official
file-size, docs-governance, and diff gates. React localized execution/results/
export, Rust parity, complete raw persistence, visual E2E, protected publication,
and epic completion remain open.

## 2026-08-12 Localized torque authoring parity (#4142)

PyQt and React can now author the two registered localized commanded-torque
variables. PyQt exposes them only in swing mode with a double-pendulum source
and bounds windows to the effective RK4 duration. React exposes them only in
its fixed 1.5 s double-pendulum swing workflow. Each row has finite half-open
start/end controls and one disabled topological selector fixed to
`joint.shoulder` or `joint.wrist`; tooltips explicitly distinguish these from
spatial `swing.*` trace IDs. Global rows keep their prior compact layout.

Load/edit/save/import retains custom spec IDs, exact high-precision locus and
scale values, groups, and unrelated plan fields. Variable changes reset the
locus atomically. Missing, reversed, off-duration, and mismatched loci fail
visibly before execution or storage mutation. A shared v2 fixture is consumed
by Python and TypeScript tests. Evidence: 49 focused Python/PyQt/core tests,
752 full React tests, TS type/lint/build, Ruff/format, changed-source MyPy, and
diff checks. React localized dynamics/results/export remain fail-closed; Rust
parity, complete raw state/event/torque persistence, visual E2E, protected
publication, and epic completion remain open.

## 2026-08-12 Localized torque static-gate closure (#4142)

The final cumulative changed-source MyPy blockers are closed without runtime
changes: `dataset_io.read_csv` explicitly types its NumPy input/success arrays,
and the Rate pipeline removes a redundant `SwingSource` cast around the already
typed source factory return. The source factory also relies on the validated
`DoublePendulumRunConfig | None` narrowing instead of recasting its non-`None`
branch. SPEC change-log rows 1.16.55 through 1.16.58 restore the monotonic audit
trail without replacing any mission text. The exact pinned Python 3.12 / MyPy
1.13 16-file delta command, 147 focused localized tests, Ruff, formatting, and
diff checks pass. UI, Rust, protected-publication, and #4142 completion gates
remain open.

## 2026-08-12 Source execution/dataset discriminator hardening (#4142)

The source factory no longer uses `run_config or default`. It validates the
raw value as `None` or `DoublePendulumRunConfig`, then defaults only the exact
`None` case. Manual and triple-pendulum sources reject prescribed mode/profile,
joint locks, and localized offsets while preserving explicit or implicit
default passive empty execution.

The outer variation-dataset JSON schema version now requires a genuine
non-Boolean integer before normalization. Boolean, float, and string lookalikes
fail closed, consistent with the strict nested plan and sibling Morris reader
contracts. Evidence is 34/34 focused and 1,483/1,483 broader shared-swing,
variation, and Rate tests, with one expected missing-Rust-wheel skip. UI, Rust,
protected-publication, and #4142 completion gates remain open.

## 2026-08-12 Localized torque source/wire hardening (#4142)

The Rate source factory now enforces the same double-pendulum-only capability
already declared by `SimulationConfig`: manual and triple-pendulum source
discriminators reject non-empty localized torque commands instead of silently
discarding them. `DoublePendulumRunConfig` validates the raw command collection
before tuple normalization, guaranteeing typed contract failures for `None`
and other malformed collection domains.

`VariationPlan.from_json_dict` no longer coerces its schema discriminator with
`int(...)`; only a genuine non-Boolean integer may select supported v1/v2
behavior. Regression evidence is 102/102 focused and 1,464/1,464 broader
shared-swing, variation, and Rate tests, with one expected missing-Rust-wheel
skip. The broader UI, Rust, protected-publication, and #4142 completion gates
remain open.

## 2026-08-12 Localized torque adversarial corrections (#4142)

The affected localized variation and helper seams now fail closed on their raw
numeric domains. `NoiseSpec`/`VariationPlan` fields reject Boolean, string, and
nonfinite values without coercion while preserving ordinary JSON integer/float
plans and v1 migration. Localized helper functions validate command
collections, base torque pairs, sample times, and durations with typed contract
errors.

One canonical fixed-step helper computes the effective RK4 duration used by
Rate request validation, `SimulationConfig`, source construction, and fallback
trace-grid construction. A locus inside the requested duration but outside the
rounded integration grid is therefore rejected during request construction,
not during a trial. The current PyQt variable picker hides
`localized_torque_only` entries until a locus editor exists; loading such a
plan remains fail-closed and atomic with an explicit locus-editor message.

Local evidence is 118/118 correction-focused tests and 1,455/1,455 broader
shared-swing/variation and Rate tests, with one expected missing-Rust-wheel
skip. PyQt/React locus authoring, Rust parity, complete raw persistence,
protected publication, and #4142 completion remain open.

## 2026-08-12 Localized double-pendulum torque execution (#4142)

Local child `codex/4142-localized-double-torque-core` starts from exact commit
`11a699155588d3d948990c5f08b72c5cc8d2c746`. It implements the first bounded
localized-perturbation execution path without widening the UI surface.

- Immutable `LocalizedTorqueOffset` commands use only the topological IDs
  `joint.shoulder` and `joint.wrist`, a required finite half-open
  `time_window_s = [start, end)`, and a finite additive torque in N.m. These IDs
  remain deliberately distinct from spatial trace points such as
  `swing.wrist`.
- Passive and prescribed double-pendulum runs add every active command at each
  Python RK4 stage. Recorded joint-torque samples use the same half-open rule;
  exact shared boundaries cannot double-apply.
- Rate variation requests map the two registered commanded-torque variables to
  exact one-point loci and deterministic pre-sampled values. Missing, multiple,
  mismatched, out-of-duration, base-only, wrong-source, and unsupported
  localized specifications fail before trial execution.
- Explicit Rust execution fails closed. `auto` selects the Python forced path
  when localized commands are present. Valid misses remain typed
  `evaluated_no_impact` results with closest-approach evidence.

Exact local evidence is 99/99 focused tests and 1,413/1,413 broader shared-
swing/variation and Rate tests (one expected Rust-wheel skip), plus Ruff, Ruff
format, and changed-source MyPy. This is a narrow core seam: PyQt/React locus
authoring and presentation, other source/locus kinds, Rust parity, complete
state/event/torque persistence, protected CI/publication, and epic completion
remain open.

## 2026-08-12 Bounded ensemble chunk lifecycle foundation (#4142 R11.5)

Local child `codex/4142-ensemble-chunks` starts from published #4405 head
`2c923fdd94ede6064cffe4847cbb56088cd78896`. It introduces an in-process,
immutable `EnsembleStreamHeader`/`SimulationResultChunk`/`EnsembleChunkSink`
lifecycle and refactors the existing public complete-ensemble runner through a
compatibility collector.

- Execution retains at most one configured chunk of complete `SimulationRun`
  captures before projection, rather than every run until the study ends.
- Chunk rows are non-empty, contiguous, canonically indexed, resource-bounded,
  immutable, and bound to the header's exact sampled-input rows plus typed
  outcome/trace/impact availability. Scientific arrays require real numeric
  domains; Boolean validity, representable integer impacts, the canonical app
  frame, and input/position cell ceilings are enforced before conversion.
- Cancellation is checked before and after each solver call and before sink
  acceptance. Sink acceptance is provisional; only commit returns authority,
  while cancellation, executor errors, or sink errors abort exactly once.
- Progress counts the accepted canonical prefix. Chunk sizes 1/2/3/>n are
  scientifically equivalent to the compatibility façade apart from elapsed
  wall time.

The collector intentionally still materializes the final four-dimensional trace
tensor, and request sampling/config construction remains eager. This is the
R11.5 execution seam, not completion of streaming persistence: a bounded source,
durable chunk archive, resume/checksum policy, full event/state/torque rows, and
measured execution-memory gate remain open. Exact local evidence is 55/55
focused lifecycle/adapter tests and 330/330 broader Rate/shared-variation tests,
plus the hosted Python 3.12 / NumPy 2.3.5 / Mypy 1.13 combination, Ruff, and
Ruff format.

## 2026-08-12 Integrated variation persistence, dispersion, and React execution (#4142)

Protected PR #4405 initially failed only its hosted `quality-gate` Mypy step:
the Linux Python 3.12 / NumPy 2.3.5 stubs are stricter than the development
runtime. NumPy array-return/allocation boundaries now carry explicit annotations
or casts, and float epsilon/tiny values are normalized before arithmetic. The
exact hosted combination (`mypy==1.13.0`, `numpy==2.3.5`, Python 3.12) passes all
nine changed production modules locally; no numerical behavior or wire shape
changed. One focused normal push is required to start exact-head protected CI.

Final independent review found and closed the last typed/wire-domain asymmetry:
`SimulationTrialOutcome` now rejects booleans and non-real scalar values, turns
accepted NumPy real scalars into finite built-in floats, and therefore guarantees
that every constructed complete outcome can cross the strict JSON writer/reader
boundary. Five new TDD cases and all 34 reader cases pass (39/39 focused).

Local branch `codex/4142-react-mc-async-integrated` is based on exact #4404
head `82e4c54c921f169227d25ece2935add4af3e721a`. It integrates the strict typed
ensemble reader/writer, confidence-scaled dispersion metrics, and asynchronous
React Monte Carlo execution plus all three independent-review hardening passes.

- One shared limit contract now governs typed Rate results and parsed archives.
  Typed results bind canonical columns, scalar outcomes, success, partial or
  unavailable trace status, impact markers, and impact-time provenance before
  serialization.
- Raw sample counts and every nested tensor axis are checked before the
  corresponding NumPy allocation. Strict finite JSON and the exact formatted
  UTF-8 byte count are preflighted before file creation.
- Deep JSON recursion, oversized JSON integers, Unicode, and normal syntax
  failures become public contract errors. Boundary tests cover each scientific
  limit, allocation order, crossed typed evidence, and writer preflight.
- Confidence ellipsoids reject invalid eigensystems and use a stable df=3
  chi-square quantile. Quiet-zone ranking supports RMS radius, largest principal
  sigma, and ellipsoid volume with explicit adequacy and deterministic ties.
- React variation runs in a per-study worker with progress, cancellation,
  immediate rerun, stale-generation protection, request/result validation, and
  single-settlement cleanup for every browser-worker failure boundary.
- This remains exact outer v1 persistence. Rejection of unknown versions is a
  fail-closed future-migration policy, not an implemented migration.

Exact integrated local evidence is 1,200/1,200 Python/PyQt/shared tests and
743/743 React tests, plus Ruff, Ruff format, CI-pinned Mypy 1.13, TypeScript,
ESLint, Vite production build, documentation governance, diff, assertion, and
changed-file size gates. The explicit Python `float` boundary on the NumPy
epsilon tolerance is typing-only; scientific and runtime behavior are
unchanged. Protected publication, UI import and dispersion controls,
cross-runtime reading, chunking, event ledgers, complete state/torque authority,
localized perturbation execution, and Playwright/screenshot gates remain open.

## 2026-08-12 Strict typed Rate ensemble reader (#4142 R11.4)

Branch `codex/4142-typed-ensemble-reader-integrated` starts from exact current
#4404 head `82e4c54c921f169227d25ece2935add4af3e721a` and remains local/unpublished.
It introduced a strict Python reader for the existing complete Rate ensemble
JSON writer without changing the version-1 wire representation; the symmetric
writer/type hardening is recorded in the newer entry above.

- Exact parsing preserves the complete plan-v2 graph, stable spec/group IDs,
  seed and sampled inputs, canonical trial indices, typed hit/no-impact/failure
  outcomes, scalar availability, point/frame/unit IDs, sample validity, impact
  markers, and all position traces.
- The reader rejects unknown/duplicate fields, coercible booleans or strings,
  nonfinite values, invalid UTF-8/truncated JSON, crossed outcome/scalar/success/
  impact evidence, corrupt axes, and noncanonical ordering. Impact indices are
  bound to typed status and the nearest recorded impact-time sample.
- External JSON is capped at 16,000,000 UTF-8 bytes. Decoded depth/nodes,
  trials, samples, points, and position cells have named pre-materialization
  bounds. Parsed arrays are owned and read-only; shared `VariationDataset`
  construction now provides the same immutable ownership everywhere.
- Migration policy is fail closed: outer ensemble v1 with exact embedded plan
  v2 only. A future schema must provide an explicit reviewed migration.

The complete local Rate plus shared-variation gate is 1,157 passing tests (15
known warnings), with Ruff and MyPy green. Final diff/size gates, independent
integration review, protected publication, browser/PyQt import surfaces,
streaming/chunking, event ledgers, and complete state/torque authority remain
open.

## 2026-08-12 Confidence-scaled dispersion and quiet metrics (#4142 R12.1/R12.2)

- The shared UI-neutral geometry layer now exposes immutable, plot-ready 3D
  Gaussian position-content ellipsoids at any declared confidence level. Axis
  lengths use the exact three-degree-of-freedom chi-square quantile and the
  existing unbiased sample covariance; the contract explicitly distinguishes
  this from a confidence region for the unknown population mean.
- Every time sample declares `estimable`, `rank-deficient`,
  `insufficient-samples`, or `invalid-covariance`. Full ellipsoid volume is
  available only with at least four valid trials and three positive principal
  variances; unavailable volume remains `NaN`.
- Quiet-zone detection can select RMS radius, largest principal sigma, or
  confidence-ellipsoid volume with explicit units. Intervals score as
  mean/threshold, sort deterministically by score then stable point/time keys,
  and exactly equal scores share a dense rank.
- Scientific adversarial review is resolved: only finite, descending,
  positive-semidefinite eigenvalues with orthonormal axes that reconstruct a
  finite symmetric covariance can supply plot geometry. Scale-aware numerical
  roundoff below zero becomes a zero-variance direction; materially invalid,
  unordered, inconsistent, or nonfinite evidence is unavailable and cannot
  qualify as quiet.
- Chi-square inversion uses SciPy's regularized-gamma inverse and remains
  accurate through the representable upper probability tail; the public
  supported domain is `[1e-12, 1)`. Criteria now
  accept only real, non-boolean values, normalize NumPy real scalars to Python
  floats, and reject malformed point IDs through the contract boundary.
- Local evidence is 27 focused tests within 189 passing scientific tests. The
  1,184-test shared-variation/full-Rate gate passed 1,183 tests with 29 known
  warnings; its one Morris child readiness timeout passed immediately in the
  permitted isolated retry. Scoped Ruff, format, and MyPy are green. PyQt/React
  controls, rendering, serialized cross-runtime fixtures, protected
  CI/publication, and #4142 epic completion remain open.

## 2026-08-12 React worker transport hardening (#4142 R14.3)

The React Monte Carlo worker client now treats the worker boundary as untrusted
runtime input. A single-settlement lifecycle terminates the worker and removes
abort/message/error handlers after success, cancellation, decoding failure,
worker failure, malformed messages, invalid progress, invalid result structure,
or a synchronous `postMessage` clone failure. Late events are inert.

Progress must advance by exactly one completed evaluation, retain the planned
total, and follow the joint-then-individual phase order. Returned plans,
datasets, sensitivity matrices, and swing-ensemble envelopes are validated
against the initiating request before results are accepted. The execution entry
point also validates the complete plan before OAT work, preserving the browser
run bound for injected and worker callers.

Direct production-transport unit coverage uses an injected Worker factory to
exercise progress/result completion, abort and late-event safety, worker and
message decoding errors, malformed progress/results, and `DataCloneError`
cleanup. This is deterministic transport coverage, not browser/Playwright or
screenshot evidence; those remain an explicit R14.5 release gate. This slice
does not complete #4142 or authorize the UpstreamDrift consumer pin.

## 2026-08-12 React Monte Carlo worker execution (#4142 R14.3)

The React Variation workspace no longer evaluates Monte Carlo studies in the
click handler. Production browsers create one bounded module worker per study;
the worker runs the unchanged seeded joint and OAT algorithms and reports
determinate progress only after each model evaluation finishes. Run exposes a
busy state and accessible progress meter, while Cancel terminates the worker,
discards partial results, and permits an immediate rerun.

An injected execution-service contract makes lifecycle behavior testable without
changing the physical plan or result schemas. Abort signals, monotonic generation
IDs, and unmount cleanup prevent a cancelled, superseded, or detached job from
updating accepted results. Configuration and workflow changes invalidate active
work. The same plan and seed produce the same datasets and sensitivity matrices
as the prior synchronous authority.

Verification: all 733 React tests pass, including injected-service component
coverage for run/progress, cancel/rerun, stale-generation suppression, and
unmount abort. TypeScript, ESLint, and the Vite production build pass; the build
emits the dedicated variation worker chunk. Browser/Playwright interaction and
screenshot coverage remain an explicit R14.5 release gate. This slice does not
complete #4142 or authorize the UpstreamDrift consumer pin.

## 2026-08-12 Integrated authority cross-review hardening (#4142)

Protected #4404 CI found that Mypy 1.13 could not infer the dtype of the two
new Morris observation `values` allocations. Both allocations now carry an
explicit `np.ndarray` annotation; the authority contract and wire output are
unchanged. Re-run the exact quality gate at the new head before relying on the
previous local green evidence.

Independent adversarial review of the combined raw-authority, PyQt plan-v2,
and pairwise-finite attribution head identified and drove closure of precision,
evidence-binding, allocation-order, unavailable-dominance, and cross-runtime
normalization defects. The final local gate is 355 Python/PyQt tests and 728
React tests, with production build, TypeScript, ESLint, Ruff, MyPy, diff, and
changed-file size checks green. Protected current-head CI and normal stacked
publication remain required; the broader open R10-R14 work is not complete.

## 2026-08-12 Pairwise-finite OAT/Spearman parity (#4142 R13.1)

- Python and React now select every input/output pair independently from
  evaluated finite observations. Misses, failures, and unavailable downstream
  cells cannot fabricate a zero/rank or invalidate another measurable pair.
- Spearman requires three paired observations and nonconstant paired columns;
  OAT spread requires two evaluated finite values per output. Unavailable
  statistics remain explicit `NaN`, cannot become a dominant input, and remain
  distinct from a genuinely finite zero-sensitivity column.
- Both runtimes consume one shared fixture covering failures, independent gaps,
  positive/negative monotonic pairs, constants, and insufficient counts.

This closes the missing-value correctness defect only. Protected publication,
localized attribution, complete raw trace authority, and #4142 remain open.

## 2026-08-12 Lossless PyQt variation-plan v2 round trip (#4142)

The integrated PyQt plan editor retains the full shared version-2 authority:
custom spec IDs, temporal/spatial locus metadata, exact unedited numeric values,
and correlation/covariance groups survive build and Save Plan. Visible numeric
edits preserve stable identity/locus fields, while unrelated selector edits do
not round any untouched numeric authority through visible controls. Load Plan preflights selectors,
registry keys, flight models, and numeric ranges before mutating the editor, so
unsupported plans fail closed with the prior runnable state intact.

Focused source evidence on the isolated slice was 161 PyQt/shared-variation/
request tests. Group matrices and loci remain retained application authority,
not editable controls. This slice does not complete #4142 or authorize an
UpstreamDrift pin before the protected Tools dependency chain lands.

## 2026-08-12 Raw Morris scalar-evidence foundation (#4142 R11)

Branch `codex/4142-morris-observation-authority` starts from exact reviewed
workspace head `ee4dfecb5e0acd1c8acd1a85d68c4d3b14113408`. It preserves the
unchanged `morris-global-sensitivity-report@1` response while adding a separate
`swing-sim/morris-observation-archive@1` authority contract.

- Every design point retains canonical ordinal, trajectory/point coordinates,
  a design-bound SHA-256 sample ID, declared physical factor values and units,
  typed hit/no-impact/numerical-failure status, every declared output with null
  availability, and bounded failure type/message diagnostics.
- The exact parser rejects schema drift, crossed identities, reordered records,
  fabricated no-impact outputs, nonfinite or altered design data, and incomplete
  diagnostics. It rejects designs above 100,000 samples and output matrices
  above 1,000,000 cells before parsing outputs or allocating observation arrays;
  archive factories enforce the same bounds. Parsed arrays are owned and
  read-only, and archive construction rejects incomplete hit outputs.
- The public Rate service still returns the unchanged aggregate report. Its
  explicit extended path returns raw scalar authority too; the job registry
  retains it only for completed jobs under a weighted cell budget without
  enlarging the existing job-envelope wire contract. Completion recomputes the
  aggregate report outside the registry mutex, then rejects crossed evidence.
- End-user raw transport/export remains open and must be separately bounded and
  connected to both clients before the authority is called UI-discoverable.
- This scalar-evidence layer is not complete R11.1 authority: full event ledgers,
  impact/shot objects, and complete pre-impact state/torque traces remain open.

Verification: 320 shared-variation and Morris application/PyQt tests pass;
focused Ruff, format, and MyPy pass. Protected CI, independent review,
publication, transport/UI integration, and UpstreamDrift remain release gates.

## 2026-08-12 Lossless Morris workspace persistence/export (#4142 R13.8)

Branch `codex/4142-morris-workspace-integration` combines exact independently
reviewed Python/PyQt commit `8968f6f3544203029fea8e07659ab494eb050c67`
and React commit `bcc0b2a0200725b6558abbe4ab056471e597aaa2`
above exact UI parent `37fe8d33bdb4ce26465f478757dfd7f081c04372`.

- The exact four-field root stores the complete authority-compatible base, all
  ten canonical factor drafts in order (including disabled and invalid raw bound
  text plus explicit validation state), bounded design controls, and either no
  evidence or one strictly bound completed request/job/report pair.
- The parser rejects unknown/duplicate keys, non-finite values, noncanonical
  factors, invalid enabled bounds, crossed base/design/source/request identities,
  non-completed evidence, excess payload depth/bytes, and ambient transport,
  credential, process, URL, environment, identity, timestamp, or path fields.
  Imported job/request IDs are inert archive provenance and never resume work.
- Cross-review hardening recursively freezes setup and evidence base mappings,
  limits documents to 2,000,000 UTF-8 bytes/25,000 decoded nodes/32 levels,
  accepts only portable decimal/exponent bound lexemes within +/-1e9, rejects
  C0/C1 controls, and constrains seeds to the signed Qt range. PyQt retains raw
  imported bound text/error state exactly and preflights all representability
  before it can invalidate an active run. Enabling a retained invalid draft
  fails closed until an explicit valid numeric edit clears the error.
- PyQt exposes Save Workspace, Load Workspace, and Aggregate CSV actions through
  a separate mixin. Import parses completely and checks the live host base before
  invalidating active work, then restores controls, every draft, and immutable
  archived evidence together. Imported results are visibly labeled archived and
  unverified-live.
- Deterministic CSV retains source/target provenance, all four Morris metrics,
  availability, adequacy, every denominator, and design/request metadata.
  Authority raw samples are not retained and are never claimed as exportable.
  The fixed export scope is `authority-base-and-morris-controls-only`; custom
  scenario/torque semantics outside the authority base are explicitly omitted.
  Text cells are spreadsheet-formula-neutralized while numeric negatives remain
  numeric.

- React provides the same strict document, archived-evidence, and aggregate-CSV
  semantics. Browser import rejects oversized files before `FileReader`, uses a
  focus-visible keyboard button, freezes the complete parsed graph, bounds the
  duplicate scanner before recursion, and preserves existing Monte Carlo state.
- Cross-runtime review pins one byte-identical fixture and identical limits,
  Unicode code-point counting, report caps, numeric grammar, formula defense,
  factor order, evidence identities, and disabled-ground-tee behavior.

This child does not complete #4142; protected CI, dependency-ordered
publication, raw-observation retention, and UpstreamDrift replacement remain
gates.


## 2026-08-12 React Morris workflow integration (#4142 R13.7)

Branch `codex/4142-morris-react-integration` stacks the independently reviewed
React Morris workflow above current PyQt parent
`9e62c9595ccfbcf7eaa14724ad7e6d65d5277cee`; reviewed PyQt production remains
the blob-exact `89eb7a0a3432158aa4ff6a3e188f874120337c28` tree plus its
test-format repair, internal immutable-constant extraction, and handoff record.
The React application owns and
injects one same-origin `MorrisAuthorityClient`; Variation exposes Monte Carlo
and Morris as explicit sibling workflows with no browser-physics fallback.

- Applicable factors retain canonical order and use base-centered, physically
  clamped shared-registry suggestions. The current club must match its complete
  canonical library specification and every unrepresented scenario field must
  match the pinned passive fixed-ball authority; unsupported context fails
  closed with an actionable message.
- Capability, create, status, and cancel operations are sequential, abortable,
  and independently bounded to 30 seconds. Run is excluded before POST; create
  must echo the submitted request ID; the accepted request/job identity remains
  pinned through every poll and cancellation response. Nonterminal cancellation
  continues polling, and unmount or a real base change aborts current work.
- Real factor or design edits invalidate the prior job, status, and report;
  no-op commits preserve evidence. Completed output is target-local and retains
  bounds/design provenance, effect uncertainty, adequacy, availability, typed
  no-impact/failure/nonfinite denominators, assumptions, and the interaction
  caveat.
- Exact React commit `eedfc24a163af736caa47c4f0c74912a7f165036`
  received independent GO after 705 full web tests and 72 reviewer-focused
  tests plus type-check, zero-warning ESLint, and production build. This
  integration commit reconciles all four handoffs without changing either
  reviewed implementation.

The post-review parent-alignment merge preserves the validated application
behavior while inheriting the PyQt child's protected-CI and file-size repairs.

Morris persistence/export, UpstreamDrift replacement, protected CI, and
parent-first merge remain open; this does not complete epic #4142.

## 2026-08-12 Standalone PyQt Morris workflow (#4142 R13.7)

Branch `codex/4142-morris-pyqt-workflow` starts at exact UI-contract parent
`71c771fb73143f1839449d1cf5a1f5472a55f098`. It adds an authority-backed
`Morris Screening` sibling under the existing Variation module; the established
`Monte Carlo & Dispersion` widget and behavior are unchanged.

- The standalone launcher owns `MorrisAuthorityRuntime` for exactly the Qt
  event-loop lifetime and injects a strict numeric-loopback client through the
  reusable `LaunchConfig.window_kwargs` and main-window constructor seams.
  Tokens are excluded from repr and never read by widgets or globals.
- The PyQt surface provides a capability gate, canonical editable factor order,
  trajectories/even-levels/seed/minimum-effects/workers, sequential background
  create/poll/cancel, pinned request/job identity, stale-generation and
  changed-input invalidation, read-only target-local ranked μ*, uncertainty,
  availability/adequacy, and all typed miss/failure denominators. Closing is
  nonblocking: every live transport thread remains owned and the window close
  is deferred until the retained workers finish.
- Unsupported current simulation semantics fail closed with a useful message;
  no local physics fallback or silent projection is introduced. Optional
  authority startup failure leaves the rest of the app usable and labels Morris
  unavailable. The established derivation-only signal remains intact; a
  separate exact-config signal updates both variation consumers on real control,
  prescribed-torque, and joint-lock edits, including an explicit invalid state.
  Monte Carlo now generation-gates worker callbacks and clears every result view
  when its base changes, preventing a cancelled prior study from resurfacing.
- Verified locally: complete `tests/rate_of_closure` 913/913 and focused
  workflow/integration/visualization 74/74; scoped Ruff and MyPy are green, plus a real
  authority-backed two-trajectory smoke rendering 17 targets.

Morris workspace persistence/export and the React workflow remain open child
slices. Protected CI, review, parent-first stack release, and UpstreamDrift
consumption remain release gates; this does not complete epic #4142.

## 2026-08-12 UI-neutral Morris application contract (#4142 R13.6)

Branch `codex/4142-morris-ui-contract` starts at exact private-authority parent
`4986b6cfe5132cd67fb7ad4b13b9a5f0208f1500`. Python and TypeScript now share
the UI-facing contract below widgets: canonical ordered ten-factor metadata,
registry-derived suggested bounds, tee/ground applicability, exact request
serialization, strict capability/job/report consumers, direct authenticated
loopback Python transport, same-origin browser transport, and target-scoped
stable `mu*` presentation with unchanged denominator diagnostics. Request
construction round-trips the complete represented `SimulationConfig` and
rejects every unrepresented semantic difference from the pinned passive,
unlocked, fixed-ball authority rather than silently discarding it.

`morris_ui_parity_v1.json` is pinned and verified against Python values and consumed by
both runtimes. Response consumers enforce exact schemas, portable IDs, complete
source-target matrices, immutable provenance, scientific availability and
adequacy, typed no-impact overlap, and the producer's sample-moment/clamp
identity. Python performs no-proxy numeric IPv4 loopback requests with copied
bearer headers; React has no cross-origin base override. Both cap successful
responses at 16 MiB and error responses at 8 KiB. No bearer enters browser
code or errors. TypeScript also mirrors authority club/flight vocabularies,
base-physics invariants, and named sample/observation-cell resource caps before
transport. Both serializers canonicalize reversed drafts before seeded design
mapping. Lazy application/Morris façades keep all four UI contract modules
importable without SciPy, FastAPI, or Uvicorn. Widgets, hooks, polling orchestration, exports, persistence,
launcher changes, local physics fallback, and #4142 completion remain open.

## 2026-08-12 Private Morris authority host (#4142 R13.5)

Branch `codex/4142-morris-authority-host` starts at exact authority-bridge
parent `3c95dcaf88c4a0eacc747b48678e1f5c225f12ec`. The standalone Rate React
launcher now owns an ephemeral child-process authority for exactly the Vite
development-server lifetime. The child binds IPv4 `127.0.0.1:0`, announces a
canonical numeric port over a bounded private pipe, proves the exact
authenticated capability document, and exits through an authenticated graceful
control request with bounded terminate/kill fallback. The bearer is redacted
from runtime representations, never uses a `VITE_` variable, and is injected
only into Vite's server-side `/api/rate-of-closure` proxy. Every response is
`no-store`/`nosniff`; there is no CORS, docs, OpenAPI, browser token, or browser
authority endpoint. Authenticated 404, validation, and sanitized unhandled-500
responses retain the same headers without exposing exception details.
`KeyboardInterrupt` and `SystemExit` during post-spawn readiness reap the child
and close the bounded readiness pipe before propagating unchanged; secondary
terminate/wait/pipe failures are contained and cannot replace the primary
startup exception.

The canonical authority prefix is `/api/rate-of-closure/v1`; capability is
`/api/rate-of-closure/v1/morris/capabilities`. Host lifespan owns the injected
registry exactly once after startup transfer; the child closes it when socket,
app, or server setup fails before lifespan. Listener and registry cleanup are
both attempted, with secondary failures contained whenever a primary setup
error is active. Optional `rate-morris-authority` dependencies are
FastAPI, Uvicorn, and SciPy. This is a local development-launch host, not a
static-preview or deployed-host contract. UI polling/presentation, export,
persistence, UpstreamDrift consumption, and completion of #4142 remain open.

## 2026-08-12 Rate Morris authority bridge (#4142 R13.5)

Exact request/job v1 contracts and `RateMorrisService` now bridge the current
ten-factor Rate adapter to the unchanged Morris report v1. Reconstruction pins
passive, unlocked, profile-free double-pendulum fixed-ball execution with no
prescribed impact time and zero time offset. The internal 113 mph scenario
speed is compatibility-only, not a new measured input or physics claim.
Wire validation is unconditional: factor ordering, pendulum/club/ball physical
domains, contextual tee factors, and both factor endpoints are checked before
shared DbC-backed constructors. WARN/OFF contract modes remain fail-closed.

The optional router strictly decodes bounded raw JSON, and its injected-clock
registry owns active/global worker budgets, TTL/retention, cancellation, and a
lock-linearized lifecycle. Cancellation registered before terminal completion
discards the report; running work stays running until acknowledgment. Expected
sample numerical failures remain completed report denominator data, while
programming failures yield only a sanitized stable job error. The TypeScript
model uses the existing report parser and an injected create/status/cancel
client with no browser physics fallback.

Open: UI/polling presentation, export, persistence, host mount, UpstreamDrift,
and a genuine fixed-ball double-pendulum hit. Cancellation latency depends on
executor observation; no partial report or per-sample diagnostic is exposed.

## 2026-08-12 Rate fixed-ball Morris evaluator (#4142 R13.3)

- Branch `codex/4142-morris-rate-adapter` starts at exact shared-executor parent
  `b2fa365087f184d9ada16a6d35b08cbce64879c6`; publication and protected
  current-head gates remain open.
- A Rate-owned injected evaluator maps ten exact global variables through the
  public immutable `apply_global_simulation_values(config, values)` seam. It
  requires a double-pendulum, fixed-ball base, registered units, unique global
  variable keys, and Tee support for tee height. The fixed-contact no-op
  `impact_time_offset_s` and all unsupported/localized factors fail closed.
- The exact current 17-scalar output order is typed as three contact scalars,
  five impact metrics, and nine shot outcomes with audited units/frame metadata.
  Extracted trial capture/projection is shared by Morris and ensemble execution,
  preserving the existing caught numerical-failure tuple and exact hit/miss/
  failure availability while allowing programming `TypeError` defects to abort.
- A genuine double-pendulum fixed-ball miss is validated end to end; a manual
  fixed hit proves source-neutral projection only. A genuine double-pendulum
  fixed hit remains an explicit physical validation gate, not an inferred claim.
- Fixed contact remains sampled clubhead-reference-point to ball-sphere
  proximity: it has no clubface mesh, swept collision between samples, or ball
  compression. Cancellation is cooperative between complete simulations, so
  one already-running simulation bounds latency. Morris observations retain
  status and values but not per-sample failure type/message diagnostics.
- UI, #4280 export, UpstreamDrift integration, protected CI, and #4142 epic
  completion remain open.

## 2026-08-12 Bounded Morris execution adapter (#4142 R13.3)

- Branch `codex/4142-morris-execution-adapter` starts from exact intended
  parent `cc572243ae0df551237265d72b9e34bff0285f01`; it must retain that normal
  history and later receive protected current-head CI before publication.
- New `morris_execution.py` evaluates the shared Morris design through an
  injected, UI-neutral typed protocol. Every immutable sample carries its
  flattened ordinal, trajectory/point coordinates, factor tuple, and exact
  physical `spec_id` mapping. Results are preallocated and written to disjoint
  canonical rows, so serial and bounded parallel execution return identical
  observation tensors.
- Evaluations retain exact output keys and finite-or-unavailable values.
  No-impact samples may retain scalar/state-point metrics but cannot fabricate
  impact/shot values; numerical failures carry no values. Only the explicitly
  injected evaluator can normalize its own domain error into that status; the
  generic executor catches no evaluator exceptions, so malformed returns and
  every thrown exception abort the study.
- Solver-shaped completed-prefix progress is drained in canonical ordinal order
  every eight samples and at final completion. Pre-start and between-sample
  cancellation raises the shared `CancelledError` and returns no partial
  observations. Named sample, output-cell, and 32-worker caps bound resources.
- This is shared execution infrastructure only: it does not import Rate, bind
  `evaluate_run`, add UI/export behavior, or complete #4142.

## 2026-08-12 Exact Morris serialized-clamp contract (#4142 R13.4)

- The producer maps `sigma` and `mu*` standard error at or below
  `64*epsilon*max(1,mu_star)` to exact zero. The consumer now rejects positive
  values inside that interval; the prior `1e-14` perturbation allowance is
  removed. Clamp uncertainty contributes to the squared identity only when the
  corresponding serialized metric is exactly zero.
- Identity arithmetic is normalized before squaring. Nonzero metrics receive
  only ordinary scale-aware rounding allowance; huge finite values that cannot
  be squared safely fail closed rather than producing `Infinity`/`NaN` that can
  evade comparisons.
- Cohesive numerical checks moved to `morrisMetricValidation.ts`, keeping the
  primary strict parser below 400 lines. Tests retain normal `n=4`/`n=12` and
  `10^6`-scale cases and add clamp-boundary and near-`1e308` adversarial cases.
- Producer calculations, UI/export/execution, and UpstreamDrift remain open and
  unchanged.

## 2026-08-12 Morris clamp-scale tolerance correction (#4142 R13.4)

- The earlier squared-space unit floor was too permissive near
  `mu_star = abs(mu)` and zero standard error. The consumer now mirrors the
  producer clamp exactly as `delta = 64*epsilon*max(1, mu_star)` and propagates
  `2*abs(metric)*delta + delta^2` through `sigma^2`, `n*SE^2`, both mean-square
  terms, and the `n/(n-1)` correction.
- A metric-level degeneracy invariant rejects `sigma` above `delta` when the
  mean-magnitude difference and standard error are within `delta`. Mutation
  tests reject `sigma=1e-8`, accept a serializer-scale `1e-14` perturbation,
  and pin realizable identities for `n=4`, `n=12`, and scale `10^6`.
- No producer, UI, export, execution, or UpstreamDrift behavior changed.

## 2026-08-12 Morris TypeScript review hardening (#4142 R13.4)

- Review mutation tests now prove the four reported statistics are jointly
  realizable for `valid_pairs`: for `n > 1`, the parser checks
  `sigma^2 - n*SE^2 = n/(n-1)*(mu_star^2-mu^2)` with a documented tolerance
  of 256 IEEE-754 epsilons scaled to the squared metrics.
- Exact zero `mu_star` requires zero `mu`, `sigma`, and standard error plus
  `constant-output`; exact zero `sigma` requires zero standard error and
  `mu_star` equal to `abs(mu)` within that tolerance. An all-zero tuple labeled
  `available` fails closed. Explicit insufficient estimates remain all-null and
  bypass finite-metric algebra without weakening denominator/adequacy checks.
- The parser accepts only ordinary or null-prototype records, rejects all C0
  and C1 control characters, and uses nested source/target sets instead of a
  delimiter-composite identity. Tests pin former NUL collisions, class/custom
  prototypes, nested missing/excess fields, stable source/target provenance,
  complete matrices, duplicate pairs, and deep immutability.
- No Python producer behavior, UI/export/execution adapter, or UpstreamDrift
  integration changes in this review fix. Later R13-R15 scope remains open.

## 2026-08-12 Strict Morris TypeScript parity contract (#4142 R13.4)

- Branch `codex/4142-morris-typescript-parity` fast-forwarded normally from
  published parent exact head `f08494f3a2698ddd69f7452dfdb1e70765388ef8`;
  no history, configured base, or parent branch was rewritten.
- The Python report now emits stable schema identity
  `swing-sim/morris-global-sensitivity-report` at `schema_version: 1` while
  `morris-elementary-effects` remains the independent scientific method value.
- A UI-neutral TypeScript parser consumes the golden fixture into immutable
  typed source/target/effect/denominator objects. It rejects unknown fields,
  coercive or non-finite values, unsupported vocabulary, broken source loci,
  units/frames/bounds, invalid Morris seed/grid/sample provenance, duplicate or
  inconsistent factors and estimate pairs, and invalid denominator cohorts.
- Unavailable effects must be four JSON `null` values paired with
  `insufficient-data`/`insufficient`; available and constant effects must be
  finite. Typed no-impact totals retain their intentional overlap with valid
  state outputs while unavailable misses, failures, and non-finite pairs remain
  mutually exclusive denominator cohorts.
- This slice deliberately does not add UI, export, simulation execution, or
  UpstreamDrift wiring owned by later R13-R15 work and PR #4280.

## 2026-08-12 Bounded Morris global-sensitivity core (#4142 R13.2-R13.4)

- Exact-head CI follow-up: the hosted changed-file MyPy gate exposed five
  NumPy inference gaps that the earlier scoped invocation did not reproduce.
  The repair adds explicit array dtypes and a typed outcome-normalization
  boundary only; scientific behavior and serialized payloads are unchanged.
- Branch `codex/4142-global-sensitivity` normally merged exact intended parent
  `feat/4144-variation-export-continuation@7fb5d7f489db49742b7bc82ef009570ad2502456`
  without rebasing, resetting, retargeting, or rewriting either history.
- New UI-neutral `swing_sim.variation` contracts generate deterministic Morris
  trajectories and report `mu`, `mu*`, `mu*` standard error, and `sigma` for
  simultaneous nonlinear/interacting bounded inputs. Source spec/locus/unit,
  target unit/frame/point/time, seed, design grid, bounds, denominators, and
  adequacy remain explicit.
- Canonical Rate trial-status wire values are accepted without a reverse shared
  package dependency. Evaluated misses retain available pre-impact/state
  outputs, while absent impact/shot effects, numerical failures, and non-finite
  values remain separate denominator cohorts with `NaN` estimates when sample
  adequacy is insufficient. No impact or shot value is fabricated.
- The deterministic report serializer maps unavailable numeric estimates to
  JSON `null`; a committed golden fixture is ready for later React consumption.
  This remains only the reusable analysis slice and does not modify PR #4280
  export/UI logic. Design execution orchestration, PyQt6/React presentation,
  and UpstreamDrift consumption remain open R13-R15 work; Morris `sigma`
  conflates nonlinearity and interaction and is not causal attribution or a
  variance decomposition.

## 2026-08-11 current workspace parent propagation (#4279 → #4280)

- PR `#4280` remains on `feat/4144-variation-export-continuation`, based on
  `feat/4218-toolstrip-workspace`; neither branch nor PR base is rewritten.
- Exact clean child head `9b45bd5beca38370c1d541f8c488ef0edad08517`
  is merged normally, child first, with exact parent head
  `983805d799b76e5e1ad1dbdc7a5ab28957d805c8`.
- Variation scatter CSV parity, typed unavailable outcomes, bounded accessible
  trial tables, linked selection, and all-trial arc analysis remain unchanged
  alongside the inherited workspace/toolstrip and registry/D-plane contracts.
- This remains a pre-manifest stack: the later strict campaign release
  manifest artifact/checker/test exists on neither side and is not recreated.
  Both histories are retained under a monotonic `1.16.20` through `1.16.0`
  sequence. Pinned Ruff `0.14.10` check/format is green across all five changed
  Python files; 87 focused Python and 25 focused React tests pass. React
  type-check, lint, and production build plus documentation, minimum-test,
  changed-file-size, module-size, SPEC-version, and diff gates are green.
  Protected checks, review, and parent-first release order remain mandatory.

## 2026-08-11 #4280 receives exact reconciled #4279 parent

- PR `#4280` retains branch `feat/4144-variation-export-continuation` and base
  `feat/4218-toolstrip-workspace`. A normal merge combines exact published child
  `e6c7460a01082631565fb9ed48aa32538bd7772c` with exact published parent
  `89af587c8f4141680bb923fc4295e261829f5c75`; no rebase, retarget, force-push,
  or parent rewrite is used.
- All implementation and test paths merge automatically. Only the two
  append-only handoffs conflict textually, and both histories are preserved.
  Variation-export behavior is unchanged while inheriting the parent's current
  workspace, kinetics, solver, layout, and Qt typing repairs.
- `SPEC.md` advances monotonically and uniquely to `1.14.20`. The later campaign
  manifest does not exist on either side of this pre-manifest stack, so there is
  no campaign artifact to update.
- Focused local verification is green: pinned Ruff 0.14.10 check/format on all
  five inherited Python files plus diff hygiene; 91 variation/workspace Python
  tests; 232 inherited workspace/kinetics/impact Python tests; and 42 React tests
  across 11 files.
- This merge is local only. Publication, protected exact-head CI, independent
  review, unresolved-thread checks, dependency order, and release remain open.

## 2026-08-11 current workspace propagation into variation PR #4280

- PR `#4280` remains on `feat/4144-variation-export-continuation`, based on
  `feat/4218-toolstrip-workspace`. Exact published child
  `3337945699966b63cb5cd8e52d7c3b194315e911` is merged first with exact newly
  published parent `efbca84095b617b4018732f7802c2da3f0525387` second by a
  normal merge; no rebase, retarget, force-push, or parent rewrite is used.
- All implementation and test paths merge automatically. The child variation
  export, typed unavailable outcomes, bounded tables, linked selection, and
  all-trial arc behavior remain authoritative while inheriting the parent's
  current workspace, launch-monitor/D-plane ancestry, split kinetics, and Qt
  typing repairs. Only append-only handoffs and SPEC conflict textually; both
  histories remain below.
- Focused pre-commit regression is green: all 56 focused Python variation tests
  pass, and all 10 tests in the three focused React variation suites pass.
- Exact post-commit PR-delta gates against the new parent are also green:
  pinned Ruff 0.14.10 checks and formats all five changed Python files, pinned
  MyPy 1.13.0 accepts all four changed production files, and pinned Bandit
  1.7.7 finds no issues in those four files. Docs governance, 500-LOC budget,
  changed-Python policy, minimum-test contract, changed-test assertions, diff
  hygiene, React type-check, React lint, and the production web build all pass.
  The manifest artifact/checker/test do not exist on either side of this
  pre-manifest stack, so that later release gate is structurally not applicable.
- This candidate remains local pending independent review, ordinary
  publication, protected
  exact-head CI, unresolved-thread checks, and dependency order.

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
## 2026-08-11 current registry parent propagation (#4203 → #4279)

- PR `#4279` remains on `feat/4218-toolstrip-workspace`, based on
  `feat/4181-launch-monitor-registry`; neither branch nor PR base is rewritten.
- Exact clean child head `89af587c8f4141680bb923fc4295e261829f5c75`
  is merged normally, child first, with exact parent head
  `1e29c6e52169de5d984144af29664c0419b51a21`.
- Workspace documents, File/View/Tools commands, module visibility/order,
  Impact/Swing/Flight navigation, deterministic playback, and independent plot
  controls remain unchanged alongside the inherited registry/D-plane history.
- This remains a pre-manifest stack: the later strict campaign release
  manifest artifact/checker/test exists on neither side and is not recreated.
  Both histories are retained under a monotonic `1.15.12` through `1.15.0`
  sequence. Pinned Ruff `0.14.10` check/format is green across all 27 changed
  Python files; 142 focused Python and 32 focused React tests pass. React
  type-check, lint, and production build plus documentation, minimum-test,
  changed-file-size, module-size, SPEC-version, and diff gates are green.
  Protected checks, review, and parent-first release order remain mandatory.

## 2026-08-11 #4279 receives exact reconciled #4203 parent

- PR #4279 keeps branch `feat/4218-toolstrip-workspace` and configured base
  `feat/4181-launch-monitor-registry`. Exact published child
  `efbca84095b617b4018732f7802c2da3f0525387` is normally merged with exact
  parent `9ce2c70f11a15420f0ba2d3b4fef6726b6eacefa`; no history or PR metadata is
  rewritten.
- Implementation merges automatically. Only the two append-only canonical
  handoffs conflict, and both histories are preserved. Workspace/toolstrip,
  navigation, playback, and independent plot behavior stay unchanged while
  the exact parent formatting repair and split-facade kinetics ancestry are
  inherited.
- Pinned Ruff 0.14.10 check/format passes all five inherited Python files.
  Regression is green for 142 workspace/plot Python tests, 125 inherited
  kinetics/impact/registry tests, and all 32 tests in the eight focused React
  files; diff checks pass. This merge is local only pending normal publication,
  protected CI, review, unresolved threads, and dependency order.

## 2026-08-11 current #4203 propagation into workspace PR #4279

- Draft PR `#4279` retains branch `feat/4218-toolstrip-workspace` and base
  `feat/4181-launch-monitor-registry`. Exact published child
  `ccd0e026c580c93038fdf5c59d5d452a85ba27a0` is merged first with exact
  newly published parent `7abce9ad767fe8311da66a1e5998b892ea3ca9de`
  second by a normal merge; no rebase, retarget, force-push, or parent rewrite
  is used.
- All implementation paths merge automatically. The child workspace,
  toolstrip, visibility, navigation, playback, and independent-plot behavior
  remain authoritative while inheriting the parent's split kinetics and four
  behavior-preserving Qt primitive-return boundaries. Only append-only
  handoffs and SPEC require textual reconciliation, and both histories remain
  below.
- Focused pre-commit regression is green: all 142 exact PR-delta Python tests
  pass, and all 32 tests in the eight changed React suites pass. Exact
  post-commit PR-delta gates against the new parent are also green: pinned Ruff
  0.14.10 checks and formats all 27 changed Python files, pinned MyPy 1.13.0
  accepts all 18 changed production files, and pinned Bandit 1.7.7 finds no
  medium/high issues in those 18 files. Docs governance, 500-LOC budget,
  changed-Python policy, minimum-test contract, changed-test assertions, diff
  hygiene, React type-check, React lint, and the production web build all pass.
  The manifest artifact/checker/test do not exist on either side of this
  pre-manifest stack, so that later release gate is structurally not applicable
  to this propagation.
- This candidate remains local pending independent review, ordinary
  publication, protected
  exact-head CI, unresolved-thread checks, and dependency order.
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
## 2026-08-11 current D-plane parent propagation (#4202 → #4203)

- PR `#4203` remains on `feat/4181-launch-monitor-registry`, based on
  `feat/4189-dplane`; neither branch nor PR base is rewritten.
- Exact clean child head `9ce2c70f11a15420f0ba2d3b4fef6726b6eacefa`
  is merged normally with exact parent head
  `9f83cd379ce8ae2805aa4a5608b5645a529f9c3c`.
- Launch-monitor convention/analytics registries, cross-runtime golden fixture,
  D-plane ndarray repair, split typed kinetics façade, and pinned Ruff
  `0.14.10` files remain unchanged. The strict campaign release manifest is
  still absent from this exact history and is not reconstructed here.
- Both handoff histories and the parent's seven post-base SPEC records are
  retained additively under new monotonic `1.14.x` revisions. Pinned Ruff
  `0.14.10` check/format is green across 18 registry, analytics, D-plane,
  delivery, and kinetics files; 79 focused Python and 31 focused React tests
  pass. Documentation, minimum-test, changed-file-size, module-size,
  SPEC-version, and diff gates are also green. Protected checks, review, and
  parent-first release order remain mandatory.

## 2026-08-11 #4203 receives exact current #4202 format repair

- Draft PR #4203 keeps branch `feat/4181-launch-monitor-registry` and base
  `feat/4189-dplane`. Exact published child
  `7abce9ad767fe8311da66a1e5998b892ea3ca9de` is normally merged with exact
  parent `ba4aa35cc384d51ed3aa52eb532a67e960669c27`; no history or PR metadata is
  rewritten.
- Both append-only handoff histories are retained. The sole code conflict is
  the already documented kinetics split-facade seam: the child keeps its typed
  `pendulum.sample(...)` call and inherits the parent's formatted geometry
  explanation. The obsolete monolithic `source.inner.sample(...)` expression
  is not restored. Physics, frames, values, and public contracts are unchanged.
- Pinned Ruff 0.14.10 check/format passes all five inherited Python files.
  Regression evidence is 81 kinetics/impact/PyQt/layout tests plus 44
  launch-registry/D-plane/delivery/contract tests, all passing. Diff checks are
  clean. No GitHub write is part of this local reconciliation; protected CI and
  review remain publication gates.

## 2026-08-11 #4203 append-only SPEC preservation repair

The first independent audit of local reconciliation candidate `e20b4f630...`
found four exact D-plane parent rows omitted from the append-only SPEC history:
2026-08-10 versions 1.13.11, 1.13.9, 1.13.7, and 1.13.6. They are restored
verbatim in the current documentation-only follow-up. Production code, tests,
merge parents, PR base, and local quality evidence are unchanged. Independent
re-review, ordinary publication, protected CI, and downstream propagation
remain open.

## 2026-08-11 #4203 current D-plane parent reconciliation

- Draft PR #4203 now requires a normal merge of exact current base
  `f3363aa88868f6a5c7e9ccfc682a9eca014e86c1` after exact published child
  `217e36dc93d30f79826847f958fbcd10805e58ed`; its base remains
  `feat/4189-dplane` and no history is rewritten.
- The parent changes nine files. Its sole textual conflict is an inherited
  kinetics formatting edit. The already reviewed split kinetics facade stays
  authoritative and uses the typed `DoublePendulumSwing` object directly;
  the parent's explanatory geometry comment is retained.
- Exact CI-pinned MyPy 1.13 validation then exposed four remaining Qt stub
  boundaries in the child delta. Responsive-event handling, legend visibility,
  ball-setup event filtering, and status text now narrow their unchanged Qt
  return values to the declared primitive contracts.
- Focused regression, full PR-delta quality gates, independent review, normal
  fast-forward publication, and fresh protected exact-head CI remain required.

## 2026-08-11 #4203 exact-head format completion

- Protected CI on exact published head
  `7d69a545ae555679f0318940e67c1786626d6794` failed only Ruff formatting.
  Reproducing the exact pinned 0.14.10 check found eleven noncompliant changed
  Python files: four inherited files plus seven altered by the automated
  pre-commit repair.
- The pending repair applies pinned Ruff formatting to exactly those eleven
  files. AST equivalence and focused tests must pass before independent review
  and ordinary fast-forward publication; no simulation, contract, or UI
  behavior is intentionally changed.

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
## 2026-08-11 pinned-Ruff parent propagation (#4179 → #4202)

- Child PR `#4202` remains on `feat/4189-dplane`, based on
  `feat/4162-wedge-impact-visualization`; neither branch nor PR base is
  rewritten.
- Exact clean child head `ba4aa35cc384d51ed3aa52eb532a67e960669c27`
  is merged normally with exact parent head
  `7e5dfecf569b39dbbf8cc2101c7426cbc53a2771`.
- The D-plane ndarray typing repair, frame-explicit geometry, pinned Ruff
  `0.14.10` files, and all impact, wedge/turf, handoff, campaign, and SPEC
  histories remain additive. No physics, frame, validation, API, schema,
  persistence, test, export, or UI behavior changes.
- Pinned Ruff `0.14.10` check/format verification and 129 focused D-plane,
  impact, solver, kinetics, PyQt, and layout tests are green. Documentation,
  minimum-test, SPEC-version, and diff gates are also green. Protected checks,
  review, and parent-first release order remain mandatory.

## 2026-08-11 PR #4202 pinned-Ruff format repair

- Exact published head `f3363aa88868f6a5c7e9ccfc682a9eca014e86c1`
  failed CI Standard run `31483390692`, job `93753191911`, only because five
  changed Python files no longer matched the workflow-pinned Ruff `0.14.10`
  formatter.
- Those five files are mechanically reformatted with Ruff `0.14.10`; numerical
  behavior, reference frames, validation, APIs, schemas, and tests are
  unchanged. This is an actionable current-head CI repair, not an expansion or
  completion of D-plane issue `#4189`.
- Verification is green for the workflow-mirrored scoped Ruff check and format
  check, `git diff --check`, and 71 focused impact, kinetics, PyQt, and layout
  tests.
- No material handoff behavior changed. The release remains parent-first and
  protected; queued checks and the ordinary repository merge gates still apply.

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
## 2026-08-11 pinned-Ruff parent propagation (#4178 → #4179)

- Child PR `#4179` remains on `feat/4162-wedge-impact-visualization`, based on
  `feat/4166-wedge-turf-physics`; neither branch nor PR base is rewritten.
- Exact clean child head `ea7acebf033379d6beefd70eb51027ebd3d01be7`
  is merged normally with exact parent head
  `188f491ccc88a335ad36afdd66b52289e2e24808`.
- The inherited Ruff `0.14.10` formatting and all visualization, turf, wedge,
  Rate, handoff, campaign, and SPEC histories remain additive. No physics,
  frame, validation, calibration, API, schema, persistence, test, export, or UI
  behavior changes.
- Pinned Ruff `0.14.10` check/format verification and 130 focused impact-scene,
  solver, kinetics, PyQt/layout, wedge-clearance, and turf-model tests are
  green. Documentation, minimum-test, SPEC-version, and diff gates are also
  green. Protected checks, review, and parent-first release order remain
  mandatory.

## 2026-08-11 PR #4179 pinned-Ruff format repair

- Exact published head `ec73b63a748347b42686758d4738c0fd2fd09332`
  failed its current CI Standard quality gate only because five changed Python
  files did not match the workflow-pinned Ruff `0.14.10` formatter.
- The files are mechanically reformatted with that exact version. No impact
  visualization, wedge or turf physics, frames, validation, APIs, schemas,
  tests, or user-visible behavior changes; this is not completion of `#4162`.
- No material handoff behavior changed. Workflow-mirrored Ruff, focused tests,
  and `git diff --check` are the local gates. Protected checks and parent-first
  release order remain the release gates.

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
## 2026-08-11 pinned-Ruff parent propagation (#4174 → #4178)

- Child PR `#4178` remains on `feat/4166-wedge-turf-physics`, based on
  `feat/4161-wedge-ground-clearance`; neither branch nor PR base is rewritten.
- Exact clean child head `ca567fe7d3fa48b1900ad3098045f4200cfe86a7`
  is merged normally with exact parent head
  `3e1b44cf42f4c0838149e0bc8e88ce4cb79b72b0`.
- The inherited Ruff `0.14.10` formatting and all wedge, turf, Rate, handoff,
  campaign, and SPEC histories remain additive. No physics, frame, validation,
  calibration, API, schema, persistence, test, or UI behavior changes.
- Workflow-pinned Ruff check/format and 127 focused turf, wedge, impact,
  kinetics, PyQt, and layout tests are green. Documentation, minimum-test,
  SPEC-version, and diff gates are also green. Protected checks, review, and
  parent-first release order remain mandatory.

## 2026-08-11 pinned-Ruff parent propagation (#4173 → #4174)

- Child PR `#4174` remains on `feat/4161-wedge-ground-clearance`, based on
  `feat/4163-impact-inspector`; neither branch nor PR base is rewritten.
- Exact clean child head `01ecf9a7b1922d1609fb99093226799a0b564704`
  is merged normally with exact parent head
  `bd48852d303db6281ed5891d4a271d99e76a94e6`.
- The inherited Ruff `0.14.10` formatting and parent handoff/spec history stay
  additive to the existing swept wedge ground-clearance contracts. No physics,
  frame, validation, API, schema, test, persistence, or UI behavior changes.
- Workflow-pinned Ruff check/format and 98 focused impact, kinetics, wedge,
  PyQt, and layout tests are green. Documentation, minimum-test, SPEC-version,
  and diff gates are also green. Protected checks, review, and parent-first
  release order remain gates.
## 2026-08-11 PR #4178 pinned-Ruff format repair

- Exact published head `b8822401f4522e867d6b160125953981a39a770d`
  failed its current CI Standard quality gate only because five changed Python
  files did not match the workflow-pinned Ruff `0.14.10` formatter.
- The files are mechanically reformatted with that exact version. No turf or
  impact physics, frames, calibration boundaries, validation, APIs, schemas,
  tests, or user-visible behavior changes; this is not completion of `#4166`.
- No material handoff behavior changed. Workflow-mirrored Ruff, focused tests,
  `git diff --check`, and 71 focused impact, kinetics, PyQt, and layout tests
  are green. Protected checks and parent-first release order remain the gates.

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

## 2026-08-11 PR #4174 pinned-Ruff format repair

- Exact published head `525696e0c1080616eb5055e2cb1c93565f98672e`
  failed CI Standard run `31485402975`, job `93759519460`, only because five
  changed Python files did not match the workflow-pinned Ruff `0.14.10`
  formatter.
- The files are mechanically reformatted with that exact version. No physics,
  frames, validation, public APIs, schemas, tests, or user-visible behavior
  changes; this is not completion of wedge issue `#4161`.
- No material handoff behavior changed. Workflow-mirrored Ruff, focused tests,
  `git diff --check`, and 71 focused impact, kinetics, PyQt, and layout tests
  are green. Protected checks and parent-first release order remain the gates.

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

## 2026-08-11 pinned-Ruff parent propagation

- Exact published head `3c19aaa9d3e812e4659053735a2955d62a080d34`
  has the same five-file Ruff `0.14.10` format mismatch proven by child CI.
- Those files are mechanically formatted with the workflow-pinned version.
  No variation, impact, plotting, persistence, API, schema, test, or UI
  behavior changes; this is not completion of issue `#4144`.
- No material handoff behavior changed. Protected CI and the ordinary carrier
  into `feat/impact-simulation-platform` remain release gates.

The formatted parent is merged normally into the impact-inspector child; no
impact-inspection or variation behavior changes.

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
