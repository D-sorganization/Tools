# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-11

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
