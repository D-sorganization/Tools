# Rate of Closure Ball-Flight Campaign Handoff

Status verified 2026-08-08. This isolated integration is published as draft
[PR #4217](https://github.com/D-sorganization/Tools/pull/4217). No source PR
branch was rewritten.

## 2026-08-09 Ground-contract stack recovery

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
fix and handoff update are committed together as `SELF`; resolve the exact head
with `git rev-parse HEAD` and push normally.

### 2026-08-07 strict flight-to-ground contract continuation

Active worktree
`C:\Users\diete\Repositories\Tools-worktrees\ground-transition-contract` on
branch `feat/4268-ground-contract` starts exactly at protected draft PR #4283
head `60ac5b46c78988225862d9b89a33ddc3656a3413`. It is the stacked implementation
for [issue #4268](https://github.com/D-sorganization/Tools/issues/4268) under
ground-model epic #4267. The implementation and this durable handoff update are
committed together as `SELF`; resolve the exact commit with `git rev-parse HEAD`.

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
