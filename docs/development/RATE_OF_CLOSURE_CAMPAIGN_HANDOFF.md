# Rate of Closure Campaign Handoff

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

- Draft child PR `#4203` retains base `feat/4189-dplane` and receives exact
  repaired parent head `7d8d2f06dc797021d01939691e58f8425b652b33`
  through a normal merge commit. No branch history, PR base, or draft state is
  rewritten.
- The inherited repair closes the parent head's two pinned MyPy
  `no-any-return` findings with explicit ndarray boundaries and makes no
  numerical, frame, schema, or UI change.
- Parent quality-gate success is not child release evidence. Current-head child
  CI, review, all earlier ancestors, and #4189 acceptance remain required.
- Reconciled child-tree evidence is 25 focused D-plane/impact tests, docs
  governance, changed-file size, and whitespace checks. The local Windows
  MyPy 1.15/installed-NumPy stub combination is incompatible with the branch's
  Python 3.11 target, while WSL currently fails to start with `E_FAIL`; the
  successful pinned typing evidence remains the exact parent hosted gate.

## 2026-08-10 Propagation into launch-monitor registry

- Immediate child PR `#4203` keeps base `feat/4189-dplane`.
- Its original head `08a2fdd8ce6bbc8fbb8f121927a677d4addb6b11`
  normally merges exact parent `#4202` head
  `b443fdbed7064c5db0320106013c8413e3e24356`; no branch rewrite, retarget,
  or force push is permitted.
- The semantic reconciliation keeps #4203's responsive
  `SimulationViewControlsMixin`, while the parent's `ImpactLayerControls`
  helper becomes the single owner of persisted D-plane checkbox state. The
  automation compatibility mapping aliases that helper state exactly.
- The launch-monitor registry, Python 3.10 compatibility layer, frame-explicit
  D-plane contracts, responsive layout, and exports remain additive. Both
  affected PyQt modules satisfy the protected 500-line limit.
- The untouched original child had three additional ungrandfathered size
  blockers: swing sources at 540 LOC, the plotting catalog at 533 LOC, and the
  main window at 528 LOC. Narrow extractions move triple-pendulum dynamics,
  plotting metadata, and versioned primary-tab state into focused modules,
  while identity-pinned re-exports preserve every established import seam. The
  resulting module pairs are 282/282, 459/98, and 494/85 lines.
- Focused evidence is 36 passing PyQt simulation/layout tests, 38
  plotting/navigation tests, and 21 simulation source/export tests. Combined
  evidence is 1,249 passing Python tests with six explicit optional skips, 521
  React tests and all web gates, 12 `swing-core` tests, real CPython 3.10
  checks, scoped static analysis, docs/minimum-test/assertion governance,
  changed-file size, detect-secrets, and diff checks. The protected size gate
  passes all 107 changed candidates; a separate full-tree audit retains two
  untouched non-candidate monoliths (`kinetics.py` and
  `torque_profile_panel.py`). Independent staged review found no actionable
  findings after 95 additional focused tests. Current-head protected CI and
  required repository review remain pending release evidence.

Status verified 2026-08-07. This isolated integration is published as draft
[PR #4217](https://github.com/D-sorganization/Tools/pull/4217). No source PR
branch was rewritten.

## 2026-08-09 Python 3.10 compatibility follow-up

The descendant Python 3.10 lane revealed seven unconditional
`enum.StrEnum` imports already present at PR #4203. The earliest owning parent
now routes runtime imports through `shared.python.compatibility.StrEnum` while
using the stdlib type only under `TYPE_CHECKING`. The change preserves every
enum value and schema while removing the Python 3.11-only import boundary.
Local evidence is 64 focused contract/physics/compatibility tests, Ruff and
format, pinned mypy 1.13 across eight changed files, and a real CPython 3.10.20
probe of the shared fallback and seven module import declarations. Parent is
published head `9dbceff76`; after a guarded normal push, propagate the new
exact head through #4279, #4280, #4281, and #4282 before retrying descendants.

The same scan found one parent-owned `datetime.UTC` import in the PyQt torque
profile controller. It now consumes the established compatibility constant,
preserving UTC serialization while allowing Python 3.10 collection. This
follow-up belongs on #4203 before the child stack is propagated.

## 2026-08-09 Launch-registry parent CI repair

The earliest campaign parent, draft PR #4203, remains based on
`feat/4189-dplane` at exact published head
`912ebc9d69b05763a76c2c8f198d943737e2d3fb`. CI run `31199764932` passed its
quality gate, then all Python versions failed before assertions because pytest
collected the flight and solver contract modules through `src.shared...` while
their absolute dotted facade aliases requested the editable `shared...`
namespace. The scoped repair converts only those two test imports to relative
package imports. Production APIs and physics are unchanged. The same run's
Rust failure is the known runner inability to link `-lpython3.11`. Validate
the two contract modules under importlib collection, publish a new exact head,
and propagate it through #4279, #4280, #4281, and #4282 in normal order.
Both modules now pass all `12` tests on Windows and WSL Python 3.11 with
importlib collection. Ruff check/format and pinned mypy 1.13 are clean. A
test-only `Any` cast keeps frozen-dataclass metadata introspection explicit
without weakening the runtime contract.

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

## 2026-08-10 PR #4202 D-plane ndarray typing repair

- The repair is based on exact published PR `#4202` head
  `b443fdbed7064c5db0320106013c8413e3e24356` and retains base
  `feat/4162-wedge-impact-visualization`.
- CI Standard run `31384810375`, job `93442745760`, reported two pinned MyPy
  1.13 `no-any-return` errors in private NumPy conversion/projection helpers.
  Explicit ndarray local boundaries resolve those errors without changing
  geometry, validation, frames, serialized contracts, or UI behavior.
- RED/GREEN evidence is exact: two errors reproduced before the edit and zero
  afterward. Twenty-four focused D-plane/impact tests, seven metadata/pre-push
  contract tests, scoped Ruff/Ruff-format/Black, docs governance, minimum-test,
  module-size, changed-file-size, and diff checks pass. Three exploratory
  CI-workflow contract tests retain unrelated older-branch toolcache/env drift;
  no workflow file is changed here.
- This repair does not alter the stacked base, publish the branch, change its
  draft state, or authorize a merge. Parent-first ordering, protected CI, and
  review remain required.

## 2026-08-10 Propagation into 3D D-plane geometry

- Immediate child PR `#4202` keeps base
  `feat/4162-wedge-impact-visualization`.
- Its original head `b4abec03bccfbdd87ddf91427159c5c2332c21dd`
  normally merges exact parent `#4179` head
  `6704a3e541a3e74c28b4a284530d1a21269dd340`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the typed,
  frame-explicit D-plane, spin-loft, visualization, and export contracts.
- Persisted D-plane layer controls are extracted into a focused helper so the
  simulation view satisfies the protected 500-line module budget without a
  behavior or compatibility-seam change.
- Combined-stack verification is green: 93 focused and 825 scoped Python tests
  (two optional `build123d` skips), 360 React tests and all web gates, real
  CPython 3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository
  governance gates. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited 17-error broad
  MyPy Qt/NumPy baseline in 11 untouched files remains outside scope.
  Protected CI and required review remain release gates.

## 2026-08-10 Propagation into wedge impact visualization

- Immediate child PR `#4179` keeps base `feat/4166-wedge-turf-physics`.
- Its original head `0eb804e70887c788421332369e42792411aff55a`
  normally merges exact parent `#4178` head
  `bfa83aedc88ead380babc73a699377d98b971006`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the
  exact-event, locked-scale, exportable impact-scene contract.
- Combined-stack verification is green: 58 focused and 739 scoped Python tests
  (two optional `build123d` skips), 347 React tests and all web gates, real
  CPython 3.10.20 compilation/UTC, scoped Ruff/Black/MyPy, and repository
  governance gates. The exact parent's 12 unchanged `swing-core` tests remain
  applicable because this child has no Rust delta. The inherited 17-error broad
  MyPy Qt/NumPy baseline in 11 untouched files remains outside scope.
  Protected CI and required review remain release gates.

## 2026-08-10 Propagation into wedge turf physics

- Immediate child PR `#4178` keeps base `feat/4161-wedge-ground-clearance`.
- Its original head `aaae3f73e17dbfaad5cca1dc6f49559b3aebe9d5`
  normally merges exact parent `#4174` head
  `9ea93e92563280ec34bca682ad44d7409edd7a02`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the passive,
  provenance-gated turf-contact model and explicit force-coupling boundary.
- Combined-stack verification is green: 56 focused and 732 scoped Python tests
  (two optional CAD-dependency skips), real CPython 3.10.20 checks, scoped
  static analysis, and repository governance gates. With no web or Rust delta,
  the exact parent's green 345 React and 12 Rust tests remain applicable. The
  inherited 17-error broad MyPy Qt/NumPy baseline in 11 untouched files remains
  outside scope.

## 2026-08-10 Propagation into wedge ground clearance

- Immediate child PR `#4174` keeps base `feat/4163-impact-inspector`.
- Its original head `880a6465fc872cf3d6650283db154ddc41793a31`
  normally merges exact parent `#4173` head
  `9ddaff3b6bca542fd7a2befc7d7b0ae53910a60a`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC repair and AST guard are inherited alongside the swept
  wedge ground-clearance model, persistence, PyQt, and React surfaces.
- Combined-stack verification is green: 56 focused and 703 scoped Python tests
  (two optional CAD-dependency skips), 345 React tests and all web gates, 12
  Rust tests, real CPython 3.10.20 checks, scoped static analysis, and
  repository governance gates. The inherited 17-error broad MyPy Qt/NumPy
  baseline in 11 untouched files remains outside scope.

## 2026-08-10 Propagation into impact inspector

- Immediate child PR `#4173` keeps base `feat/4144-variation-visualizations`.
- Its original head `3c43955aaeb3964ff8c3ef2748d626baae518b76`
  normally merges exact parent `#4167` head
  `22b66b560652b78de84141344c4ddd9a92a83b26`; no branch rewrite, retarget, or
  force push is permitted.
- The Python 3.10 UTC compatibility repair and its AST guard are inherited
  additively alongside the existing shared wedge impact-inspector contract.
- Combined-stack verification is green: 63 focused and 562 total Rate Python
  tests; 334 React tests plus type-check, lint, and production build; 12 Rust
  tests; real CPython 3.10.20 compile/UTC checks; scoped static analysis and
  repository governance gates. The broad MyPy sweep retains 17 pre-existing
  Qt/NumPy typing findings in 11 untouched files. Current-head protected CI and
  required review remain pending release evidence.

## Dependency position

PR `#4167` (`feat/4144-variation-visualizations`) is the base-most open Rate
feature above the already merged `feat/investigation-suite` carrier. Later
wedge, D-plane, launch-monitor, workspace, wind, capability, and ground work
depends on this line and must receive any repair through ordinary parent
propagation; child branches must not be rewritten.

## 2026-08-10 Python 3.10 repair

- Protected CI at exact pre-repair head
  `edaa56358a9ccf47809533fcab28e6415b336771` collected 13 Rate test modules
  unsuccessfully because `datetime.UTC` does not exist on Python 3.10.
- The torque-profile controller now consumes the repository's existing
  `shared.python.compatibility.UTC` export.
- A source-tree AST guard rejects future direct imports and unaliased or
  aliased `datetime.UTC` module-attribute access anywhere under
  `src/rate_of_closure`.
- Local evidence is green: 27 focused controller/history/AST tests and the
  complete 554-test Rate suite on Python 3.13; real CPython 3.10.20 compatibility
  import; Ruff check/format; focused pinned MyPy 1.13; detect-secrets;
  touched-file size and diff checks.

## Truthful release state

This is an actionable compatibility repair, not completion of issue `#4144` or
the variation epic. Current-head protected CI, required review, dependency
propagation, and ordinary merges remain required. Runner download timeouts,
missing toolcache/link libraries, cancelled jobs, and queued jobs are tracked
as infrastructure and are never counted as green evidence.

Every implementation commit must update this file, both other canonical
handoffs, and `SPEC.md`, or explicitly record no material handoff change and
the reason.
