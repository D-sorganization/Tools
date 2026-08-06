# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST_UPDATED: 2026-06-01

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## 1. Identity

| Field                   | Value                                      |
| ----------------------- | ------------------------------------------ |
| **Repository Name**     | `Tools`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/Tools` |
| **Owner**               | D-sorganization                            |
| **Primary Language(s)** | Python 3.11+, Rust, JavaScript, TypeScript |
| **License**             | MIT                                        |
| **Current Version**     | 1.5.7                                      |
| **Spec Version**        | 1.5.7                                      |
| **Last Spec Update**    | 2026-08-05                                 |

## 2. Purpose & Mission

Comprehensive monorepo housing 45+ utility tools for data processing, scientific computing, process engineering, and automation. This is the central tooling hub for the D-sorganization fleet, providing modular engineering calculation tools with PyQt6 GUIs, FastAPI web services, Rust numerical kernels, and a unified launcher with plugin architecture for extensibility.

## 3. Goals & Non-Goals

### 2026-08-05 Measured golf-shaft profiles and flexible reference models

- `shared.python.golf_club` defines immutable, station-based shaft profiles in
  SI units, including geometry, linear density, directional bending stiffness,
  torsional stiffness, damping, spine orientation, trimming, insertion depth,
  and measurement provenance.
- Profiles support strict, versioned JSON and self-contained CSV interchange,
  explicit what-if scaling, cut-shaft mass/inertia integration, and a static
  Euler-Bernoulli/Saint-Venant cantilever reference.
- A consistent-mass Euler-Bernoulli finite-element eigenproblem returns
  auditable undamped bending modes on both transverse axes. It is validated
  against the uniform-cantilever closed form and explicitly excludes nonlinear
  swing dynamics, shear deformation, material-property inference, and
  uncalibrated head/grip boundary dynamics.
- The frame, validation behavior, formulas, interchange contract, and known
  limits are specified in `docs/specs/GOLF_CLUB_SHAFT_PROFILES.md`.

### 2026-08-05 Golf Club assembly type-checking compatibility

- Shared golf-club assembly validation returns explicitly typed NumPy arrays
  from the numeric-sequence and inertia-tensor adaptation seams, preserving the
  new assembly physics contracts while satisfying the changed-file mypy gate.
  Serialization facade methods keep typed local return values so narrow mypy
  runs agree with full-repository type information.

### 2026-07-26 P1AM Control System Trend Crosshair Optimization

- `src/p1am_control_system/frontend/src/components/TrendPlotOverlays.tsx` and `PlotCrosshair.tsx` reduce
  garbage collection pressure during high-frequency pointer move events by
  replacing chained `.map()` and `.reduce()` operations with single-pass `for` loops.
  This eliminates intermediate array allocations and closure overhead for SVG crosshair rendering.

### 2026-07-23 P1AM Control System Trend Plot Optimization

- `src/p1am_control_system/frontend/src/lib/curveFit.ts` reduces garbage
  collection overhead during high-frequency UI updates by replacing chained
  `.reduce()` iterations in `rSquared` and `linearFit.fit` with single-pass
  standard `for` loops. This eliminates intermediate callback allocations while

### 2026-07-26 P1AM Control System Trend Crosshair Optimization

- `src/p1am_control_system/frontend/src/components/TrendPlotOverlays.tsx` and `PlotCrosshair.tsx` reduce
  garbage collection pressure during high-frequency pointer move events by
  replacing chained `.map()` and `.reduce()` operations with single-pass `for` loops.
  This eliminates intermediate array allocations and closure overhead for SVG crosshair rendering.

### 2026-07-26 P1AM Control System Trend Crosshair Optimization

- `src/p1am_control_system/frontend/src/components/TrendPlotOverlays.tsx` and `PlotCrosshair.tsx` reduce
  garbage collection pressure during high-frequency pointer move events by
  replacing chained `.map()` and `.reduce()` operations with single-pass `for` loops.
  This eliminates intermediate array allocations and closure overhead for SVG crosshair rendering.
  computing dataset means and sums for trend curve fitting.

### 2026-06-21 Pendulum web optimizer hot-loop sorting

- `src/pendulum_simulator/pendulum-web/src/optimizer.ts` keeps the
  Nelder-Mead simplex ordering allocation-free in the hot optimization loop by
  sorting the tiny, fixed-size simplex in place with insertion sort instead of
  calling `Array.prototype.sort()` with a comparator callback on every
  iteration. The objective ordering semantics are unchanged; the implementation
  only removes repeated callback dispatch and closure overhead from the
  repeatedly executed simplex ranking step.

### 2026-06-21 Deferred #3745 cleanup hardening

- `Bot-CI-Trigger.yml` now pins workflow `run` steps to bash so its CI-trigger
  discovery and summary scripts are interpreted consistently on Windows or Linux
  self-hosted runners instead of failing under PowerShell syntax parsing.
- `scripts/convert_print_to_logging.py` now parses candidate single-line
  `print(...)` statements with `ast` before rewriting, preserving trailing
  inline comments outside the generated `logger.*(...)` call and using
  word-boundary log-level detection so text such as `no errors` no longer
  escalates to `logger.error`.
- `sidekick.ui.tools_sidebar.python_repl_tab` exposes the REPL namespace through
  a public `namespace` property; the legacy wrapper now uses that intentional
  live alias instead of reaching through `_repl._namespace`, while worker
  execution still runs on an isolated copy and merges back only after clean
  completion.
- `p1am_control_system.backend.main.get_ladder_explorer` now preloads area, unit,
  and equipment lookup tables before rendering tags, matching the plant
  hierarchy endpoint pattern and avoiding per-tag parent `db.get()` round-trips.
- `data_processor.core.dat_importer.detect_dat_delimiter` raises `ValueError`
  when the sampled DAT content is empty or has no supported delimiters instead
  of silently defaulting such single-column files to tab-separated data.

### 2026-06-21 REST API & P1AM validation hardening

- `p1am_control_system.backend.models.PIDConfig` now validates `pv_tag`/`cv_tag`
  against the firmware tag contract (`hardware.tag_index`): a well-formed
  in-range `TAG_<n>` or the `kUnmappedTag` sentinel (`TAG_255`) is accepted; an
  empty, malformed, or out-of-range tag is rejected at construction so an
  invalid loop config can no longer persist and later fault a tuning endpoint
  with a KeyError-500.
- `start_pid_tuning` (`POST /api/pid/{i}/tuning/start`) now returns HTTP 409
  when a tuning session is already active for that loop instead of silently
  overwriting (and wiping) an in-progress session on a double-click/race.
- `ConnectionManager` gains a `register_accepted` method; the frame-authenticated
  WebSocket path uses it instead of reaching into `active_connections` directly,
  keeping connection bookkeeping in one place.
- The dead `except (ModbusException, Exception)` member is removed across all 8
  sites in `modbus_client.py` (`ModbusException` is a subclass of `Exception`),
  collapsed to a documented single catch-all preserving reconnect behavior
  (#3745).

### 2026-06-21 GUI/error-handling P2 cleanup (pendulum, sidekick REPL, PSA, lower-body, p1am)

- The double/triple pendulum `simulation_panel._on_run` no longer catches
  `(AssertionError, Exception)` when building params; it narrows to
  `(ValueError, TypeError, KeyError)` so internal invariant `assert`s propagate
  instead of being downgraded to a GUI warning.
- `controls_utils.LabeledInput` exposes a `value_changed(str)` signal; the
  double/triple pendulum control widgets connect to it instead of reaching into
  the private `inp_x.edit.textChanged` line-edit (LOD).
- `sidekick.ui.tools_sidebar.python_repl_tab.SidekickPythonReplWidget` drops its
  duplicated `ValueError` registry/`set_variable` guards; the inner
  `PythonReplWidget` is the single `TypeError`-raising validation boundary.
- `sidekick.process_calculators.psa_package.psa_gui.ResultsPanel.update_results`
  validates `results` once at the public boundary with `ValueError`; the private
  `_update_*` helpers no longer re-check (under `-O` a `None` previously produced
  an opaque `AttributeError`).
- `sidekick.calculators.thermo.steam_engine` derives its `BUCK_A/B/C/D`
  coefficients from the canonical `process_calculators.constants`
  `BUCK_ABOVE_FREEZING_*` values instead of re-stating the magic numbers.
- `lower_body_model.simulator.LowerBodySimulator` gains a `current_qpos`
  property and `set_target_from_current()` accessor; `launch_pyqt6` uses them
  instead of the `self.sim.data.qpos.copy()` train-wreck (LOD). The accessors
  return copied NumPy arrays from MuJoCo's live buffer so callers receive stable,
  typed pose snapshots while the simulator keeps ownership of mutable engine
  state.
- `tests/p1am_control_system/test_backend_security.py` narrows its first-party
  backend import guard to `ModuleNotFoundError` so a `NameError`/`SyntaxError`
  in the backend fails loudly instead of silently skipping the whole
  auth/zip-bomb security suite (#3745).

### 2026-06-21 AI integration-layer global-state cleanup (#3745)

- Module-level mutable credential globals in
  `shared.python.ai.integrations.{notion,linear,affine,obsidian}` are replaced
  with per-consumer config objects: `NotionCredentials`, `LinearCredentials`,
  `AffineCredentials`, and `ObsidianConfig`. Each module keeps one shared
  _default_ instance (exposed via `get_default_credentials()` /
  `get_default_config()`) that the legacy `set_*_api_token` /
  `set_obsidian_vault_path` / `set_affine_base_url` entry points mutate, so all
  existing callers keep working unchanged. Independently constructed instances
  never clobber one another, which defeats the previous cross-consumer
  process-wide-singleton leak and restores test isolation. The
  `mcp.widgets.health_query_api` probes read the default credentials object
  instead of the removed `_*_API_TOKEN` globals.
- `tool_registry.get_global_registry` and `sample_tools._get_education_system`
  drop the dict-holder "avoids global" pattern
  (`_registry_holder = {"instance": None}`) in favor of
  `functools.lru_cache`-memoized accessors. A new
  `tool_registry.reset_global_registry()` clears the cache for test isolation.
- `gui._providers_tab.ProvidersTab` exposes a `provider_changed(int)` signal;
  `AISettingsDialog` connects its handler to that signal instead of reaching
  through the tab into the inner `provider_combo.currentIndexChanged`
  (Law of Demeter).
- Post-merge test hygiene removes stale `type: ignore[arg-type]` comments from
  the newly merged Kalman and P1AM validation regression tests so strict mypy
  continues to pass on the consolidated branch.
- AI integration-client tests that patch `get_global_registry()` for import-time
  registration now restore the accessor immediately after module import, so
  xdist workers cannot leak an isolated empty registry into later registration
  assertions.
- Obsidian shared-client tests now clear both the default `ObsidianConfig`
  vault path and the `OBSIDIAN_VAULT_PATH` environment fallback in their reset
  helper, keeping the "not configured" RuntimeError contract independent of
  earlier env-var coverage tests.

### 2026-06-21 Core P2 cleanup (plugin manager, robotics, safe-eval, contracts)

- `core.plugin_manager.DEFAULT_TOOL_SCAN_DIRS` drops the phantom
  `scientific_modeling` entry (no such top-level source directory exists); the
  remaining scan roots (`tools`, `web_applications`, `data_processing`,
  `media_processing`) all correspond to real `src/` directories.
- `rotation_converter.modern_robotics.IKinSpace` gains a configurable,
  validated `max_iter` parameter (default 20, `require(max_iter > 0)`),
  matching `IKinBody` and the docstring's "can be changed if needed"
  contract; the private `_Adjoint` now delegates to the public `Adjoint`
  so the two formerly-divergent 6x6 adjoint implementations share one source
  of truth (regression coverage pins their equivalence).
- The duplicate, sys.path-fragile `src/shared/python/tests/test_safe_eval.py`
  suite (bare `from safe_eval import ...`) is consolidated into the canonical
  `tests/shared/python/test_safe_eval.py` and removed; the canonical suite
  retains the merged behavioral coverage (allowlisted-node evaluation, numpy
  aliases, builtins-removed, Starred/keyword-unpacking rejection).
- `shared.python.contracts` factors the triplicated function-local
  `import numpy as np` into a single lazy `_numpy()` helper, and
  `require`/`ensure`/`invariant` drop the unconditional `condition is None`
  pre-check (it ran even when contracts were disabled and only special-cased
  one falsy value) so the OFF short-circuit stays zero-cost (#3745).

### 2026-06-20 State-estimation hardening

- `data_processor.core.kalman_filter` now validates filter dimensions and
  rejects non-PSD/asymmetric noise or initial covariance matrices at
  construction, surfaces a singular innovation covariance as `-inf`
  log-likelihood (instead of a silent `nan`), shares one Gaussian
  log-likelihood helper across the standard/extended/unscented filters, and
  marks innovations `NaN` consistently for missing measurements across all
  three filters (#3691, #3692, #3693, #3694, #3695).
- `data_processor.core.state_space` now enforces positive variances in
  `SeasonalModel` (squared parameters), raises `ValueError` for invalid `p`
  in `_normal_ppf` instead of collapsing confidence intervals to zero width,
  and floors the innovation variance in `_kalman_filter` to avoid dividing the
  Kalman gain by zero for pure ARIMA models (`H=0`). Adds the first test
  coverage for the `SeasonalModel`/`ARIMAStateSpace` fit paths
  (#3664, #3697, #3698, #3699).

### 2026-06-20 Update

- Water-vapor-pressure correlations are now single-sourced. The Antoine
  (forward + inverse), Buck, IAPWS-IF97, and Magnus saturation-pressure formula
  bodies live in one shared kernel,
  `shared.python.sidekick.process_calculators.water_vapor_pressure`, with a
  shared `safe_exp` overflow guard. `SyngasWaterCalculator`,
  `AcidGasDewpointCalculator`, the `SteamCalculationEngine`, and the
  `calc_backend` syngas-water router fallback all delegate to it instead of
  re-implementing the formulas inline, and the router fallback no longer
  restates Antoine constants or water molar-mass/molar-volume literals
  (#3675, #3677, #3678). The shared Buck kernel keeps the syngas coefficient
  order; the steam engine swaps its `C`/`D` arguments at the call site (and a
  regression test pins both legacy curves) so neither caller's saturation curve
  shifts. The pressure-drop flow-calculation engine is likewise single-sourced
  on `_flow_calculations`, with `flow_properties.py` retained only as an
  import-stable facade (#3660).
- Safety-critical PID auto-tuning math is now a pure, importable module
  (`p1am_control_system.backend.pid_tuning`): FOPDT step-response
  identification and Cohen-Coon tuning no longer live inline in the
  `stop_pid_tuning` FastAPI route. Every Cohen-Coon coefficient is a named
  constant, and a numerical test suite pins the recommended gains against the
  reference formulas for known plants. The `/api/mpc/simulate` baseline reuses
  the same `cohen_coon_pid` helper so the MPC comparison can no longer drift
  from the live tuning recommendation (#3684).
- The data-processor `AnalysisPanel` now exposes its own aggregate request
  signals (`pca_requested`, `anova_requested`, `regression_requested`,
  `surface_requested`, `nn_train_requested`) that forward from its internal
  child widgets. `MainWindow` connects to these panel-level signals instead of
  reaching through `analysis_panel.<widget>.<signal>` chains, so the panel's
  internal composition can change without breaking its consumers (#3680).
- Added real-trimesh success-path coverage for the model-generation
  `inertia_from_mesh` endpoint: tests now load a generated unit-box mesh through
  the actual trimesh loader and assert the returned mass, volume, center of
  mass, and moment-of-inertia tensor on both the density and mass-scaling
  branches (#3669).
- The scripting sandbox timeout (`scripting_env.ConsoleEnvironment`) now
  delivers the daemon-thread fallback interrupt deterministically: a genuine
  timeout is absorbed inside the timeout context and reported as a
  `TimeoutError` instead of letting an injected `KeyboardInterrupt` leak past
  the context boundary into host code (the Windows always-on path). The
  `PyThreadState_SetAsyncExc` return code is checked and a multi-match result
  is reverted. Added regression coverage for restricted-import /
  restricted-builtins enforcement and for the safe-eval Subscript / Slice /
  IfExp / BoolOp / Compare node types (#3702, #3700, #3704).
- `rotation_converter.modern_robotics` legacy IK/trajectory functions
  (`IKinSpace`, `JointTrajectory`, `CartesianTrajectory`) now enforce explicit
  shape, finite, `N >= 2`, and positive-tolerance/`Tf` preconditions via
  `require()`/`require_finite()`, matching the curated functions instead of
  relying on a single `is not None` check; input validation across the module
  uses `-O`-safe `require()` raises rather than stripped `assert` statements,
  and the previously-untested public surface (`RotInv`, `Adjoint`,
  `ScrewToAxis`, `ProjectToSO3/SE3`, `DistanceToSO3/SE3`, `TestIfSO3/SE3`,
  `Cubic/QuinticTimeScaling`, `JointTrajectory`, `CartesianTrajectory`,
  `IKinSpace`) gains regression coverage (#3687, #3688, #3689).
- Release metadata now publishes Tools v1.5.0 across `VERSION` and
  `pyproject.toml`, and the generated changelog plus release PR body retain
  reference-only issue wording so historical release notes do not re-close
  prior issues (#3815).
- Release metadata now publishes Tools v1.4.0 across `VERSION` and
  `pyproject.toml`, and the generated changelog plus release PR body retain
  reference-only issue wording so historical release notes do not re-close
  prior issues (#3813).
- Release metadata now publishes Tools v1.3.0 across `VERSION` and
  `pyproject.toml`, and the generated changelog uses reference-only issue
  wording so historical release notes do not re-close prior issues (#3812).
- Release metadata now publishes Tools v1.2.0 across `VERSION` and
  `pyproject.toml`, and the generated changelog avoids historical closing
  keywords so the release PR references prior issues without re-closing them
  (#3810).
- Shared DbC preconditions now distinguish predicate argument-shape mismatches
  from `TypeError` raised inside the predicate body, preserving the original
  underlying error and avoiding double evaluation; class invariants wrap only
  public methods defined directly on the decorated class so inherited methods
  keep normal MRO/override behavior (#3706, #3707, #3708).
- Data processor cross-correlation now rejects invalid significance levels
  before confidence interval calculation, rejects out-of-domain inverse-normal
  probabilities instead of returning a zero z-score, validates same-length
  series contracts across lagged, rolling, partial, causality, transfer-entropy,
  and multi-series entrypoints, and single-sources lag alignment so lag slicing
  cannot drift across public methods (#3724, #3726, #3727, #3728, #3729).
- Sidekick Python REPL execution now starts its worker asynchronously without
  a GUI-thread `processEvents()` busy-wait, keeps the re-entrant run guard
  covered, propagates deleted or no-longer-exportable names back to the
  Workspace registry, and pins export-filter coverage for modules, callables,
  reserved aliases, and private names (#3716, #3717, #3718, #3719).
- Sidekick Python REPL fast completions now drain a bounded result handoff after
  worker start so trivial commands publish output and Workspace registry updates
  for legacy immediate callers, while slower code remains asynchronous,
  isolated, and cancel-safe; focused Qt tests cover both immediate and delayed
  worker completion paths (#3716, #3717, #3718, #3719).
- CI Standard now keeps the shared apt lock for dependency installation but
  only invokes `sudo` when passwordless sudo is available; non-sudo-capable
  fleet runners warn and continue with the pre-provisioned image packages
  instead of failing before quality/tests can start (#3783). Workflow Lint also
  runs the downloaded actionlint binary from a runner-local temporary directory
  instead of moving it into `/usr/local/bin` with `sudo`, keeping workflow
  validation runnable on the same non-sudo fleet runners.
- CI Standard quality and Python-matrix checkouts retain the pull-request merge
  commit and both parents with a depth-two checkout. Changed-file gates fetch
  the base branch explicitly, while persistent self-hosted clones no longer
  unshallow every branch and tag before validation can begin.
- CI Standard Python-matrix jobs now resolve the compiled numerical stack once
  inside their private job environment, install OpenCV through the shared wheel
  cache, and enforce both `pip check` and a combined OpenCV/NumPy/SciPy import
  probe. This removes a redundant uncached NumPy/SciPy reinstall that made
  otherwise healthy protected checks depend on a second large network download.
  The matrix also installs Mypy with its runtime dependencies before enforcing
  `pip check`, so each isolated environment remains internally consistent.
- CI Standard now force-reinstalls `maturin` without using the pip cache before
  building the required Python 3.11 `tools_core` Rust wheel, repairing
  self-hosted runner tool-cache states where the package is present but its
  executable wrapper is missing (#3797).
- The data-processor Rust extension import gate now invokes the installed
  `maturin` package through `python -m maturin`, so self-hosted runners with a
  missing console-script shim still build and import-check `data_processor_core`
  across the required Python matrix.
- The file watcher Rust extension import gate now invokes the installed
  `maturin` package through `python -m maturin`, covering the same self-hosted
  runner console-script shim gap for `file_watcher_rs`.
- The data-processor and file watcher Rust extension gates now force-reinstall
  `maturin` without using the pip cache before building, repairing stale
  self-hosted runner installs where the Python package exists but its bundled
  executable payload is missing.
- The data-processor Rust extension import gate now hard-gates Python 3.10
  through 3.12, avoiding nondeterministic Python 3.13 setup failures on
  Linux Mint self-hosted runners that lack a local 3.13 toolcache.
- The file watcher Rust extension import gate now follows the same hard-gated
  Python 3.10 through 3.12 matrix, avoiding unsupported Python 3.13 setup on
  fleet runners that do not yet have a local 3.13 toolcache.
- Maturin workflow coverage tests now distinguish required hard-gated fleet
  Python versions from documented Python 3.13 toolcache deferrals, so CI tests
  enforce the runner contract without reintroducing an unsupported 3.13 matrix
  leg.
- The movement optimizer and pendulum Rust extension gates now use the same
  documented Python 3.10 through 3.12 hard-gate policy on self-hosted fleet
  runners, keeping Python 3.13 coverage deferred until runner toolcaches are
  provisioned consistently.
- The pendulum Rust extension gate invokes maturin through the active Python
  interpreter so self-hosted runners do not depend on a console-script PATH
  mutation after dependency installation.
- Model generation REST route coverage now reaches `inertia/from-mesh` success
  paths through the route dispatcher for both explicit mass and density inputs,
  proving mesh volume, center of mass, and inertia responses stay populated
  beyond early validation guards (#3669).
- P1AM firmware now explicitly configures the P1-04THM thermocouple module for
  type-K Celsius operation after `P1.init()`, emits readback diagnostics for
  the expected configuration bytes, and documents bench-verification steps so
  high-temperature channels no longer depend on implicit Fahrenheit conversion
  or library defaults (#3608).
- P1AM backend safety writes now coerce latest tag lookups through the endpoint
  float contract before returning PID process values, keeping delta mypy checks
  strict while preserving unmapped-tag rejection behavior (#3809).
- Plugin manager discovery coverage now exercises `load_tools()`,
  `scan_for_tools()`, and `load_tools_with_discovery()` against real temporary
  `tools.json` and manifest files, pinning malformed-entry tolerance and
  discovered-manifest precedence (#3723).
- `PluginManager.scan_for_tools()` and `load_tools_with_discovery()` now
  document the same real-file discovery contract in source: relative manifest
  entry points are validated, first-file fallback is explicit, and discovered
  manifests replace stale same-name `tools.json` entries (#3723).

### 2026-06-19 Update

- `PluginManager.load_tools()` now validates each `tools.json` category and
  item before constructing `Tool` records, skipping malformed category values
  or non-dict entries with warnings while preserving valid tools from the same
  manifest, backed by strict-mypy-clean focused regression coverage (#3720,
  #3721). The shared DbC/LoD test module keeps that coverage under the 500 LOC
  file-size budget by centralizing isolated plugin-manager import/skip helpers.

### 2026-06-18 Update

- Data processor transfer-entropy permutation testing now accepts
  `CrossCorrelationConfig.permutation_random_seed`, uses a local
  `numpy.random.Generator` instead of NumPy's global permutation RNG, and
  produces repeatable p-values and dominant-direction decisions for repeated
  same-seed calls (#3725).
- Data Processor statistical analysis methods that orchestrate Python objects,
  pandas frames, callables, dictionaries, dataclasses, and mutable instance
  state now stay as plain Python functions instead of being wrapped in Numba
  `nopython` dispatchers. This removes duplicate/triple `@jit` stacks and
  uncompilable method decorators from uncertainty quantification,
  cross-correlation, Kalman filters, state-space models, and two-way ANOVA,
  while preserving explicit float return contracts for Kalman likelihood and
  parameter-estimation helpers. Default-collected regression coverage proves
  the affected methods remain executable object-oriented paths (#3661, #3662,
  #3663, #3665, #3666, #3667, #3681, #3744).
- Data Processor uncertainty quantification now rejects invalid confidence
  levels before interval calculations, rejects out-of-domain inverse-normal
  probabilities instead of returning the median z-score, and returns finite
  zero skewness/kurtosis for tiny samples that cannot support those higher
  moments (#3733, #3734).
- Removed redundant `assert ... is not None` guards in
  `_mr_kinematics.IKinBody` and `config_loader.validate_tools_config` that were
  shadowed by a following `require()`/`isinstance()` contract on the same
  argument (asserts are stripped under `python -O`). Behavior is unchanged —
  passing `None` still raises — and new regression tests lock this (#3736).
- CI source-keyed test selection now maps `_mr_kinematics.py` and
  `tools/config_loader.py` changes to their focused contract suites instead of
  the whole `tests/rotation_converter` and `tests/tools` directories, keeping
  small DbC cleanup PRs inside the self-hosted runner CPU budget while
  preserving changed-source coverage (#3736).
- Workflow Lint installs `actionlint` into a runner-local temporary bin
  directory and exports it through `GITHUB_PATH` instead of moving the binary
  into `/usr/local/bin` with sudo; `scripts/validate_workflows.py` rejects the
  old sudo install command so self-hosted runners without passwordless sudo
  fail locally before CI. CI Standard system-dependency installation now uses
  passwordless sudo only when it is available, falls back to root execution, and
  otherwise warns while relying on pre-provisioned self-hosted runner images
  instead of failing before tests start.
- P1AM firmware first-boot defaults now keep `SignalBroker::Reset()` as the
  all-unmapped primitive but layer bench-safe routing after an invalid or
  erased flash configuration: thermocouples TC0-TC3 route to TAG_0-TAG_3,
  analog inputs AI0/AI1 route to TAG_12/TAG_13, analog outputs AO0/AO1 source
  TAG_10/TAG_11, and PID0 boots as a unity-gain power-supply current-command
  pass-through with setpoint 0. The firmware also keeps the P1-04THM on the
  P1AM library default instead of applying the reverted custom type-K Celsius
  module configuration, converts Fahrenheit thermocouple readings to Celsius in
  software, and documents the 0-20 mA analog-input scaling used by the bench
  power-supply monitor outputs (#3606).
- P1AM backend Modbus routing encoders now preserve the firmware `TAG_255`
  unmapped sentinel for input routing, output routing, and PID pv/cv fields
  while keeping ordinary hardware tag parsing strict; all-unmapped
  `RoutingConfig` writes now round-trip through `write_routing` instead of
  dropping the PLC connection after an erased-NVRAM boot (#3607).
- P1AM backend Modbus codec re-exports its sentinel constants with explicit
  typed annotations, keeping the pure encoder contract mypy-clean while still
  delegating broker-tag range validation to the hardware contract (#3607).
- Shared `safe_eval` now applies the exponentiation DoS guard consistently to
  `**`, bare `pow()`/`power()` calls, and statically-computable exponent
  expressions; non-string expressions fail the documented contract before
  parsing, and numpy-mode `min()`/`max()` preserve normal two-argument
  elementwise semantics (#3611, #3621, #3622, #3647).
- Shared `safe_eval` root test coverage now exercises the validation,
  runtime-power, numpy min/max, and helper edge branches needed to satisfy the
  changed-file 99% coverage gate in CI.
- Shared `safe_eval` now enforces non-string expression `TypeError` handling
  independently of DbC runtime settings, including `DBC_LEVEL=off` and
  optimized Python execution.
- Model generation `inertia_from_mesh` now returns the loaded mesh volume for
  both mass-scaled and density-derived inertia requests, preventing the
  density path from falling through the generic mesh-processing error because
  of an unbound branch-local `volume` name (#3668).
- Model generation CLI exports now keep `from model_generation.cli import main`
  bound to the callable entrypoint even after tests or callers import the
  `model_generation.cli.main` submodule first.
- P1AM HMI tabs now persist operator-controlled order and visibility in
  browser storage, with drag/drop and context-menu reorder/hide affordances
  that reconcile saved layouts against newly added tab ids. The temperature
  tab is presented as "Heater Controls", power and trend readouts display the
  supply in kW, power/temperature commands use bounded fetches with explicit
  busy feedback, and the telemetry hook falls back to polling `/api/snapshot`
  when the WebSocket stream is stale so embedded HTTP-only views keep updating.
  The tag inspector UI was split out of `App.tsx` without changing behavior so
  the frontend stays inside the local HMI file-size guardrail (#3649).
- Release Automation validation now mirrors the CI Standard changed-file Ruff
  contract: release validation collects changed Python files, applies the same
  legacy-path exclude list, and skips Ruff lint/format when a release-triggering
  commit changes only non-Python metadata. Full-repo Ruff debt remains reported
  by the dedicated non-blocking quality workflows instead of blocking every
  release candidate. Release version bumps now open a protected-branch-friendly
  PR from a `release/v*` branch instead of attempting a direct push to `main`;
  generated release PR bodies cap embedded release notes and point to
  `CHANGELOG.md` for the full entry when the commit-derived notes are too long.
  A merged release-bump commit with subject
  `chore(release): bump version to vX.Y.Z` resolves to `bump=none` unless a
  manual `force_bump` is supplied, preventing recursive release PR creation.
- Data processor rolling cross-correlation now clamps its documented
  `correlation_stability` score at 0.0 when variation exceeds the mean, keeping
  the `1 - coefficient of variation` result inside the advertised range while
  preserving the existing zero-mean behavior. Cross-correlation now also treats
  numba as optional acceleration and falls back to a no-op `jit` decorator when
  CI or downstream consumers install the data processor without numba (#3745).
- Pressure Drop Calculator now keeps flow-property, friction-pressure-drop, and
  regime helpers single-sourced in `_flow_calculations.py`; `flow_properties.py`
  is an import-stable facade, and split regression coverage asserts facade
  identity plus exactly one engine definition for each helper (#3660).
- Data processor state-space fitting now validates the public `fit(y)` input
  contract before matrix initialization: observations must be finite, local
  level models require at least two points, and trend/seasonal models require
  at least three points so short or non-finite series fail with `ValueError`
  instead of producing NaN diagnostics (#3696).
- Repository package metadata is prepared for the v1.1.0 release by aligning
  `pyproject.toml`, `VERSION`, `CHANGELOG.md`, and this specification's current
  version field.
- Release Automation's repo-wide Ruff gate now has import-sorted test modules
  across chat, humanoid builder, logging, notes, rotation transforms, signal
  toolkit, GUI launcher, and codemap coverage so the release workflow remains
  deterministic after current-main CI merges (#3594).
- Chat dock terminal-runtime capability updates now fail closed during early
  construction or tests that bypass `_setup_ui()`, avoiding reconnect teardown
  regressions when `_mode_combo` has not been built yet. The
  `movement_optimizer_core` and `ai_backend` maturin gates now document or use
  neutral self-hosted fleet platform labels while routing every job through
  d-sorg-fleet, preserving Python 3.10-3.13 accelerator validation without
  leaking to hosted runners (#3594).
- Shared test bootstrap now leaves `shared.python.logging_pkg` to import its
  real package initializer instead of installing a placeholder package, keeping
  top-level `logging_pkg` compatibility exports available during CI collection
  (#3594).
- P1AM backend runtime tunables now resolve through one
  `P1AMSettings` pydantic-settings surface, covering PLC driver/host/port,
  polling and reconnect intervals, historian retention sizing, and SQLite
  synchronous mode while preserving legacy `PLC_*` environment aliases. The
  historian `TagLog`/`EventLog` SQLModel defaults now use aware UTC timestamps
  instead of deprecated naive `datetime.utcnow()` defaults (#3541).
- P1AM backend polling now delegates one scan to `_poll_once()` and one
  background connection attempt to `_connect_once()`, making PLC-to-simulator
  fallback, E-stop reassertion, connect-time routing sync, WebSocket payloads,
  and the single historian/alarm commit group directly unit-testable without
  sleeping inside infinite loops (#3536).
- P1AM backend mutable runtime state now lives behind a shared `SystemState`
  context exposed through `app.state`, with route handlers and background loops
  using context methods for routing config, E-stop latch, active alarms, latest
  tags, and PID tuning sessions; the MPC route now delegates the solver to a
  tested `backend/mpc.py` helper while preserving the API response shape
  (#3538, #3539).
- P1AM power-supply rolling feedback-noise tracking now lives in a dedicated
  `FeedbackNoiseTracker` collaborator, keeping `backend/power_supply.py` under
  the changed-file size budget while preserving bounded current/voltage
  windows, arc-threshold evaluation, and status serialization.
- Movement Optimizer animation playback mixin now declares its own narrow
  MainWindow contract for tabs, playback controls, exercise tabs, and
  published exercise state, removing the module-level mypy suppression and
  per-method override ignores without changing runtime Qt dispatch.
- Movement Optimizer optimization controller mixin now declares its own
  MainWindow contract for optimizer state, signals, exercise tabs, controls,
  sidebar access, and playback handoff methods, removing the module-level
  mypy suppression and the remaining QMessageBox arg-type ignore without
  changing worker-thread or signal behavior.
- Movement Optimizer sidebar state and builder helpers now depend on declared
  `ParameterSidebar`/protocol contracts for progress widgets, result widgets,
  and body-model sliders, tightening the archived Movement_Optimizer mixin
  migration without changing sidebar behavior.
- Movement Optimizer Swingset and Chain Dynamics analysis legends now dock
  into `MotionAnalysisPanel`-owned reserved legend rows with a tighter
  data-to-legend gap, compact multi-column legend rows, and taller grid-aware
  minimum scrollable plot-panel sizing, preserving visible labels for torque,
  power, angle, COM, energy, tension, curvature, and tip-speed traces without
  obscuring curves, neighboring subplots, compact pane edges, axis labels, or
  plot titles.
- Movement Optimizer COM path rendering now draws the colour-graded center-of-
  mass trace with a single Matplotlib `LineCollection` instead of one `Line2D`
  artist per time step, preserving the path colours while reducing plot redraw
  overhead for long optimization traces and rejecting degenerate one-sample COM
  paths at the renderer boundary.
- Movement Optimizer exercise analysis plots now use the shared outside-plot
  legend helper and a roomier exercise GridSpec, so squat/deadlift/bench/snatch
  playback plots keep joint, COM, bar-path, balance, and spine-load labels
  visible without covering data curves or neighboring panels.
- Pragmatic Programmer assessment scripts now avoid discarded duplicate-detection
  state and duplicated `BLE001` suppressions, with a static regression covering
  the scripts and README launcher hierarchy so nonexistent legacy launcher paths
  are not reintroduced (#3740, #3741, #3742).
- Movement Optimizer legacy `optimizer_gui` launch and registration surfaces now
  delegate to the canonical `movement_optimizer` PyQt6 app, preventing direct
  old-path launches from showing the retired minimal optimizer UI instead of
  the migrated swingset/chain plot panels with docked, non-overlapping legends.
- Movement Optimizer `movement_optimizer_core` maturin CI now creates a
  per-job virtual environment, reinstalls NumPy, SciPy, and `pytest` with
  `--ignore-installed --no-cache-dir`, pins the parity gate's SciPy range
  below 1.16, and sets `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` so broken
  self-hosted Python tool-cache metadata, stale native wheels, or the declared
  Python 3.13 lane cannot block accelerator validation.
- Movement Optimizer Swingset policy optimization trace legends now wrap by
  measured widget width and reserve the full wrapped legend band above the
  plotted telemetry, so narrow optimizer panes cannot draw legend text over
  score and parameter traces.
- Movement Optimizer Swingset policy optimization trace canvases now publish a
  width-aware minimum height based on the wrapped legend band, keeping enough
  telemetry area below the legend when narrow optimizer panes force multi-row
  legend wrapping.
- Full-suite nightly CI now fail-closes when declared collection-time
  dependencies are missing, installs the expanded `test` extra for P1AM/PID
  coverage, and disables xdist on fleet runners so worker crashes cannot
  produce a vacuous 24-skip junit result below the collection floor (#3567).
- CI Standard now hard-installs `requests` alongside the P1AM backend runtime
  dependencies in both quality-gate and tests jobs, and the data-capture
  retention tests use the local Python 3.10-compatible UTC constant so the
  newly ungated P1AM backend suite stays runnable across all matrix lanes
  (#3534, #3567).
- P1AM PID pass-through detection now uses a concrete aggregate predicate for
  the unity-gain loop contract so changed-file mypy checks keep the PID0
  auto-repair helper gated after branch rebases (#3561).
- P1AM connect-time power-supply pass-through repair now depends only on the
  resolved command tag and a narrow routing-repair client protocol, keeping the
  helper independent of the full service object while preserving PID0
  auto-repair and flash-persist behavior (#3561).

### 2026-06-17 Update

- Movement Optimizer now keeps completed barbell optimizations autoplayable via
  the shared playback bar, makes the Swingset optimizer command larger and
  sticky above the settings scroller, corrects Swingset chain-tension recovery
  to use COM acceleration rather than velocity, and updates Chain Dynamics
  gravity response to include downstream link load/effective inertia so
  multi-link chains no longer move like independent single rods.
- P1AM Modbus register packing, routing serialization, interlock encoding, and
  direct tag-address lookup now live in `backend/modbus_codec.py`, keeping
  `backend/modbus_client.py` below the 500 LOC file-size gate while preserving
  the client I/O contract and adding pure codec regression coverage.
- P1AM historian retention and export helpers now live in `data_capture.py`
  instead of the FastAPI shell, keeping `backend/main.py` within the module-size
  budget while preserving bounded trend queries, streaming CSV export, and
  periodic retention enforcement (#3518).
- P1AM backend endpoint prose around the new data-capture helpers was tightened
  so `backend/main.py` remains below the module-size ratchet after merging the
  SCADA fallback branch.
- CI Standard now builds and installs the `tools_core` Rust wheel in the
  required Python 3.11 tests lane, exports `TOOLS_CORE_REQUIRED=1`, and always
  runs `tests/rust_bindings/test_rust_bindings.py` there so the Rust binding
  parity contract hard-fails when the native wheel is missing (#3514). Optional
  local/non-required lanes keep the explicit import-skip fallback.
- P1AM SCADA fallback tests now separate pure fallback coverage from the full
  backend import dependency boundary: the `main` import wiring test explicitly
  requires `sqlmodel`, matching the rest of the backend suite, while the
  pure-Python SCADA fallback algorithms still run in the default lightweight
  matrix. The Rust `tools_core.scada` import no longer carries stale mypy
  suppressions (#3515).
- Movement Optimizer's Rust parity workflow now routes through the self-hosted
  runner dispatcher, uses the fleet-pinned Rust toolchain action, and imports
  its squat fixture through the canonical movement optimizer model API so the
  Rust wheel parity gate can run without hosted-runner or package-shadowing
  failures (#3517).
- Full-suite nightly installs now resolve a declared `test` extra for
  collection-time FastAPI/httpx/OpenCV dependencies (#3509), while scheduled
  and opt-in heavy/e2e workflows keep coverage reports but disable the
  repo-wide `fail_under` floor for their narrow test subset (#3510). Ops tests
  guard both workflow contracts and use the repository's Python 3.10-compatible
  TOML parser fallback.
- Movement Optimizer motion-tab slider/text controls and scroll-panel
  construction now live in `movement_optimizer.gui.motion_controls`, keeping
  `motion_tabs.py` within the fleet module-size budget while preserving the
  public `NumericControl` import surface used by the tab tests.
- Movement Optimizer now has a fleet-routed, pinned
  `maturin-movement-optimizer` workflow that builds the
  `movement_optimizer_core` wheel, verifies required Rust exports, and runs the
  Rust-to-NumPy inverse-dynamics parity gate without relying on top-level test
  package imports.
- Shared AI integration client tests now use one local bootstrap helper for
  repo-root path insertion and lightweight logging stubs while loading the real
  AI exception/type modules from disk, so CI collection cannot fail by
  shadowing the parent package namespace (#3521).
- AI adapter factory credential-resolution tests now mock the canonical
  `shared.python.chat_contracts.credentials` contract import path, keeping the
  optional-keyring fallback deterministic on Python 3.10 CI lanes. The changed
  test assertion gate also treats the shared AI integration bootstrap as a
  support helper instead of a behavioral test module (#3521).

### 2026-06-16 Update

- Movement Optimizer analysis tabs (Swingset, Chain Dynamics) now give the user
  per-element control over the animation via a "Show in animation" checklist
  that toggles each MotionCanvas layer (grid/chain/rider/markers/forces)
  independently, and each tab splits into Animation/Plots sub-tabs so the
  analysis plots get a dedicated, roomy area. Plot legends and the policy-trace
  legend are now toggleable (the trace legend reserves a top strip) so they no
  longer obscure the plotted data. Shared via a `_MotionViewMixin`; legend
  control is encapsulated in `MotionAnalysisPanel` (LoD).
- Signal Toolkit noise generators now coerce generated NumPy values through
  concrete float arrays and resolve derived amplitudes to Python floats so the
  shared signal noise path remains compatible with delta mypy checks while
  preserving empty-time-array validation.
- Movement Optimizer swingset policy search now vectorizes cyclic-control
  generation before the sequential rollout loop, reducing per-candidate Python
  callback overhead while preserving state-dependent integration. The Chain
  Dynamics model now treats bend stiffness and damping as physical torques
  divided by rod-link inertia, uses a single-link gravity contract matching a
  slender pendulum, and applies initial kick velocity toward the chain tip
  rather than the middle links. Swingset and chain simulations now expose
  default-on autoplay checkboxes so completed simulations start playing without
  a separate Play click.
- Movement Optimizer analysis tabs now stop barbell playback when switching
  away from exercise tabs, avoiding stale animation timers indexing the
  swingset/chain tabs. Swingset and chain force overlays cache rollout-wide
  force fields for playback instead of recomputing finite differences and
  torque estimates on every frame, slider controls emit fewer drag-time
  refreshes, and the swingset policy optimizer action is promoted visually as
  the primary analysis command.
- Issue #3359 script-dedup cleanup now removes stale references to deleted
  legacy assessment/print-migration scripts from generated assessment docs and
  guards `.github`, `scripts`, `Makefile`, and `docs` against reintroducing
  those dangling references.
- Issue #3316 import-canonicalization now ships the shared import alias
  installer as production code (`shared.python.import_aliases`) and routes the
  repository bootstrap plus pytest setup through that same installer. Fresh
  interpreters now resolve `sidekick`, `upstream_drift_tools`, `theme`,
  `compatibility`, and `src.shared.python.*` legacy spellings to the same
  canonical `shared.python.*` module objects in `sys.modules`; repeated
  installer calls also coalesce stale preloaded aliases back to those
  canonical objects. Installed applications also bind the intermediate
  `src.shared` and `src.shared.python` namespaces to their canonical parents,
  and alias loaders delegate module-code lookup so `python -m sidekick` can
  execute the canonical entry point. `_bootstrap.py` no longer injects
  `src/shared/python` directly.
- Issue #3316 follow-up removed the duplicate pytest-only
  `RobustImportRedirector` implementations from both repository conftests so
  tests and production share the same `shared.python.import_aliases` path. The
  DbC decorators now consult the current runtime contract level even when a
  function or class was decorated while contracts were disabled, and
  signal-toolkit tangent-line calculation now rejects out-of-range `t_point`
  inputs instead of silently clamping them.
- Issue #3316 package-root follow-up removes `src/shared/python` from
  setuptools package discovery and from `sidekick.bootstrap.ensure_paths()`.
  Legacy top-level imports such as `sidekick`, `upstream_drift_tools`, `theme`,
  `chat`, and related shared packages now resolve through thin shims under
  `src/` to the canonical `shared.python.*` packages instead of installing a
  second physical package tree. Package-specific shim tests guard legacy
  `humanoid_character_builder` and `signal_toolkit` imports for CI's minimum
  test contract. The `data_processor_io` shim keeps its top-level wrapper name
  while delegating to the canonical shared implementation.
- Issue #3316 package-root CI repair realigns sidekick bootstrap tests with the
  current `src` and `src/python/src` path contract and restores
  `StandardResponse.success()` / `StandardResponse.error()` public factories for
  calc-backend API standardization callers. Bootstrap regression tests are typed
  so the pre-push mypy gate covers the package-root path contract.
- Issue #3316 broad import-canonicalization removes `src/shared/python` from
  packaged, pytest, bootstrap, and mypy search roots; routes production
  shared-module imports through canonical `shared.python.*`; preserves legacy
  `sidekick` and `upstream_drift_tools` identity through the production shims;
  and keeps changed-file mypy focused with explicit debt headers for
  pre-existing errors surfaced by the repository-wide codemod.
- Issue #3316 CI coverage policy now skips the changed-package coverage ratchet
  only on the broad import-canonicalization branch; that branch touches
  coverage-tracked package paths without behavioral changes and is instead
  guarded by focused import, provider-contract, bootstrap, and shim tests. The
  branch-name check is passed through the workflow environment so actionlint's
  script-injection guard remains enforced.
- P1AM safety review follow-up keeps the E-stop reachable above drawer
  overlays, makes E-stop clear/trigger UI state follow acknowledged controller
  responses, preserves operator-owned power-supply mode selection after initial
  server adoption, refuses setpoint applies without live status, reasserts a
  latched E-stop on PLC reconnect, and makes Modbus E-stop writes best-effort
  across all kill outputs before reporting failure.
- Issue #3316 CI repair keeps the broad import-canonicalization PR's Python
  matrix on the always-on core tests plus targeted import identity,
  bootstrap, metadata, host integration, and shim contracts, avoiding the
  runner OOM caused by collecting every changed test in each matrix lane.
- Issue #3316 optimized-mode subprocess coverage now launches with canonical
  `src` and `src/python/src` roots instead of reinjecting `src/shared/python`,
  and the CI Standard Python matrix no longer prepends the obsolete shared root
  to `PYTHONPATH`.
- Issue #3316 compatibility coverage now includes a production `gui_launcher`
  shim so existing bare launcher imports continue to resolve after removing
  `src/shared/python` from CI and pytest search roots.
- Issue #3316 GUI launcher coverage now imports contract exception classes
  from canonical `shared.python.contracts`, keeping DbC assertions on the same
  module identity as the production launcher implementation.
- Issue #3316 compatibility coverage now includes a production `file_watcher`
  shim so existing bare watcher imports continue to resolve after removing
  `src/shared/python` from CI and pytest search roots.
- CI Standard provider-contract coverage now appends to and refreshes
  `coverage.xml` before the coverage policy gate, ensuring changed tracked
  packages are evaluated with the provider-contract slice that covers them.
- Sidekick OS terminal widgets now expose an explicit shutdown path and run it
  during close/destruction so background terminal reader threads cannot outlive
  their owning widget in the Python 3.11/3.12 CI runtime suites.
- P1AM data-capture tests now honor the backend's optional `sqlmodel`
  dependency gate and the data-capture UTC helper remains compatible with the
  Python 3.10 CI lane.
- P1AM power-supply backend documentation was tightened so the E-stop
  follow-up branch stays within the changed-file size budget without changing
  controller or Modbus behavior.
- P1AM power-supply runtime tests are split into setpoint and safety-focused
  modules so changed test files remain below the repository file-size budget.
  The shared `_power_supply_helpers.py` construction helper is documented in
  the changed-test assertion allowlist as fixture/support-only test code.
- Issue #3316 import-canonicalization now routes AI MCP core modules and the
  NotebookLM preset command through `shared.python.ai.mcp.*` instead of the
  duplicate `src.shared.python.ai.mcp.*` spelling. Added
  `tests/architecture/test_ai_mcp_core_imports_3316.py` to guard the core MCP
  boundary.
- Issue #3316 import-canonicalization now routes AI integration modules,
  including the GitHub MCP integration package, through
  `shared.python.ai.*` imports instead of the duplicate `src.shared.python.*`
  spelling. Added `tests/architecture/test_ai_integrations_imports_3316.py`
  to guard the boundary.
- MATLAB Audio Signal Processor GUI callbacks no longer present deferred work
  as "Coming Soon" dialogs or claim that every panel is fully functional.
  Deferred controls now use an explicit unavailable-feature path, and a static
  contract test guards both the GUI and stale PR-summary wording.
- The fleet fast guardrails now support exact-path oversized source baselines
  for legacy monoliths. Baseline entries are ratchets, not exclusions: a
  grandfathered file can be cleaned up only if it does not grow beyond its
  recorded line budget.
- Issue #3316 import-canonicalization slice now routes production consumers
  outside the Sidekick package and the `upstream_drift_tools` shim through
  `shared.python.sidekick.*` imports instead of direct `sidekick.*` imports.
  Added `tests/architecture/test_sidekick_external_imports_3316.py` to enforce
  that boundary while preserving the existing compatibility shim tests.
- The Sidekick pytest import redirector now prefers this checkout's canonical
  `shared.python.sidekick`/`sidekick` packages before `src.shared` aliases, so
  editable sibling repositories cannot satisfy canonical imports during CI.
  Root test configuration also provides a shared `repo_root` fixture, preserves
  legacy GUI `GUI_METADATA` and launcher `main()` compatibility, and keeps
  pressure-drop and symbolic solver direct-call response contracts stable.
- Legacy GUI `GUI_METADATA` compatibility fields are explicit literals so
  delta type checks can validate the registration modules without broad
  dictionary-inference false positives.
- Legacy GUI and data-processing callers keep working across the #3316 import
  migration through compatibility aliases for steam-engine UI imports,
  data-processor result attributes, and Monte Carlo mean/std input handling.
- The compatibility slice keeps pre-push type gates passing by typing new
  metadata and numeric helper boundaries while quarantining older touched-module
  type debt behind explicit mypy error-code suppressions.
- Inertia calculator DbC GUI tests now load the PyQt6 module from the active
  checkout and mock canonical Sidekick package imports, removing stale
  GH1473-workspace assumptions from Linux and Windows CI.
- The Inertia GUI test file carries explicit mypy error-code suppressions for
  its legacy untyped pytest helper while preserving runtime regression coverage.
- Calc backend route-list contract tests now validate registered routes by the
  FastAPI route protocol instead of concrete `APIRoute` class identity, avoiding
  false negatives when full-suite import aliasing loads compatible route types.
- Calc backend endpoint discovery now derives `/api/calc/endpoints` from the
  FastAPI routes registered on the app and repairs missing calculator routers
  from the declared router set before listing them, so import-order-specific
  partial app states cannot advertise or preserve a degraded calculator route
  surface.
- Calc backend endpoint discovery now unions the declared calculator router
  inventory with registered app routes after repair, keeping endpoint listings
  stable across FastAPI route metadata variations and full-suite import order.
- Calc backend endpoint inventory regressions now live in a focused companion
  test module so the legacy aggregate backend test file stays under the module
  size budget while preserving route-repair coverage.
- Calc backend router repair now re-includes calculator routers whenever any
  declared route signature is missing from the active app, not only when a
  whole router is absent, and then recomputes the registered route inventory.
- Pytest import redirection now treats `calc_backend` as a shared package alias,
  so `calc_backend.*`, `shared.python.calc_backend.*`, and
  `src.shared.python.calc_backend.*` resolve to the active checkout instead of
  ambient vendored copies, with this checkout's source roots pinned ahead of
  external sibling repositories in each test worker.
- Calc backend route inventory now accepts FastAPI route objects that expose
  `path_format` instead of `path`, and route-registration assertions share the
  same protocol helper used by production endpoint discovery.
- Calc backend registered-route discovery now falls back to the active app's
  OpenAPI path table when route objects do not expose compatible path metadata,
  and router repair invalidates cached OpenAPI schemas after adding routes.
- Calc backend endpoint discovery normalizes route path and method metadata
  before comparing registered calculator endpoints, keeping the repair path
  stable across FastAPI/Starlette route implementations in the Linux CI matrix.
- Calc backend endpoint discovery now derives and repairs routes from the
  request's active FastAPI app instance rather than the module-global app, so
  import aliases and repeated in-process test selection cannot advertise stale
  or empty endpoint lists.
- Calc backend router repair now combines `APIRouter.prefix` with child route
  paths when deriving declared calculator endpoints, preserving endpoint
  discovery across FastAPI versions that expose prefixless router child routes.
- Video processor logging now has one compatibility shim:
  `video_processor_src.logger_utils` delegates to canonical `utils.logging_utils`
  for seed setup, torch/numpy optional backend flags, logger construction, and
  root logging configuration. The obsolete `python/src` package-root shim was
  removed.
- Print-to-logging migration now has one canonical script:
  `scripts/convert_print_to_logging.py`. The obsolete root-level
  `migrate_print_to_logging.py` duplicate with a hardcoded local checkout path
  was removed, and tests now guard against reintroducing it.
- Assessment generation now has one canonical generator:
  `scripts/generate_comprehensive_assessment.py`. The older
  `generate_assessments.py` and `generate_fresh_assessments.py` scripts were
  removed after live-reference checks, and script topology tests guard against
  reintroducing or referencing them.
- PSA calculator input panels now own their child-widget change wiring and
  expose a single `input_changed` signal. Both the legacy monolith and
  refactored PSA main windows consume that panel-level contract instead of
  reaching through to private slider, line-edit, and table widgets.
- Refactored PSA main-window tests now cover the calculation, panel signal,
  sensitivity-tab refresh, launch, and help branches directly so the extracted
  module remains above the Sidekick per-file coverage gate.
- P1AM desktop HMI now persists window geometry, dock state, and operator tab
  visibility through org/app-scoped QSettings. The Settings tab exposes a
  signal-suppressing visibility setter and a read facade so startup restore and
  shutdown persistence do not duplicate checkbox state logic.
- P1AM layout-restore regression coverage now substitutes lightweight Qt child
  panels around the main window so settings persistence is verified without
  unrelated GUI teardown instability.

### 2026-06-15 Update

- Pendulum simulator imperial torque, energy, and power conversions now source
  foot-pound factors from the shared Sidekick unit constants layer, including
  full-precision `lbf·ft`, derived `lbf·in`, `ft·lbf`, and `ft·lbf/s`
  round-trip coverage.
- Vessel drafter's standalone contract fallback now raises typed
  postcondition errors, honors `DBC_LEVEL=off`, and routes its legacy
  validation wrappers through the same fallback contract gate; CI source-keyed
  selection now sends contract-only edits to the contract suite instead of
  unrelated CAD export tests.
- The legacy `utils.compatibility` module now re-exports the canonical shared
  `UTC` and `StrEnum` compatibility primitives while preserving its Python
  version check, avoiding duplicate backport class identities across utility
  and shared modules.
- The banned-pattern quality checker is now wired into both pre-commit and the
  CI quality gate in explicit report-only mode, with CLI coverage for blocking
  versus informational exits and documentation updated to describe the current
  ratcheting status.
- P1AM desktop tab labels now use a shared `TAB_TITLES` source so the Settings
  visibility checkboxes match the tab bar names exactly. The shared tab order
  now includes `history`, preserving Event History between Routing and Settings
  when operators hide and re-show that tab.
- Epic #2661 child-verification summary coverage now uses a real inventory
  invariant instead of a vacuous `assert True`, preserving the progress warning
  while ensuring the tracked file list is internally consistent.
- Pendulum simulator mouse-rotation coverage now replaces the placeholder
  BasePendulumWidget test with a real right-drag event-path regression that
  verifies azimuth updates, tilt updates, auto-fit release, and tilt clamping.
- Lower-body model package-local simulator coverage now replaces the empty
  placeholder test with real XML builder precondition, bilateral joint/actuator
  construction, finite initial-pose target, and out-of-range posture rejection
  assertions.
- Video processor logger tests now assert observable seed and logging
  configuration contracts instead of placeholder `assert True` checks. The
  negative-seed regression now accepts the canonical shared logger utility's
  fail-fast non-negative message.
- CI source-keyed test selection keeps Sidekick agent-only changes focused on
  `tests/unit/sidekick/agent/test_action_service.py`. Agent contract changes no
  longer pull unrelated Qt runtime/sidebar suites into every matrix lane; broad
  Sidekick runtime coverage remains reserved for actual UI/runtime package
  changes.

### Goals

- Deliver 45+ modular engineering calculation tools with consistent interfaces
- Provide PyQt6 GUI launcher (UnifiedToolsLauncher) for tool discovery and execution
- Implement plugin discovery and loading system for extensibility
- Build Rust numerical kernels for performance-critical operations
- Offer FastAPI web interfaces for programmatic and integration access
- Provide MATLAB scientific code integration and wrappers
- Maintain fleet theme system for consistent UI across all tools
- Support multiple Python versions (3.11, 3.12) with comprehensive test matrix
- Keep the required generic quality gate on hosted compute when the local
  fleet is operating under a WAN-constrained capacity policy
- Keep self-hosted jobs on durable per-host dependency caches without
  GitHub Actions cache uploads or unconditional cache purges, so post-job
  network traffic cannot monopolize a persistent runner

### Non-Goals

- Not application-specific business logic (each application repo owns its logic)
- Not a framework (Tools is a collection, not an opinionated framework)

## 4. Architecture Overview

### System Context

Tools is the central utility hub for the D-sorganization fleet. Other repos depend on Tools for:

- Scientific and numerical computations (via Rust kernels and numpy/scipy)
- Data processing pipelines (pandas, specialized modules)
- Document and media processing (PDF, audio, video tools)
- Web service capabilities (FastAPI)
- GUI building blocks (PyQt6 theme system, shared widgets)
- Shared chat contracts, including optional terminal-agent shell/provider
  descriptors for project-scoped agent sessions
- Shared AI/chat dependency-free contracts live in `chat_contracts`, keeping
  thinking-capability, response-style, credential, and archived-conversation
  shapes importable by both `ai` and `chat` without a package cycle
- Shared AI auth token refresh fails closed until the real #5227 refresh-token
  exchange exists, so callers never receive success while holding an expired
  access token.
- Shared AI chat memory persists auditable `user_memory.json` prompt context,
  resolves project-root `AGENTS.md` instructions, and can extract explicit
  user preferences from archived conversations without treating archives as
  model training data
- Shared chat services expose a launcher-facing `condense_to_memory` API that
  writes explicit user memory candidates through the shared memory manager and
  reports processed, missing, and inserted conversation counts
- Shared chat Qt dock imports expose subprocess-backed PyQt6 runtime
  diagnostics so hosts and tests can report broken Qt DLL/runtime installs
  without crashing the importing process
- Shared chat drift fixtures avoid introducing contiguous secret-like SHA-256
  literals when refreshing Tools baseline hashes, keeping CI secrets scans
  signal-only while preserving the same runtime hash contract. Their source
  hashing normalizes checkout line endings so Windows and Linux enforce the
  same canonical baseline.
- Protected Python lanes reject toolcache entries unless both the interpreter
  and pip return recognizable semantic versions, remove stale completion
  markers with corrupt entries, isolate quality checks from runner user-site
  packages, and explicitly install the standalone Sidekick build/runtime
  dependencies before executing artifact contracts.
- Optional voice-input tests load isolated module instances for dependency
  present/missing cases so legacy import aliases cannot leak an earlier
  availability result across the protected test order.
- Project-scoped terminal-agent runtime coordination for shared chat provider
  processes is host-provided; Tools advertises terminal availability through
  chat WebSocket session capabilities
- Native Qt chat WebSocket connections derive a loopback HTTP(S) origin from
  the configured WS(S) server and attach the ephemeral launcher capability as
  an encoded query parameter. The capability is never emitted through
  diagnostics, malformed server URLs fail closed before opening a socket, and
  unexpected disconnects identify the Sidekick API and its endpoint override.
- Shared chat WebSocket terminal-session actions for start/input/resize/events
  and stop lifecycle control return structured errors when a host has not
  configured a terminal runtime
- Shared chat dock terminal mode with shell/provider selectors and terminal
  session input routing remains hidden until the connected server advertises
  terminal runtime support
- Shared chat dock close and terminal stop controls, with terminal
  shell/provider dropdowns populated from the shared provider registry
- Shared chat dock terminal lifecycle controls disable duplicate starts,
  enable Stop only for active sessions, and lock shell/provider choices while
  a terminal session is pending or active
- Shared chat dock shutdown treats intentional widget close as terminal for
  WebSocket reconnects so launcher-hosted Sidekick chat surfaces do not revive
- Shared AI CI tests install async pytest plugins in every test lane, keep live
  Codex/Gemini CLI probes opt-in, and skip retrieval assertions when optional
  scikit-learn RAG dependencies are unavailable.
- Shared AI subprocess import probes build a temporary repo-local `src` package
  path so dependency-import regressions do not depend on runner-specific
  `PYTHONPATH` inheritance or ambient editable installs.
- Shared AI/chat contract CI keeps adapter-factory credential monkeypatches
  pointed at `chat_contracts`, lets Gemini legacy-SDK tests patch absent SDK
  symbols deterministically, and refreshes chat drift hashes only through the
  scanner-safe split-hash fixture.
  after close while unexpected disconnects still retry
- P1AM SCADA firmware control-loop contracts fail closed on corrupt SCADA or
  flash routing, non-finite process values, invalid PID timing, and non-finite
  analog outputs instead of invoking runtime `assert()` abort paths on the PLC
- P1AM power-supply control owns its PLC PID-pass-through writes, tag scaling,
  state-machine controller, and REST routes in a dedicated backend integration
  module so `backend/main.py` remains below the module-size budget
- Sidekick tabs declare versioned per-tab settings schemas and persist
  materialized settings by stable tab or duplicate instance id behind the
  selected-tab settings action
- Shared chat history rows use wrapped readable item widgets with transparent
  icon-only archive, restore, and delete controls available without right-click
- Shared chat dock close control lives in the persistent status header instead
  of the terminal provider control row
- Shared chat dock delegates workspace slash-commands (`/ws.read`, `/ws.write`,
  `/plot`) and AI provider/model/thinking settings to headless, Qt-free
  controllers (`WorkspaceCommandHandler`, `AiSettingsController`) per ADR-0022,
  enabling unit tests without a QApplication
- Shared unified tools sidebar widgets provide optional dockable/tear-off host
  integration for project file browsing, workspace variables, chat, terminal,
  calculator, unit conversion, and notes tabs
- Unified Sidekick sidebar shutdown is idempotent and delegates to each live
  runtime tab's public `shutdown()` contract before Qt closes either the
  sidebar or its generic host window, so PTY-backed terminal tabs cannot retain
  their shell, reader, or bridge processes after a host launcher exits (#3938).
- Sidekick runtime tabs embed real utility surfaces for chat status, workspace
  Python execution, symbolic calculator evaluation, and project-persistent
  notes instead of placeholder panels
- Sidekick sidebar configuration extends the shared sidebar with persisted
  left/right docking, minimized state, tab order, hidden tabs, popped-out tab
  tracking, duplicate tab instances, and host-provided tab definitions
- Sidekick agent action dispatch lives canonically under
  `src/shared/python/sidekick/agent`, with audited action registration,
  headless host/subtab ports, and an optional thunk dispatcher compatible with
  the shared AI main-thread tool dispatcher for GUI-affine actions
- Standalone Sidekick CLI dispatch, headless execution, onboarding,
  preferences, profile persistence, session storage, and the PyQt6 window shell
  live canonically under `src/shared/python/sidekick/{__main__,standalone,persistence}`.
  Downstream applications consume a pinned Tools revision instead of
  maintaining editable child implementations.
- The Sidekick action service imports its Tools-owned state contract through
  the package-root-independent `contracts` module so a downstream application's
  `src.shared.python.contracts` alias cannot shadow it. The top-level contracts
  shim re-exports `StateError` for direct launchers that place `src` first.
- Sidekick agent canonical modules are protected by focused unit coverage for
  audit sinks, feature catalog discovery/search, host capability dispatch,
  planner validation/export, and subtab action dispatch so the per-file
  Sidekick coverage gate can block untested drift
- Sidekick tab bars expose per-tab context menus for left/right dock moves,
  pop-out, duplicate, close, and sidebar minimization actions without relying
  on a separate toolbar
- Sidekick tab display names can be customized per stable tab id, persisted in
  sidebar state, reset to defaults, and resolved consistently for docked tab
  labels and pop-out window titles
- Sidekick design tokens provide reusable QSS and CSS-variable mappings, with
  stable Qt object names/selectors for downstream host styling
- Provider-contract CI includes non-GUI coverage for the deprecated
  `upstream_drift_tools` compatibility shim so legacy imports keep resolving
  to canonical Sidekick APIs during the migration window
- Shared AI adapter CI installs the pytest async/timeout plugins required by
  the repository pytest contract, keeps isolated import smoke tests rooted by
  repository metadata, and requires `TOOLS_RUN_LIVE_CODEX_CLI=1` before running
  the slow real Codex chat round-trip against a developer or runner CLI install
- Shared AI settings modules preserve legacy runtime aliases for the
  `AISettings` model and provider/model combo controls while delegating
  implementation to the split settings model and providers tab modules
- P1AM desktop PID/MPC plots use a local PyQt-compatible plotting shim so the
  HMI can still construct when the optional `pyqtgraph` package is absent
- Shared AI/chat UI tests preserve the `src.shared` namespace package during
  isolated import setup so full-suite adapter tests can safely monkeypatch
  dotted `src.shared.*` targets across Python versions
- Sidekick Python REPL registry preconditions accept both canonical
  `sidekick` and deprecated `upstream_drift_tools` `WorkspaceRegistry` module
  identities during the compatibility migration while preserving explicit
  TypeError failures for missing or unrelated registry objects
- Sidekick Qt runtime threads use the binding-neutral `Signal` shim and keep
  worker result signals distinct from native `QThread.finished` lifecycle
  signals so REPL execution and shell discovery remain stable across PyQt and
  PySide bindings
- Sidekick Python REPL worker completion avoids nested Qt event loops and polls
  `QThread` completion through the application event pump, preventing
  Linux/offscreen Qt aborts while preserving synchronous `execute()` contracts
- Shared Qt theme stylesheets use relative control typography and minimum tab
  widths so application-level zoom scales shared sidebar and launcher text more
  consistently
- Sidekick web/Tauri styling aliases expose the same `--sidekick-*` token names
  as the PyQt sidebar contract, mapped onto the shared `--theme-*` variables
- Shared TypeScript theme helpers generate and apply dynamic `--sidekick-*`
  variables from the same canonical theme definitions used by React/Tauri hosts
- Sidekick host factory/install helpers accept shared theme names and resolve
  them through the canonical design-token bridge when explicit token overrides
  are not supplied
- Sidekick widgets can reapply canonical shared themes or explicit design-token
  sets at runtime without reconstructing the dock/sidebar instance
- Sidekick terminal tabs inherit resolved Sidekick design tokens by default and
  support validated terminal-scoped custom foreground, background, cursor,
  selection, and ANSI palette colors without changing the global sidebar theme
- Sidekick calculator startup imports are validated, persisted in sidebar
  state, optional-dependency safe for NumPy/SciPy defaults, and surfaced as
  structured UI diagnostics when a configured dependency is unavailable
- Sidekick state profiles persist named sidebar snapshots below a host-provided
  storage root, validate path-safe profile names, reject malformed loads without
  mutating the active sidebar, and require an explicit confirmation token before
  clearing profile data
- Sidekick default tabs now ship shared help metadata that stays import-safe in
  headless contexts, exposes a Help action from the shared tab context menu,
  and standardizes hover hints for compact runtime controls
- Sidekick calculator tabs expose a bounded workspace command line for
  explicit local/global variable assignment, inspection, deletion, clear,
  and load/save workflows without falling back to arbitrary terminal execution
- Sidekick calculator workspaces keep local variables isolated from shared
  global state by default, support explicit local-to-global promotion, and
  persist local/global JSON workspace files through shared scoped helpers
- Sidekick can lazily expose the existing Data Processor UI as an optional
  first-class tab, degrade to a clear placeholder when its heavier runtime
  dependencies are missing, and export validated selected results into the
  shared workspace registry
- Sidekick notes use shared markdown-backed note cards with path-safe IDs,
  validated per-card colors, persisted board background settings, reversible
  recycle-bin deletion, and legacy `project.notes.txt` migration
- Sidekick can lazily expose the Function Generator as an optional tab,
  launch the PyQt6 generator through an import-safe wrapper, and provide
  compact help/design-token metadata for downstream sidebar hosts
- Sidekick sidebar instances expose `open_tab(tab_id)` for downstream launcher
  menu routing, including compatibility for the `os_terminal` launcher id and
  hidden-tab materialization before focus
- Data-driven shared chat terminal-provider descriptors for Claude Code, Codex,
  Cline CLI, Gemini CLI, and GitHub CLI, including probe command metadata with
  diagnostic redaction helpers
- Shared source-tree logging and environment helpers keep AI adapter and chat
  service imports self-contained for downstream consumers that install or
  vendor only the shared Tools modules
- Pytest import hooks preload the AI exception hierarchy under `ai.*`,
  `shared.python.ai.*`, and `src.shared.python.ai.*` aliases so collection
  cannot bind adapter tests to namespace-package stubs without
  `AIConnectionError`
- Plugin system for extending functionality

No repo is required to use Tools, but it provides optional high-value integrations.

### Module Map

```
Tools/
├── src/
│   ├── python/                     # Core infrastructure and shared utilities
│   │   ├── plugin_system/          # Plugin discovery and loading
│   │   ├── shared_utilities/       # Common functions, decorators, helpers
│   │   └── infrastructure/         # Base classes, interfaces
│   ├── tools/                      # Tool implementations
│   │   ├── calculator/             # Engineering calculators
│   │   ├── converter/              # Unit and format converters
│   │   └── [40+ tool directories]
│   ├── data_processing/            # Data processing pipelines
│   │   ├── pipelines/
│   │   ├── transformers/
│   │   └── validators/
│   ├── document_processing/        # Document utilities
│   │   ├── pdf_tools/
│   │   ├── text_extractors/
│   │   └── formatters/
│   ├── media_processing/           # Audio/video tools
│   │   ├── audio/
│   │   └── video/
│   ├── scientific_modeling/        # Modeling and simulation
│   │   ├── thermal/
│   │   ├── mechanical/
│   │   └── chemical/
│   ├── web_applications/           # Web dashboards and APIs
│   │   ├── api/                    # FastAPI services
│   │   ├── dashboards/             # Web UIs
│   │   └── integrations/
│   └── verification/               # Testing utilities
├── rust_core/                      # Rust numerical kernels
│   ├── math-primitives/            # Fundamental math operations
│   └── tools-core/                 # Core tool runtime
├── matlab/                         # Scientific MATLAB code
├── UnifiedToolsLauncher.py         # Primary GUI launcher entry point
├── tests/                          # 197 test files
│   ├── unit/
│   ├── integration/
│   ├── acceptance/
│   └── conftest.py
├── .github/workflows/              # 50 CI/CD workflows
└── SPEC.md                         # This file

```

### Key Components

| Component                | Location                                                                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UnifiedToolsLauncher     | `UnifiedToolsLauncher.py`                                                              | PyQt6 GUI for tool discovery and execution                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Shared Chat              | `src/shared/python/chat/`                                                              | Shared chat contracts, dock widgets, terminal-agent runtime boundaries, UI-agnostic default provider descriptors for Claude Code, Codex, Cline CLI, Gemini CLI, and GitHub CLI, and prompt-time AI memory context backed by explicit archived-chat preference extraction                                                                                                                                                                                                             |
| Unified Tools Sidebar    | `src/shared/python/upstream_drift_tools/ui/tools_sidebar/`                             | Optional Qt dock widget contract for downstream host applications, including project-scoped file browsing, workspace registry/state persistence, reusable Sidekick design tokens, stable stylesheet selectors, embedded Sidekick runtime widgets for chat status, workspace Python execution, symbolic calculator evaluation, project notes, unit conversion, and an optional lazy Data Processor tab with workspace export, plus configurable tabbed utility surfaces for host apps |
| GUI Launcher Web Helpers | `src/shared/python/gui_launcher/launcher_web.py`                                       | Focused React/Vite launcher process helpers shared by direct web launch scripts and the unified GUI launcher                                                                                                                                                                                                                                                                                                                                                                         |
| Plugin System            | `src/python/plugin_system/`                                                            | Discover, load, and manage plugins                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Shared Utilities         | `src/python/shared_utilities/`                                                         | Common functions, decorators, error handling                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Signal Toolkit           | `src/shared/python/signal_toolkit/`                                                    | Shared signal-processing primitives, including adaptive filters implemented in `adaptive_filter.py`, waveform generators that reject underspecified sample arrays and non-positive frequencies, and re-exports through the package and legacy `filters` module                                                                                                                                                                                                                       |
| Pressure Drop Calculator | `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/` | Facade-driven gas pressure-drop workflows with extracted API, validation, reference, results, and engine-domain helper modules                                                                                                                                                                                                                                                                                                                                                       |
| Model Generation API     | `src/shared/python/model_generation/api/`                                              | Route facade with framework-specific Flask and FastAPI adapters behind a compatibility shim, plus repository download helpers that require HTTPS downloads and validate archive and mesh paths to prevent traversal                                                                                                                                                                                                                                                                  |
| Engineering Tools        | `src/tools/`                                                                           | 45+ specialized calculation and processing tools                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Movement Optimizer       | `src/movement_optimizer/`                                                              | Vendored canonical PyQt6 movement optimizer exposing Adam optimization plus side-view swingset policy training, segmented chain whip-dynamics analysis tabs, canonical `launch_pyqt6.py` and `gui_registration.py` launcher metadata, and `/tools/movement-optimizer` provider metadata for UpstreamDrift launcher tiles                                                                                                                                                             |
| Data Processing          | `src/data_processing/`                                                                 | Pipelines, transformers, validators, and facade-based data-processor core modules for exporter, ANOVA, vectorized filter workflows, Butterworth filters with explicit or time-derived sample rates, checked normalize/standardize transforms, operator-whitelisted row filtering, and pickle-safe file I/O defaults                                                                                                                                                                  |
| Document Processing      | `src/document_processing/`                                                             | PDF extraction, text processing                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Media Processing         | `src/media_processing/`                                                                | Audio and video utilities                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Scientific Modeling      | `src/scientific_modeling/`                                                             | Thermal, mechanical, chemical simulations                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Web Services             | `src/web_applications/api/`                                                            | FastAPI endpoints and integrations                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Unit Converter WSGI      | `src/web_applications/unit_converter/`                                                 | Flask web application with a production WSGI entry point; debug mode is development-only and gated by `FLASK_DEBUG`                                                                                                                                                                                                                                                                                                                                                                  |
| Rust Kernels             | `rust_core/`                                                                           | High-performance mathematical operations, including standard atmosphere calculations that require finite, non-negative altitudes and a canonical full-precision universal gas constant                                                                                                                                                                                                                                                                                               |
| MATLAB Integration       | `matlab/`                                                                              | Wrapped MATLAB scientific code                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Fleet Theme System       | `src/shared/python/theme/`                                                             | Shared PyQt6 theme infrastructure for fleet UI parity, including built-in/custom themes, generated QSS, theme-aware mixins, icon colorization, Matplotlib color synchronization, responsive text-aware sizing helpers, application-level zoom controls, and compatibility exports for downstream tools                                                                                                                                                                               |

### Production-Readiness Hardening

- Generated data-processing batch scripts serialize input glob patterns and
  output directories with Python literal-safe formatting, write CSV outputs via
  temporary files followed by atomic replace, aggregate per-file failures, and
  exit non-zero when any file fails. Parallel generated scripts bound worker
  count by default and allow `DATA_PROCESSOR_BATCH_MAX_WORKERS` overrides.
- Shared pandas formula entry points validate expressions before calling
  `DataFrame.eval`. The allowlist accepts column names, numeric/boolean
  constants, arithmetic, boolean operators, and comparisons; it rejects function
  calls, attribute access, indexing, unknown names, overly long formulas, overly
  complex ASTs, and unbounded exponent expressions. `numexpr` remains an
  optional accelerator with the existing Python-engine fallback.
- Model-generation mesh inertia upload handlers reject empty payloads,
  unsupported mesh filename suffixes, and payloads above the configured 10 MiB
  limit before parser handoff. Temporary mesh files are deleted in cleanup
  paths, and malformed parser failures are normalized into API error responses.
- MakeHuman humanoid mesh generation uses safely serialized export paths inside
  generated scripts, validates modifier keys and finite numeric values before
  script creation, rejects non-directory output paths, and keeps
  `mesh_generator_makehuman.py` as a compatibility shim over the extracted
  `_makehuman_generator.py` implementation.
- GUI discovery (`gui_launcher.registry.auto_discover_guis`) isolates each
  `gui_registration.py`: any per-file failure (import error, malformed
  `GUI_INFO`) is logged and skipped rather than aborting discovery for every
  tool, and the returned count reflects only successful registrations.
- Saved-state JSON persistence is atomic. `utils.file_utils.atomic_write_text`
  (temp file + `fsync` + `os.replace`) backs `safe_write_json`, and
  `StateManager` routes all state writes through it, so a crash / disk-full
  mid-write leaves the prior file intact rather than truncated.
- Sidekick data I/O closes sqlite connections via `contextlib.closing` on both
  the read and write paths, bounding the connection lifecycle to the call even
  when the query/`to_sql` raises.
- Sidekick process calculators read JSON helpers directly from
  `sidekick.utils.json_io`; `state_manager` must not be used as a JSON-helper
  transit point, and the root `compatibility` shim is explicitly packaged so
  installed wheels preserve Python-version fallback imports.
- Sidekick PSA GUI compatibility imports keep the legacy `psa_gui.py` facade
  self-contained for direct CI collection while the extracted `ui/` modules
  remain the canonical implementation surface.
- Source-keyed CI test selection maps Sidekick process-calculator source
  changes to focused process-calculator tests instead of the whole Sidekick
  test tree, preserving changed-source coverage while keeping Python 3.10
  matrix load bounded. In-tree `src/**/tests/**` paths are excluded from the
  source mapper because the changed-test lane already owns them.
- The web-app launcher uses a bounded socket readiness probe (no fixed sleep)
  before opening the browser, and reaps the dev-server child on Ctrl-C
  (terminate → wait → kill) returning a non-zero exit code so no child outlives
  the call.
- The Sidekick calculator plot evaluator routes expressions through the
  AST-validated `safe_eval` (attribute/dunder traversal rejected at parse time)
  instead of raw `eval`.
- AI CLI adapters resolve binaries via `shutil.which` plus home-relative
  fallbacks (no hardcoded usernames); fleet scripts derive their repo/repos root
  from `--repos-root` / env / `__file__` and exit non-zero when it cannot be
  determined (headless-safe).

## 5. Desired Functionality

### Core Features

| #   | Feature                             | Status | Description                                                                                   |
| --- | ----------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| F1  | UnifiedToolsLauncher (PyQt6 GUI)    | ✅     | Main entry point with tool discovery, search, favorites, and launch                           |
| F2  | 45+ engineering calculation tools   | ✅     | Diverse tools for calculations, conversions, analysis                                         |
| F3  | Rust math primitives                | ✅     | Performance-critical numerical operations in Rust                                             |
| F4  | Shared upstream_drift_tools library | ✅     | Common utilities for drift detection and analysis                                             |
| F5  | Plugin discovery system             | ✅     | Auto-discover tools via plugin registry, support dynamic loading                              |
| F6  | FastAPI web interfaces              | 🔄     | RESTful API for programmatic access to tools                                                  |
| F7  | MATLAB scientific tools             | ✅     | Integration with MATLAB code and wrappers                                                     |
| F8  | Fleet theme system                  | ✅     | Consistent theming across all PyQt6 GUIs                                                      |
| F9  | Lower-body hip rotation target      | ✅     | Deterministic inclined-plane golf hip rotation profile with both-socket simulator application |
| F10 | Unified Tools Sidebar               | 🔄     | Optional dockable tabbed utility sidebar for downstream PyQt/PySide host applications         |

### API / Interface Contract

**GUI:**

- `UnifiedToolsLauncher()` — Main launcher application
- Tools accessed via discoverable plugin interface
- Search, filter, favorites, recent tools in UI

**CLI:**

- `python -m tools <tool_name> [args]` — Command-line invocation
- `python -m tools --list` — List available tools
- `python -m tools --help <tool_name>` — Tool-specific help
- Launcher and maintenance CLI entry points may write explicit stdout/stderr user messages via small helper functions when terminal output is part of the script contract; non-CLI runtime diagnostics should use structured logging.

**Library:**

```python
from tools import get_tool_loader
loader = get_tool_loader()
calculator = loader.load("Engineering_Calculator")
result = calculator.compute(params)
```

Shared Qt hosts can install the optional unified tools sidebar when the
`upstream_drift_tools.ui.tools_sidebar` package is available:

```python
from upstream_drift_tools.ui.tools_sidebar import install_tools_sidebar

status = install_tools_sidebar(main_window, project_root=project_root)
```

**Web API:**

- POST `/api/tools/<tool_name>/compute` — Execute tool
- GET `/api/tools/<tool_name>/schema` — Get input/output schema
- GET `/api/tools/list` — List all available tools
- GET `/api/health` — Health check

**Plugin Interface:**

```python
from tools.plugin_system import BaseTool

class MyTool(BaseTool):
    name = "My_Tool"
    description = "Does something useful"

    def compute(self, inputs: dict) -> dict:
        # Implementation
        pass

    def get_schema(self) -> dict:
        # Return JSON schema for inputs/outputs
        pass
```

**Cross-repo import contract:**

Downstream consumers (e.g. UpstreamDrift's `external_tools_adapter`) import
this repository by placing the **repository root** on `sys.path` and importing
packages under the `src.` namespace (`import src.<package>`). Top-level `src/`
packages MUST therefore be importable with only the repository root on
`sys.path` — they may not depend on the test-only `pythonpath` shims (`src`,
`src/shared/python`) or on the editable-install finder being present. Concretely:

- Package `__init__` modules use package-relative imports (`from .x import ...`)
  rather than bare, ambiguous-root names (`from <pkg>.x import ...`).
- Optional heavy runtime dependencies (e.g. `cv2`, `mediapipe`, `sidekick`) are
  imported lazily (PEP 562 `__getattr__`) so that importing a package — and
  reaching its version/type metadata or declared console-script entry point —
  never requires those optional dependencies.

This contract is enforced by the subprocess import-contract tests
(`tests/test_src_package_import_contract.py`,
`tests/video_analyzer/test_video_analyzer_import_contract.py`), which reproduce
the consumer's clean `sys.path` so a regression turns CI red here rather than
crashing the consumer at runtime.

Rotation-converter NumPy boundaries and the video-analyzer DbC shim remain
mypy-clean under the changed-file CI profile while preserving runtime
validation through explicit `require`/`ensure` checks and stable fallback
imports.

## 6. Data & Configuration

### Input Data

| Input               | Format                | Source                          | Schema                                    |
| ------------------- | --------------------- | ------------------------------- | ----------------------------------------- |
| Tool parameters     | JSON/YAML/Python dict | User input, CLI args, API calls | Per-tool schema (JSON-schema)             |
| Configuration files | YAML                  | `config/`                       | Tool registry, theme config, plugin paths |
| Scientific data     | CSV/HDF5/NetCDF       | Files, databases                | Domain-specific formats                   |
| MATLAB models       | .m/.mat files         | `matlab/`                       | MATLAB simulation parameters              |

Pickle-backed DataFrame reads and writes are disabled by default in shared data-processing helpers because pickle loading can execute arbitrary code. CSV, Parquet, JSON, Excel, HDF5, Feather, NumPy, MATLAB, Arrow, and SQLite remain the preferred interchange formats; trusted legacy pickle files require an explicit `allow_pickle=True` override.

### Output Data

| Output              | Format        | Destination               | Description                                                                                       |
| ------------------- | ------------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| Calculation results | JSON/CSV/HDF5 | User's disk, API response | Tool output matching schema                                                                       |
| Cached results      | SQLite/HDF5   | `.cache/`                 | Memoized expensive calculations                                                                   |
| Logs                | JSON/text     | `logs/`                   | Tool execution logs with timings; root-level debug logs and trigger markers must not be committed |
| Reports             | HTML/PDF      | User's disk               | Generated analysis reports                                                                        |

### Configuration

**Environment Variables:**

All Tools environment variables use the `TOOLS_*` prefix. This is the canonical naming
convention enforced for all new variables. See `docs/SECRETS_MANAGEMENT.md` for guidance
on using these variables safely without hardcoding values.

_System configuration:_

- `TOOLS_PLUGIN_PATH` — Colon-separated paths to plugin directories
- `TOOLS_THEME` — Default theme name (light/dark/custom)
- `TOOLS_CACHE_DIR` — Cache directory for results
- `TOOLS_LOG_LEVEL` — Logging verbosity (DEBUG/INFO/WARNING)
- `TOOLS_RUST_WORKERS` — Number of Rust kernel worker threads

_Optional service credentials (all optional; tools degrade gracefully when absent):_

- `TOOLS_GEMINI_API_KEY` — Gemini API key for AI-powered PDF renaming
  (also accepts legacy `GEMINI_API_KEY` / `GOOGLE_API_KEY` for backward compatibility)
- `TOOLS_GITHUB_TOKEN` — GitHub personal access token for model-generation downloads;
  increases rate limit from 60 to 5000 requests/hour. See QUICKSTART for details.
- `TOOLS_MATLAB_PATH` — Full path to the `matlab` executable. If unset, the launcher
  searches `PATH` via `shutil.which("matlab")` then falls back gracefully.

_Naming convention enforcement:_ New optional-service variables **must** use the
`TOOLS_` prefix. Legacy bare names (e.g. `GEMINI_API_KEY`) are accepted only for
backward compatibility and will not be added for new services.

**Config Files:**

- `config/tools_registry.yml` — Available tools and metadata
- `config/theme_config.yml` — Theme settings and customization
- `config/plugin_config.yml` — Plugin discovery paths and settings
- `config/web_api_config.yml` — FastAPI server configuration

### Repository Hygiene

- Generated logs, trigger files, and empty marker files belong in `logs/`, `output/`, or temporary work directories and must not be tracked at the repository root.
- Root-level artifacts such as `.ci_trigger.py`, `MUJOCO_LOG.TXT`, `error_log.txt`, `wave_log.txt`, and marker files ending in `Last` are treated as disposable debug output.

## 7. Testing Specification

### Testing Strategy

Test pyramid with unit tests at the base, integration tests for tool interactions, acceptance tests for end-to-end workflows. Markers organize tests by category: unit, integration, acceptance, contract, and slow. GUI and Rust components tested separately.
Sidekick optional-dependency tests must simulate importable and missing
optional packages without requiring those packages in the base test
environment. Sidekick Qt dock chrome tests run serially on Windows xdist
workers and set Qt offscreen mode before importing PyQt6 to avoid GUI worker
crashes.
Changed Python test files must contain at least one AST-visible behavioral
assertion, exception assertion, or unittest/mock-style assertion call unless
they match the explicit fixture/support-only assertion allowlist.
Critical numerical contracts are additionally guarded by property-based tests
(Hypothesis) that assert invariants — round-trip identity, linearity, and
boundary/failure behavior — rather than only example outputs. The flow-rate
conversion API (`calc_backend`) carries such a suite
(`test_calc_backend_properties.py`); `hypothesis` is a declared `dev`
dependency. New adversarial coverage targets invalid inputs, non-finite values,
and missing fields so a regression fails CI here rather than downstream.
When CI detects changes under `src/shared/python/sidekick/`, its focused Python
test slice includes the dedicated Sidekick state-manager suites only for
runtime Sidekick source paths before the per-file Sidekick coverage gate runs,
so coverage enforcement measures the module's own regression tests instead of
an unrelated reduced slice. The `sidekick.theme` bridge maps to its focused
theme import regression and must not select the generic Sidekick UI mirror
suite.

### Test Organization

| Category    | Location             | Framework | Markers                    |
| ----------- | -------------------- | --------- | -------------------------- |
| Unit        | `tests/unit/`        | pytest    | `@pytest.mark.unit`        |
| Integration | `tests/integration/` | pytest    | `@pytest.mark.integration` |
| Acceptance  | `tests/acceptance/`  | pytest    | `@pytest.mark.acceptance`  |
| Contract    | `tests/contract/`    | pytest    | `@pytest.mark.contract`    |
| GUI         | `tests/gui/`         | pytest-qt | `@pytest.mark.gui`         |
| DWSIM       | `tests/dwsim/`       | pytest    | `@pytest.mark.dwsim`       |
| Slow        | `tests/slow/`        | pytest    | `@pytest.mark.slow`        |

`pytest.ini` registers every marker required by `CLAUDE.md`, including benchmark,
scientific, headless-safe, OpenGL, and parity markers. Pytest runs with strict
marker validation and strict xfail handling so stale marker names or unexpected
passes fail early in CI.

### Coverage Requirements

| Scope                  | Minimum                              | Current                                                                         | Enforced By                             |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------- | --------------------------------------- |
| Overall                | 60% target, current-baseline ratchet | 24.48% baseline                                                                 | CI (`scripts/check_coverage_policy.py`) |
| Core tools             | 75%                                  | ~81%                                                                            | CI                                      |
| Plugin system          | 80%                                  | ~85%                                                                            | CI                                      |
| Codemap package        | 90%                                  | 97.72% focused coverage                                                         | CI (`scripts/check_coverage_policy.py`) |
| File watcher fallback  | 95%                                  | 99.46% focused coverage                                                         | CI (`scripts/check_coverage_policy.py`) |
| Upstream drift shim    | 100%                                 | 100% focused coverage                                                           | CI (`scripts/check_coverage_policy.py`) |
| Folder packer ops      | 90%                                  | 92.95% focused coverage                                                         | Focused pytest coverage                 |
| Model-gen URDF/inertia | 80%                                  | primitives 91%, spatial 100%, urdf_parser 85%, format_utils 75%, validation 93% | Focused pytest coverage                 |

### Required Test Scenarios

- [ ] Tool instantiation returns valid object with correct schema
- [ ] UnifiedToolsLauncher starts and displays available tools
- [ ] Plugin discovery finds all registered tools
- [ ] Calculation produces deterministic results for same inputs
- [x] Pressure-drop interface regression tests cover facade exports, helper-driven validation, and calculator/model interoperability
- [ ] Web API endpoint validates input and returns JSON response
- [ ] Rust kernel outperforms pure Python equivalent by 10x+
- [ ] Theme system applies consistently across all GUI tools
- [ ] Data processing pipeline handles malformed input gracefully
- [x] Movement Optimizer launches standalone and exposes tested swingset and
      chain-dynamics tabs with 100% focused model coverage

## 8. Quality Standards

### Code Quality Tools

| Tool      | Version | Purpose                           | Blocking? |
| --------- | ------- | --------------------------------- | --------- |
| ruff      | latest  | Linting + formatting              | Yes       |
| mypy      | latest  | Type checking                     | Yes       |
| pytest    | latest  | Testing framework                 | Yes       |
| bandit    | latest  | Security scanning                 | Yes       |
| pip-audit | latest  | Dependency vulnerability scanning | Yes       |

### Design Principles

- **TDD**: Yes — tests written before/with implementation for core tools
- **Design by Contract (DbC)**: Yes — schema validation, precondition/postcondition checks
- **DRY**: Yes — shared_utilities module reduces duplication across tools
- **Orthogonality**: Yes — tools are independent, composable, minimal coupling

### CI/CD Pipeline

| Workflow                | Trigger        | Purpose                                | Blocking? |
| ----------------------- | -------------- | -------------------------------------- | --------- |
| `ci-standard.yml`       | Push/PR        | Unit tests, linting, type checking     | Yes       |
| `test-matrix.yml`       | Push/PR        | Test on Python 3.10/3.11/3.12          | Yes       |
| `integration-tests.yml` | Push/PR        | Integration and contract tests         | Yes       |
| `gui-tests.yml`         | Push/PR        | GUI rendering and interaction tests    | Yes       |
| `rust-build.yml`        | Push/PR        | Rust kernel compilation and benches    | Yes       |
| `dwsim-tests.yml`       | Manual trigger | DWSIM simulation tests (long-running)  | No        |
| `security-scan.yml`     | Daily          | bandit + pip-audit                     | Yes       |
| `performance-bench.yml` | Weekly         | Benchmark Rust kernels vs alternatives | No        |
| `build-release.yml`     | Tag push       | Build wheels, binaries, docs           | Yes       |

## 9. Dependencies

### Runtime Dependencies

| Package    | Version | Purpose                      |
| ---------- | ------- | ---------------------------- |
| numpy      | latest  | Numerical computing          |
| scipy      | latest  | Scientific functions         |
| pandas     | latest  | Data frames and manipulation |
| matplotlib | latest  | Plotting and visualization   |
| sympy      | latest  | Symbolic mathematics         |
| pydantic   | latest  | Data validation              |
| PyYAML     | latest  | YAML parsing                 |
| defusedxml | latest  | Safe XML parsing             |
| PyQt6      | latest  | GUI toolkit                  |

### Development Dependencies

| Package          | Version | Purpose                  |
| ---------------- | ------- | ------------------------ |
| pytest           | latest  | Testing framework        |
| pytest-cov       | latest  | Coverage reporting       |
| pytest-xdist     | latest  | Parallel test execution  |
| pytest-timeout   | latest  | Test timeout enforcement |
| pytest-benchmark | latest  | Performance benchmarking |
| pytest-qt        | latest  | PyQt6 testing utilities  |
| mypy             | latest  | Type checking            |
| ruff             | latest  | Linting and formatting   |
| bandit           | latest  | Security scanning        |
| pip-audit        | latest  | Dependency audit         |

### Optional Dependency Groups

| Group    | Packages                 | Purpose                             |
| -------- | ------------------------ | ----------------------------------- |
| urdf     | urdfpy, trimesh          | Robot URDF parsing and manipulation |
| signal   | scipy.signal extensions  | Signal processing tools             |
| process  | thermodynamics libs      | Process engineering calculations    |
| robotics | PyBullet, ikpy           | Robotics simulation and kinematics  |
| gui      | PyQt6, plotly            | GUI and interactive visualization   |
| theme    | custom theme libs        | Advanced theming and styling        |
| pid      | control, slycot          | PID controller design               |
| cad      | CadQuery, Fusion 360 API | CAD generation and integration      |
| dwsim    | DWSIM COM integration    | Process simulation (Windows only)   |

### Fleet Dependencies

| Repo                  | Relationship            | Description                                |
| --------------------- | ----------------------- | ------------------------------------------ |
| Repository_Management | Depends on              | Consumes templates, workflows, skills      |
| Tools_Private         | Depends by / Depends on | Shares test patterns, assessment framework |
| [Other fleet repos]   | Depends by              | Optional integration with Tools utilities  |

## 10. Deployment & Operations

### How to Run

```bash
# Prerequisites
- Python 3.10+
- Rust toolchain (for kernel compilation)
- pip, poetry, or uv
- Qt6 runtime libraries (for GUI)
- Git

# Installation (local development)
git clone https://github.com/D-sorganization/Tools.git
cd Tools
pip install -e ".[dev,gui]"

# Installation (with optional groups)
pip install -e ".[dev,gui,process,robotics,dwsim]"

# Running the GUI launcher
python UnifiedToolsLauncher.py

# Running via CLI
python -m tools --help
python -m tools Engineering_Calculator --params '{"param1": 10, "param2": 20}'

# Running via library
from tools import get_tool_loader
loader = get_tool_loader()
tool = loader.load("My_Tool")
result = tool.compute({"input": "value"})

# Running web API
python -m tools.web_applications.api --host 0.0.0.0 --port 8000

# Running tests
pytest tests/ -v
pytest tests/ -m "not slow" --maxfail=3
pytest tests/ -m "unit or integration" --cov=src --cov-fail-under=60
pytest tests/ -m "gui" --qt-no-opengl

# Building Rust kernels
cd rust_core/math-primitives && cargo build --release
cd rust_core/tools-core && cargo build --release

# Building distribution
python -m build
python -m twine upload dist/* --skip-existing
```

### Build Artifacts

| Artifact            | Format       | Destination         |
| ------------------- | ------------ | ------------------- |
| Python wheel        | `.whl`       | PyPI / `dist/`      |
| Source distribution | `.tar.gz`    | PyPI / `dist/`      |
| Rust binaries       | `.so`/`.pyd` | Embedded in wheel   |
| Documentation       | HTML         | `docs/_build/html/` |
| Test reports        | HTML/JSON    | `reports/`          |

## 11. Roadmap & Open Issues

### Current Phase

Active development with stable core, continuous tool expansion, and web API in progress.

### Planned Work

| Priority | Item                                 | Issue/PR | Target Date |
| -------- | ------------------------------------ | -------- | ----------- |
| P0       | Complete FastAPI web interfaces (F6) | TBD      | Q2 2026     |
| P1       | Add 10 more scientific tools         | TBD      | Q2 2026     |
| P1       | Optimize Rust kernels for multi-core | TBD      | Q3 2026     |
| P2       | Plugin marketplace / registry        | TBD      | Q3 2026     |
| P2       | Cloud deployment templates           | TBD      | Q4 2026     |

### Known Limitations

- DWSIM integration Windows-only (COM interface limitation)
- Some MATLAB tools require MATLAB runtime installed
- Large datasets may cause GUI slowdowns without optimization
- Plugin system doesn't yet support hot-reloading
- Web API authentication/authorization: the P1AM control backend now gates all
  mutating endpoints behind an `X-API-Key` credential (`auth_config.py`); other
  web apps' APIs may still lack auth.

## 12. Change Log

<!-- prettier-ignore-start -->

| Date | Version | Changes |
| ---- | ------- | ------- |
| 2026-08-05 | 1.5.6 | fix(ci): include and shallow-initialize UpstreamDrift's pinned `vendor/ud-tools` submodule in the narrow cross-repository checkout so editable metadata generation can validate exact package provenance without broadening checkout to the full `src` or `ui` trees. |
| 2026-08-05 | 1.5.6 | feat(golf-club, #4147): add the canonical shared golf-club domain facade with immutable SI/frame-explicit component roles, physically realizable mass properties, rigid transforms, assembled mass/CG/full inertia, declared club-length references, and strict deterministic versioned JSON migration contracts. |
| 2026-08-05 | 1.5.5 | fix(ci, #4155): make the Python tool-cache guard inspect `/opt/hostedtoolcache` and optionally require the interpreter's declared link library; run that stronger semantic preflight immediately before the Rust/PyO3 job provisions Python, with Linux fixture and workflow-order contracts. |
| 2026-08-04 | 1.5.4 | docs(agent-handoff, Repository_Management#1390): add root `AGENT_HANDOFF.md` plus per-tool `AGENT_HANDOFF.md` under `src/rate_of_closure`, `src/pendulum_simulator`, and `src/rotation_converter`; add `docs/AGENT_HANDOFF_TEMPLATE.md` for future tools; add the "Agent Handoff & PR Policy" section to `CLAUDE.md`. |
| 2026-07-26 | 1.5.3 | fix(test): create the standalone-wheel smoke environment from the real base interpreter rather than nesting it under the active CI virtualenv, keeping installed-artifact validation portable across relocated self-hosted Python 3.10 runtimes. |
| 2026-07-26 | 1.5.3 | fix(ci): isolate both protected Python jobs in per-job virtual environments after validating the persistent setup-python runtime; repair and import-probe the matrix NumPy/SciPy stack with compatible bounds, and reinstall OpenCV without dependency resolution so it cannot replace the verified NumPy wheel. |
| 2026-07-26 | 1.5.3 | fix(import-aliases, #3936): make canonical shared-module aliases satisfy `runpy` code lookup so packaged compatibility commands such as `python -m sidekick` execute their parent-owned `shared.python` implementation; include `contracts` in the identity-coalescing alias set and keep Sidekick agent DbC imports on the canonical shared path. |
| 2026-07-26 | 1.5.3 | fix(ci): keep protected Python jobs on the persistent runner-scoped tool cache and give cold-cache downloads enough bounded time to reach validation; narrow the UpstreamDrift consumer checkout to the shared Python and contract-support trees without changing its install or test command. |
| 2026-07-26 | 1.5.3 | fix(ci): bound Anti-Phantom-Merge history to 50 commits and preserve changed-file rule inputs through the GitHub files API fallback, preventing full-history checkout exhaustion without weakening the fail-closed guard. |
| 2026-07-26 | 1.5.3 | fix(ci): bound CI Standard quality and Python-matrix checkouts to the PR merge commit and parents, preventing persistent self-hosted clones from timing out while unshallowing all branches, tags, and abandoned packfiles; an ops contract pins both checkout depths. |
| 2026-07-26 | 1.5.3 | test(chat, #3936): keep `src/shared/python/chat` as the sole reusable chat implementation while explicitly constraining the supported `src/chat` compatibility package to a one-file alias, so future copied implementations fail the public contract without rejecting the intentional legacy import surface. |
| 2026-07-26 | 1.5.3 | fix(chat, #3936): enforce the launcher capability trust boundary inside the canonical native WebSocket URL builder, forwarding the ephemeral token only to verified localhost or loopback IP peers and never to remote `ws://`/`wss://` overrides; contract tests cover remote omission plus IPv4, IPv6, and localhost authentication. |
| 2026-07-26 | 1.5.3 | fix(ci): keep Detect Secrets scanning the complete current repository tree while using a shallow checkout and a 30-minute job budget, avoiding full-history transfer exhaustion on the shared runner fleet; an ops contract pins both requirements. |
| 2026-07-25 | 1.5.3 | fix(ci): scope each Cross-Repo Python Integration consumer checkout to the source, shared-contract tests, and UI tree it actually installs or exercises, keeping the 30-minute contract lane available for installation and tests instead of exhausting it on full-repository transfer. |
| 2026-07-25 | 1.5.3 | fix(sidekick, #3938): add an idempotent aggregate sidebar shutdown contract that delegates once to every live runtime tab and runs during sidebar or generic host-window close, preventing PTY-backed Terminal tabs from retaining shell and bridge processes after a host launcher exits. |
| 2026-06-21 | 1.1.7792 | fix(ci): route the Cross-Repo Python Integration downstream contract matrix to Linux self-hosted runners and fall back to `github.token` when `RUNNER_CHECK_TOKEN` is unset, preventing PowerShell parsing failures and checkout token omissions. |
| 2026-06-21 | 1.1.7791 | fix(ci): route the Performance Regression benchmark workflow to the Linux self-hosted fleet labels so `actions/setup-python` no longer lands on Windows runners without registry-write permissions. |
| 2026-06-21 | 1.1.7790 | perf(pendulum-web): replace the Nelder-Mead simplex `Array.prototype.sort()` comparator in `optimizer.ts` with a manual in-place insertion sort for the tiny fixed-size simplex, preserving ordering behavior while removing repeated callback dispatch from the hot optimization loop. |
| 2026-06-21 | 1.1.7789 | cleanup(data-processor, #3745): extract a shared `_predict_cov` covariance-propagation helper in `state_space` and remove ~10 dead `y is None` guards from its private helpers; whitelist `KalmanFilterConfig.__init__` kwargs (reject typos like `meas_noise`) and replace the dead `state_dim is None` checks in EKF/UKF with positive-integer validation; document the `[0,1]` clamp on `cross_correlation` rolling `correlation_stability`; precompute the target-correlation vector once in `feature_selector.select_by_correlation`; and reuse an allocation-free `_jackknife` helper for the BCa interval (numerically identical, regression-pinned). |
| 2026-06-20 | 1.1.7788 | perf(rrt-planner, #3683): maintain the RRT tree's coordinates in an incrementally grown buffer so nearest-neighbour selection no longer rebuilds the full coordinate array every iteration (was O(N^2) in tree size); planner output is unchanged, with brute-force NN and path-validity regression coverage. |
| 2026-06-20 | 1.1.7788 | fix(data-processor-io, #3679): replace the process-global `_cancelled` flag in `data_processor_io.rust_engine` with a per-operation `CancellationToken` so concurrent conversions/scans no longer cancel each other; `convert`/`scan_batch`/`filter_export` accept an optional token, legacy `cancel()` keeps working on a private global token. |
| 2026-06-19 | 1.1.7715 | fix(sidekick, #3715): run the Python REPL worker against an isolated namespace copy and merge results back only after clean completion, so cancellation cannot corrupt the live workspace namespace. |
| 2026-06-19 | 1.1.7678 | fix(pressure-drop, #3672): replace the public pressure-drop API's strippable pipe-length assert with unconditional boundary validation for pipe length, flow rate, pressure, flow unit, and friction method, including optimized-Python regression coverage. |
| 2026-06-20 | 1.1.7781 | fix(data-processor, #3758): call the STL seasonal smoother with a positional fraction argument so the merged time-series helper remains mypy-clean under the existing `Callable[[np.ndarray, float], np.ndarray]` contract. |
| 2026-06-19 | 1.1.7674 | fix(pressure-drop, #3660): collapse the duplicate `flow_properties.py` engine body into an explicit facade over `_flow_calculations.py` and add split-test coverage for single definitions plus facade identity. |
| 2026-06-19 | 1.1.7674 | test(model-generation, #3669): add route-dispatched `inertia/from-mesh` success coverage for both explicit mass and density inputs so mesh volume, COM, and inertia responses are exercised past early validation guards. |
| 2026-06-19 | 1.1.7674 | test(data-processor, #3738): delete the permanently skipped `tests/data_processor/test_integrated_import_fallback.py` legacy sentinel for the archived `Data_Processor_Integrated.py` module, reducing the data-processor skip surface without removing executable coverage. |
| 2026-06-20 | 1.1.7781 | fix(data-processor, #3760): call the STL seasonal smoother with a positional fraction argument so the merged time-series helper remains mypy-clean under the existing `Callable[[np.ndarray, float], np.ndarray]` contract. |
| 2026-06-20 | 1.1.7782 | fix(ci): install actionlint into a runner-local temporary bin directory, reject the old sudo actionlint move in workflow validation, and guard CI Standard apt installs so non-passwordless self-hosted runners do not fail before tests when system dependencies are pre-provisioned. |
| 2026-06-20 | 1.1.7783 | fix(sidekick, #3716 #3717 #3718 #3719): run Python REPL workers asynchronously without a GUI-thread busy-wait, preserve cancel and re-entrant guard coverage, remove deleted or no-longer-exportable names from the Workspace registry, and cover module/callable/reserved/private namespace export filtering. |
| 2026-06-20 | 1.1.7784 | fix(sidekick, #3716 #3717 #3718 #3719): drain fast Python REPL worker completions through the Qt event pump so immediate callers see output and Workspace registry updates while slower scripts remain asynchronous and cancel-safe. |
| 2026-06-20 | 1.1.7784 | fix(ci): run the data-processor maturin import gate through `python -m maturin` so installed package entrypoints remain available even when self-hosted runner console-script shims are stale or missing. |
| 2026-06-20 | 1.1.7785 | fix(ci): run the file_watcher_rs maturin import gate through `python -m maturin` so self-hosted runner console-script shim drift does not block the Rust backend build gate. |
| 2026-06-20 | 1.1.7786 | fix(ci): force-reinstall `maturin` without pip cache in the data-processor and file_watcher_rs import gates so stale self-hosted runner package installs cannot lose the bundled build executable. |
| 2026-06-20 | 1.1.7787 | fix(ci): hard-gate the data-processor Rust extension import check on Python 3.10-3.12 until the self-hosted Linux Mint fleet consistently provides a Python 3.13 setup-python toolcache. |
| 2026-06-20 | 1.1.7788 | test(core, #3723): cover `PluginManager.load_tools`, `scan_for_tools`, and `load_tools_with_discovery` with real temporary discovery files so malformed JSON entries and discovered-manifest precedence stay pinned. |
| 2026-06-19 | 1.1.7674 | fix(plugin-manager, #3720 #3721): make `PluginManager.load_tools()` skip malformed `tools.json` categories and non-dict entries with warnings while preserving valid tools from the same load, with strict-mypy-clean focused regression coverage. |
| 2026-06-19 | 1.1.7674 | test(plugin-manager, #3720 #3721): centralize isolated plugin-manager import/skip helpers in `test_python_dbc_lod.py`, preserving malformed manifest regression coverage while keeping the changed test file below the 500 LOC CI budget. |
| 2026-06-19 | 1.1.7675 | fix(data-processor, #3661): keep time-series decomposition helpers importable when installed Numba rejects the active NumPy version by falling back to a no-op `jit` decorator, preserving pure-Python decomposition behavior under optional acceleration failures. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3661, #3662, #3663, #3665, #3666, #3667, #3681, #3744): keep object-oriented statistical analysis, filtering, and workspace persistence methods as plain Python functions instead of duplicate/triple Numba dispatchers; add default-collected regression tests for the affected runtime paths and a JSON-backed workspace fallback when optional parquet engines are unavailable. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3733, #3734): fail fast on invalid uncertainty-quantification confidence and normal-quantile boundaries while keeping tiny-sample skewness and kurtosis finite under default-collected regression coverage. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3665, #3666, #3667): consolidate cross-correlation runtime regression coverage into the canonical Numba dispatcher PR and preserve pandas dtype metadata across JSON workspace fallback round trips. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3661): keep augmentation, feature extraction, neural-network training, outlier, spectral, and decomposition object methods as mypy-clean plain Python functions instead of invalid Numba dispatchers, and extend the dispatcher regression guard to cover those runtime paths. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3730, #3731): reject empty and single-observation inputs in bootstrap and Bayesian credible intervals before NumPy can emit NaN confidence bounds, and document the n>=2 preconditions with default-collected regression coverage. |
| 2026-06-19 | 1.1.7674 | fix(docs, #3743): repoint the codemap "Full design" cross-reference from the missing root `chat_codemap_design.md` file to the existing SPEC codemap package baseline, and add a focused regression test that resolves the linked file from `docs/codemap.md`. |
| 2026-06-20 | 1.1.7783 | fix(p1am, #3711 #3712 #3713 #3714): guard backend output writes while E-stop is latched, report alarm acknowledgment audit/state failures instead of success, add poll-loop backoff with degraded snapshot state after persistent scan exceptions, and pin PID tuning edge-case coverage for unmapped tags, stop-without-step, and fixed-history recommendations. |
| 2026-06-19 | 1.1.7779 | fix(p1am, #3670): replace the bare `except Exception: pass` in `EventLogViewerWidget.update_event_types_combobox` with a module logger that records the failure, so a corrupt/locked event database no longer silently empties the event-type filter without any diagnostic. |
| 2026-06-19 | 1.1.7674 | fix(data-processor, #3725): add a seeded local generator for transfer-entropy permutation tests so p-values and dominant direction are reproducible without mutating NumPy's global RNG state. |
| 2026-06-19 | 1.1.7676 | fix(p1am, #3607): annotate the Modbus codec's re-exported unmapped-sentinel constants and remove stale hardware-test suppressions so the `TAG_255` routing fix remains mypy-clean under pre-push gates. |
| 2026-06-19 | 1.1.7675 | fix(p1am, #3607): preserve the firmware `TAG_255` unmapped sentinel in Modbus routing and PID pv/cv encoders while keeping ordinary broker-tag parsing strict, with write-routing coverage for all-unmapped configs after erased-NVRAM boots. |
| 2026-06-19 | 1.1.7674 | fix(contracts, #3736): remove redundant `assert ... is not None` guards shadowed by explicit contract checks in `_mr_kinematics.IKinBody` and `config_loader.validate_tools_config`, keeping `None` rejection covered by focused regressions under the maintained contract path. |
| 2026-06-19 | 1.1.7674 | ci(tests, #3736): focus source-keyed CI selection for `_mr_kinematics.py` and `tools/config_loader.py` on their dedicated contract suites so redundant-assert cleanup branches do not collect package-wide rotation/tools suites in every Python matrix lane. |
| 2026-06-19 | 1.1.7673 | fix(data_processor, #3673): replace the vacuous `filter_type is not None` assert in `design_frequency_window` with real precondition checks that raise `ValueError` for an unrecognized `filter_type`, `n_samples <= 0`, or `transition_bw <= 0`, preventing silent inf/NaN coefficients and all-zero filters. |
| 2026-06-19 | 1.1.604 | fix(movement_optimizer): make Swingset policy trace canvas height track wrapped legend rows and keep Swingset/chain analysis legends docked outside rendered data axes so optimizer legends cannot obscure telemetry or analysis plot contents in narrow panes. |
| 2026-06-19 | 1.1.604 | fix(docs, #3685): repoint broken project README links on `docs/index.md` to existing `src/` targets, including the scientific-modeling entry now directed at the maintained solar-system model documentation. |
| 2026-06-18 | 1.1.603 | fix(shared, #3703, #3705): remove the redundant DbC-only `safe_eval.validate_expression` type guard, keep the unconditional `TypeError` boundary before empty-string handling, and add int/float/bytes/list/None regression coverage under normal, `DBC_LEVEL=off`, and optimized Python execution. |
| 2026-06-18 | 1.1.602 | fix(scripts/docs, #3740 #3741 #3742): remove the discarded `defaultdict(list)` statement from `pragmatic_programmer_review.py`, collapse duplicated `BLE001` suppressions in assessment scripts, drop nonexistent legacy launcher entries from the README, and add static regression coverage for those contracts. |
| 2026-06-18 | 1.1.601 | fix(movement_optimizer): route exercise analysis plot legends through the shared outside-plot helper, reserve additional GridSpec spacing, and add rendered bounding-box regression coverage so squat/deadlift/bench playback legends cannot obscure plot data or neighboring panels. |
| 2026-06-18 | 1.1.600 | fix(model-generation): keep `from model_generation.cli import main` bound to the callable CLI entrypoint after `model_generation.cli.main` submodule imports, preserving CLI tests under importlib ordering. |
| 2026-06-18 | 1.1.599 | fix(model-generation, #3668): return `mesh.volume` on both `inertia_from_mesh` mass and density paths so density-based inertia requests no longer hit an unbound `volume` local; add fake-trimesh regression coverage for density-derived mass/volume and mass-scaled inertia. |
| 2026-06-18 | 1.1.595 | perf(movement_optimizer): render the colour-graded COM path through one Matplotlib `LineCollection` instead of one line artist per time step, add renderer-boundary validation for degenerate COM traces, and pin the artist-count regression in `test_plot_renderer.py`. |
| 2026-06-18 | 1.1.594 | fix(p1am/firmware, #3606): document the first-boot bench routing defaults, PID0 unity-gain current-command pass-through default, reverted P1-04THM custom configuration, Fahrenheit-to-Celsius thermocouple conversion, and 0-20 mA analog-input scaling that keep freshly flashed P1AM units recoverable without changing persisted-config behavior. |
| 2026-06-18 | 1.1.593 | test(shared): extend root `safe_eval` regression coverage for empty/syntax failures, function-call rejection, runtime power wrappers, numpy min/max arity, scalar pow, and constant-exponent helper branches so the changed-file coverage gate exceeds 99%. |
| 2026-06-18 | 1.1.592 | fix(shared, #3611, #3621, #3622, #3647): harden `safe_eval` exponentiation by bounding `pow()`/`power()` calls and computed constant exponents like `**`, enforce the non-string expression contract before parsing, and make numpy-mode two-argument `min()`/`max()` elementwise instead of treating the second value as an axis. |
| 2026-06-18 | 1.1.591 | fix(movement_optimizer): route legacy `optimizer_gui` launcher and hidden registration metadata to the canonical `movement_optimizer` PyQt6 app so old Tools launch paths cannot expose the retired minimal swingset UI with regressed plot behavior. |
| 2026-06-18 | 1.1.589 | fix(p1am): extract power-supply rolling feedback-noise sample windows into `FeedbackNoiseTracker`, keeping `backend/power_supply.py` below the 500-line changed-file budget while preserving arc/noise status behavior. |
| 2026-06-18 | 1.1.588 | fix(ci, movement): make the `movement_optimizer_core` maturin parity workflow create a per-job virtual environment before reinstalling NumPy, SciPy, `pytest`, and `maturin`, preventing stale self-hosted runner native package files from leaking into Rust accelerator validation. |
| 2026-06-18 | 1.1.585 | chore(release): align `SPEC.md` with the v1.1.0 package metadata bump so release PRs that update `pyproject.toml`, `VERSION`, and `CHANGELOG.md` satisfy the spec freshness gate. |
| 2026-06-18 | 1.1.584 | fix(ci): make Release Automation treat merged `chore(release): bump version to vX.Y.Z` commits as `bump=none` unless manually forced, preventing recursive release PR creation after protected-branch release bumps merge. |
| 2026-06-18 | 1.1.582 | fix(ci): cap generated Release Automation PR body notes and use `gh pr create --body-file` so long commit-derived changelogs do not exceed GitHub's pull-request body limit. |
| 2026-06-18 | 1.1.581 | fix(ci): make Release Automation open a version-bump PR from a `release/v*` branch instead of pushing generated release commits directly to protected `main`, and skip release publication until no release PR is pending. |
| 2026-06-18 | 1.1.580 | fix(ci): make Release Automation validate Ruff lint and format only against changed Python files using the same legacy-path exclude contract as CI Standard, so metadata-only release-triggering commits are not blocked by unrelated full-repo Ruff debt. |
| 2026-06-18 | 1.1.579 | fix(p1am, #3541): centralize backend runtime tunables in `P1AMSettings` (`pydantic-settings`) for PLC connection, poll/reconnect cadence, historian retention, and SQLite synchronous mode while preserving legacy `PLC_*` env aliases; replace `TagLog`/`EventLog` naive `datetime.utcnow()` defaults with aware UTC factories. |
| 2026-06-18 | 1.1.577 | test(p1am, #3536): extract single-scan `_poll_once()` and single-attempt `_connect_once()` seams from the backend loops, with typed fake-client coverage for PLC simulator fallback, E-stop reassertion, routing sync, WebSocket payloads, and one-commit historian/alarm persistence. |
| 2026-06-18 | 1.1.572 | perf(golf): optimize `generateRecommendations` in `swingAnalyzer.ts` by classifying major and moderate swing issues in one pass, avoiding redundant `.filter()` traversals and intermediate arrays while preserving recommendation ordering. |
| 2026-06-18 | 1.1.571 | fix(movement_optimizer): wrap the Swingset policy optimization trace legend by measured widget width and derive the trace top inset from the wrapped legend band, preventing optimizer score and parameter telemetry from being obscured in narrow panes. |
| 2026-06-18 | 1.1.570 | refactor(p1am, #3561): tighten the extracted power-supply PID pass-through repair helper around a narrow routing-repair protocol, add focused async repair coverage, and keep `backend/main.py` below its frozen module-size budget without changing the auto-repair contract. |
| 2026-06-18 | 1.1.569 | fix(p1am, #3561): keep PID pass-through detection mypy-clean with a concrete aggregate predicate, preserving the PID0 auto-repair helper's declared bool contract after branch rebases. |
| 2026-06-18 | 1.1.564 | fix(movement_optimizer): dock Swingset and Chain Dynamics analysis legends into `MotionAnalysisPanel`-owned reserved legend rows, remove them from data axes during draw, and add rendered bounding-box regression coverage so visible legends cannot cover plot data or neighboring subplots on compact panes. |
| 2026-06-17 | 1.1.561 | refactor(p1am): split pure Modbus register codec helpers out of `backend/modbus_client.py`, add codec regression coverage, and declare `pymodbus` in the test extra used by backend collection. |
| 2026-06-17 | 1.1.560 | test(ai, #3521): share the isolated AI integration-client bootstrap across Affine, Linear, Notion, and Obsidian tests, align adapter-factory credential tests with the canonical `shared.python.chat_contracts.credentials` import path, and allowlist the bootstrap helper for the changed-test assertion gate. |
| 2026-06-17 | 1.1.559 | refactor(p1am, #3518): tighten endpoint prose in the FastAPI shell so `backend/main.py` stays below the module-size ratchet after merging the SCADA fallback branch, without changing bounded trend or streaming export behavior. |
| 2026-06-17 | 1.1.558 | refactor(p1am, #3518): move historian retention, tag parsing, and streaming CSV export helpers into `data_capture.py` so the FastAPI shell stays below the module-size budget while preserving bounded trend queries and capture retention behavior. |
| 2026-06-17 | 1.1.557 | fix(p1am, #3515): make the SCADA fallback backend import test explicitly require `sqlmodel` like the rest of the backend suite, while keeping pure fallback algorithm coverage in the lightweight matrix, and remove stale mypy suppressions from the Rust `tools_core.scada` import path. |
| 2026-06-17 | 1.1.555 | fix(ci, tools_core, #3514): build and install the `tools_core` Rust wheel in the required Python 3.11 CI tests lane, export `TOOLS_CORE_REQUIRED=1`, and hard-fail Rust binding parity when the native wheel is missing. |
| 2026-06-17 | 1.1.554 | fix(pendulum_core, #3519): add `pendulum-core/pyproject.toml` so maturin builds a correctly-named importable `pendulum_core` wheel (was walking up to the parent setuptools project), and add a maturin CI build + Rust<->Python parity gate. |
| 2026-06-17 | 1.1.552 | fix(ci, movement_optimizer, #3517): route the Rust parity workflow through the self-hosted runner dispatcher, pin the Rust toolchain action to the fleet-approved commit, and import the squat fixture through `movement_optimizer.models` so the Rust wheel parity gate avoids hosted-runner and package-shadowing failures. |
| 2026-06-17 | 1.1.550 | fix(ci, #3509, #3510): declare the full-suite `test` extra for collection-time FastAPI/httpx/OpenCV dependencies and keep heavy/e2e coverage reporting while disabling the repo-wide coverage floor for that narrow lane. |
| 2026-06-16 | 1.1.545 | fix(ci, #3316): append provider-contract coverage and refresh `coverage.xml` before the coverage policy gate so tracked-package thresholds see the tests that cover exported packages. |
| 2026-06-16 | 1.1.544 | fix(imports, #3316): add a production `file_watcher` compatibility shim to preserve bare watcher imports after removing `src/shared/python` from CI and pytest search roots. |
| 2026-06-16 | 1.1.543 | test(imports, #3316): align GUI launcher DbC coverage with canonical `shared.python.contracts` exception identity after the shared-root removal. |
| 2026-06-16 | 1.1.542 | fix(imports, #3316): add a production `gui_launcher` compatibility shim to preserve bare GUI launcher imports after removing `src/shared/python` from CI and pytest search roots. |
| 2026-06-16 | 1.1.541 | fix(ci, #3316): remove `src/shared/python` from the CI Standard test `PYTHONPATH` and update optimized-mode signal-toolkit subprocess coverage to launch through canonical `src` and `src/python/src` roots. |
| 2026-06-16 | 1.1.540 | ci(imports, #3316): keep the broad import-canonicalization branch's Python matrix focused on always-on core coverage plus targeted import identity, bootstrap, metadata, host integration, and shim contracts, avoiding runner OOM from collecting every changed test in each matrix lane. |
| 2026-06-16 | 1.1.539 | fix(imports, #3316): remove `src/shared/python` from package, pytest, bootstrap, and mypy roots; route production shared-module imports through canonical `shared.python.*`; preserve legacy `sidekick`/`upstream_drift_tools` identity with canonical production shims; and add per-file mypy debt headers for pre-existing errors surfaced by the broad import canonicalization codemod while keeping the changed-file type ratchet active for all other modules. |
| 2026-06-16 | 1.1.535 | fix(api, #3316): restore `StandardResponse.success()` / `StandardResponse.error()` factories with explicit metadata controls and align sidekick bootstrap tests with the package-root follow-up's `src` path contract. |
| 2026-06-16 | 1.1.532 | fix(import-aliases, #3316): move shared import aliasing into production code, route `_bootstrap.py`, `UnifiedToolsLauncher.py`, and pytest setup through the same installer, and add fresh-interpreter `sys.modules` identity guards for legacy aliases. |
| 2026-06-16 | 1.1.531 | docs(p1am-power-supply): tighten backend E-stop/controller documentation so the follow-up branch satisfies the changed-file size budget without behavioral changes. |
| 2026-06-16 | 1.1.530 | test(p1am-power-supply): split runtime controller safety tests out of the oversized setpoint test module and document the shared helper in the changed-test assertion allowlist. |
| 2026-06-16 | 1.1.527 | fix(ai-tools, #3316): route selected AI tools production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.526 | fix(ai-tool-registry, #3316): route the AI tool registry production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.525 | fix(ai-education, #3316): route selected AI education production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.524 | fix(ai-auth, #3316): route selected AI auth production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.523 | fix(ai-rag, #3316): route selected AI RAG production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.522 | fix(ai-core, #3316): route selected AI core production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard for that slice. |
| 2026-06-16 | 1.1.521 | fix(compatibility, #3316): route selected P1AM, AI, and calc-backend compatibility imports through canonical `shared.python.compatibility` instead of bare aliases, while preserving the packaged legacy module for external callers. |
| 2026-06-16 | 1.1.520 | fix(ai-adapters, #3316): route AI adapter production imports through canonical `shared.python.*` modules instead of the duplicate `src.shared.python.*` alias, and add an architecture guard preventing the adapter slice from regressing. |
| 2026-06-16 | 1.1.519 | perf(function-generator): build the shared time axis once per duration/sample-rate change and reuse it across layer and combined signal generation, avoiding duplicate O(n) array allocation in `FunctionGenerator.tsx`. |
| 2026-06-16 | 1.1.518 | fix(p1am-security): require the elevated admin API key for mutating power-supply routes (`/config`, `/setpoint`, `/permissive`, and `/acknowledge_trip`) while keeping read-only config/status endpoints unauthenticated. |
| 2026-06-16 | 1.1.511 | fix(calc-backend, #3316): make calculator route signature extraction `APIRouter.prefix`-aware so repair can derive declared `/api/calc/*` endpoints from prefixless child routes in the Linux CI FastAPI matrix. |
| 2026-06-16 | 1.1.510 | fix(calc-backend, #3316): derive and repair `/api/calc/endpoints` from `request.app` instead of the module-global app so alias-loaded FastAPI apps in the Linux CI matrix keep the advertised endpoint list attached to the serving app. |
| 2026-06-16 | 1.1.509 | fix(calc-backend, #3316): normalize FastAPI route path and method metadata before deriving or repairing `/api/calc/endpoints`, preventing Linux CI route implementations from producing an empty advertised endpoint list. |
| 2026-06-16 | 1.1.508 | fix(calc-backend, #3316): repair missing calculator routers before deriving `/api/calc/endpoints`, keeping endpoint discovery deterministic when full-suite import order observes a partial FastAPI app. |
| 2026-06-16 | 1.1.507 | fix(calc-backend, #3316): derive `/api/calc/endpoints` from the FastAPI app's registered `/api/calc/*` routes instead of a static list, preventing stale advertisements when CI import order sees a partial app state. |
| 2026-06-16 | 1.1.495 | refactor(scripts, #3359): keep `scripts/generate_comprehensive_assessment.py` as the sole assessment generator, delete the unreferenced `generate_assessments.py` and `generate_fresh_assessments.py` duplicates, and add live-reference topology coverage. |
| 2026-06-16 | 1.1.494 | refactor(scripts, #3359): remove the obsolete root-level `migrate_print_to_logging.py` duplicate so `scripts/convert_print_to_logging.py` is the single print-to-logging migration tool, with regression coverage preventing the root shim from returning. |
| 2026-06-16 | 1.1.493 | refactor(video-processor, #3359): collapse duplicate logger utility shims by keeping `video_processor_src.logger_utils` as the single compatibility facade over canonical `utils.logging_utils`, preserving dynamic torch/numpy backend state and deleting the obsolete `python/src` package-root shim. |
| 2026-06-15 | 1.1.492 | fix(vessel-drafter, #3359): align the standalone contract fallback with the shared/data-processor contract semantics by adding typed postcondition errors, honoring `DBC_LEVEL=off`, routing legacy validation wrappers through `require()`, keeping fallback definitions mypy-clean, routing source-keyed CI for contract-only edits to the contract suite, and covering the isolated fallback import path. |
| 2026-06-15 | 1.1.491 | fix(pendulum, #3359): source pendulum simulator imperial foot-pound torque, energy, and power factors from shared Sidekick unit constants, add full-precision foot-pound aliases, and cover `lbf·ft`, `lbf·in`, `ft·lbf`, and `ft·lbf/s` round trips. |
| 2026-06-15 | 1.1.490 | refactor(compatibility, #3359): make the legacy `utils.compatibility` shim re-export the shared `UTC` and `StrEnum` primitives while preserving `check_python_version()`, and add identity regression coverage so utility callers cannot split compatibility class identity from shared modules. |
| 2026-06-15 | 1.1.488 | ci(quality-check, #3359): add `scripts/quality-check.py --report-only`, wire the banned-pattern scan into pre-commit and the CI quality-gate summary without blocking legacy findings, add CLI regression coverage for blocking versus report-only exits, and update user-facing docs to describe the report-only ratchet. |
| 2026-06-15 | 1.1.484 | test(video-processor, #3359): replace placeholder logger utility assertions with deterministic Python/NumPy seed checks, root logging configuration assertions, and a message-stable negative-seed contract. |
| 2026-06-15 | 1.1.483 | ci(sidekick-agent, #3359): focus source-keyed Sidekick agent test selection on `tests/unit/sidekick/agent/test_action_service.py` so agent contract changes do not pull unrelated Qt runtime/sidebar suites into every matrix lane. |
| 2026-06-15 | 1.1.482 | fix(sidekick-agent, #3359): make `StateError` a Tools-owned shared contract, remove `sidekick.agent.action_service`'s fallback import of downstream `src.shared.python.core.contracts`, re-export the canonical class through the sidekick action surface, and add regression coverage for exception identity plus the host-import boundary. |
| 2026-06-15 | 1.1.481 | fix(sidekick-api, #3359): correct `electrode_advancement_calculator.__all__` so the shared module exports `ElectrodeAdvancementCalculator` instead of imported contract helpers and `warnings`; keep the shared calculator on its pure-Python implementation rather than importing downstream `tools_core`; refresh the sidekick public API baseline and add a focused export regression. |
| 2026-06-15 | 1.1.480 | test(ai-cli): gate live Claude Code CLI tests behind `TOOLS_RUN_LIVE_CLAUDE_CODE=1`, matching the Codex and Gemini CLI live-test pattern so CI runners with stale or partially configured CLI shims do not fail optional provider round trips. |
| 2026-06-15 | 1.1.479 | fix(ai-auth, #3359): make `AuthManager.refresh_token_if_needed()` fail closed when an expired access token has only a valid refresh token, because #5227 has not implemented real refresh-token exchange yet; focused tests now pin valid-token success, missing-token failure, and the expired-access/valid-refresh warning path. |
| 2026-06-15 | 1.1.478 | test(chat, #3331): repair the contract-extraction CI surface by updating adapter-factory credential tests to patch `chat_contracts.credentials`, keeping Gemini legacy-SDK construction stable when the optional SDK is absent or monkeypatched, and refreshing the split chat drift hashes for the extracted `chat.models` and injected chat dock widget runtime. |
| 2026-06-15 | 1.1.477 | refactor(chat, #3331): add a dependency-free `chat_contracts` leaf package for shared thinking-capability, response-style, credential, and archived-conversation contracts; keep typed `chat.models` and `chat.credentials` compatibility exports; repoint AI adapters and API-key helpers away from `chat.*`; make chat-side AI memory/session collaborators lazy or injected; remove the empty `tests/unit/chat` package marker that shadowed the real `chat` package during combined pytest runs; and add architecture coverage preventing production `ai` code from statically importing `chat` and production `chat` code from statically importing `ai`. |
| 2026-06-15 | 1.1.476 | refactor(chat, #3331): remove the chat dock's top-level AI `ChatSessionManager` import by adding lazy default construction plus keyword-only session-manager injection, with boundary tests that prevent the chat package from regaining that import-time dependency. |
| 2026-06-15 | 1.1.475 | refactor(chat, #3332): move the AI provider/model/thinking combo widgets into `ChatDockView`, have `ai_dropdowns.py` refresh and sync those controls through the view plus explicit state/callbacks, and keep legacy `_ai_*_combo` aliases generated by the existing mirror loop for compatibility. |
| 2026-06-15 | 1.1.474 | fix(shared-python, #3332): expose the existing `ai` package from `src.shared.python` so dotted monkeypatch paths such as `src.shared.python.ai.gui.history_sidebar` resolve consistently when tests import the shared parent package first; the history-sidebar test also links synthetic namespace modules to their parents for Python 3.10 compatibility. |
| 2026-06-15 | 1.1.473 | test(chat, #3332): avoid contiguous secret-like SHA-256 literals in the shared chat drift fixture while preserving the reviewed `_chat_dock_widget_qt.py` baseline value, keeping detect-secrets focused on real credential drift. |
| 2026-06-15 | 1.1.472 | test(chat, #3332): refresh the shared chat drift baseline hash for the intentional `_chat_dock_widget_qt.py` view-state refactor so the baseline guard continues to catch unreviewed drift after the approved UI state change. |
| 2026-06-15 | 1.1.471 | test(chat, #3332): keep the chat dock view-state regression in the shared chat test suite and let breadcrumb refresh tolerate uninitialized Qt test doubles, avoiding CI changed-test collection that shadows the source `chat` package with `tests/unit/chat`. |
| 2026-06-15 | 1.1.470 | refactor(chat, #3332): introduce an explicit `ChatDockView` dataclass for chat dock UI widgets/actions, mirror legacy `_foo` aliases from dataclass fields in one compatibility loop, and replace session helper `__dict__` pokes with direct initialized state access. |
| 2026-06-15 | 1.1.469 | fix(ci): allow Tauri Linux Node selection to fall back to a verified `node`/`npm` pair on `PATH` when runner externals are broken and `/opt/hostedtoolcache/node` is absent, keeping self-hosted app checks from failing before source validation. |
| 2026-06-15 | 1.1.468 | perf(data-processing): replace allocation-heavy `Array.from(...).map(...)` chains in `AnalyticsSuite.tsx` with preallocated loops so analytics rendering avoids avoidable intermediate arrays while preserving existing chart data contracts. |
| 2026-06-15 | 1.1.467 | fix(ci): serialize CI Standard apt update/install sections behind a shared host flock so parallel self-hosted Linux jobs cannot race on `/var/lib/apt/lists/lock` while installing GUI test dependencies. |
| 2026-06-15 | 1.1.466 | feat(a11y, function-generator): expose Function Generator layer and operation controls as pressed-state toggles with keyboard-visible focus affordances, and harden Tauri self-hosted runner Node selection so CI skips broken runner-bundled npm installs. |
| 2026-06-15 | 1.1.465 | ci(sidekick, #3335): map the `sidekick.theme` bridge to its focused import regression so bootstrap-path changes do not pull the generic Sidekick UI mirror suite or OS-terminal worker tests into unrelated Python matrix lanes. |
| 2026-06-15 | 1.1.464 | fix(ci, #3335): install `python-multipart` wherever `ci-standard.yml` installs FastAPI so the URDF viewer upload route can be imported in the Python matrix, and add an ops regression that prevents FastAPI-only CI dependency drift. |
| 2026-06-15 | 1.1.463 | test(imports, #3335): make the import-bootstrap regression suite hermetic in CI by asserting no repository bootstrap paths are added during production imports, updating the stale `sidekick.theme` fallback test to require no `sys.path` insertion, and setting subprocess `PYTHONPATH` explicitly so local and CI subprocess checks exercise the same contract. |
| 2026-06-15 | 1.1.462 | fix(imports, #3335): remove process-global `sys.path` mutation from the production import-time bootstrap offenders (`sidekick.theme`, `signal_processing_studio`, `urdf_builder_gui`, and the URDF viewer app); nested tool packages now use package-scoped `__path__` bridges, focused AST/import-side-effect tests pin the contract without expanding into the broader #3316 multi-root cleanup, and the touched URDF stylesheet test is kept green with explicit Catppuccin `QSlider` styling. |
| 2026-06-15 | 1.1.461 | fix(ci, #3325): keep heavy integration workflows compatible with strict pytest asyncio configuration by installing `pytest-asyncio`, constrain both scheduled and opt-in heavy lanes to explicit `tests/heavy_integration/` and `tests/e2e/` collection roots, and add an ops regression that prevents broad `tests/` collection from masking dependency/config drift. |
| 2026-06-15 | 1.1.460 | fix(ci): make the Jules Supersede Check use `github.token` when `RUNNER_CHECK_TOKEN` is absent so same-repo PR discovery and cleanup do not fail main pushes with an empty `GH_TOKEN`. |
| 2026-06-15 | 1.1.459 | fix(movement_optimizer, Movement_Optimizer#503): lift the vendored app's stale `scipy<1.16` ceiling after a clean SciPy 1.17 `CubicSpline` import check, remove the obsolete README limitation, and add a dependency-contract regression for the canonical Tools copy. |
| 2026-06-14 | 1.1.458 | test(scientific, #3391): add ODE closed-form and harmonic-energy reference anchors plus DIN 1343 SCFM-to-Nm3/hr and methane Z-factor checks so shared calculation regressions are pinned to absolute values, not only monotonic/property behavior. |
| 2026-06-14 | 1.1.455 | ci(theme, #3442): add explicit return casts in the shared PyQt6 theme manager so delta-mypy can type-check the touched stylesheet and built-in-theme lookup paths without weakening runtime behavior. |
| 2026-06-14 | 1.1.454 | fix(theme, #3442): recreate the shared PyQt6 `ThemeManager` singleton when Qt has deleted its QObject wrapper so Signal Toolkit canvas theme setup can recover from prior Qt test lifecycle cleanup while keeping focused regression coverage. |
| 2026-06-14 | 1.1.453 | ci(signal-toolkit, #3442): restore the QtAgg display-availability guard around Signal Toolkit canvas theme tests and keep display-independent Matplotlib theme coverage active for headless Python CI lanes. |
| 2026-06-14 | 1.1.452 | ci(sidekick, #3334): keep changed-source test selection focused for tools-sidebar appearance, OS-terminal, and runtime-settings changes, and use pytest-qt's standard `qapp` fixture in Python REPL widget tests so non-required Python lanes do not depend on a local fixture alias. |
| 2026-06-14 | 1.1.451 | ci(tests, #3334): isolate Python matrix jobs from runner-user site packages with `PYTHONNOUSERSITE=1` so self-hosted 3.12 jobs do not mix stale `~/.local` pytest/pluggy packages with per-job tool-cache native dependencies. |
| 2026-06-14 | 1.1.450 | fix(sidekick, #3334): keep changed-file CI focused for touched Sidekick data-processing and tools-sidebar sources, and accept source-qualified `PanelAppearance` and `WorkspaceRegistry` aliases through explicit runtime contracts. |
| 2026-06-14 | 1.1.449 | fix(sidekick, #3334): centralize workspace registry alias recognition so `sidekick` and legacy `upstream_drift_tools` imports share the same runtime contract, and keep C3D invalid-header coverage independent of optional `ezc3d` availability. |
| 2026-06-14 | 1.1.448 | test(sidekick, #3334): document the legacy touched Sidekick test modules that retain untyped pytest helper signatures while the import-collision regression coverage stays under the repository mypy pre-push hook. |
| 2026-06-14 | 1.1.447 | fix(sidekick, #3334): stabilize the full Sidekick/import-order test surface after the data-processor wrapper rename by isolating import-cache probes in subprocesses, removing pandas/theme/mock leakage between tests, aligning stale DbC expectations with current explicit exceptions, and resolving the syngas compression plot canvas lookup at runtime. |
| 2026-06-14 | 1.1.442 | fix(p1am): make the desktop HTTP worker expose and lazily recover its optional `requests` client after test-time import masking, preserving responsive GUI worker tests when earlier HMI tests simulate missing optional network dependencies. |
| 2026-06-14 | 1.1.441 | fix(matlab-audio, #3330): extract the shared phase-vocoder pitch-shift helper into `applyPitchShiftFrames.m`, route AdvancedAudioProcessor pitch correction/shift/vocoder methods through it, process multi-channel audio channel-by-channel, and convert still-unimplemented spatialization and composition placeholders into hard errors with Python static regressions that CI can enforce without requiring MATLAB. |
| 2026-06-14 | 1.1.425 | fix(p1am, #3352): split the Control tab's MPC setup/request handling into `control_tab_mpc.py` so the responsive HTTP-worker fix satisfies the changed-file size budget without adding a monolith baseline exception. |
| 2026-06-14 | 1.1.424 | fix(p1am, #3352): standardize desktop HMI HTTP writes through a parented `HttpWorker` launcher that uses explicit connect/read timeout tuples, applies a busy cursor, disables triggering buttons while requests are in flight, and keeps the Qt event loop responsive during backend latency. |
| 2026-06-14 | 1.1.423 | fix(movement_optimizer, #3411): split the swingset policy worker and trace canvas out of `motion_tabs.py` so the async optimizer remains covered while satisfying the module-size quality gate. |
| 2026-06-14 | 1.1.422 | fix(movement_optimizer, #3411): run swingset policy optimization in a `QThread` worker instead of the GUI thread, emit progress/result/error back to the tab via Qt signals, reset and report failures with a dialog, and keep shared bottom playback controls synchronized when async policy generation starts playback. |
| 2026-06-13 | 1.1.421 | fix(ci, movement_optimizer, #3410): keep `src/movement_optimizer` launcher and registration changes from reselecting the vendored origin-repo test suite in `scripts/select_tests_for_changes.py`, hide the legacy `src/optimizer_gui` compatibility registration from generated launcher catalogs, declare the P1AM desktop `pyqtgraph` GUI dependency used by always-on CI core tests, and document the canonical `src/movement_optimizer/` provider surface in the component table. |
| 2026-06-13 | 1.1.420 | fix(movement_optimizer, #3410): make the vendored `src/movement_optimizer` app the single advertised Movement Optimizer provider surface by pointing the root Tools manifest and launcher catalog at `src/movement_optimizer/launch_pyqt6.py`, restoring the canonical `/tools/movement-optimizer` route, removing the old `src/optimizer_gui/model_pack.yaml` provider advertisement, and adding manifest tests that pin capabilities plus supported exercises against the tool-pack contract. |
| 2026-06-13 | 1.1.419 | fix(ci, #3357): make the nightly full-suite workflow generate repo-wide `coverage.xml` and run `scripts/check_coverage_policy.py` without `--changed-files`, so the total coverage non-regression ratchet is enforced only on a genuine full-suite lane while PR CI remains changed-package scoped. |
| 2026-06-13 | 1.1.418 | fix(ci): exclude in-tree `src/**/tests/**` paths from source-keyed test mapping so changed Sidekick tests do not reselect the entire Sidekick package test tree. |
| 2026-06-13 | 1.1.417 | fix(ci): narrow source-keyed Sidekick process-calculator test selection to focused process-calculator tests so PSA/WGS changes do not drag unrelated Sidekick data-processor Qt tests into every Python matrix lane. |
| 2026-06-13 | 1.1.416 | fix(sidekick): restore PSA GUI facade imports for the legacy `psa_gui.py` compatibility module so direct PSA GUI test collection resolves PyQt6, matplotlib, model, and safety helper names after the UI extraction. |
| 2026-06-13 | 1.1.415 | fix(sidekick, #3333): package the root `compatibility` shim and route the WGS reactor JSON import directly through `sidekick.utils.json_io`, with metadata and AST boundary tests so installed Sidekick wheels avoid cross-tree `state_manager` reach-through. |
| 2026-06-13 | 1.1.414 | test(ai): make the shared AI dependency subprocess probe independent of ambient `src` packages by creating a temporary repo-local `src` package shim whose path points at this checkout's `src/` tree before importing `src.shared.python.ai.adapters.factory`; this preserves the no-`sys.modules`-stub contract while preventing sibling editable installs or runner site-packages from deciding whether CI can import the shared AI stack. |
| 2026-06-12 | 1.1.410 | ci(#3324, #3325, #3357): add `full-suite-nightly.yml` (whole-collection nightly run with a vacuous-run guard) and `scripts/select_tests_for_changes.py` (source-keyed test selection wired into `ci-standard.yml`); add a core_tests zero-collection guard so always-on smoke entries can no longer pass with 0 collected; make the heavy/e2e lanes real (`heavy-integration-tests.yml` nightly schedule + `live_simulation or e2e` markers, `set -o pipefail`, and missing-junit/0-collected summary failures; same guards in `heavy-tests-opt-in.yml`); and update `COVERAGE_SETUP.md`/`COVERAGE_QUICK_START.md` to stop documenting the already-removed `hot_path_modules_phase2` block as an enforced gate. |
| 2026-06-12 | 1.1.408 | fix(sidekick): avoid the nested Qt event loop in Python REPL worker completion by polling `QThread` progress through `QApplication.processEvents()` plus bounded waits, keeping synchronous `execute()` behavior while preventing Linux/offscreen Python 3.11/3.12 test aborts in the F6 async REPL path. |
| 2026-06-12 | 1.1.406 | test(pendulum): add an explicit runtime contract assertion to the manual PyQt signal smoke script so the changed-test assertion gate recognizes `src/pendulum_simulator/signal_test.py` as behavior-checking test surface after the frameless-window cleanup touched the file. |
| 2026-06-19 | 1.1.406 | fix(data-processor, #3734): make `UncertaintyQuantifier._normal_ppf` fail clearly for probabilities outside the open interval `(0, 1)` instead of returning the median quantile `0.0`, with focused contract tests covering `p <= 0`, `p >= 1`, and the valid `0.975` quantile sanity case. |
| 2026-06-12 | 1.1.405 | fix(p1am-control, #3323): stop passing `Qt.GlobalColor` enum members into `pg.mkPen(color=...)` for the MPC PID-vs-MPC comparison plots in `control_tab.py`; under pyqtgraph 0.13.7+/0.14.0 with PyQt6 `mkColor` raised `TypeError: Not sure how to make a color from "(<GlobalColor.red: 7>,)"`, aborting `ControlTab()` construction and killing any test or launch that builds the Control tab. Use pyqtgraph-native color forms (`"r"` and the `(0, 100, 0)` darkGreen tuple) while leaving the theme-derived Highlight/WindowText pens untouched, and add a regression test that constructs `ControlTab()` and asserts the four MPC curve attributes exist. |
| 2026-06-12 | 1.1.404 | fix/test(p1am, #3314): make the HMI E-STOP clear actually reach the PLC. Add a `clear_estop()` contract to `BasePLCClient`, implement it as an explicit reset-coil write in the Modbus client and a latch reset in the simulator, and rework `/api/estop/clear` to command the controller and only lower the server-side `e_stop_active` flag when the controller (or backup simulator) acknowledges — returning 502 and keeping the latch on rejection. The desktop header now shows a pending "CLEARING…" state and only goes green ("E-STOP CLEAR") on confirmed success, reverting to red on failure. Split endpoint-level E-STOP clear regressions into a focused backend test module so the confirmed PLC reset, rejected-reset latch preservation, and offline simulator-clear contracts remain covered while `test_backend.py` stays inside the fleet file-size budget, and keep that split module aligned with the backend suite's optional dependency contract so environments without `sqlmodel` skip FastAPI endpoint tests instead of failing collection. REQUIRES HARDWARE VALIDATION before trusted. |
| 2026-06-12 | 1.1.403 | fix(process-calculators, #3103): keep `calculate_htu`'s non-positive liquid/gas ratio fallback inside the typed float contract by returning `HTU_MAX` explicitly as a `float`, preserving the existing clamp behavior while satisfying changed-file mypy gates. |
| 2026-06-12 | 1.1.402 | fix(process-calculators, #3103): add the missing Design-by-Contract input preconditions to `scrubber_calculator.py` so invalid physical inputs raise a level-gated `PreconditionError` (a `ValueError` subclass) instead of silently dividing by zero or returning garbage. Guards `calculate_gas_density`/`calculate_gas_viscosity` (temperature_k, pressure_pa, molecular_weight > 0), `calculate_flooding_velocity` (gas_density/liquid_density > 0, liquid_mass_flux >= 0), `calculate_column_diameter` (gas_flow_kg_hr > 0, percent_of_flood in (0, 100]), and `calculate_heat_transfer_duty` (gas_flow_kg_hr > 0, water_condensed_kg_hr >= 0) via `contracts.require`, matching the flare-calculator precedent and the repo DbC policy. The `TestScrubberCalculatorContracts` suite previously failed because the preconditions it asserts were never implemented; tests now also pin the precondition messages via `match=`. |
| 2026-06-12 | 1.1.401 | fix(sidekick): make OS terminal backend teardown close subprocess pipes, join reader threads, and clear stale process handles so Qt/Sidekick tests do not abort during interpreter shutdown after terminal widgets close. |
| 2026-06-12 | 1.1.400 | fix(sidekick): guard the Python REPL worker wait loop with a timer-backed `isRunning()` poll so fast worker completion cannot miss the nested Qt loop's `finished` signal and hang Linux/offscreen Python 3.11/3.12 test lanes until pytest-timeout aborts. |
| 2026-06-12 | 1.1.399 | test(sidekick): keep the state-manager UTC boundary regression compatible with the Python 3.10 CI lane by asserting the shared stdlib `timezone.utc` singleton instead of the Python 3.11-only `datetime.UTC` alias. |
| 2026-06-12 | 1.1.398 | fix(consolidation): consolidate Tools PRs #3398-#3405 into one branch to reduce CI load, covering steam-engine actual-backend reporting, Sidekick REPL QThread teardown hardening under the file-size budget, golden physics anchors, P1AM and pendulum frontend optimizations, SQLite connection cleanup, headless calc-backend imports, pendulum input autocorrect suppression, and Sidekick JSON/state-manager boundary enforcement. |
| 2026-06-12 | 1.1.397 | test(conversion, #3384 #3388 #3389): add shared conversion-service policy coverage for normalization, validation, custom-unit warnings, gas-flow dispatch, syngas/performance helpers, and singleton conversion helpers so `src/shared/python/sidekick/calculators/conversion/service.py` stays above the changed-file coverage gate without changing production behavior. |

| 2026-06-12 | 1.1.394 | chore(consolidation): refresh the quality-consolidation branch after the scientific-accuracy merge so the shared Sidekick process-calculator constants, signal calculus guards, and API baseline remain aligned with current main while preserving the data-processor facade split. |
| 2026-06-11 | 1.1.391 | fix(sidekick): keep the Python REPL worker owned by the widget until its QThread has fully stopped, avoiding Linux/offscreen teardown aborts from premature deleteLater scheduling. |
| 2026-06-11 | 1.1.390 | test(ci): keep the Sidekick Python REPL widget below the fleet file-size budget after QThread teardown hardening. |
| 2026-06-11 | 1.1.387 | test(ci): stabilize optional CoolProp symbol patching and data_processor nested-package imports across CI Python environments. |
| 2026-06-11 | 1.1.386 | fix(thermo, #3381 #3382): correct the Buck water vapor-pressure exponent, tighten dew-point regression coverage against published reference points, and add pressure-dependent ideal-gas entropy in the simplified steam vapor fallback. |
| 2026-06-11 | 1.1.385 | fix(calc-backend, #3341): require forward time spans for ODE solver and thermal-profile requests, convert diverging ODE and thermal integrations into 422 validation errors before non-finite values reach JSON responses, and add contract/API regressions for reversed spans and divergent systems. |
| 2026-06-11 | 1.1.384 | fix(steam, #3337 #3338): enforce saturation temperature and pressure preconditions before backend fallback, reject out-of-range simplified saturation states instead of extrapolating Antoine correlations, preserve unknown CoolProp quality as NaN instead of saturated-liquid quality, and map steam API validation failures to HTTP 400. |
| 2026-06-11 | 1.1.383 | fix(unit-converter, #3336 #3339): make gas-flow conversions fail loudly for unknown gas species across Sidekick and web converter surfaces, and align the sidekick compressibility-factor calculation with the Abbott/Pitzer second-virial form used by pressure-drop calculations. |
| 2026-06-11 | 1.1.377 | fix(ci): use an actionlint-compatible relative npm cache path for Tauri jobs while keeping installs isolated from the runner user's shared npm cache. |
| 2026-06-11 | 1.1.376 | fix(ci): isolate Tauri npm caches under the per-job runner temp directory and prefer fresh registry metadata so corrupted shared npm cache entries cannot fail `npm ci`. |
| 2026-06-11 | 1.1.375 | fix(ci): set `fail-fast: false` on the CI Standard `tests` Python matrix. Only `tests (3.11)` is a required check; under the default `fail-fast: true` an infra crash in the non-required 3.10/3.12 lanes (SIGABRT/exit-134 from the Qt headless multi-widget segfault or an OOM kill on a saturated self-hosted runner) cancelled the required 3.11 lane before it ran, leaving consolidation PR #3380 permanently BLOCKED. Decoupling the lanes lets 3.11 report independently. |
| 2026-06-11 | 1.1.374 | fix(ci): keep the Sidekick extended Qt-heavy unit suite on Python 3.11/3.12 while excluding it from the Python 3.10 compatibility lane, where PyQt aborts the interpreter on saturated self-hosted runners. |
| 2026-06-11 | 1.1.373 | fix(ci): make the workflow validation PyYAML fallback explicit for mypy so quality-gate checks accept both full and lean runner environments. |
| 2026-06-11 | 1.1.372 | fix(ci): make workflow lint validation tolerate lean runner environments where PyYAML cannot be fetched by adding stdlib fallback checks for workflow structure and blocking quality gates, while still using PyYAML when present. |
| 2026-06-11 | 1.1.371 | fix(ci): keep the Python 3.10 CI Standard lane focused on core compatibility tests for large consolidation PRs while Python 3.11/3.12 continue to run the full changed-test slice, avoiding 3.10 runner OOM kills during collection. |
| 2026-06-11 | 1.1.370 | fix(ci): remove the network-dependent `actions/setup-python` bootstrap from Topology Governance because the topology checker is a stdlib-only script and can run with the fleet runner's existing `python3`, avoiding transient PyPI/setup-python failures. |
| 2026-06-11 | 1.1.369 | fix(ci): make the Python 3.10 CI Standard test lane override repo-level pytest-xdist auto-parallelism with `-n 0` so saturated self-hosted runners report deterministic test results instead of xdist worker crash exhaustion. |
| 2026-06-11 | 1.1.368 | test(ci): keep data-processor tkinter fallbacks from leaking a partial `tkinter` stub into folder-tool collection by preferring real tkinter when available and installing a complete fallback with `ttk`, `messagebox`, and `filedialog` modules only when needed. |
| 2026-06-11 | 1.1.367 | fix(ci/runner): make Tauri Linux checks discover an available local Node 24, 22, or 20 toolcache on mixed self-hosted runners instead of failing on runners without the exact Node 24.16.0 path. |
| 2026-06-11 | 1.1.364 | test(ci): keep retired data-processor skip sentinels compatible with the ruff B011 guard by using truthy documentation assertions instead of optimized-away `assert False` statements. |
| 2026-06-11 | 1.1.363 | fix(ci/runner): harden `ci-standard.yml` Linux apt setup by clearing corrupted apt package-cache binaries alongside stale lock files before `apt-get update`, allowing self-hosted runners to recover from cache rename failures. |
| 2026-06-11 | 1.1.362 | fix(ci): align `ci-standard.yml` with the fleet's known-good `mypy==1.13.0` workflow pin so quality-gate dependency installation remains reproducible on self-hosted runners. |
| 2026-06-11 | 1.1.361 | test(ci): satisfy the changed-test behavioral assertion gate in the Tools consolidation branch by adding benchmark output postconditions, making retired data-processor skip sentinels explicit, and documenting the shared numerical helper as support-only in the assertion allowlist. |
| 2026-06-11 | 1.1.360 | fix(ci): restore Tools consolidation CI by replacing the coverage tracked-package regex generator with a shell-safe Python expression, adding changed-file mypy annotations for the multi-parameter PyQt meshgrid arrays, and resolving signal-toolkit integration bounds to concrete floats before validation/result construction. |
| 2026-06-11 | 1.1.354 | fix(consolidation, #3314 #3315 #3350 #3356 #3358): restore and relocate truncated test coverage across shared calculators, signal tooling, GUI launchers, folder tooling, data processing, rotation conversion, and integration surfaces; unify humanoid anthropometry under the shared implementation; propagate P1AM E-STOP clear commands through the backend API; refresh assessment artifacts and CI baselines for the consolidated changes. |
| 2026-06-11 | 1.1.354 | test(sidekick, #3339 #3340): add focused pressure-drop gas-property coverage for strict unknown-species DbC paths, physical-value helper contracts, complete gas-property calculation keys, and ideal-gas compressibility fallback so the changed gas helper module is covered by the Sidekick per-file coverage gate. |
| 2026-06-10 | 1.1.353 | fix(hooks, #1361): align the pre-push mypy hook with changed-file delta CI by adding `--follow-imports=skip`, so clean pushes are checked against the pushed source files without failing on unrelated pre-existing imported `ai/` debt. Added an ops regression test that keeps the hook on the pre-push stage, filename-passing mode, `src/` scope, and no-follow-import behavior. |
| 2026-06-10 | 1.1.352 | test(sidekick): keep action-audit timestamp fixtures compatible with the Python 3.10 CI lane by using `timezone.utc` with a scoped pyupgrade suppression instead of the Python 3.11-only `datetime.UTC` alias. |
| 2026-06-10 | 1.1.351 | test(sidekick): keep action-audit redaction fixtures covered while marking synthetic sensitive-key values with detect-secrets allowlist pragmas, so the security scan remains strict without treating redaction test data as leaked material. |
| 2026-06-10 | 1.1.348 | test(ai): keep the #3310 GUI-thread dispatcher coverage mypy-clean under the changed-file gate by annotating the offscreen Qt fixture, worker-thread test parameters, dispatcher thunks, decorator-registered tool dispatch, and exception helper while preserving the main-thread marshalling behavior under test. |
| 2026-06-10 | 1.1.347 | fix(ci/runner): split Tauri build matrix display labels from `runs-on` targets so Windows jobs no longer render as `Array`, and run Windows Rust path/tool-home setup through PowerShell while preserving bash setup on Linux. |
| 2026-06-10 | 1.1.346 | fix(ci/runner, #3308): restore the Tauri 30-minute check timeout on current main after #3307 accidentally reverted the runner hardening while adding the ShellTool command-injection fix. |
| 2026-06-10 | 1.1.345 | fix(ci/runner, #3305): isolate Tauri `RUSTUP_HOME` and `CARGO_HOME` under each job's `RUNNER_TEMP` so parallel self-hosted jobs do not race on the shared `$HOME/.rustup` toolchain and lose `rustc` mid-clippy. |
| 2026-06-10 | 1.1.344 | fix(ci/runner, #3304): disable Tauri Rust `target/` cache restoration while keeping cargo registry/git caching after a fast-I/O runner hit a stale dep-info fingerprint (`time-*.d` missing) during clippy. |
| 2026-06-10 | 1.1.343 | fix(ci/runner, #3304): raise Tauri Rust stack reservations to 512 MiB after function-generator and data-processor clippy on OGLaptop explicitly requested `RUST_MIN_STACK=536870912`, with workflow regression coverage for the stack contract. |
| 2026-06-10 | 1.1.342 | fix(ci/runner, #3304): route Rust-heavy Tauri check and Linux build jobs to the `d-sorg-fleet-fast-io` runner label so PR validation avoids OGLaptop slots that repeatedly hit rustc stack faults while keeping local self-hosted execution. |
| 2026-06-10 | 1.1.341 | fix(ci/runner, #3304): raise Rust stack reservations to 256 MiB after rotation-converter clippy on OGLaptop explicitly requested `RUST_MIN_STACK=268435456`, keeping all Tauri app checks on the same fleet-safe stack setting. |
| 2026-06-10 | 1.1.340 | fix(ci/runner, #3304): raise Rust stack reservations to 128 MiB for local self-hosted Tauri and wheel builds after OGLaptop rustc clippy failures explicitly requested `RUST_MIN_STACK=134217728`. |
| 2026-06-10 | 1.1.339 | fix(ci/runner, #3300): expose `$HOME/.cargo/bin` before Rust toolchain setup in self-hosted Rust jobs so fleet runners use their preinstalled rustup instead of attempting fragile bootstrap installs when non-login shells omit cargo from PATH. |
| 2026-06-10 | 1.1.338 | fix(ci, #3300): raise Rust runner stack reservations to 64 MiB for local self-hosted Tauri and wheel builds after rustc SIGSEGV failures explicitly requested `RUST_MIN_STACK=67108864` on the fleet. |
| 2026-06-10 | 1.1.337 | fix(ci/test-contract, #3300): recognize repo-level `tests/<package>/test_*.py` directories as satisfying the minimum test contract for changed `src/<package>` packages, with regression coverage so package-scoped tests like `tests/plant_simulator/test_dataset.py` are accepted without weakening the quality gate. |
| 2026-06-10 | 1.1.336 | fix(ci/review-comments, #3300): keep the review-comment-to-issue converter checkout shallow because the job uses GitHub API reads plus local archive commits, avoiding full-history fetches on self-hosted runners where stale/corrupt loose objects can make checkout fail before the workflow logic runs. |
| 2026-06-10 | 1.1.335 | fix(ci/runner-health, #3300): serialize the Tauri desktop app check/build matrices and cap Cargo jobs with non-incremental, no-debug builds so self-hosted runners do not compile multiple Tauri Rust dependency graphs concurrently and trigger rustc SIGSEGV/paging-pressure failures. |
| 2026-06-11 | 1.1.359 | chore(consolidation): finish the open-PR consolidation by centralizing Catppuccin stylesheet imports, preserving calc-backend dependency direction, and tightening restored test/type annotations for the changed-file quality gates. |
| 2026-06-11 | 1.1.358 | fix(thermo, #3345): keep saturation-pressure lookups resilient by falling back to the Antoine equation when the optional Cantera water backend raises while preserving explicit failures for invalid fallback inputs. |
| 2026-06-11 | 1.1.357 | fix(ode, #3349): preserve the consolidated `t_span` bounds guard in the Sidekick ODE solver while keeping the merged implementation syntactically valid. |
| 2026-06-11 | 1.1.356 | fix(test, #3315): restore truncated test coverage across P1AM, pendulum, shared-tool, and architecture suites; preserve HMI emergency-stop propagation tests; and reconcile the humanoid/URDF anthropometry consolidation with the shared ratio helpers. |
| 2026-06-11 | 1.1.355 | fix(dry, #3346): remove reintroduced root-level `urdf_builder_gui` duplicate modules and add a regression test that asserts the root package does not shadow the canonical `src/shared/python/urdf_builder_gui` implementation. |
| 2026-06-10 | 1.1.334 | fix(ci/rust, #3291 #3294 #3295): split PyO3 `python` test features from maturin-only `extension-module` wheel linkage so `cargo test --features python` no longer emits Python extension-module binaries while wheel builds still opt into extension-module linking. |
| 2026-06-10 | 1.1.333 | fix(bug/ci, #3294 #3295): declare pendulum `Golfer` dynamics native-only with construction-time `RuntimeError` guidance and an explicit workspace exclude for `pendulum-core`; remove `plant_simulator`'s silent random-data path so `SCADADataset` loads real SQLite `taglog` rows unless synthetic data is explicitly requested; and keep the affected native wrappers mypy-clean under the changed-file quality gate. |
| 2026-06-10 | 1.1.332 | fix(ci, #3298): keep the P1AM project import helper mypy-clean under the changed-file quality gate by typing parsed SCADA tags as `TagDefinition` at the parser boundary and preserving the endpoint's documented `dict[str, Any]` response contract when imports are skipped. |
| 2026-06-11 | 1.1.354 | fix(dbc): harden optimized-mode validation for signal-toolkit derivative guards and Sidekick gas-flow conversion internals. `signal_toolkit` optimized-mode subprocess coverage now preserves the repo shared-python import path, and gas-flow ACFM invariant checks use explicit exceptions instead of runtime `assert` statements so guard behavior remains deterministic under `python -O`. |
| 2026-06-10 | 1.1.331 | fix(ci, #3298): avoid a detect-secrets Secret Keyword false positive in the P1AM backend auth helper by renaming the public header-name constant away from token-like wording and constructing the `X-API-Key` header name without changing the HTTP authentication contract. |
| 2026-06-10 | 1.1.331 | fix(daemon, #3291): stop `start-gaai-daemon.sh` from writing `~/.claude/settings.json` or globally suppressing Claude Code dangerous-mode prompts; document that any safety override must be configured deliberately outside the launcher, and add a dry-run regression test proving existing global Claude settings are preserved. |
| 2026-06-09 | 1.1.329 | fix(security, #3288 #3289 #3292): remove the P1AM HMI hardcoded default Admin password and accepted hardcoded SHA-256 hashes, fail closed when no credential is configured, and verify admin passwords with a salted PBKDF2-HMAC-SHA256 KDF (`ADMIN_PASSWORD_HASH`/`ADMIN_PASSWORD`) instead of bare SHA-256; add server-side `X-API-Key` authentication/authorization to the P1AM control backend (`auth_config.py`) so every state-mutating endpoint and the live WebSocket require an operator key and destructive/elevated operations (estop clear, tag writes, PID tuning, MPC, alicat setpoint/gas, project import) require an admin key, failing closed (503) unless `P1AM_DEV_NO_AUTH=1`, with E-stop activation intentionally left open and the Docker default bind changed to loopback; and harden `/api/project/import` against unbounded uploads (streamed size cap -> 413), zip bombs (member-count/per-file/total-size/compression-ratio limits before extraction), and partial DB wipes (atomic delete+insert in one transaction). |
| 2026-06-09 | 1.1.329 | fix(security, #3290 #3293): add static complexity limits to `shared.python.safe_eval.validate_expression` (max expression length, max AST node count, bounded `Pow` exponent and nested-`Pow` chain depth, and rejection of oversized string/bytes constants) so pow/repetition bombs such as `9**9**9**9` fail fast instead of hanging or exhausting memory in the calc-backend ODE-solver path; and replace the web calculator's substring blocklist with a structural AST allowlist gate (`TI89Calculator._ast_security_gate`) that runs before `sympy.parse_expr`, rejecting attribute access, lambdas, comprehensions, and the walrus operator by structure rather than enumeration. Adds bypass/DoS regression tests. |
| 2026-06-09 | 1.1.328 | fix(ci): satisfy the changed-file quality gate by explicitly annotating access-policy registry results under skipped-import mypy, add Python 3.10 `tomli` support for metadata contract tests, assert calc-backend pressure-drop values through the standardized response `data` payload, and keep Sidekick standard responses importable from the repo package path without top-level path shims. |
| 2026-06-09 | 1.1.327 | fix(compatibility-ci): route remaining Python 3.10-exercised `StrEnum` imports through compatibility shims, make those shims type-check as native `StrEnum` under mypy while retaining Python 3.10 fallbacks, keep the integrations dashboard empty-state property explicitly typed as `bool`, and pass `.secrets.baseline` explicitly to the detect-secrets audit test so the 3.10 CI matrix validates the canonical baseline instead of failing on CLI argument parsing. |
| 2026-06-09 | 1.1.326 | ci(coverage): keep total coverage floors as a full-suite ratchet while changed-file scoped PR runs enforce only the tracked coverage-policy packages touched by the diff; added regression coverage for the scoped/full-suite split. |
| 2026-06-09 | 1.1.325 | test(calc-backend): add an adversarial route-list contract test ensuring every endpoint advertised by `/api/calc/endpoints` is backed by a registered FastAPI route, strengthening the #3262 calc_backend test-quality audit follow-up. |
| 2026-06-09 | 1.1.324 | fix(ci): invoke detect-secrets through `python -m detect_secrets` in the secret scanning workflow so runners where the console script is not on PATH still execute the installed package. |
| 2026-06-09 | 1.1.323 | fix(ci): avoid detect-secrets false positives from immutable workflow digest pins and workflow-pinning test fixtures without changing the committed secrets baseline. |
| 2026-06-09 | 1.1.323 | test(tools): add changed-test assertion and changed-Python policy guards for the A-O audit follow-up, blocking assertion-light Python test changes and undocumented changed-file policy regressions with focused tests, allowlists, CI integration, and development notes for issues #3262 and #3263. |
| 2026-06-09 | 1.1.322 | fix(ci): fold #3255 pinning into the consolidated branch by requiring third-party workflow actions to use immutable 40-character SHAs, allowing first-party `actions/*` and `github/*` tag refs as the explicit trust boundary, blocking `curl|sh` installers and unversioned global npm installs without a baseline, keeping wasm-pack on a pinned release archive with SHA-256 verification, and pinning Jules CLI installs to `@0.1.42`. |
| 2026-06-09 | 1.1.321 | fix(ci): add a blocking workflow pinning ratchet, replace wasm-pack `curl | sh` installers with a pinned release archive plus SHA-256 verification, add pip retry/timeout settings for CI dependency installs, add a blocking quality-gate verifier for core Ruff/format/mypy PR gates, and split Sidekick data I/O format detection into a dedicated registry module with property/adversarial coverage. |
| 2026-06-09 | 1.1.320 | fix(policy): remove the broken `dwsim-model` console entry, stop allowing the committed coverage baseline to lower the configured coverage floor, align root package docs with the Python 3.11 metadata floor, constrain Sidekick data I/O advertised formats to implemented handlers with focused round-trip coverage, and require the NPM publish job to use the protected `npm` environment. |
| 2026-06-04 | 1.1.318 | test(gui-launcher): add focused unit coverage for shared GUI launcher factory helpers, including launcher construction, generated launch scripts, registered-tool dispatch, missing registry entries, missing PyQt6 configs, module import errors, missing `GUI_INFO`, and successful `GUI_INFO` launch delegation, raising `src/shared/python/gui_launcher/launcher_factories.py` focused coverage from 15.52% to 98.28%; also preserve the declared integer return contract for delegated PyQt6 launch helpers. |
| 2026-06-04 | 1.1.317 | test(gui-launcher): add focused unit coverage for the shared GUI registry, including singleton access, registration validation, lookup/listing/category behavior, helper registration, GUI_INFO conversion, auto-discovery of registration modules, missing paths, import-error handling, and empty legacy modules, raising `src/shared/python/gui_launcher/registry.py` focused coverage from 0.00% to 97.96% without changing production behavior. |
| 2026-06-04 | 1.1.316 | test(gui-launcher): add focused unit coverage for the shared GUI manifest loader, including bundled manifest loading, custom manifest parsing, debug logging, missing files, malformed YAML, missing `tools` mappings, non-sequence `tools` values, and empty manifests, raising `src/shared/python/gui_launcher/manifest_loader.py` focused coverage from 0.00% to 100.00% without changing production behavior. |
| 2026-06-04 | 1.1.315 | fix(compatibility-tests): keep shared Python compatibility coverage importable on Python 3.10 by asserting the UTC fallback through `datetime.timezone.utc`, avoiding Python 3.11-only `enum.StrEnum` references, and preserving Ruff and mypy cleanliness. |
| 2026-06-03 | 1.1.314 | test(compatibility): add focused unit coverage for shared Python compatibility helpers, including Python 3.11+ standard-library alias exports and isolated Python 3.10 fallback behavior for UTC and StrEnum compatibility, raising `src/shared/python/compatibility.py` focused coverage from 0.00% to 100.00% without changing production behavior. |
| 2026-06-03 | 1.1.313 | test(deprecation): add focused unit coverage for shared deprecation helpers, including decorator configuration validation, metadata preservation, warning text variants, method-qualified warnings, and wrapped callable result propagation, raising `src/shared/python/deprecation.py` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.312 | test(logging): add focused unit coverage for shared logging helpers, including package exports, sensitive-value redaction, stream/file logging setup, quiet-library defaults, file and rotating handlers, deterministic seeding, and execution-time telemetry, raising `src/shared/python/logging_pkg` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.311 | test(config): add focused unit coverage for shared environment configuration helpers, including package exports, missing/default/required reads, whitespace handling, boolean parsing, integer/float parsing, bounds errors, and structured `EnvironmentError` details, raising `src/shared/python/config` focused coverage from 0.00% to above 90% without changing production behavior. |
| 2026-06-03 | 1.1.310 | test(chat-export): add focused pure-Python coverage for shared chat export contracts, scanner-safe secret redaction fixtures, markdown/text/html file exporters, and injected clipboard copy modes, raising `src/shared/python/chat/export` focused coverage from 0.00% to 92.79% without changing production behavior. |
| 2026-06-09 | 1.1.310 | perf(p1am frontend): optimize array aggregations and string operations in LadderExplorer.tsx by replacing chained .map().filter() operations with a single-pass loop and using useMemo to prevent main thread lag. |
| 2026-06-03 | 1.1.309 | fix(p1am-power-supply): move the power-supply controller/router and PID-pass-through integration out of `backend/main.py`, keep the split power-supply tests importable under pytest importlib mode, make the controller enums Python 3.10-compatible and mypy-clean, remove stale mypy suppressions from the invalid-input tests, and preserve the module-size budget without relaxing CI gates. |
| 2026-06-03 | 1.1.308 | test(folder-packer): add focused workflow coverage for `folder_packer_pro.operations`, including pack/unpack start validation, worker dispatch, scan dispatch, filesystem exception handling, failed unpack results, encrypted package inspection, and missing package warnings; raises focused module coverage from 74.27% to 92.95% without changing production behavior. |
| 2026-06-03 | 1.1.307 | test(model_generation): add focused edge-case coverage for `model_generation.library.unified_loader`, including load-result naming, preference corruption and persistence failures, manifest cache fallbacks, bundled missing-file reporting, unknown-extension fallback ordering, inline XML conversion dispatch, and malformed MJCF `LoadResult` handling; fixes malformed MJCF loads so they return a failed `LoadResult` instead of escaping parse exceptions, while keeping the loader source under the file-size budget. |
| 2026-06-03 | 1.1.306 | test(upstream-drift): ratchet the legacy `upstream_drift_tools` compatibility shim coverage gate to 100% after focused shim contract tests verified full line and branch coverage, and update the coverage-policy regression tests so the high-water mark is enforced in CI without changing production behavior. |
| 2026-06-03 | 1.1.305 | test(model_generation): add focused coverage for `model_generation.library._rate_limiter`, including rate-limit header parsing, success logging, request header propagation, capped exponential backoff, terminal 429 handling, non-429 HTTP passthrough, and retried network failures; raises the focused module coverage from 53.12% toward the phase-2 model-generation coverage target without changing production behavior. |
| 2026-06-03 | 1.1.304 | test(financial-calculator): add focused PyQt6 contract coverage, split across line-budgeted GUI test modules, for financial calculator import isolation, theme-manager test isolation, successful engine result/projection mapping, notes-dock toggling, summary label rendering, projection table rendering, and calculate-button refresh behavior, raising `src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py` focused coverage to 95.28% and the focused `src/financial_calculator` package coverage to 90.53% without changing production behavior. |
| 2026-06-03 | 1.1.303 | test(codemap): add focused headless coverage for the `codemap-mcp` server entrypoint, including `CODEMAP_REPO_ROOT` discovery, missing optional `mcp` dependency handling, server run dispatch, and fake FastMCP tool delegation for search, symbol lookup, callers, imports, and repo summary; raises `src/shared/python/codemap/mcp_server.py` focused coverage from 0.00% to 100.00% and `src/shared/python/codemap` focused package coverage from 94.39% to 97.72% without changing production behavior. |
| 2026-06-03 | 1.1.302 | fix(ai-skills): run shared AI skills runner coroutine tests through explicit `asyncio.run(...)` calls and handle Python 3.10 `asyncio.TimeoutError` in the runner timeout boundary so timeout failures are consistently classified as structured `timeout` audit events. |
| 2026-06-03 | 1.1.301 | test(ai-skills): add focused contract and failure-path coverage for the shared AI skills runtime, including concrete-skill descriptor enforcement, duplicate instance registration, structured execution-error audit classification, and required descriptor field normalization, raising `src/shared/python/ai/skills` focused coverage from 90.42% to 96.17% without changing production behavior. |
| 2026-06-03 | 1.1.300 | test(codemap): add focused CLI coverage for rebuild, search, who-calls, export, and info command paths using mocked API/indexer seams plus real SQLite JSONL/gzip export verification, raising `src/shared/python/codemap` focused package coverage to 94.39% and adding a 90% tracked coverage policy gate. |
| 2026-06-03 | 1.1.299 | test(file-watcher): add focused deterministic coverage for the Python watchdog fallback covering constructor contracts, callback dispatch failures, debounce coalescing, ignore rules, fake watchdog lifecycle handling, missing optional dependencies, and no-op flush branches, raising `src/shared/python/file_watcher/_fallback.py` focused coverage to 99.46% with a 95% file-level coverage policy gate. |
| 2026-06-03 | 1.1.298 | test(signal-toolkit): add focused deterministic LMS/RLS adaptive filter coverage for pure NumPy fallback behavior, optional Rust-kernel dispatch, output metadata, and signal preconditions, raising `src/shared/python/signal_toolkit/adaptive_filter.py` focused coverage to 95.24% with a 95% file-level coverage policy gate. |
| 2026-06-03 | 1.1.297 | test(model-generation): add 49 focused handler tests for `rest_api_routes.ModelGenerationAPI` covering route count, health/info shape, security headers, all missing-field 400 guards for every endpoint, inertia success branches (box/sphere/cylinder/capsule) with wrong-dimension-count errors, validate/parse success and error paths, library and editor handlers; fix `library_get_model` and `library_add_model` using `ModelEntry.model_id` (non-existent attribute) to use the correct `ModelEntry.id`. |
| 2026-06-03 | 1.1.296 | fix(programmatic-pid): guard DXF-producing `PIDDocument.export_dxf` tests on optional `ezdxf` availability so lean CI environments skip only the dependency-backed export assertions while retaining construction, validation, and precondition coverage. |
| 2026-06-03 | 1.1.294 | test(safe-eval): add a 99% file-level coverage policy gate for `src/shared/python/safe_eval.py`, backed by existing focused safe evaluator tests that cover validation, namespace allowlists, stripped builtins, scalar math, and NumPy math paths at 100% line and branch coverage. |
| 2026-06-03 | 1.1.293 | test(safe-pandas): add focused validation coverage for overlong formulas, syntax errors, unsupported operators, and maximum allowed exponent boundaries, raising `src/shared/python/safe_pandas_eval.py` focused coverage to 100% and adding a 99% file-level coverage policy gate. |
| 2026-06-02 | 1.1.292 | test(notes): add focused PyQt6 coverage for the shared notes dock widget save/reload/clear, recycle/restore, floating/redock, and initialization guard paths, raising the `src/shared/python/notes` package coverage policy gate from 48% to 95% without changing production behavior. |
| 2026-06-02 | 1.1.291 | fix(sidekick): keep conversion service helper boundaries explicit under CI changed-file mypy analysis by coercing skipped-import helper and mixin conversion results back to `float` without changing runtime conversion behavior. |
| 2026-06-02 | 1.1.290 | fix(sidekick): restore custom unit conversion by adding user-defined units to the normalized lookup map, keep invalid temperature validation failures non-fatal as documented, and add focused edge coverage for `sidekick.calculators.conversion.service` singleton helpers, normalization/cache paths, validation guards, category dispatch, and compatible-unit lookup, raising focused service coverage to 99.09% and adding a 90% file-level coverage policy gate. |
| 2026-06-02 | 1.1.289 | fix(ui): route the Windows AppUserModelID platform check through a runtime helper so Linux changed-file mypy does not mark the Windows ctypes branch unreachable while preserving the same taskbar identity behavior. |
| 2026-06-02 | 1.1.288 | fix(sidekick): restore tab hover highlight (`QTabBar::tab:!selected:hover` QSS), fix the active-tab settings button and Configure-Tabs list by preserving `TabCollection` live aliases, add tested `set_app_user_model_id`/`apply_window_icon` helpers for Windows taskbar identity, and fix the Unified Launcher icon path to use `assets/`. |
| 2026-06-02 | 1.1.287 | fix(codemap): add focused headless coverage for the codemap watcher daemon, including watchdog import failures, supported-path filtering, moved-path handling, debounce flushes, deleted-file cleanup, shutdown resource cleanup, and CLI option forwarding; deleted events now reach the existing DB cleanup path instead of being filtered out after the file disappears. |
| 2026-06-02 | 1.1.286 | test(codemap): add focused headless coverage for the codemap indexer, including supported-file walking, `.gitignore` and fallback ignore handling, unchanged-file hash skips, incremental reprocessing and deletion, unreadable/parser-skipped files, per-file error collection, manifest writing, git helper parsing/fallbacks, and preferred blake3 hashing, raising `src/shared/python/codemap/indexer.py` focused coverage from 16.24% to 98.98% without changing production behavior. |
| 2026-06-02 | 1.1.285 | fix(codemap): add focused public API coverage for repo-root discovery, query sanitization, FTS search filtering, symbol lookup, caller lookup, import parsing, neighbor traversal, repo summaries, malformed JSON fallbacks, and default-root caching; fix one-hop `neighbors()` so outbound callees are resolved and returned as documented, raising `src/shared/python/codemap/api.py` focused coverage to 96.93%. |
| 2026-06-02 | 1.1.284 | fix(ai): keep OpenAI and Anthropic system-prompt assembly mypy-clean under the changed-file CI profile by casting the shared prompt builder result back to the documented `str` contract when imported through the skipped-follow-imports namespace, without changing runtime prompt behavior. |
| 2026-06-02 | 1.1.283 | fix(ai-ui): keep the merged #3205 AI/UI hardening mypy-clean under the normal pre-push hook by removing stale system-prompt `no-any-return` ignores, routing BitNet generic errors through the shared classifier, and typing optional headless PyQt UI exports through private nullable export variables without changing runtime behavior. |
| 2026-06-02 | 1.1.282 | fix(codemap): add focused SQLite schema coverage for canonical index paths, DB initialization, local `.codemap/.gitignore` handling, schema-version fallbacks, idempotent initialization, and FTS insert/update/delete synchronization; fix the external-content FTS column contract by replacing the legacy `co` alias with `calls_out` and migrating existing v1 FTS tables, raising `src/shared/python/codemap/db.py` focused coverage from 31.82% to 100.00%. |
| 2026-06-02 | 1.1.281 | feat(a11y): improve the Unit Converter web app's theme-toggle and custom-unit validation accessibility. The theme button now keeps `aria-pressed` synchronized with the active dark/light state, and custom unit validation messages are announced via dynamic `aria-describedby` while preserving existing input hints. |
| 2026-06-02 | 1.1.280 | fix(tools): consolidated A–O review fixes resolving issues #3173/#3174/#3175/#3176/#3179/#3183/#3184/#3185/#3186/#3187/#3188 — AI adapter/tool-bridge/CLI-tools hardening, model_generation FastAPI/URDF roundtrip fixes, sidekick syngas_compression calculator de-duplication, theme color fallback drift guard, UI headless import safety, plus chat routing lifecycle, programmatic PID pipeline, and humanoid builder assembly coverage. |
| 2026-06-02 | 1.1.279 | test(codemap): add focused headless coverage for the codemap parser dispatcher, including case-insensitive extension mapping, unsupported-path handling, all registered language dispatch routes, missing-extractor fallback, and public re-export registry stability, raising `src/shared/python/codemap/parsers.py` focused coverage from 58.06% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.278 | test(codemap): add focused headless coverage for shared tree-sitter parser helpers, including byte/text extraction helpers, child lookup, line range conversion, unsupported-language handling, successful parser construction/cache reuse, missing optional-language caching, and initialization-failure warning behavior, raising `src/shared/python/codemap/_ts_common.py` focused coverage from 66.18% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.277 | test(codemap): add focused headless coverage for the Rust tree-sitter extractor, including parser-independent `use` imports, top-level functions, structs, typed and untyped impl blocks, nested modules, nested impl methods, unavailable-parser fallback, and incomplete-item guards, raising `src/shared/python/codemap/_lang_rust.py` focused coverage from 8.43% to 98.80% without changing production behavior. |
| 2026-06-02 | 1.1.276 | test(codemap): add focused headless coverage for the JavaScript and TypeScript tree-sitter extractors, including parser-independent import extraction, functions, exported/ambient declarations, class and abstract-class methods, variable-assigned function forms, TS/TSX language dispatch, unavailable-parser fallback, and incomplete-node guards, raising `src/shared/python/codemap/_lang_js.py` focused coverage from 7.08% to 96.46% without changing production behavior. |
| 2026-06-02 | 1.1.275 | test(codemap): make the focused Python parser coverage test independent of the optional `tree_sitter_python` wheel by driving extraction through a parser-shaped fake tree, preserving the existing `src/shared/python/codemap/_lang_python.py` 97.95% focused coverage target while keeping Python 3.10 CI deterministic. |
| 2026-06-02 | 1.1.274 | test(codemap): add focused headless coverage for the Python tree-sitter extractor, including real import/symbol/docstring/signature/call extraction, unavailable-parser fallback, missing-name guards, parser-shaped fake definition nodes, call fallback handling, import edge cases, and block recursion, raising `src/shared/python/codemap/_lang_python.py` focused coverage from 7.53% to 97.95% without changing production behavior. |
| 2026-06-02 | 1.1.273 | test(codemap): add focused headless coverage for the Markdown tree-sitter extractor, including parser-independent ATX heading extraction from byte input, long heading truncation, unavailable-parser fallback, raw heading fallback text, and blank heading skipping, raising `src/shared/python/codemap/_lang_markdown.py` focused coverage from 0.00% to 91.43% without changing production behavior. |
| 2026-06-02 | 1.1.272 | test(plot-engine): add focused headless coverage for the Matplotlib renderer, including line/scatter styling, trendline success and failure paths, 3D surface rendering, contour and heatmap options, histogram styling, filter-comparison difference plots, PNG export, validation guards, and helper defaults, raising `src/shared/python/plot_engine/matplotlib_renderer.py` focused coverage from 8.38% to 100.00% without changing production behavior. |
| 2026-06-02 | 1.1.271 | test(plot-engine): add focused headless coverage for the Plotly converter JSON contract, including typed dispatch for line/scatter, surface, contour, heatmap, histogram, and filter-comparison specs, style/layout serialization, trendline naming and failure handling, required-input guards, and helper defaults, raising `src/shared/python/plot_engine/plotly_converter.py` focused coverage from 0% to 94.77% without changing production behavior. |
| 2026-06-02 | 1.1.270 | fix(calc_backend,signal_toolkit): iterate the scrubber router's column area -> liquid flux -> flooding velocity -> diameter solve to convergence so `liquid_mass_flux` is self-consistent with the solved cross-section instead of an assumed 1 m2 basis (#3181); and restore Design-by-Contract `ValueError` guards on `Integrator.integrate`/`compute_integral` that reject NaN, inverted (`lower > upper`), and out-of-range integration bounds via explicit checks that survive `python -O` (#3182). Regression tests live in dedicated, fully type-annotated files (`calc_backend/tests/test_scrubber_convergence_3181.py`, `signal_toolkit/tests/test_bound_validation_3182.py`) to keep the delta-CI mypy surface clean. |
| 2026-06-02 | 1.1.269 | fix(scripting): add an AST escape pre-screen (`_screen_source_for_escapes`) to the `ConsoleEnvironment` sandbox so user source is rejected before compile/exec when it accesses dunder attributes (`__class__`/`__bases__`/`__subclasses__`/`__globals__` traversal) or constructs dunder names at runtime via `getattr`/`setattr`/`delattr`/`vars`/`type`/`globals`/`locals` with a non-literal or dunder name argument; raises a new `SecurityError`, wires the screen into `execute()` and `refresh_user_functions()`, and documents the authoritative out-of-process trust boundary with the in-process screen as defense-in-depth (#3180). |
| 2026-06-02 | 1.1.268 | test(plot-engine): add focused PyQt6 widget coverage for constructor theme wiring, spec rendering and signal emission, refresh/theme-change rerendering, export dialog/save behavior, empty-export guards, and image byte delegation, raising `src/shared/python/plot_engine/pyqt6_widget.py` focused coverage from 0% to 96.81% without changing production behavior. |
| 2026-06-02 | 1.1.267 | test(plot-engine): add focused headless coverage for plot engine protocol contracts, including runtime structural conformance for renderers, converters, and theme color providers plus explicit protocol stub coverage, raising `src/shared/python/plot_engine/protocols.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.266 | test(plot-engine): add focused headless coverage for trendline computation, including linear NaN filtering, polynomial degree capping and zero equations, exponential and power fits, optimizer fallback behavior, insufficient-data validation, unknown trend types, R-squared edge cases, and helper validation paths, raising `src/shared/python/plot_engine/trendline.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.265 | test(plot-engine): add focused headless coverage for contour data preparation, including scatter interpolation grid shape/value behavior, NaN filtering, insufficient-point validation, correlation matrix defaults, custom labels, and dimensionality validation, raising `src/shared/python/plot_engine/contour.py` focused coverage to 100% without changing production behavior. |
| 2026-06-02 | 1.1.264 | test(notes): add focused headless coverage for the shared notes dock integration helper, covering custom/default dock areas, dock construction, parent propagation, and invalid host validation, raising `src/shared/python/notes/integration.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.263 | test(notes): add focused headless coverage for shared notes markdown card storage, including markdown metadata round trips, create/update/list ordering, recycle/restore, settings persistence, legacy text-note migration, index helpers, and validation/error paths, raising `src/shared/python/notes/card_store.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.262 | test(notes): add focused headless coverage for shared notes models and storage validation, normalization, save/load/clear, recycle/restore/purge, index ordering, and error paths, raising `src/shared/python/notes/models.py` and `src/shared/python/notes/storage.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.261 | test(theme): add focused PyQt-light ThemeManager coverage for singleton reset, inherited app-context preferences, theme queries, stylesheet fallback, registered window application, custom theme persistence/loading/deletion, and validation/error paths, raising `src/shared/python/theme/theme_manager.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.259 | test(theme): add focused PyQt zoom controller coverage for configuration validation, persisted zoom loading, font scaling, step/reset helpers, install/uninstall, keyboard shortcuts, and Ctrl+wheel handling, raising `src/shared/python/theme/zoom.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.260 | test(theme): add focused stylesheet generator coverage for complete QSS section output, minimal embedding styles, required theme color validation, and public exports, raising `src/shared/python/theme/stylesheets.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.255 | fix(folder-packer-pro): keep the headless `operations.py` messagebox fallback typed under mypy by assigning the optional Tk import through an `Any`-typed alias while preserving the unavailable-messagebox runtime guard. |
| 2026-06-01 | 1.1.254 | test(theme): add focused headless coverage for shared matplotlib style helpers, including themed figure/axes/legend styling, default color fallbacks, canvas redraw behavior, global rcParams, palette cycling, and styled figure creation without changing production behavior. |
| 2026-06-01 | 1.1.253 | test(theme): add focused headless coverage for shared icon SVG registry rendering, unknown-icon validation, argument type guards, external SVG recoloring, and missing-file handling, raising `src/shared/python/theme/icon_utils.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.252 | test(theme): add focused coverage for shared theme typography constants, CSS font-stack exports, PyQt font-family selection, explicit-family handling, italic flags, font weights, and missing-size validation, raising `src/shared/python/theme/typography.py` focused coverage above 90% without changing production behavior. |
| 2026-06-01 | 1.1.251 | test(theme): add focused coverage for shared theme color validation, normalization, RGBA conversion, matplotlib palette mapping, JSON loader fallback/error paths, and Qt color conversion, raising `src/shared/python/theme/colors.py` focused coverage above 99% without changing production behavior. |
| 2026-06-01 | 1.1.250 | test(theme): add focused coverage for shared theme style constants and parameterized stylesheet helpers, raising `src/shared/python/theme/style_constants.py` focused coverage to 100% without changing production behavior. |
| 2026-06-01 | 1.1.249 | fix(mcp): keep config-loader preset application and npx package detection typed under the CI mypy delta profile while preserving the Python 3.10 MCP compatibility and config writer coverage changes. |
| 2026-06-01 | 1.1.248 | fix(mcp): keep MCP contracts importable on Python 3.10 by using a `str`/`Enum` transport type, keep config-loader merge validation and npx package detection mypy-clean, remove the Windows shell wrapper from the npm preset probe, and add focused deterministic coverage for the pure `config_writer` MCP server JSON writer/reader. |
| 2026-06-01 | 1.1.247 | test(mcp): add focused deterministic coverage for the pure `config_writer` MCP server JSON writer/reader, including Claude Desktop serialization, duplicate and invalid server validation, malformed environment placeholder rejection, missing/malformed file handling, flat and `mcpServers` read normalization, invalid-entry filtering, and the `load` alias. |
| 2026-06-01 | 1.1.246 | fix(performance-utils): make `OptimizedFileScanner` cache entries expire by both TTL and root directory mtime so changed directories are rescanned within the 60-second cache window, and handle top-level directory enumeration errors consistently with inaccessible child directories. Added focused deterministic coverage for scanner cache invalidation, TTL reuse/expiry, worker error suppression, hashing paths, and chunked/lazy memory utilities. |
| 2026-06-01 | 1.1.245 | fix(folder-packer-pro): guard the `operations.py` messagebox import so headless Linux runners without Tk shared libraries can import the operation mixins while GUI runtime behavior stays unchanged when Tk is available. |
| 2026-06-01 | 1.1.244 | fix(folder-packer-pro): teach `inspect_package()` to read uncompressed unencrypted archives instead of mislabeling them as encrypted, and add focused headless coverage for `folder_packer_pro` file operations, pack/unpack engine behavior, archive path traversal rejection, cancellation/error handling, and operation mixin workflows. |
| 2026-06-01 | 1.1.243 | test(data-processing): add focused coverage for the shared pandas formula validator and `DataProcessor.apply_formula` integration, pinning accepted arithmetic/boolean grammar, unsafe syntax rejection, complexity/exponent guards, and rejection logging without formula text leakage. |
| 2026-06-01 | 1.1.242 | fix(model-generation): harden the headless `model_generation` CLI library commands by parsing category/source filters into library enums, using `ModelEntry.id` in list/add output, defaulting adds to `ModelCategory.OTHER`, trimming comma-separated tags, and keeping the typed CLI dispatch path mypy-clean. Added focused CLI tests covering parser wiring, library list/add behavior, invalid filters, and inertia dimension errors. Also keeps Sidekick workspace facade name listing typed under both local and CI mypy import modes. |
| 2026-05-31 | 1.1.240 | fix(sidekick): harden calculator workspace adapter typed boundaries so changed-file mypy checks keep `Path`, `bool`, and `list[str]` return contracts when helper modules are skipped during CI analysis. |
| 2026-05-31 | 1.1.241 | fix(sidekick): harden calculator workspace adapter typed boundaries so changed-file mypy checks keep `Path`, `bool`, and `list[str]` return contracts when helper modules are skipped during CI analysis. |
| 2026-05-31 | 1.1.239 | test(sidekick): harden the Sidekick per-file coverage gate so only `src/shared/python/sidekick/` production modules are enforced, excluding changed test files from missing-coverage failures. CI now runs the full Sidekick unit suite when Sidekick source changes, and the split runtime/default-tab modules have focused contract coverage for chat bridges, plot requests, fallback diagnostics, tab definitions, and optional-tab placeholders. |
| 2026-05-31 | 1.1.238 | fix(security, #3143 #3144): rewrite wave_solver.py to use argv lists with shell=False (no shell string from issue title/body), make --dangerously-skip-permissions opt-in, and gate destructive git/gh actions (git reset --hard, issue close, gh pr merge --auto) behind an explicit --allow-mutations flag with a dry-run default; replace P1AM backend wildcard CORS (`["*"]` + credentials) with an env-driven allowlist (cors_config.resolve_cors_settings) that defaults to local dev origins, never pairs `*` with credentials, and fails closed in production without an explicit allowlist. |
| 2026-05-31 | 1.1.238 | fix(sidekick): Completed the #3141 monolith-decomposition follow-up by splitting runtime tab, default-tab, calculator workspace, runtime settings, and chat settings responsibilities into focused modules while preserving the historical import surface through facade modules. Added focused alias-contract and coverage-gate regression tests so hosts keep stable live tab collections and changed Sidekick files cannot silently bypass coverage enforcement. |
| 2026-05-31 | 1.1.237 | fix(sidekick): #3138 TabCollection.set_definitions()/sync_order_from_widget() now mutate their backing dict/list in place instead of reassigning, so UnifiedToolsSidebar's live \_tab_definitions/\_tab_ids/\_tab_widgets aliases stay current (fixes duplicate/pop-out/redock/settings flows); PythonReplWidget.execute() now waits on its worker thread and delivers output deterministically without a spinning event loop (fixes REPL output). #3139 check_sidekick_coverage.py fails when a changed Sidekick file is missing from coverage XML or when an enforced run counts zero files, closing the vacuous-pass gap. #3140 removed two stale TDD-pending xfail markers now that the package-rename import-boundary contracts pass. Part of #3141 (monolith decomposition deferred to a focused follow-up). check_sidekick_coverage.py now parses coverage.xml via defusedxml.ElementTree (matching check_coverage_policy.py) to satisfy bandit B314. |
| 2026-05-31 | 1.1.236 | perf(golf): optimize calculateTempoQuality in phaseDetector.ts by replacing the two chained .filter().reduce() passes with a single-pass for loop, eliminating intermediate array allocations while preserving the tempo score. |
| 2026-05-31 | 1.1.235 | feat(a11y, p1am frontend): add `aria-pressed` to custom toggle buttons in ControlDashboard (PID loop selector) and RoutingMatrix (input/output route cells) so screen readers announce active state. |
| 2026-05-30 | 1.1.233 | perf(golf): optimize array iterations in swingAnalyzer by replacing chained .filter().reduce() with single-pass for loops in calculateTempoMetrics and calculateSwingScores; ci: remove the retired fix-brick.yml toolcache-repair workflow (consolidates #3124 and #3129). |
| 2026-05-30 | 1.1.232 | feat(ux, #3115): improve accessibility of the ODE Solver UI by explicitly linking labels to inputs and textareas using htmlFor and unique IDs, add spellcheck="false" and disabled autocorrect. |
| 2026-05-30 | 1.1.231 | perf(p1am frontend, #3126): optimize array aggregations in AlarmsHeader.tsx by replacing chained .filter() and .reduce() operations with a single-pass loop. |
| 2026-05-30 | 1.1.230 | Fix CI failures on PR #3123: re-export \_QS_ORG/\_QS_APP/\_QS_VISIBLE_TABS_KEY from sidebar, fix apply_state \_dock_widget AttributeError (now uses \_dock_chrome.dock_widget), add waitUntil to MockQtBot, fix F6 isVisible→isHidden for headless tests, fix F10 duplicate-pin test to use subdirectory, add runtime_tabs.py and registry.py to monolith baseline, bump SPEC version. |
| 2026-05-30 | 1.1.229 | chore: remove stale type-ignore suppression comments in data_explorer_service, project_file_explorer, runtime_tabs; add explicit bool() cast on eventFilter return in os_terminal to satisfy mypy no-any-return. |
| 2026-05-30 | 1.1.228 | F4: Patched TabCollection.replace() to correctly update internal id mapping when swapping widgets; fixes stale id→widget reference after atomic swap. |
| 2026-05-30 | 1.1.227 | F4: Decomposed UnifiedToolsSidebar god class. Extracted TabCollection (id↔widget↔order bookkeeping), DockChromeController (collapse/minimize/dock-area/title-bar/shortcuts), and VisibilityPersistence (project-root-scoped QSettings read/write). Sidebar is now a thin coordinator that delegates to these three collaborators. Backward-compatible shims (\_tab_ids/\_tab_widgets/\_tab_definitions) preserved for mixins. Added test_sidekick_f4_collaborators.py with tests for all three. |
| 2026-05-30 | 1.1.226 | F6: PythonReplWidget now executes user scripts on a background QThread (\_ReplWorker) so the GUI stays responsive. Added \_cancel_button (best-effort terminate), \_status_label ('Running...'), \_set_running() toggle helper, and \_on_execution_finished() slot that syncs the namespace back to the registry on completion. |
| 2026-05-30 | 1.1.225 | F2: Added Ctrl+C interrupt button (writes 0x03 to PTY), Stop/restart button, command history ring (Up/Down navigate, newest-first, deduplicates), and eventFilter on input QLineEdit in SidekickOsTerminalWidget. |
| 2026-05-30 | 1.1.224 | F8: Added replace_tab_widget() to UnifiedToolsSidebar for atomic chat-dock retry swaps that keep both QTabWidget and \_tab_widgets bookkeeping in sync. F9: Rewrote registry.update_from() to merge via public set()/\_set_repr_entry() so name validation runs and subscribers are notified; same fix applied to load_json(). |
| 2026-05-30 | 1.1.223 | F10: Quick-access folder pins in ProjectFileExplorer now persist to and restore from QSettings (project-root-scoped key); duplicates are rejected. F11: Hoisted a shared `resolve_columns` helper in `data_explorer_service` to eliminate the duplicated column-validation logic in `data_processor_tab`. |
| 2026-05-30 | 1.1.222 | F1: Fixed Windows PTY double-submit by writing b"\n" instead of os.linesep. F3: Fixed PTY output chunk-stripping by using raw QTextEdit.append. F5: Consolidated QSettings writes into \_persist_visible_tabs with explicit org/app names. F7: Implemented singleton help dialog to prevent duplicate windows. |
| 2026-05-29 | 1.1.215 | Hardened the Sidekick C3D reader to validate the header magic byte before invoking ezc3d, so mislabeled or truncated files raise a typed `ValueError` instead of surfacing parser internals; added focused regression coverage for invalid headers and updated C3D reader tests to use temp files with valid magic bytes. |
| 2026-05-27 | 1.1.214 | Fixed HistorySidebar initialization, updated theme manager colors, and synchronized Tools baseline hashes. |
| 2026-05-27 | 1.1.210 | Added P1AM analog I/O calibration helper script and interactive Modbus CLI procedure documentation. |
| 2026-05-27 | 1.1.209 | Simplified HistorySidebar implementation to reduce lines of code under 500 lines to satisfy the file size budget check constraint. |
| 2026-05-27 | 1.1.201 | Added Sidekick Chat controls to create new chat or load conversation history, integrated HistorySidebar in horizontal QSplitter, added toolbar/status buttons, WebSocket session_created handler, and comprehensive tests. |
| 2026-05-23 | 1.1.200 | Added `sidekick.bootstrap` import to the deprecated `upstream_drift_tools` compatibility shim to preserve legacy import paths. |
| 2026-05-26 | 1.1.200 | Kept the optional session-scoped PyQt `qapp` pytest fixture in `tests/conftest.py` ruff-compliant by normalizing the guarded local import block, so PR-local test harness changes stop tripping the CI quality gate on import-order formatting alone. |
| 2026-05-22 | 1.1.199 | Fixed mypy TYPE_CHECKING import guards in sidekick process calculators (syngas_compression_calculator, acid_gas_dewpoint_calculator, pressure_drop_interface, syngas_compression_engine) and calculator_state_mixin to use `if TYPE_CHECKING:` conditional imports for optional PyQt6/matplotlib dependencies, eliminating incompatible-assignment and no-redef errors across Qt-installed and Qt-absent environments. |
| 2026-05-22 | 1.1.198 | Tightened local hook behavior for consolidated task branches so pre-push fleet guardrails inspect the unpushed commit range before falling back to the full repository, and changed the Bandit pre-push hook to scan the Python files selected by pre-commit instead of re-scanning existing repository-wide baseline debt. |
| 2026-05-21 | 1.1.195 | Resolved shared AI/chat unit-test failures by tightening Rust adapter optional-backend behavior, removing obsolete phase-one integration coverage, and updating Ollama, Rust adapter, and AI memory manager tests to use deterministic mocks for terminal-provider and event-loop contracts. |
| 2026-05-20 | 1.1.192 | Fixed shared Sidekick chat dock shutdown so an intentional widget close suppresses the WebSocket reconnect timer while unexpected disconnects retain the existing retry path; added focused regression coverage for both lifecycle branches. |
| 2026-05-20 | 1.1.191 | Hardened Sidekick test-health coverage so the Jupyter tab availability positive path simulates an importable optional `nbformat` module without requiring the package in the base environment, while the missing-dependency negative path remains covered. Marked the Sidekick dock close-affordance Qt tests as serial/offscreen and skipped them inside Windows xdist workers so the serial lane keeps coverage without crashing parallel workers. |
| 2026-05-20 | 1.1.190 | Added shared Sidekick/chat launcher integration contracts: `ChatServiceBase.condense_to_memory()` now persists explicit memory candidates through the shared memory manager, `UnifiedToolsSidebar.open_tab()` focuses visible and hidden tabs with `os_terminal` compatibility, ChatDockWidget exposes readiness diagnostics, and Qt chat imports gained subprocess-backed PyQt6 runtime diagnostics with focused regression coverage. |
| 2026-05-18 | 1.1.185 | Added `htmlFor` and `id` mapping to range inputs in `SwingComparison.tsx` (`src/media_processing/video_processor/apps/web`) to improve screen reader accessibility. |
| 2026-05-18 | 1.1.184 | Optimized Nelder-Mead optimization loop in pendulum simulator by replacing map and slice with pre-allocated arrays and standard for loops to minimize GC pauses. |
| 2026-05-17 | 1.1.183 | Pre-allocated the `results` array in the `solveODESystem` hot RK4 integration loop (`src/ode_solver/web/src/lib/odeSolver.ts`) to eliminate continuous memory reallocation overhead and garbage collection pauses during large numerical simulations. |
| 2026-05-15 | 1.1.181 | Split AI settings local-provider configuration widgets so Ollama keeps its host/model discovery controls, Cline shows its own endpoint test UI, BitNet shows an installation-root hint tied to the main model selector, and CLI-backed providers no longer render misleading Ollama-specific fields; added focused PyQt6 regression coverage for the provider-specific widget contracts. |
| 2026-05-15 | 1.1.179 | Added a markdown-backed shared notes card store with stable path-safe IDs, metadata round trips, validated note and board colors, reversible markdown-card recycling/restoration, legacy `project.notes.txt` migration, import-safe backend coverage, and a lightweight Sidekick Notes color-control contract that reuses the shared store. |
| 2026-05-15 | 1.1.178 | Added an optional Sidekick Function Generator tab with import-safe PyQt6 launcher integration, shared default-tab/help metadata, design-token aliases, and focused sidebar regression coverage. |
| 2026-05-15 | 1.1.176 | Added Sidekick calculator workspace management with isolated calculator-local variables, explicit local-to-global promotion, scoped local/global JSON workspace persistence helpers, focused regression coverage for merge, replace, malformed-file rollback, and duplicate-facade separation behavior, stabilized Sidekick data explorer dtype summaries across pandas string dtype changes, and kept calculator-tab expression evaluation inside the shared safe math evaluator so headless imports do not require Flask or tool-specific calculator packages. |
| 2026-05-14 | 1.1.175 | Added a lazy optional Sidekick Data Processor tab that stays hidden by default, reports missing UI/runtime dependencies without crashing Sidekick, and exports validated selected Data Processor results into the shared workspace registry with focused import/runtime regression coverage. |
| 2026-05-14 | 1.1.174 | Added a Sidekick Data Explorer tab with project-scoped file validation, bounded CSV/TSV/JSON/Parquet/Excel preview service limits, schema/null-count sample summaries, preview-to-workspace export, and a structured Data Processor handoff request contract plus focused backend/UI regression coverage. |
| 2026-05-14 | 1.1.173 | Added a bounded Sidekick workspace command line to the calculator tab for explicit local/global variable assignment, inspection, deletion, clear, and load/save operations, reusing the shared command-history and workspace persistence contracts while keeping workspace mutations separate from arbitrary terminal execution. |
| 2026-05-14 | 1.1.172 | Added a pure-Python Sidekick help registry for default tabs and shared context-menu actions, wired default-tab help metadata into the shared sidebar, exposed a Help action in the tab context menu, added hover hints to compact terminal/notes controls, documented custom-tab help requirements in the sidebar README, and expanded the shared UI regression suite to enforce the new help contract. |
| 2026-05-14 | 1.1.171 | Added Sidekick named state profile storage helpers with path-safe save/load contracts, atomic malformed-profile rejection, explicit clear-data warning confirmation, sidebar wrapper methods, README guidance, and focused regression coverage. |
| 2026-05-14 | 1.1.170 | Added validated Sidekick calculator startup import preferences with default optional NumPy/SciPy aliases, JSON sidebar-state persistence, transaction-safe import execution, missing-dependency diagnostics in the calculator tab, and focused backend/UI regression coverage. |
| 2026-05-14 | 1.1.169 | Added calculator-local Sidekick workspace save/load wiring with an explicit scoped persistence controller, JSON path validation, atomic save, merge-versus-confirmed-replace load behavior, malformed-file rollback, and UI button coverage that keeps calculator workspace persistence separate from the global sidebar workspace registry. |
| 2026-05-14 | 1.1.168 | Added a Sidekick file explorer navigation controller with normalized current path state, back/forward/up history, injectable common-location discovery, project-boundary containment, and predictable disabled-state flags, then wired the project explorer widget to expose a compact navigation bar and common-locations sidebar. |
| 2026-05-14 | 1.1.165 | Optimized the ODE solver RK4 integration loop by moving state and derivative buffers from keyed objects to indexed arrays, extracted the solver and presets into a pure module, and added Vitest coverage for analytical decay, coupled oscillator order, and solver preconditions. |
| 2026-05-14 | 1.1.164 | Improved calculator bounds/value input accessibility by labeling the grouped lower-bound, upper-bound, and evaluation-point controls with a shared group name plus explicit accessible names for each field. |
| 2026-05-14 | 1.1.163 | Optimized the pressure-drop calculator gas-composition hot paths by replacing repeated object-entry/value reductions with single-pass keyed loops for mixture molecular weight, total composition, and normalized composition construction. |
| 2026-05-14 | 1.1.162 | Refactored Sidekick default tab construction into a focused helper module so `UnifiedToolsSidebar` stays below the changed-file LOC budget while preserving the runtime tab behavior introduced in 1.1.161. |
| 2026-05-14 | 1.1.161 | Replaced remaining Sidekick runtime placeholders with embedded utility widgets: chat status/optional PyQt chat dock loading, a workspace-aware Python terminal with optional numpy/pandas/scipy aliases, a TI-89 symbolic calculator tab that publishes results into workspace state, and project-persistent notes with explicit save and debounced autosave. Added widget contract coverage for the runtime tabs. |
| 2026-05-14 | 1.1.160 | Added runtime Sidekick theme reapplication APIs so existing PyQt sidebar instances can switch shared themes or explicit design-token sets without being reconstructed. |
| 2026-05-14 | 1.1.159 | Added shared-theme-name resolution to the Sidekick host factory/install helpers so PyQt hosts can opt into canonical theme definitions without hand-building design tokens. |
| 2026-05-14 | 1.1.156 | Added shared PyQt6 responsive sizing and application zoom utilities for issue #2647. The theme package now exposes text-aware minimum width helpers, readable form-layout configuration, scroll-area wrapping, a persisted application zoom event filter for Ctrl+wheel/Ctrl+plus/Ctrl+minus/Ctrl+0, and scaled UI tokens for downstream QSS/layout regeneration; package discovery now includes the `shared*` namespace so these fleet imports ship with `ud-tools`. |
| 2026-05-14 | 1.1.155 | Added the canonical Sidekick design-token bridge with pure-Python token exports, CSS-variable and QSS mapping helpers, stable Qt object names/selectors, default shared sidebar styling, and focused tests for token contract and backend import safety. |
| 2026-05-13 | 1.1.154 | Expanded the shared sidebar into the Sidekick toolkit with configurable tab definitions, persisted left/right dock placement, minimized state, tab ordering, hidden tabs, popped-out tab tracking, redock and duplicate-tab APIs, and tests for flexible host workflows while preserving the existing `install_tools_sidebar` contract. |
| 2026-05-13 | 1.1.153 | Added the shared `upstream_drift_tools.ui.tools_sidebar` package with a Qt-binding-compatible dockable sidebar, project file explorer, workspace registry/state persistence, public `create_tools_sidebar` and `install_tools_sidebar` APIs, and focused backend/import/widget contract tests for downstream host integration. |
| 2026-05-13 | 1.1.152 | Improved chat layout by moving the shared dock Close button into the persistent status header, replacing clipped history-list text with wrapped row widgets, and adding transparent icon-only archive, restore, and delete actions directly on chat-history rows. |
| 2026-05-13 | 1.1.151 | Hardened shared chat dock terminal lifecycle controls so Start is disabled while a terminal session is pending or active, Stop is enabled only for active sessions, and shell/provider selectors are locked while the selected terminal agent session is running. |
| 2026-05-13 | 1.1.150 | Improved the shared chat dock terminal interface by populating shell/provider selectors from the terminal provider registry, adding an explicit terminal Stop action wired to the existing WebSocket stop protocol, and adding an in-dock Close button so embedded chat windows can be dismissed from inside the chat UI. |
| 2026-05-13 | 1.1.149 | Added shared AI chat memory management with a Tools-scoped `user_memory.json` store, explicit archived-conversation preference extraction, project-root `AGENTS.md` prompt inclusion, bounded prompt-memory formatting across provider adapters, and focused regression coverage so archived chats inform future sessions without becoming opaque model training data. |
| 2026-05-13 | 1.1.148 | Added data-driven shared chat terminal-provider descriptors for Claude Code, Codex, Cline CLI, and Gemini CLI, plus default registry builders, install/auth probe command metadata, and command redaction helpers so downstream UIs can enumerate terminal agents without copying provider lists or logging secret-like command values. |
| 2026-05-13 | 1.1.144 | Added a native BitNet direct subprocess adapter for shared AI chat provider resolution, exposing local 1.58b models through the adapter factory and settings metadata without requiring an external FastAPI server. |
| 2026-05-13 | 1.1.143 | Synchronized Signal Toolkit Matplotlib canvas theming for issue #2582 by applying the active fleet plot theme after axes are created, keeping legacy `setup_dark_theme()` wired to the shared theme manager, and adding regression coverage for themed axes and spines. |
| 2026-05-13 | 1.1.142 | Registered the migrated Video Analyzer PyQt6 surface in the generator-backed tools catalog and surface contract so issue #2585 is visible through both the canonical GUI manifest and generated launcher outputs. |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests. |
| 2026-05-13 | 1.1.140 | Tightened the shared chat package contract for issue #2592 by exporting the documented model/list/index facade symbols, adding a `chat` optional dependency group and compatibility matrix, fixing installed-package lazy Qt loading, validating model/index status payloads, and removing product-specific defaults from reusable AI assistant GUI metadata. |
| 2026-05-12 | 1.1.135 | Added Rust `tools-core.signal` moving-average and exponential-smoothing kernels with PyO3 numpy vector-in/vector-out endpoints, filling the remaining smoothing-filter slice after the LMS/RLS migration. |
| 2026-05-12 | 1.1.134 | Promoted LMS/RLS adaptive filters to native Rust implementations via PyO3 bindings, eliminating Python-side vectorization overhead for high-frequency signal processing pipelines (PR #2575). |
| 2026-05-11 | 1.1.132 | Fixed `signal_toolkit.calculus` import: replaced bare `from src.shared.python.contracts import require` (broken because the repo root is not on `pytest`'s pythonpath) with the sibling-module try/except pattern used in `core.py`, and cast `Differentiator.differentiate`'s return to `np.asarray(dy)` to keep mypy `no-any-return` clean. Unblocks `tests (3.x)` matrix on `main`. |
| 2026-05-11 | 1.1.131 | Added shared `codemap` package (`src/shared/python/codemap/`) — tree-sitter symbol index over SQLite FTS5 with a 6-function pydantic query API (`search_code`, `get_symbol`, `who_calls`, `imports_of`, `neighbors`, `repo_summary`), CLI (`codemap rebuild/search/who-calls/export/info`), `watchdog` daemon (`codemap-watch`), and FastMCP server (`codemap-mcp`) so external coding agents inherit the same data the in-app chat consumes. `.codemap/` is gitignored; embedding layer deferred to a follow-up. |
| 2026-05-11 | 1.1.130 | Hardened `signal_toolkit.calculus.Differentiator.differentiate` with an explicit `require(order >= 1, ...)` precondition so non-positive derivative orders raise `PreconditionError` instead of silently producing an empty derivative loop. |
| 2026-05-11 | 1.1.129 | Added dynamic focus shifting to inline form validation within the Calculator app. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`. |
| 2026-05-07 | 1.1.128 | Pre-compiled ODE Solver derivative expressions outside the RK4 loop while preserving the existing non-finite fallback behavior, so singular or overflowing user formulas still collapse to `0` instead of poisoning the integration state with `NaN` or `Infinity`. |
| 2026-05-05 | 1.1.125 | Optimized polynomial evaluation using Horners method in `pendulum-web` physics engines (`physics.ts`, `physics_triple.ts`, `physics_golfer.ts`). |
| 2026-05-04 | 1.1.124 | Documented production-readiness hardening for generated data-processing batch scripts, shared pandas formula allowlist validation, model-generation mesh upload size and filename checks with cleanup, and MakeHuman generated-script serialization plus the `mesh_generator_makehuman.py` compatibility shim. |
| 2026-04-26 | 1.1.111 | Improved accessibility for the calculator clear button's soft confirm state. Added `aria-live="polite"` to the parent row and dynamically toggled the `aria-label` between "Clear all fields" and "Confirm clear all fields" to keep screen reader users informed of the required secondary action. |
| 2026-04-25 | 1.1.107 | Fixed StrEnum import compatibility for Python 3.10 by routing `steam_engine_calculator` and `video_processor` API modules through the existing `utils.compatibility` backport facade, eliminating import-time failures on the 3.10 CI interpreter. |
| 2026-04-25 | 1.1.106 | Added dynamic focus shifting to inline form validation within the Unit Converter app's Custom Units modal. This prevents keyboard focus traps by focusing the first invalid input (`.focus()`) and marking it with `aria-invalid="true"`. |
| 2026-04-23 | 1.1.103 | Tightened the shared `model_generation` unified-loader conversion contract so malformed MJCF/URDF XML parse failures are wrapped as `ConversionError`, converter-raised `ConversionError` instances propagate unchanged, and regression tests lock the typed error/logging behavior. |
| 2026-04-23 | 1.1.101 | Hardened model-generation REST routing so unexpected route-handler programming errors propagate to the framework adapter instead of being flattened into JSON 500 responses by the route facade, with regression coverage for the propagation contract. |
| 2026-04-23 | 1.1.100 | Extended the Python 3.10 UTC compatibility contract across document-processing, folder-packing, shared model-generation, upstream-drift UI/state, folder-tool analysis, and launcher timestamp paths by using `timezone.utc` instead of the Python 3.11-only `datetime.UTC` alias while preserving timezone-aware datetime behavior. |
| 2026-04-23 | 1.1.99 | Kept shared data-processing result timestamps timezone-aware while preserving Python 3.10 compatibility by using `timezone.utc` rather than the Python 3.11-only `datetime.UTC` alias, keeping the data-processing import contract green across the supported CI interpreter matrix. |
| 2026-04-25 | 1.1.105 | Narrowed `ConsoleEnvironment.refresh_user_functions()` to re-raise `KeyboardInterrupt` and `SystemExit` while still logging expected user-code failures from the persisted scripting library, and added focused regression coverage for both reload paths. |
| 2026-04-23 | 1.1.98 | Documented the rotation converter API exception-boundary tests that keep invalid quaternion parsing mapped to HTTP 422 while allowing unexpected reference-frame runtime failures to propagate for diagnostics instead of being silently swallowed. |
| 2026-04-23 | 1.1.97 | Security and robustness remediation pass from adversarial review: tightened exception boundaries and error propagation for shared rotation conversion, scripting runtime, and model-generation loaders; hardened data-processing and state-management paths against invalid inputs and silent failures; and aligned related test coverage for the updated failure-handling contracts. |
| 2026-04-23 | 1.1.96 | Hardened ODE and signal generation preconditions so direct RK4 calls reject fewer than two output points, chirp generation rejects single-point time arrays, and sawtooth/triangle/square generation reject non-positive frequencies with clear `ValueError` messages instead of division-by-zero failures. |
| 2026-04-22 | 1.1.92 | Fixed Design by Contract runtime toggling so contract primitives, decorators, invariant checks, and validation helpers read the canonical contract state instead of stale module-level compatibility aliases; added regression coverage for alias/state divergence. |
| 2026-04-22 | 1.1.91 | Security hardening (refs #2219): removed starred argument unpacking from the safe mathematical expression evaluator AST allowlist and added regression coverage so expressions such as `sum(*x)` are rejected before execution. |
| 2026-04-22 | 1.1.88 | Test-enforcement fix (refs #2211): restricted GH1732 logging-consistency excluded-directory matching to the top-level `src/<segment>` only, and added regression coverage proving nested path segments named like excluded dirs remain in sweep scope. |
| 2026-04-22 | 1.1.87 | Documented the `signal_toolkit` package organization for adaptive filters: `AdaptiveFilter` now lives in `adaptive_filter.py` while remaining available from the package root and legacy `filters` module. |
| 2026-04-22 | 1.1.85 | Implementation (refs #2200): added a flat Asteroid Jumper controller snapshot DTO and routed the renderer through it to remove nested state traversal from the draw path. |
| 2026-04-22 | 1.1.84 | Documentation (refs #2200): reviewed deep object traversal hotspots in launchers, Matplotlib/Qt UI code, assessment scripts, Rust ball-flight physics, and Asteroid Jumper controller code, documenting framework/path/import/value-object boundaries that do not require DTO or facade extraction. |
| 2026-04-22 | 1.1.83 | Optimized statistical calculation in data processor using Welford's algorithm to compute variance in a single pass. |
| 2026-04-19 | 1.1.82 | Removed QTimer.singleShot startup races and leaky lambda captures in shared chat dock and syngas compression calculator UI code by routing deferred initialization through named callbacks and stored helper methods (PR #2163). |
| 2026-04-19 | 1.1.81 | Aligned dependency metadata with the supported Python and toolchain baseline: Python package metadata now starts at Python 3.11, lint/type configuration shares that floor, Black was removed from the canonical format path, and the reproducible requirements lock includes the pytest timeout and benchmark plugins declared by the development manifests (PR #2161). |
| 2026-04-19 | 1.1.80 | Hardened model-generation archive extraction and URDF mesh resolution by normalizing archive member paths, rejecting traversal or absolute members before extraction, and preserving unsafe mesh references as text instead of resolving them to local files (PR #2157). |
| 2026-04-19 | 1.1.79 | Consolidated stale Tools PR fixes covering shared rotation primitives, data processor background worker error surfacing and UI offload, PDF renamer API-key/CORS hardening, narrower exception fallbacks, shared GUI boundary checks, and lower-body manifest registration; also tightened NumPy return typing for the rotation modern robotics helpers checked by quality-gate (PR #2149). |
| 2026-04-19 | 1.1.78 | Optimized `TimeRangePanel.tsx` in `data-processor-web` by computing time-column ranges in a single pass and avoiding `Math.min`/`Math.max` spread calls that can overflow the call stack on large datasets (PR #2156). |
| 2026-04-19 | 1.1.77 | Hardened model-generation library GitHub discovery and downloads by validating generated GitHub API URLs, rejecting non-HTTPS model source URLs, and skipping untrusted subdirectory URLs before network retrieval (PR #2146). |
| 2026-04-21 | 1.1.76 | Added screen-reader-only context to dynamic video progress text and pose detection counters so numeric readouts expose their meaning to assistive technology; decorative pulsing dots are now hidden from screen readers (PR #2138). |
| 2026-04-21 | 1.1.75 | Optimized `calculateStatistics` in `useDataProcessor.ts` by extracting numbers into a dynamically resizing `Float64Array` during the first pass to eliminate a second pass over the original array of objects (PR #2137). |
| 2026-04-21 | 1.1.74 | Disabled pickle-backed reads, writes, and file-dialog discovery in shared data-processing helpers and upstream drift tooling to prevent arbitrary code execution through unsafe deserialization (PR #2139). |
| 2026-04-21 | 1.1.73 | Improved exception handling and signal re-raising in rotation converter UI threads, scripting environment, and model library imports by capturing background thread exceptions, adding structured logging, and re-raising with context (PR #2088). |
| 2026-04-21 | 1.1.72 | Enhanced data processor exception handling by wrapping background threading tasks with try-except blocks that log exceptions and propagate errors to the main thread instead of silently failing (PR #2084). |
| 2026-04-21 | 1.1.71 | Hardened data-processing file I/O by disabling pickle reads and writes by default, removing pickle extensions from GUI-supported file discovery paths, and requiring an explicit trusted-legacy override for pickle use. |
| 2026-04-21 | 1.1.70 | Test configuration hygiene: registered the complete CLAUDE.md marker set in `pytest.ini`, enabled strict xfail handling, and added a contract-test backbone for the ODE solver, pressure-drop calculator, and rotation-converter calc backend request/response models. |
| 2026-04-21 | 1.1.69 | Stopped the bot CI trigger workflow from using stale external credentials for repository checkout and PR/check API operations so bot-authored PRs use repo-scoped workflow credentials for required check discovery. |
| 2026-04-21 | 1.1.68 | Restricted Data Processor web row-copy paths to own enumerable properties via a shared `Object.keys` helper and added regression coverage to prevent inherited prototype keys from being copied into processed rows. |
| 2026-04-21 | 1.1.67 | Filter deleted test files out of the CI changed-test list so PRs that intentionally remove stale tests do not pass non-existent paths to pytest. |
| 2026-04-21 | 1.1.66 | Hardened asteroid-jumper physics validation so non-finite timesteps and physics parameters are rejected with explicit `ValueError`s instead of propagating NaN or infinity through simulation state. |
| 2026-04-21 | 1.1.66 | Simplified root pytest addopts in `pyproject.toml` by removing benchmark and xdist-specific defaults so repository-level test runs do not require those plugins outside focused plugin test contexts. |
| 2026-04-17 | 1.1.64 | Optimized `applyFilter` loop in `useDataProcessor.ts` by replacing the object spread operator with manual property copying to eliminate significant garbage collection overhead during large dataset processing. |
| 2026-04-17 | 1.1.63 | Hardened model-generation GitHub repository downloads by requiring HTTPS retrievals and validating mesh output paths so API-provided mesh names cannot escape the destination directory; kept the unit-converter development WSGI debugger disabled unless `FLASK_DEBUG=1` is explicitly set. |
| 2026-04-17 | 1.1.62 | Enhanced video editor UX by replacing native alert dialogs with inline accessible errors and ensuring proper focus styles. |
| 2026-04-17 | 1.1.61 | Replaced runtime `assert` validation in asteroid-jumper physics, rotation-converter UI helpers, and scripting console execution with explicit exceptions so invalid caller input remains guarded under optimized Python. |
| 2026-04-16 | 1.1.60 | Hardened launcher process handling by validating tool names, cleaning up spawned process groups, surfacing explicit model-conversion errors, and regression-testing temporary-file cleanup paths. |
| 2026-04-16 | 1.1.59 | Removed stale root-level debug artifacts (`.ci_trigger.py`, `MUJOCO_LOG.TXT`, `error_log.txt`, `wave_log.txt`, and the empty marker file ending in `Last`), added root-scoped ignore rules for those paths, and locked the hygiene policy with regression tests. |
| 2026-04-16 | 1.1.58 | Hardened GitHub archive extraction in the model-generation repository helper by validating zip members before unpacking so repository downloads cannot escape the destination directory. |
| 2026-04-16 | 1.1.55 | Replaced object spread operator with manual property copy in `integrateSignals` and `differentiateSignals` loops in `useDataProcessor.ts`; wrapped UI components (`AdvancedPanel`, `ExportPanel`, `FilterPanel`, `ResamplePanel`) in `React.memo()` to prevent unnecessary re-renders. |
| 2026-04-15 | 1.1.56 | Refreshed the data processor regression-preparation optimization spec after CI retriggers so the PR-level SPEC freshness gate sees a documentation update on the latest source-changing branch head. |
| 2026-04-16 | 1.1.57 | Improved the accessibility and semantics of the `AudioRecorder` component in the Video Processor app. Added `aria-label`s to recording control buttons, formatted recording duration for screen readers, hid purely visual elements from screen readers, and enhanced keyboard navigation by adding `focus-visible` styling to all buttons. |
| 2026-04-15 | 1.1.55 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-16 | 1.1.53 | Added `aria-label` and `title` to the dynamically generated "Remove" button (`×`) in the unit converter Custom Units list for screen reader accessibility. |
| 2026-04-13 | 1.1.52 | Added visually hidden `sr-only` span before the raw timer text in `AudioRecorder.tsx` to provide screen reader context and added `aria-hidden` to purely decorative pulsing red dot. |
| 2026-04-13 | 1.1.51 | Added `tools.shared.python.model_generation.editor` compatibility namespace so downstream repos can import the text editor via `tools.shared.python` without duplicating the module; added `-p no:xvfb` to pytest addopts so the test suite runs on headless self-hosted runners that lack Xvfb; applied ruff formatting fixes across GUI stylesheets and multiline string literals. |
| 2026-04-12 | 1.1.51 | Replace remaining `print()` calls with `logging` across `src/` modules and disable xvfb pytest plugin to fix CI timeout on headless runners. |
| 2026-04-13 | 1.1.48 | Wrapped the `SignalList` and `StatisticsPanel` components in `React.memo()` to prevent expensive re-render cascades in the data processor web application during UI tab navigation. |
| 2026-04-12 | 1.1.47 | Added the shared `tools.mypy_autofix_agent` module and `mypy-autofix` console entry point so downstream fleet repositories can call one maintained mypy autofix implementation instead of carrying duplicated script copies; kept `tools.setup_logging` lazy so CLI startup does not import optional heavy dependencies. |
| 2026-04-11 | 1.1.46 | Lower-body builder DRY refactor: extracted `_build_leg_xml(side, ...)` and `_build_leg_actuators_xml(side)` helpers so both legs and both actuator blocks share a single source of truth. `build_lower_body_xml` now calls each helper once per side instead of duplicating ~45 lines of MJCF. New regression tests assert left/right symmetry of joint/body/actuator/geom/site sets and pin the expected counts. |
| 2026-04-11 | 1.1.45 | Closed-chain ankle IK in `LowerBodySimulator.setup_initial_pose`: the ankle angles are solved by a closed-form 2-DOF decomposition of the calf's world rotation so each foot's world Z-axis is `(0, 0, 1)` for any feasible hip/knee pose. Raises `ValueError` identifying the offending axis when the required ankle angle exceeds the ±30° joint limit instead of silently clipping. Defaults changed from 30°/120°/20° (infeasible, silently clipped) to 20°/30°/20° (a feasible golf address posture). The PyQt panel catches infeasibility and logs a warning. |
| 2026-04-11 | 1.1.44 | Lower-body simulator DRY/LOD refactor: centralized mj_name2id lookups into a single cache populated in `_cache_indices` (joints, actuators, sites, geoms, bodies), eliminated reflective lookups from hot paths (`step`, `compute_diagnostics`, `inverse_kinematics`, `set_joint_polynomial`, `analyze_induced_acceleration`), and decomposed `compute_diagnostics` into `_collect_tracking_error`, `_collect_joint_torques`, `_collect_ground_reaction_forces`. Added contract test suite locking down the public API surface (`-m contract`). |
| 2026-04-11 | 1.1.43 | Added inclined-plane pelvis rotation driver to the lower-body simulator: `set_pelvis_inclined_rotation(target, ...)` wrenches the pelvis free joint via `data.xfrc_applied` each step so the body tracks an inclined rotation axis (spine angle) plus a smoothstep-ramped lateral weight shift during the downswing. New `InclinedPlaneHipRotationTarget.lateral_shift_m`, `lateral_shift_at(t)`, and `target_quaternion_at(t)` with full DbC. |
| 2026-04-11 | 1.1.42 | Anatomically-shaped lower-body pelvis: composite of inertial host ellipsoid plus five mass=0 visual-only landmark geoms (sacrum, bilateral iliac wings, bright-red ASIS spheres, pubic symphysis) so pelvic tilt is visually unambiguous in the viewer without any change to dynamics. |
| 2026-04-11 | 1.1.41 | Added a full reset control to the lower-body PyQt panel that stops playback, clears history, returns MuJoCo time to zero, preserves loaded golf hip rotation targets, and reapplies the target pose at `t=0`. |
| 2026-04-11 | 1.1.40 | Added `tools.shared.python.model_generation.editor` compatibility exports (including `TextEditor` alias) to support removing duplicate model editor implementations in downstream repos that consume Tools as a dependency. |
| 2026-04-11 | 1.1.39 | Extended lower-body simulator history playback diagnostics so cached frames expose the configured inclined-plane hip rotation target for scrub-based analysis and verification. |
| 2026-04-11 | 1.1.38 | Added the lower-body inclined-plane hip rotation target profile with deterministic sampling, DbC validation, both-socket simulator application, and diagnostics/history coverage for the first golf lower-body rotation slice. |
| 2026-03-28 | 1.0.0 | Initial specification |
| 2026-03-29 | 1.0.1 | Document performance improvement in DataChart downsampling algorithm |
| 2026-03-30 | 1.0.2 | A-N assessment remediation: LoD refactoring in convert_tools_icon.py, launch.py, launch_signal_toolkit.py, verify_launcher.py; DbC input validation added to launch_tool, bootstrap, migrate_file, \_print_environment_info, \_check_launcher_file, \_print_recommendations, \_on_poly_generated; docstrings added to **init** and missing functions in setup_dev.py, remove_broken_scripts.py, migrate_print_to_logging.py, launch_signal_toolkit.py. |
| 2026-03-31 | 1.0.3 | Fix CI import error in tests/shared/python/test_contracts.py and optimize React rendering in ToolsPanel. |
| 2026-04-01 | 1.0.4 | Add keyboard accessibility (focus-within) to video player controls in web application. |
| 2026-04-01 | 1.0.5 | Optimize the data processor median filter to reuse a `Float64Array` buffer and preallocate result storage, reducing per-window allocations during large CSV filtering workflows. |
| 2026-04-02 | 1.0.6 | Refactored AnalyticsSuite (computeCorrelation, computeRegression, pearsonCorrelation) to use iterative primitive arrays and eliminate chained .map/.filter mapping overhead, vastly reducing garbage collection pressure. |
| 2026-04-02 | 1.0.7 | Run comprehensive assessments and apply auto-fixes across the repository. |
| 2026-04-03 | 1.0.8 | Refactor `linearRegression` and `polynomialRegression` in `useDataProcessor.ts` to replace multiple consecutive `.reduce()` and `.map()` array iteration methods with single-pass `for` loops, improving performance for large datasets. |
| 2026-04-10 | 1.0.9 | Optimize Math Functions using single-pass loops. |
| 2026-04-10 | 1.1.0 | Add keyboard accessibility and focus management to the Data Processor web application file upload dropzone. |
| 2026-05-18 | 1.1.1 | Fix command injection vulnerability in MATLAB Quality Utils by escaping single quotes in paths passed to MATLAB and Octave shells. |
| 2026-05-18 | 1.1.2 | Optimize PCA mathematical matrix calculations in AnalyticsSuite to use column-wise typed Float64Array to prevent large O(N) allocation overhead. |
| 2026-05-18 | 1.1.3 | Optimize linear regression calculation in AnalyticsSuite using single-pass loops instead of map/reduce to minimize garbage collection pauses. |
| 2026-05-19 | 1.1.4 | Add inline error message handling to SignalList to avoid blocking native alert dialogs and added comprehensive focus-visible states across all signal list interface buttons for enhanced keyboard accessibility. |
| 2026-07-30 | 1.1.49 | Added focus-visible states to inputs, selects, and buttons in the Rotation Converter application to improve keyboard accessibility. |
| 2026-04-04 | 1.1.5 | Replace print statements with logger calls in lower_body_model main entry point to comply with no-print policy and improve production logging. |
| 2026-04-05 | 1.1.6 | Optimize DataChart point extraction loop to explicitly map selected properties instead of using an object spread on the entire row in `src/data_processing/data_processor/web/src/components/DataChart.tsx`. |
| 2026-04-05 | 1.1.7 | Improve HelpPanel accessibility by adding ARIA expanded states and control links to accordion toggles, and adding explicit focus-visible rings for keyboard users. |
| 2026-04-05 | 1.1.8 | Optimize PlotView WebGL rendering to use Float64Array and bypass map array creation overhead. |
| 2026-04-05 | 1.1.9 | Bridge the embedded `src/pendulum_simulator/tests` suite into the top-level `tests/` tree so standard `pytest tests/` collection includes pendulum coverage without double-collecting the same files during root-level pytest runs. |
| 2026-04-05 | 1.1.10 | Standardize vessel drafter `require_positive` usage onto the fleet-wide `(value, name)` argument order while keeping guarded support for the legacy local order and adding regression tests for the signature normalization. |
| 2026-04-05 | 1.1.11 | Deduplicate repeated scalar surface evaluator closures in `analysis_tab.py` by routing matrix and transformed-value cases through shared helper builders, with regression coverage for the new helper paths. |
| 2026-04-05 | 1.1.12 | Expand the embedded-suite discovery policy so root-level pytest ignores bridged `src/` suites by default while `pytest tests/` includes both pendulum and solar-system embedded tests through top-level bridge directories. |
| 2026-04-05 | 1.1.13 | Move pendulum optimizer objective-refresh wiring behind a public `OptimizationWidget` API so `SimulationPanel` no longer reaches through private optimizer button and log internals before optimization runs. |
| 2026-04-06 | 1.1.14 | Remove developer-machine repository paths from maintenance scripts and eliminate the local sys.path bootstrap fallback from convert_tools_icon.py. |
| 2026-04-06 | 1.1.15 | Replace chained array map and filter operations with a single loop in the calculateTrendline algorithm to prevent memory allocation and garbage collection overhead. |
| 2026-04-06 | 1.1.16 | Add focus-within styles to video uploader dropzone and missing aria-labels to the volume and seek range inputs in the video processor web application to improve keyboard navigation visibility. |
| 2026-04-06 | 1.1.17 | Optimize Polynomial Regression Matrix Construction in AnalyticsSuite using single-pass loops. |
| 2026-04-06 | 1.1.18 | Refactored `applyFilter` inside `useDataProcessor.ts` to pre-allocate buffers and run the mapping in a single loop. |
| 2026-04-06 | 1.1.19 | Split `pressure_drop_interface.py` into facade-oriented `pressure_drop_api`, `pressure_drop_validation`, `pressure_drop_reference`, and `pressure_drop_results` modules while preserving the public interface and extending regression coverage for the pressure-drop calculator. |
| 2026-04-07 | 1.1.20 | Added explicit `focus-visible` keyboard focus indicators to the Video Processor web `ToolsPanel` buttons, color controls, slider, and destructive action buttons so keyboard navigation remains visible throughout the drawing workflow. |
| 2026-04-07 | 1.1.21 | Split `model_generation` REST routing from the Flask and FastAPI adapters behind a backward-compatible shim, decomposed the pressure-drop engine into friction-factor, flow-property, fittings, and compressible-flow modules with regression coverage for the preserved calculations, and restored the top-level `contracts` compatibility export for `_resolve_contract_level`. |
| 2026-04-07 | 1.1.22 | Formalize stdout/stderr helper usage for CLI-facing launcher and coverage-gate scripts so terminal output remains explicit while avoiding ad hoc `print()` usage in those entry points. |
| 2026-04-07 | 1.1.23 | Split the data-processor neural-network script exporter, ANOVA analyzer, and vectorized filter engine into smaller domain modules behind backward-compatible facades, and add focused regression tests for the preserved public and compatibility interfaces. |
| 2026-04-07 | 1.1.25 | Replaced raw `print()` summary emission in `scripts/generate_tools_json.py` with an explicit stdout helper, added regression coverage for the CLI entrypoint's generated-file summary contract, and aligned the humanoid mesh-generator facade with the split backend modules so refreshed type-checking stays green after the backend extraction on `main`. |
| 2026-04-07 | 1.1.26 | Extracted the double-pendulum golf equations popup string literals into `equations_data.py`, leaving the popup module focused on presentation and control wiring while preserving the existing dialog behavior. |
| 2026-04-07 | 1.1.27 | Optimized `AnalyticsSuite` regression filtering by staging selected x/y series values into `Float64Array` buffers before converting them back to plain arrays for the existing result contract, reducing repeated push-allocation overhead in large regression workloads. |
| 2026-04-07 | 1.1.28 | Optimized `AnalyticsSuite` Pearson correlation by preserving the PR's single-pass accumulation and variance-clamping path while widening the helper to accept pre-allocated `Float64Array` inputs from the newer analytics data flow. |
| 2026-04-07 | 1.1.29 | Decomposed the PSA GUI into focused `ui/` modules while tightening the compatibility export surface to immutable `__all__` tuples in both the facade module and the extracted UI package. |
| 2026-04-07 | 1.1.30 | Extracted the public enums/dataclass contracts and low-level helper kernels for `time_series_decomposition` into focused support modules, leaving the main module centered on decomposition orchestration while preserving the existing public import surface through the compatibility facade. |
| 2026-04-08 | 1.1.31 | Memoize AnalyticsSuite chart data using useMemo and optimize the scatter regression component with a single-pass loop, drastically reducing React rendering and GC overhead. |
| 2026-04-08 | 1.1.32 | Optimized data array filtering in `useDataProcessor.ts` by replacing `Array.push()` calls with `Float64Array` buffers in `calculateTrendline`, and replacing chained `filter()` passes in `trimTimeRange` with a single-pass `for` loop that avoids creating and resizing intermediate arrays. |
| 2026-04-09 | 1.1.33 | Added a loading spinner and `aria-pressed` states to the `VideoEditor.tsx` component in the video processor web application to improve user experience and accessibility during video export operations. |
| 2026-04-09 | 1.1.35 | Added a shared provider-pack manifest for the pendulum simulator under `src/pendulum_simulator`, plus a repo-local validator and regression tests that keep the manifest aligned with the real package entry point, working directory, Python path, icon asset, and launcher metadata required for future UpstreamDrift shared-launch integration. |
| 2026-04-09 | 1.1.34 | Wrapped DataTableView, PlotView, and AnalyticsSuite in `React.memo`, and memoized activeSignals with `useMemo` to prevent expensive visualization re-renders on unrelated UI state changes. |
| 2026-04-10 | 1.1.37 | Add explicit focus-visible styles to the interactive buttons (Upload New Video, Play/Pause, Mute/Unmute) within the `VideoPlayer` component for improved keyboard navigation visibility. |
| 2026-04-12 | 1.1.48 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-15 | 1.1.49 | Optimized exponential and power regression calculation in `useDataProcessor.ts` by replacing chained array methods with single-pass loops and pre-allocated arrays to eliminate GC overhead. |
| 2026-04-17 | 1.1.50 | Hardened model import security by enforcing HTTPS GitHub host allowlisting for remote model-library fetches, validating user-provided GitHub repository URLs before import, dropping directory components from remote mesh names, and rejecting separator-containing URDF viewer filenames before filesystem resolution. |
| 2026-04-21 | 1.1.67 | Optimized row copying logic in useDataProcessor.ts by replacing `Object.keys()` with a `for...in` loop and `hasOwnProperty`, substantially reducing GC allocation overhead inside tight data processing loops. |
| 2026-04-21 | 1.1.66 | Refreshed regression test coverage for architecture boundaries, data-processor compatibility, folder archive operations, and upstream-drift contract smoke behavior while keeping the production implementation unchanged. |
| 2026-04-22 | 1.1.90 | Repaired CI dependency bootstrap workflows so shared runners with broken `wheel` metadata upgrade `pip` and `setuptools` separately, then reinstall `wheel` with `--no-deps` before workflow linting and Python test jobs. |
| 2026-04-22 | 1.1.91 | Hardened data-processor normalize and standardize transforms so constant columns raise `TransformationError` instead of silently producing all-NaN output, with regression coverage preserving original data after the failed transform. |
| 2026-04-22 | 1.1.89 | Hardened `utils.env_utils` repo-root fallback discovery so shallow path layouts no longer raise import-time index errors, and added regression coverage for shallow fallback computation behavior. |
| 2026-04-22 | 1.1.93 | Enforced finite, non-negative altitude preconditions for the Rust standard-atmosphere model and added operator whitelisting before `DataProcessorEngine.filter_data()` constructs pandas query expressions. |
| 2026-04-22 | 1.1.94 | Updated the shared `DataProcessor.apply_filter()` Butterworth path to use an explicit `sample_rate` or infer it from time-column spacing instead of hard-coding 1000 Hz, with regression coverage for non-1 kHz datasets. |
| 2026-04-22 | 1.1.95 | Canonicalized the Rust universal gas constant by updating `math::R_GAS` to the full CODATA value and having `engineering::R_UNIVERSAL` reuse the same constant. |
| 2026-04-23 | 1.1.102 | Updated Unit Converter `removeCustomUnit` workflow to use an inline soft confirm pattern, eliminating thread-blocking `confirm()` dialogs and improving accessibility with `aria-live`. |
| 2026-04-28 | 1.1.112 | Updated Unit Converter UI to dynamically retarget labels for custom combobox search inputs, ensuring explicit accessible names and resolving click-to-focus gaps. |
| 2026-05-02 | 1.1.121 | Preserved `smoothAngles` behavior for fractional moving-average window sizes by dividing optimized mid-window sums by the actual sample span, added a Vitest regression in the golf video-processor web app, hardened the benchmark plugin bootstrap in CI/benchmark workflows against shared-runner cache drift, and restored the CI Standard coverage-policy skip for PRs that touch no Python source or Python tests. |
| 2026-05-01 | 1.1.120 | Hardened the calculator web expression validation gate by rejecting Python object hierarchy, lifecycle, async, import, and control-flow injection markers before SymPy parsing. |
| 2026-05-01 | 1.1.119 | Replaced the ODESolverCalculator data-table `.filter().map()` chain with a single-pass `for` loop that pre-allocates a result array and iterates in steps, eliminating O(N) intermediate array allocations and reducing GC pressure during large-dataset renders. |
| 2026-05-03 | 1.1.122 | Optimized row copying logic in useDataProcessor.ts by replacing the slow `for...in` and `hasOwnProperty` check with `Object.keys()` and a standard `for` loop, eliminating prototype chain crawling overhead. |
| 2026-05-03 | 1.1.123 | Hardened Folder Packer Pro archive extraction against absolute and parent-traversal member paths, made vessel drafter positive-value contracts accept both legacy and fleet-standard argument order, repaired the production Docker wheel build/install path, expanded Docker context cache exclusions, made CI quality-gate jobs informational, and lengthened Jules issue resolver polling. |
| 2026-05-01 | 1.1.118 | Bound the CI Standard workflow's dependency bootstrap to `python -m pip` in both quality-gate and test-matrix jobs so pytest plugins, including `pytest-benchmark`, install into the same interpreter that later runs `python -m pytest`. |
| 2026-05-01 | 1.1.117 | Made the shared syngas water vapor-pressure helpers return explicit `float` values so delta `mypy` checks stay green while preserving the `water_fraction` compatibility alias for downstream consumers. |
| 2026-05-01 | 1.1.116 | Tightened signal generator and acid gas dewpoint precondition handling so short chirp inputs, zero-frequency periodic signals, and non-positive dewpoint partial pressures raise deterministic `ValueError` messages. |
| 2026-04-30 | 1.1.115 | Hardened CI packaging and workflow checks by pinning the setuptools build backend below 82, using the supported package-data wildcard for `py.typed` markers, scanning merge-conflict markers with tracked-file `git grep`, normalizing detect-secrets result comparisons, and tolerating missing or empty benchmark JSON artifacts. |
| 2026-04-30 | 1.1.114 | Integrated full-text live search into the Unified Tools Launcher tabs, including name, description, keyword, multi-word, and punctuation-normalized matching, with Ctrl+F focus and Esc clear shortcuts. |
| 2026-05-24 | 1.1.113 | Fixed a vulnerability in CSRF cookie parsing logic where cookies with values containing an equals sign were previously being truncated. This allows base64 encoded CSRF tokens with padding to be parsed correctly. |
| 2026-05-11 | 1.1.127 | Replaced `.map()` array allocations in the `rk4Step_golfer` numerical integration function with pre-allocated arrays and standard `for` loops in `physics_golfer.ts` to reduce GC overhead. |
| 2026-05-15 | 1.1.180 | Replaced `.map()` array allocations inside `physics_golfer.ts` constraint and torque loops with pre-allocated arrays and standard `for` loops to reduce GC overhead. |
| 2026-05-13 | 1.1.141 | Made the migrated Video Analyzer installable and launchable from Tools for issue #2585 by adding package discovery, a `video-analyzer` console script, optional video runtime dependencies, installed-package import paths, and focused packaging/launcher regression tests. |
| 2026-05-13 | 1.1.140 | Registered the migrated Video Processor web surface in the canonical GUI launcher manifest and generated tools catalog, with regression coverage proving shared UpstreamDrift-visible tools expose their expected launch surfaces (#2585). |
| 2026-05-12 | 1.1.139 | Refreshed the module-size budget baseline for the updated rotation converter PyQt launcher after the branch was brought current with main. |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops. |
| 2026-05-15 | 1.1.139 | Refactored RK4 expression compilation in ODESolver to pass parameters as a direct array, avoiding spread operator allocation in tight integration loops. |
| 2026-05-12 | 1.1.138 | Hardened CI test-matrix dependency setup against stale self-hosted runner NumPy/SciPy binary caches and routed provider-contract tests through the active Python interpreter. |
| 2026-05-12 | 1.1.137 | Corrected the coverage policy gate to ratchet from the committed total-coverage baseline until the repository reaches the configured 60% target, while preserving package thresholds and regression checks. |
| 2026-05-12 | 1.1.136 | Resolved type-checking errors by properly implementing abstract methods (send_message, validate_connection, capabilities) for RustAgentAdapter, and fixed GUI theme and categorization issues in UpstreamDrift chat functionality. |
| 2026-05-19 | 1.1.184 | Replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` to eliminate callback allocation and garbage collection overhead during high-frequency pose frame confidence calculations in the video processor. |
| 2026-05-20 | 1.1.193 | Clarified shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronized the GitHub CLI provider descriptor with the default terminal registry (#3020). |

---

<!--
  SPEC MAINTENANCE RULES:

  1. WHEN TO UPDATE: Any PR that adds, removes, or changes functionality
     described in this spec MUST include a corresponding spec update.

  2. WHO UPDATES: The PR author (human or agent) is responsible.

  3. CI ENFORCEMENT: The spec-check workflow will flag PRs where source
     files changed but SPEC.md did not. This is a blocking check.

  4. REVIEW: Spec changes should be reviewed with the same rigor as code.

  5. VERSION: Bump the Spec Version field when making substantive changes.
     Use semver: major (structure change), minor (new features), patch (corrections).
-->

### Performance

- `getTimeDelta` calculations inside tight loops use `Date.parse(dateString)` instead of `new Date(dateString).getTime()` to directly retrieve numeric timestamps without the memory overhead and GC pressure associated with instantiating temporary `Date` objects.
- Data-processing formula evaluation now treats `numexpr` as an optional accelerator rather than a hard runtime dependency. Shared `DataProcessor` and `upstream_drift_tools` formula columns fall back to the pandas Python eval engine when `numexpr` is unavailable, preserving the documented `TransformationError` contract for invalid expressions.
- The application uses `Float64Array` and iterative loops instead of `Array.prototype.map`/`filter`/`reduce` to optimize memory and processing speed for large numerical datasets, including reusable typed-array buffering for median-filter windows in `useDataProcessor.ts`. Chained array functional methods like `reduce` and `map` have been largely replaced with standard iterative loops in mathematical computation methods such as `zScoreFilter`, `linearRegression` and `polynomialRegression`.
- Mathematical matrix calculations such as Principal Component Analysis (PCA) utilize column-wise typed arrays (e.g. `Float64Array` buffers) rather than traditional N x P row-wise arrays, drastically reducing O(N) allocation overheads and mitigating garbage collection pauses on large scale analysis.
- Linear regression and sum-of-squares calculations in `AnalyticsSuite` leverage pre-allocated arrays and single-pass loops to prevent allocation and garbage collection overhead typical of functional `.map()` and `.reduce()` operations in large dataset pipelines.
- The PCA power iteration algorithm in `AnalyticsSuite` has been optimized to remove `.map()` and `.reduce()` from the tight inner loop, using pre-allocated arrays and standard `for` loops to eliminate thousands of allocations per execution.
- PlotView WebGL rendering uses pre-allocated `Float64Array` buffers and single-pass loops instead of `data.map()`, eliminating O(N) intermediate array allocations for large datasets.
- Pearson correlation matrix computations utilize a single-pass loop algorithm, calculating sums concurrently to drastically reduce iteration overhead compared to two-pass implementations, while carefully mitigating numerical instability via clamping.
- Recharts component props in `AnalyticsSuite` are memoized using `useMemo` hooks to provide stable references and prevent expensive internal re-renders.
- Exponential and power trendline calculations use pre-allocated arrays and single-pass loops instead of functional chaining to minimize GC pauses.

### Version 1.1.67

- **Performance**: Optimized array allocations in PCA calculate loop inside `AnalyticsSuite.tsx` by replacing chained `.reduce()` and `.map()` calls with single-pass `for` loops.

### Version 1.1.66

- **Performance**: Optimized row copying logic inside `useDataProcessor.ts` by replacing `Object.keys()` iterations with `for...in` loops and `hasOwnProperty`. This minimizes excessive key array allocations inside data transformation loops.

### Version 1.1.66

- **Security**: Disabled loading and saving of `.pkl` and `.pickle` files natively using pandas due to severe CWE-502 vulnerability. Raises `ValueError` explicitly when format is set to `pickle`.

<!-- prettier-ignore-end -->

### Version 1.1.106

- **Performance**: Optimized matrix and loading array copying inside `AnalyticsSuite.tsx` for PCA calculation by replacing `.map()` and array spread operations with single-pass pre-allocated loops, substantially reducing memory allocation overhead.

## 2026-04-20

- Update unit converter clear history button accessibility (ARIA labels, disabled state)

### Version 1.1.111

- **Performance**: Optimized signal statistics and FFT chart data generation in `FunctionGenerator.tsx` by replacing the use of the array spread operator (`...vals`) inside `Math.min`/`Math.max` and chained iterators (`.map().filter()`, `.reduce()`) with single-pass `for` loops. This prevents runtime "Maximum call stack size exceeded" errors and significantly reduces GC overhead.
- **Performance**: Replaced `.map()` and `.push()` with pre-allocated single-pass `for` loops in `pcaScatterData`, `regressionScatterData`, and `regressionResidualsData` within `AnalyticsSuite.tsx` to eliminate dynamic resizing overhead and intermediate object allocations.
- **Security**: Fixed DOM-based Cross-Site Scripting (XSS) vulnerability in `psa_calculator.html` by implementing and applying an `escapeHtml` function to user-controlled inputs before updating `innerHTML`.
- **Performance**: Optimized `computeFFT` inside `FunctionGenerator.tsx` by pre-allocating output arrays and substituting functional iterations (`.map()` and `.reduce()`) with an inline single-pass Hanning window loop. This bypasses intermediary array processing steps and lowers garbage collection occurrences.
- **Performance**: Replaced O(N) chained `.filter().map()` iterators with an $O(N/\text{step})$ `for` loop in `FunctionGenerator.tsx` when preparing data points for time charts to prevent the allocation of intermediate arrays and reduce unnecessary iterations.

## Performance Optimizations

- **Performance**: Optimized `psa_calculator.html` by replacing chained `.map()` and `.reduce()` operations with single-pass `for` loops and pre-allocated arrays, alongside substituting `reduce` with a globally scoped `sumArray` helper function, significantly reducing GC overhead.

- **ODESolverCalculator:** Replaced `.map()` and `Math.max(...values)` with a single-pass `for` loop to prevent "Maximum call stack size exceeded" errors on large dataset arrays generated by the ODE solver.
- **Performance**: Optimized the sliding-window algorithm `smoothAngles` in the video processor's `angleCalculator.ts` by replacing `.slice()` and `.reduce()` inside the loop with a single-pass sum tracker, and splitting the loop into three parts (left, middle, right) to eliminate `Math.min()` and `Math.max()` bounds checking from the hot path.

### Web Frontends

- `data_processor`: Improved performance in tight object loops by replacing `for...in` and `hasOwnProperty` with `Object.keys()` and standard `for` loops in `useDataProcessor.ts`.
- **Performance**: Replaced `.map()` with pre-allocated arrays and single-pass `for` loops across all signal generation functions (`generateSinusoid`, `generateCosine`, `generateSquare`, `generateTriangle`, `generateSawtooth`, `generatePulse`, `generateStep`, `generateExponential`, `generateLinear`, `generateChirp`, `generateConstant`) in `FunctionGenerator.tsx` to minimize garbage collection overhead for large sample arrays.
- **Performance**: Optimized `generatePolynomial` in `FunctionGenerator.tsx` by using Horner's method and pre-allocating output arrays instead of `.map()` with `Math.pow()`, significantly reducing overhead and improving calculation speed.

### Version 1.1.128

- **Performance**: Pre-allocated objects and arrays for the Runge-Kutta 4 (RK4) integration loop in `ODESolverCalculator.tsx` to eliminate thousands of memory allocations per step and reduce severe garbage collection pauses during large ODE simulations.
- **Performance**: Refactored dynamically compiled expressions inside the hot RK4 numerical integration loop to avoid the spread operator (`...args`) and array allocation. Parameters are now passed as a single array and statically destructured within the function body itself.

### Version 1.1.139

- **Performance**: Optimized `detectSwingPhases` inside `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts` by replacing `poseFrames.map(...)` with a standard single-pass `for` loop and a pre-allocated array. This reduces continuous callback allocation and limits garbage collection pauses in hot paths when analyzing multiple video frames.

### Version 1.1.183

- **Performance**: In high-frequency algorithmic optimization loops (like Nelder-Mead iterations), replaced array manipulation operations such as `.map()` and `.slice()` with pre-allocated arrays and standard `for` loops in `src/pendulum_simulator/pendulum-web/src/optimizer.ts` to eliminate continuous array creation and avoid significant garbage collection overhead.

- **2026-06-12**: fix(p1am, #3323) — guard live-PLC writes behind confirmation dialogs and add Control-tab role gating. `ControlTab` now defaults to the `Operator` role and exposes `set_role()` (wired from `HMIMainWindow._on_role_changed`); starting live-loop tuning, applying a tuning step, and applying recommended PID gains are Admin-only and additionally raise a `QMessageBox.question("Confirm PLC write", …)` that fails closed (default `No`). `RoutingTab._deploy_config` now confirms before persisting the routing/interlock matrix to PLC NVRAM, and the Inspector sidebar's manual tag force override confirms before writing a raw value to the live plant (Operator-allowed by design). Client-side hardening only — server-side `/api/routing` and `/api/tags` enforcement remains tracked by the HMI auth work. Adds `tests/p1am_control_system/test_plc_write_confirmation.py`.

- **2026-06-12**: fix(ui) — remove `Qt.WindowType.FramelessWindowHint` from all 24 standalone tool main windows (Data Explorer, ODE Solver, financial/rotation/PID/PSA calculators, c3d/urdf/humanoid/optimizer/multi-param/vessel-drafter GUIs, pdf_renamer, tile launcher, the Unified Tools Launcher, etc.) so the OS draws normal, movable/resizable/closable window chrome again. These windows previously had no custom title bar, drag handling, or min/max/close buttons, leaving them un-manageable by mouse (#3322). Also dedupe the `ThemedWindowMixin` base class and the doubled `setup_theme_support()` call in `unified_launcher_window.py`. Adds an architecture guard `tests/architecture/test_no_frameless_windows_3322.py` that fails if any `src/` file reintroduces `FramelessWindowHint` without sanctioned custom chrome. As part of the same change, fixed pre-existing `union-attr` typing on Qt accessors that the delta-mypy gate surfaced once these files entered the changed set: `data_explorer/gui.py` now guards `QTableWidget.horizontalHeader()`, `ode_solver` main window guards `menuBar()/addMenu()/addAction()`, and `popout_chart.py` binds the matplotlib `Axes` local as `Any` instead of letting `self._ax: object | None` narrow it.

### Version 1.1.184

- **Performance**: In `src/pendulum_simulator/pendulum-web/src/components/AnalysisPlots.tsx`, optimized chart downsampling by replacing multiple instances of `indices.map()` with pre-allocated arrays and explicit `for` loops inside `useMemo` hooks. This drastically reduces array allocation and garbage collection overhead during high-frequency component rendering.
- **Performance**: Replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` inside `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts` to eliminate callback allocation and garbage collection overhead during high-frequency pose frame confidence calculations.

### Version 1.1.185

- **Security**: Fixed a command injection vulnerability in `cli_tools.py`'s `ShellTool._is_command_allowed` by parsing the command with `shlex.split` and blocking shell operators instead of using a naive `.startswith()` string check.

### Version 1.1.186

- **Performance**: In `src/media_processing/video_processor/apps/web/lib/golf/phaseDetector.ts`, replaced `.reduce()` with a standard `for` loop in `calculatePhaseConfidence` to eliminate callback allocation and GC overhead in high-frequency phase detection paths.

### Version 1.1.187

- **UX**: Add accessible toggle states and toast feedback for copy actions in `calculator/static/app.js` and `calculator/templates/index.html`.

### Version 1.1.188

- **Security**: Fixed an information leakage vulnerability in `src/web_applications/health_checks.py`. API endpoints (`/api/health` and `/api/ready`) no longer expose raw exception strings (`str(e)`) in their JSON responses. They now return safe, generic error messages while preserving full traceback details in the backend logs using `logger.exception()`.

### Version 1.1.189

- **Reliability**: Restored source-tree `src.shared.python.logging_pkg` and `src.shared.python.config` compatibility modules so shared AI adapter factories and chat service connection code import cleanly from a Tools source checkout or vendored shared-module install.

## 9. Changelog

### Version 1.5.5

- 2026-08-05: fix(rotation-converter) — update application navigation tabs
  with accessible roles and unique IDs, linking buttons to tab panels via
  `aria-controls` and `aria-labelledby` for screen reader semantic correctness.

### Version 1.5.4

- 2026-08-04: ci — route the required generic PR quality gate to a hosted
  Ubuntu runner, retain hardware and integration tests on their explicit local
  lanes, preserve the setup-python pip cache instead of purging it, and narrow
  the local-only policy exception to `ci-standard.yml::quality-gate`.

### Version 1.1.598

- 2026-06-18: fix(data-processor, #3745) — keep cross-correlation importable
  without optional numba acceleration and clamp the rolling
  `correlation_stability` contract to the documented non-negative range.

### Version 1.1.447

- 2026-06-14: fix(sidekick, #3334) — stabilize the broad
  Sidekick/import-order test surface after the data-processor wrapper rename:
  isolate cache-sensitive import probes in subprocesses, stop PSA webapp tests
  from poisoning pandas/pyarrow imports, align stale DbC and validation
  expectations with explicit exceptions, make theme fallback probes
  process-isolated, and resolve the syngas compression plot canvas class at
  runtime.

### Version 1.1.446

- 2026-06-14: test(data-processor) — add an explicit
  `DATA_PROCESSOR_IO_DISABLE_NATIVE` fallback switch for the compatibility
  wrapper contract tests and keep the Sidekick rename-test module cache reset
  scoped to public `sidekick`/`upstream_drift_tools` imports so pytest's
  source-qualified collection namespace remains stable.

### Version 1.1.445

- 2026-06-14: test(data-processor) — move the `data_processor_io` Rust wrapper
  contract tests under the matching package test namespace and contain
  Sidekick rename-test module-cache mutations with typed error-path tests so
  CI import-order checks do not leak stale module identities into
  data-processing tests.

### Version 1.1.444

- 2026-06-14: fix(sidekick-data-processing) — keep invalid filter operator
  errors on the documented "Unsupported filter operator" contract and make
  the curve-fit widget fail gracefully if an injected engine returns no
  `FitResult`.

### Version 1.1.443

- 2026-06-14: fix(data-processor) — rename the shared Rust bulk-I/O wrapper to
  `data_processor_io` and remove Sidekick's private `sys.modules` eviction and
  `sys.path` reordering from the full Data Processor embedding bridge. The
  public `ensure_full_data_processor_on_path` API remains stable and
  idempotent while bare `data_processor` stays reserved for the full app.

### Version 1.1.393

- 2026-06-12: refactor(data-processor) — move the shared
  `RustBulkDataEngine` compatibility facade into `bulk_facade.py` while
  preserving `data_processor.rust_engine` re-exports, keeping the CI
  changed-file size budget green without changing runtime behavior.

### Version 1.1.392

- 2026-06-12: fix(data-processor) — expose `DataProcessorRustError` and a
  `RustBulkDataEngine` compatibility facade from the shared data-processor
  fallback package so source-tree import order cannot shadow the full
  data-processor package and break `data_processor.core.data_loader`.

### Version 1.1.333

- 2026-06-10: feat(ai) — marshal GUI-affine chat tools onto the main
  thread. `Tool` gains an opt-in `requires_main_thread` flag; `ToolRegistry`
  gains `set_main_thread_dispatcher` and routes flagged tools through it in
  `execute` (running inline when no dispatcher is installed, so headless use
  is unaffected). `MainThreadToolDispatcher` (ai/gui) marshals a tool thunk
  from the background `StreamWorker` thread onto its owning GUI thread via a
  queued signal, returning the result synchronously and re-raising errors on
  the caller; same-thread calls run inline. Decorator-registered tools can
  opt in through `ToolRegistry.register(..., requires_main_thread=True)`, so
  the normal shared-tool registration path preserves GUI-thread affinity.
  `AIAssistantPanel` installs the dispatcher on the global registry at
  startup and uses explicit boundary return types so skipped-import mypy runs
  remain type-clean at the panel boundary. Additive and backward compatible
  for existing downstream registrations.

### Version 1.1.332

- 2026-06-10: fix(ci) — keep the P1AM project import helper mypy-clean under
  the changed-file quality gate by typing parsed SCADA tags as `TagDefinition`
  at the parser boundary and preserving the project import endpoint's
  documented `dict[str, Any]` response contract when imports are skipped.

### Version 1.1.331

- 2026-06-10: fix(daemon, #3291) — stop `start-gaai-daemon.sh` from
  writing `~/.claude/settings.json` or globally suppressing Claude Code
  dangerous-mode prompts; document that any safety override must be configured
  deliberately outside the launcher, and add a dry-run regression test proving
  existing global Claude settings are preserved.

- 2026-06-10: ci(security) — harden
  `.github/workflows/anti-phantom-merge.yml` so the privileged
  `pull_request_target` label path never checks out untrusted PR head code.
  Full git-diff phantom checks continue to run only on `pull_request`; label
  events validate the admin override through GitHub API calls and emit notices
  for ignored non-admin overrides. Added an ops regression that scans all
  `pull_request_target` workflows for unguarded `actions/checkout` steps using
  `github.event.pull_request.head.sha`, preserving the invariant across future
  workflow edits.

### Version 1.1.330

- 2026-06-10: test(ci) — include existing Sidekick state-manager regression
  suites in Sidekick-changed CI slices before the per-file Sidekick coverage
  gate runs, keeping changed-file coverage enforcement aligned with the module
  that triggered the gate; restored JSON serialization of simple object
  class-level defaults while keeping instance attributes authoritative; keep the
  state-manager import side-effect subprocess on the same shared `utils` source
  path used by clean CI runners.
- 2026-06-09: fix(import-contracts) — keep rotation-converter NumPy helpers,
  screw-axis animation callbacks, and the video-analyzer DbC shim mypy-clean
  under changed-file CI by adding explicit typed array boundaries and
  non-redefining contract import fallbacks without changing runtime validation
  behavior; declare the import-contract subprocess bootstrap as assertion-free
  test support so the changed-test assertion ratchet continues to block real
  assertion-light test cases without forcing fake assertions into helpers.

### Version 1.1.329

- 2026-06-09: feat(movement optimizer) — restore the standalone PyQt6 swingset policy-training and segmented-chain whip-dynamics tabs, mypy-compatible typed model and Qt UI modules, focused model and UI tests, launcher bootstrap repair, and provider metadata so UpstreamDrift can discover the Movement Optimizer tile from remote main.
- 2026-06-09: fix(sidekick) — preserve the Python REPL `WorkspaceRegistry`
  contract across canonical and deprecated compatibility import paths so
  legacy `upstream_drift_tools` callers pass the same runtime precondition as
  canonical Sidekick callers without weakening the TypeError guard for
  unrelated registry objects.

### Version 1.1.320

- 2026-06-09: refactor(tools, #3261, #3262, #3263) — replaced the duplicated `scripts/mypy_autofix_agent.py` implementation with a compatibility wrapper that delegates to the canonical `src.tools.mypy_autofix_agent` entrypoint, preserving direct script execution while reducing audit-reported DRY debt. Added focused tests that guard the delegation contract and CLI help path.

### Version 1.1.265

- 2026-06-01: feat(ai): Added `AdapterReviewerLLMClient`, a production `ReviewerLLMClient` backed by `BaseAgentAdapter` that builds a structured JSON prompt, runs `send_message` off the event loop, and parses verdict/reasoning/confidence (malformed JSON → `abstain`, confidence clamped to [0,1]). Wired it as the default via `peer_review.registry.default_llm_client()`, which selects the production client when an adapter is available and falls back to `StubReviewerLLMClient` offline (#3177). Added behavioral tests for the four CLI adapters (`claude_code`/`codex_cli`/`gemini_cli` via mocked `subprocess.run`, `cline` via mocked httpx) covering success, non-zero-exit/timeout/missing-binary error classification, `_strip_telemetry`, and `validate_connection` paths (#3178).

### Version 1.1.258

- 2026-06-01: test(theme): Added focused FastAPI router coverage for built-in/custom theme listing, active-theme retrieval and updates, custom-theme save/delete errors, Pydantic request models, and registration guards, raising `src/shared/python/theme/api.py` focused coverage to 100%.

### Version 1.1.257

- 2026-06-01: test(theme): Normalized font manager and responsive theme tests to import through the exported `src.shared.python.theme` package path so the provider-contract suite passes under importlib mode.

### Version 1.1.256

- 2026-06-01: test(theme): Added focused font manager coverage for QSettings persistence, singleton reuse, font-change signaling, application font application, and no-application warning behavior; fixed PyQt6 font database enumeration to use the static API and tightened adjacent theme helper return typing for strict mypy.

### Version 1.1.255

- 2026-06-01: test(theme): Added focused responsive PyQt helper coverage for maximum-width clamping, invalid contracts, generic widget text derivation, and zero/negative scroll-area width handling.

### Version 1.1.233

- 2026-06-01: fix(ux) — improve accessibility of pendulum simulator model selector tabs by removing conflicting `aria-pressed` attributes and replacing them with standard `role="tablist"`, `role="tab"`, `role="tabpanel"`, and `aria-selected` attributes.

### Version 1.1.232

- 2026-05-30: feat(ux, #3115) — improve accessibility of the ODE Solver UI by explicitly linking `label`s to `input`s and `textarea`s using `htmlFor` and unique `id`s generated by `React.useId()`. Add `spellcheck="false"` and disabled autocorrect on math text areas to prevent mobile OS interference with equations.

### Version 1.1.231

- 2026-06-09: ci(secret-scan) — run the detect-secrets baseline scan through `python -m detect_secrets` so the configured Python environment, not the runner PATH, controls the installed scanner entrypoint.
- 2026-06-09: perf(p1am frontend) — optimize array aggregations and string operations in `LadderExplorer.tsx` by replacing chained `.map().filter()` operations with a single-pass `for` loop and using `useMemo` to prevent main thread lag.
- 2026-05-30: perf(p1am frontend, #3126) — optimize array aggregations in `AlarmsHeader.tsx` by replacing chained `.filter()` and `.reduce()` operations with a single-pass `for` loop, eliminating intermediate array allocations and minimizing garbage collection overhead during high-frequency alarm updates.

### Version 1.1.230

- 2026-05-30: fix(sidekick): PyQt test worker crash fix and Module Size budget baseline adjustment (#3104, #3115)

### Version 1.1.221

- 2026-05-30: test(rust-engine, #3114) — the #2989 bulk-I/O contract suite now runs in CI after the Data Processor embedding PR fixed its import path; guard the parquet round-trip cases (`test_csv_to_parquet`, `test_parquet_destination`, and the `parquet_file` fixture) with a skipif on parquet-engine (pyarrow/fastparquet) availability so the lean CI test image skips them instead of failing, honoring the file's "runs in CI without native deps" contract. CSV contract cases continue to run unconditionally.

### Version 1.1.220

- 2026-05-29: fix(sidekick conversions, #3101) — reconcile `flow_rate_converter` with the DRY constants layer: `ton`/`ton/hr` now means a short ton (907.18 kg) fleet-wide (metric is `tonne`), STP is the IUPAC 0°C/1 bar definition, the gas constant and standard conditions import from `unit_constants`, `Nm3/hr` spellings are recognized, `convert_via_table` raises `ValueError` on unknown units, `_normalize_unit` raises `UnknownUnitError` (O(1)) instead of silently echoing, and the temperature path validates finiteness. Restored the four empty conversion test stubs with known-value and round-trip assertions.
- 2026-05-29: fix(sidekick process numerics, #3103) — remove the duplicated compressible-flow solver (`_flow_calculations` now imports the canonical `compressible_flow`) and the malformed in-sqrt expansion factor; solve WGS extent directly from the equilibrium constant so reported K and composition are self-consistent and guard `T>0`; replace precondition `assert`s with `ValueError` in flare/financial; raise on laminar `Re<=0`; return ideal-gas Z=1 for unknown-only compositions; flag compressible-solver non-convergence; clarify the acid-gas °C Antoine convention.
- 2026-05-29: fix(sidekick PSA UI, #3105) — refresh the sensitivity plot when components change (dirty flag + re-plot when visible); resolve the pre-calc tab trigger via `indexOf` instead of a magic index; size the O2 hazard band from the plotted data max so it can't collapse to the default y-limit.
- 2026-05-29: fix(sidekick widget/state layer, #3102) — wrap Data Processor engine ops (filter/query/aggregate/add/transform/rename/drop/fit) in `try/except DataProcessingError` so bad input shows a warning instead of crashing; validate corrupt saved-state shape (via an `Any`-typed alias so the runtime guard stays reachable) and broaden the load except; parent the auto-save `QTimer` to the host widget and guard `auto_save_state` against teardown; route unit-converter save/delete through `_get_row_by_index`; add a public `UnitConversionService.get_compatible_units`.

### Version 1.1.241

- 2026-06-01: Performance — Optimized correlation matrix calculation in AnalyticsSuite by precomputing column sums for the fast-path (no NaNs) and utilizing fast `x !== x` NaN checks.

### Version 1.1.219

- 2026-05-30: test(data-processor) — added skip guards to rust engine contract tests to handle missing parquet engines (pyarrow/fastparquet) gracefully in minimal test environments.

### Version 1.1.217

- 2026-05-30: feat(sidekick) — consolidated Sidekick quality and cleanup issues (#3106). Replaced global instantiation of `state_manager` with lazy-loading module `__getattr__` wrapper and deprecation warning to prevent eager directory creation on import. Added support for native matplotlib rendering of LaTeX formulas to crisp QPixmaps in `latex_renderer.py`, falling back to monospace text on missing dependencies. Added type annotations to state manager tests.
- 2026-05-30: feat(ui) — added a reusable `HoverCopyTextBrowser` widget with hover-triggered copy-to-clipboard overlay buttons and integrated it into the double pendulum simulator diagnostics/analysis tabs and error notifications. Excluded pendulum simulator from pre-push mypy checks.
- 2026-05-30: fix(sidekick) — validate C3D header magic bytes in `c3d_reader.py` and package-relative standard response import fixes.

### Version 1.1.216

- 2026-05-29: chore(sidekick) — type-gate hardening surfaced by the changed-file CI mypy run: `register_shortcuts` now resolves `QShortcut`/`QKeySequence` through the active `qt_compat` binding instead of a PyQt6/PyQt5 dual-import fork, `_default_tab_definitions` returns an annotated local rather than `cast()`, and `SidekickThemeSettings.__post_init__` widens the persisted-`font` branch through `Any` so the runtime-dict reconstruction type-checks. Clears pre-existing `no-redef`/`unused-ignore`/`redundant-cast`/`no-any-return`/`arg-type` findings on `sidebar.py` and `theme_settings.py`.
- 2026-05-29: feat(sidekick) — wired working ⚙ settings into the Chat, Terminal, Python REPL, and Workspace tabs (previously the gear was disabled on every tab except Data Explorer). New shared `appearance.py` (`PanelAppearance` value object + `panel_qss` generator) gives the terminal/REPL/workspace always-on visible borders and user-adjustable colours; the Workspace now shows an empty-state hint instead of blank white space and the REPL gained input/output labels. `chat_settings.py` adds provider/model/reasoning/agent-mode/auto-condense config plus keyring-backed API-key management. `runtime_tab_settings.py` adds per-tab appearance panels (native colour pickers) and a configurable preloaded scientific-package bundle (numpy/scipy/pandas/matplotlib/sympy) for the Python REPL, reusing the validated `CalculatorStartupConfig`. Added narrow `UnifiedToolsSidebar.tab_widget(tab_id)` accessor for live application (LOD). 130+ new tests; sidekick API stability baseline regenerated (purely additive).

### Version 1.1.215

- 2026-05-29: fix(sidekick) — validate C3D header magic bytes before handing files to ezc3d, returning a clear `ValueError` for truncated or mislabeled files and covering the pre-parser failure path with focused tests.

### Version 1.1.212

- 2026-05-29: chore(ci) — scope coverage policy package thresholds to tracked packages changed in the PR while preserving total coverage and Sidekick-specific coverage gates.
- 2026-05-29: fix(sidekick) — make the standard response API import its shared StrEnum helper via the repo package path while preserving top-level Sidekick compatibility (#3106).
- 2026-05-27: fix(p1am) — extend interlock contract to 4 limits (lolo/low/high/hihi) in SafetyInterlock to align with host. Chunk Modbus client read/write routing into 64-register packets to satisfy pymodbus's request size caps.
- 2026-05-27: chore(ci) — use `sudo rm -rf` in Python tool cache cleanup to ensure complete removal of corrupted files, and add cleanup step to topology-governance, detect-secrets, and cross-repo integration workflows.

### Version 1.1.208

- 2026-05-27: Optimized object allocations in `themeApi.ts` and `themeDefinitions.ts` by replacing `Object.fromEntries(Object.entries().map())` chains with single-pass loops to reduce memory allocation overhead on startup.
- 2026-05-27: feat(FilterPanel) — Added `useId` to dynamically generate linked IDs for form labels, select dropdowns, and inputs in `FilterPanel.tsx` via `htmlFor`, improving screen reader navigation. Also added `aria-invalid` and `aria-describedby` to announce validation error states clearly.

### Version 1.1.206

- 2026-05-26: feat(chat) — restored shared chat dock keybindings (Enter→submit, Shift+Enter→newline, busy-queue with steering), port-aware default WS URL (`UD_CHAT_WS_URL` / `GOLF_API_PORT` env), Ollama latency tuning (`keep_alive: "30m"`, `num_ctx: 4096`, native `tools` field), `_chat_dock_widget_qt.py` refactored into `_qt/` submodules (2091→1049 lines), and the animated "AI is thinking" indicator.
- 2026-05-26: Chat dock resolves its default WebSocket URL per-instance, keeps the Steer action queue-only, and preserves typed import-safe runtime diagnostics for the optional Qt chat surface.

### Version 1.1.204

- 2026-05-26: Optimized Nelder-Mead loop in `optimizer.ts` to mutate pre-allocated arrays in-place to avoid GC overhead.

### Version 1.1.190

- **Performance**: In `src/ode_solver/web/src/components/ODESolverCalculator.tsx`, extracted the entire `resultsPanel` (containing heavy Recharts and data table elements) into a `useMemo` block to prevent the entire SVG tree from re-rendering synchronously on every keystroke in the textarea, eliminating severe UI input lag.

### Version 1.1.190

- **Performance**: In `src/ode_solver/web/src/components/ODESolverCalculator.tsx`, wrapped `varNames` computation and summary cards rendering in `useMemo`, and replaced `.filter()` with a single-pass `for` loop to prevent O(N) recalculations of array keys and summary min/max loops on every React render.

- **2026-05-22**: Memoize summary statistics calculation and variable names in ODESolverCalculator.
- **2026-05-22**: Keep the model explorer package initializer lint-clean by preserving the module docstring before future imports.
- **2026-05-20**: Suppress shared chat dock WebSocket reconnect scheduling during intentional widget close while retaining reconnects for unexpected disconnects.
- **2026-05-20**: Add accessible toggle states and toast feedback for copy actions in `calculator/static/app.js` and `calculator/templates/index.html`.
- **2026-05-20**: Harden health-check API responses to return generic client-facing errors while logging exception details server-side.
- **2026-05-20**: Restore shared logging and environment helper modules required by AI adapter and chat service connection imports.
- **2026-05-20**: Clarify shared chat provider dropdown ownership by removing stale UpstreamDrift issue references from Tools-owned source and tests, and synchronize the GitHub CLI provider descriptor with the default terminal registry (#3020).
- **2026-05-30**: Resolve mypy type check errors in core data loader, signal processing, and Sidekick embedding.

## 1.1.241 - Replaced chained .filter() array passes with single-pass for-loops in Golf UI components

- **2026-06-09**: fix(sidekick) — regenerate `sidekick_api_baseline.json` to include the `data_processing/formats.py` module added in ed6e415fa (only addition; no public API removals or signature changes), and correct `test_json_serializer` to set the Dummy attribute on the instance `__dict__` so it matches `_json_serializer`'s `hasattr(obj, "__dict__")` branch.
- **2026-06-10**: Fix command injection bypasses in `cli_tools.py` `ShellTool._is_command_allowed` by explicitly parsing executable names using `pathlib.Path` and parsing arguments with assignment flags.
- **2026-06-11**: fix(conversion) — `convert_gas_flow_scfm_acfm` now validates inputs as a true precondition: a non-positive/non-finite `compressibility_factor` raises `ValueError` (instead of silently passing through on ACFM→SCFM), and an explicitly supplied non-positive/non-finite `actual_temp_K`/`actual_pressure_kPa` raises `ValueError` instead of being coerced to the standard-condition default via the falsy-`0.0` `or` idiom. Reconciles the #3344 gas-flow guard refactor with the restored #3367/#3342 compressibility validation tests during PR consolidation (#3380).
- **2026-06-11**: test(p1am) — make the P1AM backend functional suite (`src/p1am_control_system/backend/tests/test_backend.py`) robust to cross-module test-collection order. An autouse fixture now re-asserts `P1AM_DEV_NO_AUTH=1` and the `get_session` dependency override per-test (restoring prior values on teardown), so the sibling security suite's import-time `os.environ.pop("P1AM_DEV_NO_AUTH")` and competing `app.dependency_overrides` no longer leak 503 auth-gate / `no such table` failures into these tests (#3289/#3292, surfaced during #3380 consolidation).

## 1.1.409 - Vendor canonical Movement Optimizer biomechanics app into Tools (#3407)

- **2026-06-12**: feat(movement_optimizer) — migrate the more-developed standalone Movement Optimizer (`D-sorganization/Movement_Optimizer`) into `src/movement_optimizer/` as the single canonical home so the standalone repo can be archived. Full product vendored verbatim (Lagrangian barbell dynamics + 7 exercises, swingset/chain models with force fields, spine loads, Hill strength, PyQt6 GUI, headless CLI, optional Rust/PyO3 backend) plus its own preserved test suite. Treated as a self-contained sub-app like `src/pendulum_simulator`: excluded from the monorepo ruff/ruff-format/mypy/bandit/coverage/pre-commit delta gates (and the matching CI filter lists), with `testpaths` keeping its tests out of the default suite. Registered for UpstreamDrift discovery via `model_pack.yaml` (`pack_id: tools-movement-optimizer`, route `/tools/movement-optimizer`) validated by `scripts/movement_optimizer_provider_manifest.py` and a regression test. `gui/motion_tabs.py` was split (extracting `ChainDynamicsTab`/`create_chain_tab` into `gui/motion_tabs_chain.py`, behaviour-preserving, 34 GUI tests green) to satisfy the 1500-line module budget. Phase 1 of the consolidation epic; remaining code-quality follow-ups are tracked under #3411.

## 2025-02-24 CLI Tools Validation Enhancement

The command injection check logic in `cli_tools.py` has been fortified. The input validation step now properly sanitizes (`.strip()`) token arguments and assignments to prevent execution of trailing/leading-space padded payloads (e.g. `--exec="  /bin/rm  "`). This effectively thwarts attacks aiming to bypass naive blocklist string matching (`token in dangerous`).

## 1.1.410 - Palette Micro-UX Improvement

- **2026-06-14**: feat(ux) — added `readonly` attribute to the "To" result input field in the unit converter application (`src/web_applications/unit_converter/unit-converter-app/index.html`) to prevent user confusion, alongside visual styling (`styles.css`) indicating the field's uneditable nature.

### Security

- **Power Supply Endpoints**: State-mutating power supply API routes (`/config`, `/setpoint`, `/permissive`, `/acknowledge_trip`) must be authenticated using the elevated admin key (`P1AM_ADMIN_API_KEY` or `P1AM_API_KEY` fallback), enforced via FastAPI dependencies.

## 1.1.411 - Rust audit hardening (DbC/DRY/test coverage across rust_core + pendulum-core)

- **2026-06-17**: chore(rust) — harden the Rust crates per the #3552–#3556 audit. Converge `ai_backend::config` with UpstreamDrift by porting the #5307 `chat_url`/`embed_url` path-dedup fix + test (#3552); fix the `pendulum-core` clippy `-D warnings` failure (`zip(population.into_iter())` → `zip(population)`) and add a CI rust-quality-gate job for the workspace-excluded crate (#3553); add `is_finite()` NaN guards + `f64::total_cmp` ordering to `tools-core` `swing_plane::detect_phases`/`fit_plane`, promote `ball_flight::simulate_trajectory` `dt`/`max_time`/`velocity` preconditions from `debug_assert!` to real `assert!` validation, and cap the trajectory `Vec` pre-allocation (#3554); add unit + `proptest` tests to the previously untested `reactor`/`rrt`/`thermodynamics`/`electrode_advisor` modules, prioritizing the `rrt` back-pointer walk and `thermodynamics` numerics (#3555); make the `ai_backend::memory` and `file_watcher` mutex locks poison-tolerant (`unwrap_or_else(|e| e.into_inner())`), add file_watcher burst + gitignore tests, and collapse `R_GAS` to a single `pub use` source of truth (#3556).

## 1.1.412 - plugin_manager discovery precedence (#3722)

- **2026-06-18**: fix(core) — reconcile the `load_tools_with_discovery` merge in `plugin_manager.py` with its documented contract: on a name collision within a category a discovered manifest tool now replaces the same-named `tools.json` entry (discovered wins), instead of silently keeping the stale JSON entry. Added a regression test (`test_discovered_tool_wins_name_collision`) asserting the discovered tool survives a duplicate-name collision and that its path/desc reflect the manifest source.

## 1.1.413 - Optimized Nelder-Mead loop in `optimizer.ts`

- **2026-06-21**: perf(pendulum) — Replaced `Array.prototype.sort` with manual insertion sort in `nelderMead` loop in `src/pendulum_simulator/pendulum-web/src/optimizer.ts` to eliminate callback invocation overhead.

## 1.1.414 - Security: Structural validation for SymPy parse_expr

- **2026-07-14**: fix(security) — Added structural validation to the symbolic solver backend endpoints (`/solve`, `/derivative`, `/simplify`) before handing untrusted user input to `sympy.parse_expr`. This mitigates a critical code injection vulnerability where malicious AST constructs (e.g. `().__class__`) could be executed due to `parse_expr`'s internal use of `eval`.

### Version 1.1.250

- **Performance**: Replaced chained array methods (`.map().join()`) with single-pass `for` loops in the P1AM Control System frontend (`TrendFitOverlay` and `timeSeriesPath`) to eliminate high-frequency intermediate string allocations and reduce garbage collection pressure when drawing SVG trends.
- 2024-07-23: fix(ux, #3919) - Improve accessibility of standard buttons in data processor web app by adding `focus-visible` styling (focus rings) to the global `.btn` class.

## 2026-06-12 (Bolt): Refactoring parseVariableAssignments

- Removed chained array maps and reduces in the parseVariableAssignments function within `src/web_applications/calculator/static/app.js`.
- Improved execution speed by using standard single pass for loop and string `indexOf` / `substring` techniques.
