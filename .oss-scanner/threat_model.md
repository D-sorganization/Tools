<!-- Threat model for D-sorganization/Tools, in the anthropics/oss-scanner template layout. Free text; the scanner reads it before it starts. Paths are relative to the repository root (/src in the scan image). -->
# Threat Model

## What This Project Does and Where Untrusted Input Enters

Tools is a monorepo of engineering and scientific applications: process calculators, signal and data
processing, URDF/MJCF robot-model builders, biomechanics (C3D, swing and putting analysis), a P&ID
generator, and a few web companions. Most of it is PyQt6 desktop apps behind one launcher
(`UnifiedToolsLauncher.py`, `src/shared/python/gui_launcher/`). It also contains a Python library
(`ud-tools`) that UpstreamDrift and Gasification_Model import, FastAPI/Flask backends, React/TypeScript
front ends, a Rust workspace (`rust_core/`) with PyO3 bindings, and Arduino/C++ PLC firmware.

The main attacker is someone who gets a user to **open a crafted file** (model, dataset, project
archive) or, for the server components, someone who can **reach a listening port**. Treat as untrusted:

- **Robot and simulation model files (XML).** URDF, MJCF, SDF/OpenSim-style and Simscape `.mdl` input:
  `src/shared/python/model_generation/converters/urdf_parser.py`, `mjcf_converter.py`,
  `converters/simscape/mdl_parser.py`, `model_generation/library/unified_loader.py`,
  `model_generation/editor/text_editor_validation_mixin.py` (all use `defusedxml`), and
  `src/shared/python/swing_sim/model_interchange/parsers.py`, which deliberately uses **stdlib**
  `xml.etree.ElementTree` (no external entities, but check entity expansion and size limits). Also
  `src/shared/python/codemap/parsers.py`. Look for any XML path that bypasses defusedxml, for
  `mesh filename=`/`package://` paths that escape the model directory, and for mesh loading via trimesh.
- **Data files.** CSV, Excel (`pandas.read_excel`/openpyxl), MATLAB `.mat` (`scipy.io.loadmat`),
  NumPy `.npy/.npz`, HDF5 and JSON, in `src/data_processing/data_processor/python/data_processor/file_utils.py`,
  `.../core/dataset_manager.py`, `src/shared/python/sidekick/data_processing/io.py`,
  `src/shared/python/signal_toolkit/io.py` (calls `np.load` without an explicit `allow_pickle`),
  `src/shared/python/swing_sim/variation/dataset_io.py` (h5py), `src/rate_of_closure/variation/_durable_ensemble_io.py`.
  Any route from a user-chosen file to pickle (`allow_pickle=True`, `pickle`, `joblib`, `torch.load`
  without `weights_only=True`) is a finding.
- **Images, video and C3D motion capture.** `src/c3d_viewer/`, `src/media_processing/video_processor/`,
  `src/video_analyzer/`, `src/putting_launch_monitor/` (OpenCV, MediaPipe, ffmpeg via `imageio-ffmpeg`).
- **Project archives.** P1AM project import unpacks an uploaded ZIP:
  `src/p1am_control_system/backend/project_import.py` (`_safe_extract_zip`: path, member-count, size and
  ratio budgets), then `parsers/indusoft_parser.py` and `parsers/plc_map_parser.py`.
- **Expressions typed or posted by users.** SymPy `parse_expr`/`sympify` (both reach `eval`) in
  `src/web_applications/calculator/calculator.py` (with a deny-list in `webapp.py`),
  `src/shared/python/calc_backend/routers/symbolic_solver.py`, `src/shared/python/sidekick/symbolic_engine.py`,
  `sidekick/latex_renderer.py`, `sidekick/process_calculators/ode_solver.py`,
  `src/shared/python/signal_toolkit/polynomial_generator.py`. AST-whitelist evaluators:
  `src/shared/python/safe_eval.py`, `src/p1am_control_system/backend/data_explorer_expression.py`.
  A sandbox escape in any of these from a network request is the most important class of bug here.
- **Network servers.**
  - P1AM SCADA backend `src/p1am_control_system/backend/main.py` (FastAPI, REST and `/api/stream`
    WebSocket, Modbus to real PLCs; `auth_config.py` API keys with `P1AM_DEV_NO_AUTH` bypass). It controls
    physical process hardware; the container binds 0.0.0.0 and relies on compose publishing to loopback.
  - Calculation backend `src/shared/python/calc_backend/app.py` and its routers.
  - URDF viewer `src/web_applications/urdf_viewer/app.py` (uploads, `get_safe_path`; `uvicorn.run(host="0.0.0.0")`).
  - Video processor API `src/media_processing/video_processor/python/video_processor_src/api.py` (500 MB uploads).
  - Rate of Closure `src/rate_of_closure/web_authority/api.py`, `web_companion/app.py`,
    `application/morris/host.py` (loopback authority for the React app), Flask calculator
    `src/web_applications/calculator/webapp.py`, steam engine `src/steam_engine_calculator/python/steam_engine_calculator/api.py`,
    model-generation REST `src/shared/python/model_generation/api/`.
  - Codemap MCP server `src/shared/python/codemap/mcp_server.py` (stdio, used by coding agents).
- **Subprocesses.** No `shell=True` was found in `src/`. Check argv built from user data in
  `src/shared/python/ai/tools/cli_tools.py` (`ShellTool` allowlist, `shlex` tokenizing),
  `ai/adapters/*_cli_adapter.py` and `github_cli_provider.py`, `gui_launcher/launcher_web.py`,
  `src/tools/launch_utils.py`, `document_processing/pdf_renamer/`, `folder_packer_pro/app.py`.
- **AI/chat input.** LLM output and chat messages that reach tools or files: `src/shared/python/chat/`,
  `src/shared/python/ai/` (knowledge packs, tool adapters). Treat model output as attacker-controlled.
- **Code loading.** `gui_launcher/registry.py` imports every `gui_registration.py` under its search
  paths; `src/shared/python/sidekick/bootstrap.py` `exec`s a bootstrap file. Worth checking that search
  paths cannot include a user-writable or downloaded directory.
- **Front ends.** TypeScript/React in `src/rate_of_closure/web/`, `src/p1am_control_system/frontend/`,
  `src/media_processing/video_processor/apps/web/`, `src/pendulum_simulator/pendulum-web/`: XSS from
  server data or file names, unsafe `dangerouslySetInnerHTML`, open proxying in `authorityProxyConfig.ts`.
- **Native code.** Rust crates in `rust_core/` and `src/pendulum_simulator/pendulum-core/`,
  `src/movement_optimizer/rust_core/` exposed to Python through PyO3 (`unsafe`, unchecked array shapes and
  lengths from NumPy). C++ firmware `src/p1am_control_system/firmware/` parses serial/Modbus frames.

## Components That Matter Most / Least

Most important:
1. P1AM SCADA backend and its project import, auth, WebSocket and Modbus write paths (network-facing,
   drives hardware).
2. Expression evaluation reachable from HTTP (calc_backend, Flask calculator, data explorer).
3. Other HTTP servers that accept uploads (URDF viewer, video processor) and the Rate of Closure authority.
4. File importers a user opens from the desktop apps (model XML, `.mat/.npz/.h5`, Excel, C3D, video).
5. PyO3 boundaries in the Rust crates and the AI tool adapters that spawn CLIs.

Less important but in scope: desktop-only calculators whose input is typed by the local user, the
deliberately unsandboxed-by-design local scripting console `src/shared/python/scripting/scripting_env.py`
(report only a bypass of its documented restrictions, or a way to reach it from a file or network
input), launcher and packaging scripts.

Out of scope: `manuals/` (generated design manual), `docs/`, `assessments/`, `drafts/`, `output/`,
`visual-baseline-candidates/`, `assets/`, `matlab/`, `examples/`, `benchmarks/`, built web bundles
(`**/dist/`), lock files, `.github/` workflow configuration, and third-party packages installed in the
image (report those upstream).

## How to Exercise It

- The image has the package installed editable at `/src` with extras `all,gui,test,dev,p1am`, plus a
  built Rust workspace (`/src/target`). Run Python as `python`; Qt is headless (`QT_QPA_PLATFORM=offscreen`,
  `MPLBACKEND=Agg`). Never start a GUI without offscreen.
- Tests: `python -m pytest tests/<area> -n 4 --timeout 120` (pytest config in `pyproject.toml`). Useful
  suites: `tests/p1am_control_system/` (including `test_backend_security.py`), `tests/calc_backend/`,
  `tests/data_processor_io/`, `tests/humanoid_character_builder/`, `tests/rate_of_closure/`,
  `tests/rust_bindings/`. Rust: `cargo test --workspace` (offline; crates were fetched into `CARGO_HOME` at build time).
- Servers can be driven in-process with FastAPI `TestClient` (httpx is installed), for example
  `from p1am_control_system.backend ...` via `sys.path` entries listed under `[tool.pytest.ini_options]`
  `pythonpath`, or by running `uvicorn` on 127.0.0.1 inside the container.
- Parsers can be called directly: `URDFParser().parse(...)`, `scipy.io.loadmat`, the signal_toolkit
  and data_processor `file_utils` loaders, `project_import._safe_extract_zip`.
- The audit runs offline. Nothing needs real PLCs, cameras, GPUs, API keys or internet access.

## How You Rate Severity

These are mostly local desktop tools, with a few servers that are meant to sit on loopback or a
trusted plant network.

- **Critical:** unauthenticated remote code execution or arbitrary file write in a network server; an
  unauthenticated or auth-bypass path that issues Modbus writes, changes setpoints or clears an e-stop on
  the P1AM backend; auth bypass that works with `P1AM_DEV_NO_AUTH` unset.
- **High:** code execution from opening a crafted file in a desktop app or library loader (pickle, an
  `eval` reachable from file content, a sandbox escape in `safe_eval`/data_explorer/SymPy deny-list);
  authenticated RCE in a server; path traversal or zip-slip that writes outside the destination; XXE or
  SSRF that reads local files; memory-unsafety reachable through PyO3 from Python data.
- **Medium:** reading files outside the intended directory; stored or reflected XSS in a front end;
  missing auth on a read endpoint that exposes plant data; argument injection into a spawned CLI;
  resource exhaustion (zip or XML bombs, unbounded uploads) against a network server.
- **Low:** DoS or crash from a crafted file in a desktop app; information leaks in error messages; issues
  that need the attacker to already control the local user account or the environment variables.

## Anything to Leave Alone

- `scripting_env.py` intentionally runs user-typed Python in-process; that by itself is not a finding.
- The P1AM container binding 0.0.0.0 is intentional (isolation is at `docker-compose.yml` publish,
  `127.0.0.1:8000:8000`); report it only if a shipped config exposes it beyond loopback.
- `pickle` between `src/rate_of_closure/ui/pyqt6/plots_process_worker.py` and its parent is internal IPC
  over a private pipe; report it only if an untrusted party can feed that pipe.
- `nn_script_exporter_renderers.py` emits `torch.load(...)` inside generated user scripts; that text is
  not executed by Tools.
- Code style, typing, test coverage, GUI cosmetics, numerical accuracy and performance are out of scope.
- Do not report vulnerable versions of third-party dependencies unless Tools code makes them reachable.
