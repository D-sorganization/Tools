# SOLID Principles Adherence Review — Tools Repository

**Assessment Date**: 2026-02-19
**Scope**: Tools repository (`/home/user/Tools/src/`)
**Methodology**: Manual code review with AST-guided file discovery
**Cross-references**: Cross_Repo_Quality_Assessment_2026-02-17, Comprehensive_Assessment_2026-02-16, ARCHITECTURE_QUALITY_ASSESSMENT_2026-02-12

---

## Executive Summary

| Principle | Score | Verdict |
| :--- | :---: | :--- |
| **S** — Single Responsibility | 4/10 | Multiple critical violations in calculators, API classes, and GUI modules |
| **O** — Open/Closed | 5/10 | Several long if/elif dispatch chains; plugin system partially redeems |
| **L** — Liskov Substitution | 6/10 | Builder hierarchy has return-type contract changes; most ABCs are sound |
| **I** — Interface Segregation | 4/10 | `CalculatorStateMixin` (790 lines, 40+ methods) is a textbook fat interface |
| **D** — Dependency Inversion | 6/10 | Protocols are well-designed but inconsistently adopted; hardcoded instantiation in launchers |
| **Overall** | **5/10** | Good infrastructure (Protocols, ABCs, Command pattern) undermined by monolithic modules and fat mixins |

---

## S — Single Responsibility Principle

### Score: 4/10

The Cross-Repo Assessment (2026-02-17) already identified 633 functions exceeding 50 lines (7.2% of total). This SOLID review reveals the class-level picture is equally concerning.

### Critical Violations

#### 1. `ModelGenerationAPI` — 928 lines, 27 methods, 10+ concerns

**File**: `src/shared/python/model_generation/api/rest_api.py:96-1023`

This single class handles:

| Concern | Methods |
| :--- | :--- |
| Route registration | `_register_core_routes`, `_register_inertia_and_library_routes`, `_register_editor_routes`, `_register_routes`, `add_route` |
| Request dispatching | `handle_request`, `get_routes` |
| Security | `_add_security_headers` |
| URDF generation | `generate_humanoid`, `generate_from_params` |
| Format conversion | `convert_simscape_to_urdf`, `convert_mjcf_to_urdf`, `convert_urdf_to_mjcf` |
| Validation | `validate_urdf` |
| Parsing | `parse_urdf` |
| Inertia calculation | `calculate_inertia`, `inertia_from_mesh` |
| Library CRUD | `library_list_models`, `library_get_model`, `library_add_model`, `library_remove_model`, `library_download_model` |
| Composition | `compose_models` |
| Diffing | `diff_urdfs` |
| Health/info | `health_check`, `get_api_info` |

**Recommendation**: Extract into focused handler classes — `GenerationHandler`, `ConversionHandler`, `LibraryHandler`, `CompositionHandler` — orchestrated by a thin router.

#### 2. `pressure_drop_interface.py` — 1,377 lines, 6 mixed concerns

**File**: `src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/pressure_drop_interface.py`

A single module mixes:
- Help/documentation display (`show_help`)
- Data access (gas properties, pipe specs, fittings database)
- Input validation
- Core calculations (`calculate_pressure_drop`, `calculate_pressure_drop_custom_gas`, `calculate_pressure_drop_syngas`)
- Unit conversion
- Output formatting (`print_results`)

**Recommendation**: Separate into `PressureDropCalculator` (core), `PressureDropValidator`, `GasPropertiesProvider`, `PipeGeometryResolver`, `PressureDropFormatter`.

#### 3. `VectorizedFilterEngine` — 930 lines, 18 methods

**File**: `src/data_processing/data_processor/python/data_processor/vectorized_filter_engine.py:88-1017`

Every filter type (moving average, Butterworth, median, Hampel, z-score, Savitzky-Golay, Gaussian, FFT) is embedded as a private method in one class. Each filter implementation could be a separate class implementing a common `FilterStrategy` interface.

#### 4. `CompressionCalculationWorker` — QThread + business logic

**File**: `src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py:498`

Extends `QThread`, mixing threading infrastructure (progress signals, error handling) with compression calculations. The worker should delegate to a pure calculation engine.

### Positive Examples

| Pattern | File | Notes |
| :--- | :--- | :--- |
| Command pattern | `src/data_processing/.../core/undo_redo.py` | `Command` ABC (27 lines) with 7 focused implementations (12-68 lines each) |
| State space models | `src/data_processing/.../core/state_space.py` | `BaseStateSpaceModel` ABC with `LocalLevelModel` (38 lines), `LocalLinearTrendModel` (54 lines), etc. |
| Mesh generator factory | `src/shared/python/humanoid_character_builder/generators/mesh_generator.py:1067` | `MeshGenerator` factory delegates to backend-specific generators |

---

## O — Open/Closed Principle

### Score: 5/10

### Violations

#### 1. Tool launch dispatcher — 5+ elif branches

**File**: `src/tools/launch_utils.py:258-276`

```python
if tool_type == "python":
    launch_python_tool(...)
elif tool_type == "matlab":
    launch_matlab_tool(...)
elif tool_type in ("web", "browser", "html"):
    launch_browser_tool(...)
elif tool_type == "bat":
    launch_batch_tool(...)
elif tool_type == "file":
    # platform-specific branching inside
else:
    raise LaunchError(...)
```

Adding new tool types (Docker, Node.js, executable) requires modifying this function. A `ToolLauncher` protocol with a registry would make it extensible without modification.

#### 2. Tile launcher type dispatch

**File**: `src/python/src/tile_launcher/ui.py:291-304`

Same pattern — `if app.launch_type == LaunchType.PYTHON ... elif LaunchType.BAT ... elif LaunchType.HTML ...` — the UI class must be edited for every new launch type.

#### 3. Signal type generation — 11+ branches

**File**: `src/shared/python/signal_toolkit/widget_processing.py:63-140`

A 26-branch if/elif chain dispatching on `signal_type` strings (`"Sinusoid"`, `"Cosine"`, `"Polynomial"`, `"Exponential"`, `"Square"`, `"Triangle"`, `"Chirp"`, etc.). A second chain (lines 158-184) does the same for fitting types. Adding a new signal type means editing two separate if/elif chains.

**Recommendation**: A `SignalTypeRegistry` mapping type names to generator callables would make this open for extension.

#### 4. Test data type dispatch

**File**: `src/python/src/utils/test_utils.py:76-95, 107-146`

Parallel if/elif chains for `"int"`, `"float"`, `"string"`, `"mixed"` in both `generate_sample_data` and `generate_edge_case_data`.

### Positive Counterexamples

| Pattern | File | Assessment |
| :--- | :--- | :--- |
| Plugin manager | `src/python/src/core/plugin_manager.py` | Tools register via `tools.json` manifest — new tools need no code changes |
| GUI registry | `src/shared/python/gui_launcher/registry.py` | Auto-discovery via `auto_discover_guis()` — fully OCP-compliant |
| Model generation plugin ABC | `src/shared/python/model_generation/plugins/__init__.py` | `ModelGenerationPlugin` ABC with `name`, `version`, `initialize` |

---

## L — Liskov Substitution Principle

### Score: 6/10

### Violations

#### 1. Builder hierarchy return-type contract change

**Base**: `BaseURDFBuilder.add_link(link: Link) -> None` (`src/shared/python/model_generation/builders/base_builder.py:154`)

**Subclass**: `ManualBuilder.add_link(link: Link) -> ManualBuilder` (`src/shared/python/model_generation/builders/manual_builder.py:89`)

The base class returns `None`; the subclass returns `self` to enable fluent method chaining. While return-type covariance is technically valid in Python, the **semantic contract changes**: code written against `BaseURDFBuilder` that ignores the return value will work, but code that type-checks the return or relies on the `None` contract may break when handed a `ManualBuilder`.

#### 2. ParametricBuilder divergent API

**File**: `src/shared/python/model_generation/builders/parametric_builder.py:147`

`ParametricBuilder` inherits from `BaseURDFBuilder` but introduces `add_segment(name, parent, mass_ratio, ...)` with 10+ parameters. It never overrides `add_link`/`add_joint` — clients using the base class API get no validation or parametric behavior. The substitutability promise is broken: passing a `ParametricBuilder` where a `BaseURDFBuilder` is expected produces semantically different behavior.

### Sound Hierarchies

| Hierarchy | File | Assessment |
| :--- | :--- | :--- |
| `MeshGeneratorInterface` → `PrimitiveMeshGenerator`, `MakeHumanMeshGenerator`, `SMPLXMeshGenerator` | `src/shared/python/humanoid_character_builder/generators/mesh_generator.py` | All implement `generate()`, `get_supported_segments()`, `backend_name`, `is_available` consistently |
| `BaseStateSpaceModel` → `LocalLevelModel`, `LocalLinearTrendModel`, `SeasonalModel`, `ARIMAStateSpace` | `src/data_processing/.../core/state_space.py` | Uniform `fit`/`forecast` interface; factory provides correct instantiation |
| `Command` → `DataFrameCommand`, `FilterCommand`, `ColumnOperationCommand`, ... | `src/data_processing/.../core/undo_redo.py` | All implement `execute`/`undo`/`name` identically |
| `BaseCalculationEngine` → individual calculators | `src/shared/python/upstream_drift_tools/calculators/base.py` | Clean single-method `calculate()` interface |

---

## I — Interface Segregation Principle

### Score: 4/10

### Critical Violations

#### 1. `CalculatorStateMixin` — 790 lines, 40+ methods

**File**: `src/shared/python/upstream_drift_tools/ui/mixins/calculator_state_mixin.py`

This is a textbook fat interface. A calculator that only needs state persistence must also inherit:

| Concern | Methods |
| :--- | :--- |
| State management | `save_calculator_state`, `load_calculator_state`, `get_calculator_state`, `set_calculator_state`, `auto_save_state` |
| Copy/paste | `copy_selected_text`, `copy_all_results`, `paste_text`, `copy_widget_text`, `copy_to_clipboard`, `show_context_menu`, `show_widget_context_menu` |
| Splitter management | `register_splitter`, `register_input_widget`, `register_copyable_widget`, `save_splitter_states`, `restore_splitter_states`, `on_splitter_moved` |
| UI widget creation | `create_copy_button`, `create_save_load_buttons`, `setup_copy_paste`, `setup_shortcuts` |
| Auto-registration | `auto_register_widgets` |

The mixin relies on duck-typing with `type: ignore[attr-defined]` and `cast()` to access Qt methods, confirming it assumes a very specific consumer shape rather than defining a clean interface.

**Recommendation**: Split into composable protocols:

- `IStateManager` — save/load/get/set state
- `ICopyPasteable` — clipboard operations
- `ISplitterManager` — splitter state tracking
- `IWidgetFactory` — UI creation helpers

#### 2. `SignalToolkitWidget` — 2,400+ lines across 3 mixins

**Files**:
- `src/shared/python/signal_toolkit/widget.py:171-176`
- `src/shared/python/signal_toolkit/widget_ui.py` (893 lines)
- `src/shared/python/signal_toolkit/widget_processing.py` (620 lines)
- `src/shared/python/signal_toolkit/widget_plotting.py` (109 lines)

```python
class SignalToolkitWidget(UISetupMixin, ProcessingMixin, PlottingMixin, QWidget):
```

Each mixin references attributes created by the other mixins via `type: ignore[attr-defined]`. `ProcessingMixin` calls `self._update_plot()` (defined in `PlottingMixin`); `PlottingMixin` accesses `self.canvas` (defined in `UISetupMixin`). This creates a tightly coupled set of interfaces that cannot be used independently. A client that only needs signal generation gets 2,400 lines of UI, plotting, and processing code.

#### 3. Duplicate `show_error`/`show_info` across widget bases

**Files**:
- `src/shared/python/upstream_drift_tools/ui/widgets/base_calculator_widget.py:35-68`
- `src/shared/python/upstream_drift_tools/ui/mixins/base_calculator_mixin.py:17-48`

Both `BaseCalculatorWindow(QMainWindow, BaseCalculatorMixin)` and `BaseCalculatorWidget(QWidget, BaseCalculatorMixin)` independently define identical `show_error`/`show_info` methods. `BaseCalculatorMixin` separately defines `log_info`/`log_warning`/`log_error`. The relationship between dialog-based error display and logger-based error logging is ambiguous.

### Positive Examples

| Interface | File | Size | Assessment |
| :--- | :--- | :--- | :--- |
| `ProcessCalculator` protocol | `src/shared/python/upstream_drift_tools/protocols.py` | 1 method | Focused — `calculate(inputs) -> results` |
| `DataTransformer` protocol | same file | 1 method | Focused — `transform(data) -> data` |
| `StateSerializable` protocol | same file | 2 methods | Focused — `save_state`/`restore_state` |
| `UnitConverter` protocol | same file | 1 method | Focused — `convert(value, from, to)` |
| `ThemeProvider` protocol | `src/shared/python/theme/protocols.py` | 3 methods | Appropriately scoped |
| `ThemeSwitcher` protocol | same file | 2 methods | Extends `ThemeProvider` behavior only |
| `StylesheetGenerator` protocol | same file | 1 method | Single concern |

These protocol definitions are exemplary. The gap is that the mixin classes that implement them do not follow the same discipline.

---

## D — Dependency Inversion Principle

### Score: 6/10

### Violations

#### 1. Hardcoded `FileLayoutStore` in tile launcher

**File**: `src/python/src/tile_launcher/manager.py:76-81`

```python
@staticmethod
def _default_store() -> LayoutStore:
    from tile_launcher.models import FileLayoutStore
    return FileLayoutStore(path=DEFAULT_LAYOUT_PATH)
```

While `LayoutStore` is defined as a Protocol (good), the manager creates a concrete `FileLayoutStore` directly. Tests or alternative configurations (in-memory, database-backed) require patching rather than injection.

#### 2. `UnifiedLauncherWindow` directly imports concrete config and help modules

**File**: `src/tools/gui/windows/unified_launcher_window.py:38-44, 122-124`

```python
from tools.config_loader import CATEGORY_ORDER, load_tools_config
tools_config = load_tools_config(self.repo_root)

try:
    from python.src.help import get_help_manager
    HELP_AVAILABLE = True
except ImportError:
    HELP_AVAILABLE = False
```

The high-level launcher window depends directly on low-level file-loading modules and conditionally imports the help subsystem by path. Both should be injected via a `ConfigProvider` and `HelpSystem` abstraction.

#### 3. Shared library importing tool-specific code (layer boundary violation)

**File**: flagged by `test_shared_does_not_import_tool_packages` (failing — 3 violations)

The Cross-Repo Assessment (2026-02-17, Section 4.3) already identifies this. Shared libraries should never depend on tool-specific packages. This is a direct DIP violation: the lower-level reusable layer depends on the higher-level application layer.

#### 4. `PluginManager` not consistently injected

**File**: `src/python/src/core/plugin_manager.py`

The `PluginManager` is well-designed internally (loads tools from manifest), but high-level consumers create it directly rather than receiving a `ToolRepository` abstraction. The tile launcher creates `AppManager` with hardcoded catalog paths; config loaders import `tools.json` directly.

### Positive Patterns

| Pattern | File | Assessment |
| :--- | :--- | :--- |
| `LayoutStore` Protocol | `src/python/src/tile_launcher/models.py:40-47` | Clean `load`/`save` interface |
| Protocol-driven shared library | `src/shared/python/upstream_drift_tools/protocols.py` | 4 well-scoped protocols enabling loose coupling |
| Theme protocols | `src/shared/python/theme/protocols.py` | `ThemeProvider`, `ThemeSwitcher`, `StylesheetGenerator` — GUI code can depend on abstractions |
| `BaseCalculationEngine` ABC | `src/shared/python/upstream_drift_tools/calculators/base.py` | Calculators depend on abstraction, not each other |
| `GUIRegistry` | `src/shared/python/gui_launcher/registry.py` | Auto-discovery means no hardcoded tool references |

---

## Cross-Principle Issues

### 1. Mixin anti-pattern (violates S, I, and D simultaneously)

The `CalculatorStateMixin` is a 790-line God mixin that:
- Violates **SRP** by handling state, clipboard, splitters, and UI creation
- Violates **ISP** by forcing all 40+ methods on every consumer
- Violates **DIP** by assuming concrete Qt widget internals via duck typing

### 2. Signal toolkit composition (violates S, I, and L)

`SignalToolkitWidget` composes three large mixins with implicit cross-references:
- Violates **SRP** — each mixin is 100-900 lines
- Violates **ISP** — clients get all 2,400 lines regardless of need
- Violates **LSP** in spirit — the mixins cannot be substituted or used independently despite appearing composable

### 3. Builder hierarchy (violates L and I)

`BaseURDFBuilder` has non-abstract methods (`add_link`, `add_joint`) whose contract is changed by `ManualBuilder` (return type) and ignored by `ParametricBuilder` (uses `add_segment` instead), violating both **LSP** and **ISP** (the base interface is too broad for `ParametricBuilder`).

---

## Remediation Priorities

### P0 — Must Fix

| # | Principle | Target | Action | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **I** | `CalculatorStateMixin` (790 lines) | Split into `IStateManager`, `ICopyPasteable`, `ISplitterManager`, `IWidgetFactory` | Decouples 40+ methods into composable concerns |
| 2 | **S** | `ModelGenerationAPI` (928 lines) | Extract `GenerationHandler`, `ConversionHandler`, `LibraryHandler`, `CompositionHandler` | Each handler becomes independently testable |
| 3 | **D** | Shared → tool imports (3 violations) | Remove reverse dependencies; test is already catching these | Restores layer boundary integrity |

### P1 — High Priority

| # | Principle | Target | Action | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 4 | **O** | `launch_utils.py` tool dispatch | Replace if/elif with `ToolLauncher` protocol + registry | New tool types need no code modification |
| 5 | **O** | `widget_processing.py` signal dispatch | Create `SignalTypeRegistry` mapping names → generators | New signal types need only registration |
| 6 | **L** | `ManualBuilder.add_link` return type | Make `BaseURDFBuilder.add_link` return `Self` (PEP 673), or document the covariance | Consistent contract across hierarchy |
| 7 | **S** | `VectorizedFilterEngine` (930 lines) | Extract each filter into a `FilterStrategy` class | Filters become independently testable and composable |

### P2 — Medium Priority

| # | Principle | Target | Action | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 8 | **S** | `pressure_drop_interface.py` (1,377 lines) | Separate calculator, validator, formatter, data provider | Module becomes maintainable |
| 9 | **I** | `SignalToolkitWidget` 3-mixin composition | Replace mixins with composition; extract UI factory | Eliminates `type: ignore[attr-defined]` proliferation |
| 10 | **D** | `UnifiedLauncherWindow` hardcoded imports | Inject `ConfigProvider` and `HelpSystem` abstractions | Enables testing and alternative configurations |
| 11 | **L** | `ParametricBuilder` API divergence | Either override `add_link`/`add_joint` or extract to separate hierarchy | Clear substitutability |
| 12 | **D** | `_default_store()` hardcoded `FileLayoutStore` | Accept store as constructor parameter with default | Standard dependency injection |

---

## Scorecard vs. Previous Assessments

| Dimension | Architecture Assessment (2026-02-12) | Cross-Repo (2026-02-17) | This Review (SOLID) |
| :--- | :---: | :---: | :---: |
| SRP / Function Size | 3 (Changeability) | 5 | **4** |
| OCP / Extensibility | 6 (Reusability) | 8 | **5** |
| LSP / Hierarchy | — | — | **6** |
| ISP / Interface Design | 4 (Orthogonality) | 6 | **4** |
| DIP / Decoupling | 4 (LoD) | 6 | **6** |

The scores here are generally consistent with prior assessments. The ISP score is lower because this review specifically examined mixin interfaces, which the previous assessments did not focus on.

---

## Methodology

- **SRP**: Identified classes/modules >500 lines; counted distinct concerns (UI, validation, calculation, I/O, formatting)
- **OCP**: Searched for if/elif chains >4 branches dispatching on type strings or enums; checked for registry/strategy alternatives
- **LSP**: Examined all ABC/Protocol hierarchies for contract changes in overriding methods (return types, exceptions, signatures)
- **ISP**: Measured mixin method counts; identified `type: ignore[attr-defined]` as a proxy for implicit interface assumptions
- **DIP**: Traced import directions between layers; checked for concrete instantiation vs. injection in high-level modules

---

_Assessment conducted 2026-02-19._
