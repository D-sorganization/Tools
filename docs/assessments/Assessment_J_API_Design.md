# Assessment J: API Design

## Executive Summary
This assessment evaluates the internal module interfaces, object-oriented design, and the use of formal contracts within the `Tools` repository.
The repository is highly modular, with over 2063 classes defined, separating UI logic (PyQt) from core business logic (calculators/engines). However, the API design heavily relies on implicit contracts (duck typing) rather than explicit interfaces. The lack of Python `Protocol` or `ABC` (Abstract Base Class) definitions in the `src/shared` library means that integrating new tools relies on trial and error or reading implementation code, increasing the friction for developers.

## Scorecard
- **Grade: 7.0/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| J-001 | Major | Contracts | `src/core/plugin_manager.py` | Fragile tool loading | Plugins duck-type their entry points | Define a formal `ToolPlugin(Protocol)` | M |
| J-002 | Major | REST API | `src/shared/python/model_generation/api/rest_api.py` | Missing OpenAPI schemas | Implicit request bodies | Migrate to `FastAPI` / `Pydantic` | L |
| J-003 | Medium | Coupling | UI Layer (`main_window.py` variants) | Hardcoded business logic inside UI callbacks | Lack of Controller/Presenter pattern | Refactor to MVC/MVP patterns | H |
| J-004 | Minor | Extensibility | `src/shared/python/signal_toolkit/` | Difficult to add new filters | Monolithic filter registry | Implement a plugin/registry pattern for filters | M |

## Refactoring Plan
- **Short Term**: Address J-001 by formalizing the plugin contract using `typing.Protocol`. All new tools must conform to this protocol for the `UnifiedToolsLauncher` to register them.
- **Medium Term**: Implement a decorator-based registry pattern in `signal_toolkit` (J-004) so that users can define custom filters without modifying the core library code.
- **Long Term**: Decouple the PyQt6 UI files from the calculation engines. Ensure all UI logic delegates to a dedicated "Controller" or "ViewModel" class (J-003), making the business logic easily testable via CLI or REST APIs (J-002).
