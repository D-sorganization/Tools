# Assessment J: Tools Repository API Design & Extensibility Review

## 1. Executive Summary

- The repository lacks a standardized plugin system for UI injection, relying heavily on hardcoded switch statements in `UnifiedToolsLauncher.py`.
- Shared API usage (`src/shared`) is strong, but many utility methods are highly coupled.
- Abstract methods exist (e.g., `base_builder.py`, `repository.py`), offering a framework for extensibility, but their actual implementation across new tools is inconsistent.
- **Top Risk**: Scaling to 50+ tools will break the unified launcher logic as `if-elif` statements grow uncontrollably without dynamic module discovery.

## 2. Scorecard (0-10)

| Category                     | Description                                   | Score |
| ---------------------------- | --------------------------------------------- | ----- |
| Public API Consistency       | Standardization of data input/output          | 8     |
| Plugin Architecture          | Dynamic module discovery implementation       | 4     |
| Decorators & Meta-classes    | Abstractions simplifying tool creation        | 6     |
| Semantic Versioning (SemVer) | Predictability of internal shared libraries   | 8     |
| Interface Completeness       | `NotImplementedError` gaps                    | 5     |

*Evidence for Plugins (4)*: `UnifiedToolsLauncher` manually maps string tool names to import calls.
*Evidence for Interfaces (5)*: `signal_toolkit/io.py` and `format_utils.py` previously contained `NotImplementedError` stubs which blocked downstream implementations.

## 3. API Gap Table

| ID    | Severity | Domain/File | Description | Fix Recommendation | Effort |
| ----- | -------- | ----------- | ----------- | ------------------ | ------ |
| J-001 | Major    | Launcher Core | Switch statement routing | Build `plugin_manager.py` with dynamic `importlib` scanning | M |
| J-002 | Minor    | `shared/` | Shared libraries lack stable API boundaries | Deprecate older utility functions | S |
| J-003 | Major    | `media_processing` | Missing backend API | Implement RESTful DB connection | M |

## 4. Remediation Plan

**Immediate (48 Hours):**
- Fix any remaining `NotImplementedError` in `signal_toolkit` or document them formally as unsupported.

**Short-Term (2 Weeks):**
- Establish a REST API format standard for any tools relying on backend data (e.g., Video Processor, Document OCR).

**Long-Term (6 Weeks):**
- Migrate the UI launchers to use a declarative JSON registry mapping tool modules dynamically, bypassing brittle `elif` blocks.
