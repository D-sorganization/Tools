# Assessment: Logging (Category L)

## Grade: 6/10

## Analysis
Logging is an area with room for improvement.
- **Mixed Usage**: There is a mix of `logging` (good) and `print` statements (bad) in the codebase.
- **Configuration**: While some launchers configure logging, it's not universally standardized across all modules.
- **Structure**: Logs often lack structured data (JSON formatting), making them harder to parse programmatically.

## Recommendations
1. **Ban Print**: Enforce a linting rule (like `T201` in flake8/ruff) to forbid `print` statements in the `src/` directory (excluding CLI entry points).
2. **Structured Logging**: Move to structured logging (e.g., using `structlog` or `python-json-logger`) to make logs machine-readable.
3. **Context**: Ensure logs include context (trace IDs, module names) to aid debugging.
