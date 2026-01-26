# Assessment: Data Handling

## Grade: 4/10

## Analysis
Data handling practices need improvement:
- **Hardcoded Paths**: Tests and scripts often rely on hardcoded paths (e.g., `../../src/...`), leading to brittleness and `NameError` / `ModuleNotFoundError` in tests.
- **Format Dependency**: Heavy reliance on CSV files without a unified schema or data abstraction layer.
- **State Management**: The `unit_converter` uses `localStorage` correctly, but Python state management is often global or file-based.

## Recommendations
1. **Use `pathlib`**: Replace string-based path manipulation with `pathlib.Path`.
2. **Configurable Paths**: Move file paths to a configuration file or environment variables.
