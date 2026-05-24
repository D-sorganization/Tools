# Assessment B Results: Documentation

## Executive Summary

- **Docstrings**: 7946 functions are missing docstrings.
- **Overall**: Documentation coverage is spotty in utility modules.

## Scorecard

| Category             | Description         | Weight | Score | Evidence                 |
| -------------------- | ------------------- | ------ | ----- | ------------------------ |
| Inline Documentation | Docstrings present? | 2x     | 5     | 7946 missing docstrings. |

## Findings Table

| ID    | Severity | Category | Location                                    | Symptom                                      | Root Cause     | Fix             | Effort |
| ----- | -------- | -------- | ------------------------------------------- | -------------------------------------------- | -------------- | --------------- | ------ |
| B-000 | Minor    | Docs     | `src/pid_generator/ui/pyqt6/main_window.py` | Missing docstring for `__init__`             | Technical Debt | Write docstring | S      |
| B-001 | Minor    | Docs     | `src/pid_generator/ui/pyqt6/main_window.py` | Missing docstring for `_build_ui`            | Technical Debt | Write docstring | S      |
| B-002 | Minor    | Docs     | `src/pid_generator/ui/pyqt6/main_window.py` | Missing docstring for `_browse_spec`         | Technical Debt | Write docstring | S      |
| B-003 | Minor    | Docs     | `src/pid_generator/ui/pyqt6/main_window.py` | Missing docstring for `_browse_out`          | Technical Debt | Write docstring | S      |
| B-004 | Minor    | Docs     | `src/data_explorer/gui.py`                  | Missing docstring for `__init__`             | Technical Debt | Write docstring | S      |
| B-005 | Minor    | Docs     | `src/data_explorer/gui.py`                  | Missing docstring for `_on_choose_directory` | Technical Debt | Write docstring | S      |
| B-006 | Minor    | Docs     | `src/data_explorer/gui.py`                  | Missing docstring for `_populate_table`      | Technical Debt | Write docstring | S      |
| B-007 | Minor    | Docs     | `src/data_explorer/gui.py`                  | Missing docstring for `__init__`             | Technical Debt | Write docstring | S      |
| B-008 | Minor    | Docs     | `src/data_explorer/_embed_adapter.py`       | Missing docstring for `__init__`             | Technical Debt | Write docstring | S      |
| B-009 | Minor    | Docs     | `src/data_explorer/_embed_adapter.py`       | Missing docstring for `embed_capabilities`   | Technical Debt | Write docstring | S      |
