# Assessment D Results: User Experience & Developer Journey

## Executive Summary

- **Confusing Entry Points**: The presence of `Launcher.py`, `UnifiedToolsLauncher.py`, `run_tile_launcher.py`, and `tools_launcher.py` (legacy) creates immediate friction for new users.
- **Installation Friction**: Users must navigate multiple `requirements.txt` files or rely on a `setup_dev.py` that isn't the standard `pip install .`.
- **Feedback Loops**: CLI tools use `print()` extensively, which is good for feedback but bad for programmatic integration. GUI tools lack consistent error reporting.
- **"Works on My Machine"**: Hardcoded paths and reliance on specific directory structures make the repo fragile across environments.

## Time-to-Value Metrics

| Stage             | Time (P50) | Blockers Found | Notes |
| ----------------- | ---------- | -------------- | ----- |
| Installation      | 15 min     | 2              | Multiple requirements files. |
| First run         | 10 min     | 1              | "Which launcher?" |
| First result      | 5 min      | 0              | Once running, tools seem to work. |
| Understand output | 5 min      | 0              | GUIs are intuitive enough. |

## Friction Point Heatmap

| Stage     | Friction Points             | Severity | Fix Effort |
| --------- | --------------------------- | -------- | ---------- |
| Install   | No root `setup.py`          | MAJOR    | M          |
| First run | Ambiguous Launchers         | CRITICAL | S          |
| Usage     | Hardcoded Paths             | MAJOR    | M          |
| Debugging | "Print" debugging           | MINOR    | L          |

## User Journey Map

```
[Install] → 😐 (Confused by multiple requirements)
[First run] → 😡 (Tried Launcher.py, failed? Tried UnifiedToolsLauncher, worked?)
[Learn concepts] → 😐 (Docs are sparse)
[Custom workflow] → 😡 (No API docs)
```

## Scorecard

| Category              | Score (0-10) | Evidence | Remediation |
| --------------------- | ------------ | -------- | ----------- |
| Installation Ease     | 5/10         | No single install command. | Create `pyproject.toml`. |
| First-Run Success     | 4/10         | Ambiguous entry points. | Delete legacy launchers. |
| Documentation Quality | 4/10         | Sparse. | See Assessment C. |
| Error Clarity         | 6/10         | `print()` is okay for humans, bad for machines. | Use `logging`. |
| API Ergonomics        | 3/10         | Untyped, un-docstringed. | Add types/docs. |
| **Overall UX Score**  | **4.4**      |          |             |

## Remediation Roadmap

**48 hours:**
-   **Delete `Launcher.py` and `run_tile_launcher.py`**. Leave only `UnifiedToolsLauncher.py`.
-   Update `README.md` to say "Run `python UnifiedToolsLauncher.py`".

**2 weeks:**
-   Create a root `pyproject.toml` to install all dependencies via `pip install .`.

**6 weeks:**
-   Add a "First Run Wizard" to the launcher to check environment health.
