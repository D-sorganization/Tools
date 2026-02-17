# Launcher Hierarchy

This repository utilizes a specific hierarchy for launching tools to ensure compatibility and ease of use.

## Primary Launcher

**`UnifiedToolsLauncher.py`**

- **Type:** PyQt6 (Modern GUI)
- **Role:** The **canonical entry point** for the repository. It provides a modern, tabbed interface to access all tools, including Python, MATLAB, and Web Applications.
- **Requirement:** Python 3.11+
- **Usage:**
  ```bash
  python UnifiedToolsLauncher.py
  ```

## Legacy Launcher (Fallback)

**`launch_tools_main.py`**

- **Type:** Tkinter (Legacy GUI)
- **Role:** A robust fallback launcher used if PyQt6 is unavailable or if the unified launcher fails. It focuses on core data processing tools.
- **Usage:**
  ```bash
  python launch_tools_main.py
  ```

## Specialized Launchers

Individual tools may have their own specific launchers (e.g., `src/web_applications/urdf_viewer/main.py`), but users are encouraged to use the `UnifiedToolsLauncher.py` for a unified experience.

## Launcher Selection Logic

1. Users should attempt to run `UnifiedToolsLauncher.py` first.
2. If dependencies (PyQt6) are missing or the environment is restricted, `launch_tools_main.py` serves as a reliable alternative using the standard Tkinter library.
