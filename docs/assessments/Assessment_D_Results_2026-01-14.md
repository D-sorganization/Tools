# Assessment D: User Experience & Developer Journey Results

**Date:** 2026-01-14
**Assessor:** Jules

## 1. Installation & Environment Setup
**Score: 4/10**

*   **Complexity**: High. Requires Python, Node.js (pnpm), and potentially MATLAB.
*   **Instructions**: Scattered. `requirements.txt` exists but is fragmented across subprojects.
*   **Time-to-Value**: >30 minutes likely due to environment hell (version mismatches).

## 2. First Run Experience
**Score: 5/10**

*   **Unified Launcher**: `UnifiedToolsLauncher.py` is the intended entry point, which is good.
*   **Web Apps**: `unit_converter` requires a separate build/serve process not clearly integrated into the Python launcher.
*   **Failures**: High risk of `ImportError` or `ModuleNotFound` due to `sys.path` hacks in legacy code.

## 3. API Ergonomics
**Score: 5/10**

*   **Inconsistency**: `Data_Processor` uses complex config objects. `Calculator` is a web app. `Solar System` is a GUI.
*   **Usability**: The "Unified" experience is a veneer over disparate tools.

## Remediation Roadmap
*   **Immediate**: Create a `setup_dev.sh` (or `py`) script that installs ALL dependencies (Python & Node) in one go.
*   **Short-term**: Update `UnifiedToolsLauncher.py` to gracefully handle missing dependencies (e.g., disable buttons if Node is missing).
*   **Long-term**: Dockerize the web applications for instant zero-setup usage.
