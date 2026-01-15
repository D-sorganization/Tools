# Assessment D Results: User Experience & Developer Journey

## Executive Summary

-   **Unified Launcher**: The `UnifiedToolsLauncher` significantly improves the UX by providing a central point of access.
-   **GUI Focus**: The heavy use of GUIs (PyQt, Tkinter, Web) makes the tools accessible to non-technical users.
-   **Developer Experience**: The "Control Tower" architecture and strictly defined agent roles create a structured dev environment.
-   **Onboarding**: The polyglot nature (Python, MATLAB, Node) creates friction for new developers setting up the repo.

## Top 10 UX Risks

1.  **Installation Friction (Severity: High)**: Need to install Python, Node, and MATLAB is a high bar.
2.  **Launcher Dependencies (Severity: Medium)**: Launcher requires PyQt6.
3.  **Missing Tools Feedback (Severity: Medium)**: Launcher buttons show "Missing" which is good, but "Why" might be unclear to users.
4.  **Inconsistent UI (Severity: Low)**: Tkinter vs PyQt vs Web creates disjointed experience.
5.  **Console Output (Severity: Low)**: Launcher log area is small.
6.  **Shortcut Creation (Severity: Low)**: PowerShell scripts for shortcuts work only on Windows.
7.  **Web App Launching (Severity: Medium)**: Launching web apps involves opening a browser, which disconnects from the launcher flow.
8.  **Error Messages (Severity: Low)**: Need to ensure friendly error messages across all tools.
9.  **Accessibility (Severity: Medium)**: Web apps have ARIA roles, but desktop apps (PyQt) might lack accessibility features.
10. **Theme (Severity: Low)**: Dark mode in launcher is nice, but might not match OS theme.

## Scorecard

| Category             | Score | Evidence & Remediation                                    |
| -------------------- | ----- | --------------------------------------------------------- |
| Time-to-value        | 7/10  | High for pre-configured machines, low for fresh clones.   |
| Onboarding           | 7/10  | Docs help, but environment complexity is high.            |
| Friction Points      | 8/10  | Launcher removes friction of finding scripts.             |
| UI/UX Consistency    | 6/10  | Mix of technologies.                                      |
| Developer Journey    | 9/10  | Strong guardrails and agent support.                      |

## Findings Table

| ID    | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
| ----- | -------- | -------- | -------- | ------- | ---------- | --- | ------ |
| D-001 | Medium   | UX       | Repo Root | Complex setup | Polyglot stack | Containerize (Docker) | L |

## Refactoring Plan

**48 Hours**:
-   None.

**2 Weeks**:
-   Create a `devcontainer` definition to standardize the development environment.

**6 Weeks**:
-   Explore Electron or similar for a more unified cross-platform launcher experience? (Maybe too much effort). Stick to improving PyQt launcher.
