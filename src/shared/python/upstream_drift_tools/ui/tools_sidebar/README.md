# Unified Tools Sidebar

This package provides the first shared slice of the Tools sidebar epic.

## Import contract

Pure-Python state and workspace registry APIs are safe in headless contexts:

```python
from upstream_drift_tools.ui.tools_sidebar import SidebarState, WorkspaceRegistry
```

Qt widgets are lazy imports:

```python
from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar
```

## Host integration

```python
sidebar = UnifiedToolsSidebar(project_root=project_root)
dock = sidebar.install_as_dock(main_window, area="right", state_path=state_path)
sidebar.file_open_requested.connect(open_file)
sidebar.context_updated.connect(update_terminal_context)
```

Call `sidebar.save_state(state_path)` during host shutdown to persist dock area,
floating state, dock size, and active tab.

The project file explorer is read-only in this first slice. It emits
`file_open_requested` for files under the scoped project root and intentionally
does not expose destructive file actions yet.
