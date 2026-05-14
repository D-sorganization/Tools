# Sidekick Unified Tools Sidebar

This package provides the shared Sidekick sidebar toolkit for host
applications that need project-scoped utilities beside their primary workflow.

## Import contract

Pure-Python state and workspace registry APIs are safe in headless contexts:

```python
from upstream_drift_tools.ui.tools_sidebar import SidebarState, WorkspaceRegistry
```

Qt widgets are lazy imports. `SidekickSidebar` is an alias for the stable
`UnifiedToolsSidebar` class:

```python
from upstream_drift_tools.ui.tools_sidebar import SidekickSidebar
```

## Host integration

```python
sidebar = UnifiedToolsSidebar(project_root=project_root)
dock = sidebar.install_as_dock(main_window, area="right", state_path=state_path)
sidebar.file_open_requested.connect(open_file)
sidebar.context_updated.connect(update_terminal_context)
```

Call `sidebar.save_state(state_path)` during host shutdown to persist dock area,
floating state, minimized state, dock size, active tab, tab order, hidden tabs,
and popped-out tab ids.

## Flexible workflows

Sidekick supports modern tab workflow controls:

- dock on either side with `set_dock_area("left" | "right")`
- collapse/expand without losing state via `set_minimized(...)`
- reorder tabs with `move_tab(...)` or drag the movable tab bar
- hide/show configured tabs with `set_tab_visible(...)`
- pop a tab into its own window with `pop_out_tab(...)`
- redock popped-out tabs with `redock_tab(...)`
- duplicate tabs that opt in through `SidebarTabDefinition.duplicate_enabled`

Hosts can pass `tab_definitions=[...]` to `UnifiedToolsSidebar`,
`create_tools_sidebar`, or `install_tools_sidebar` to choose which utilities are
enabled for a specific application while keeping the default import contract.

The project file explorer is read-only in this first slice. It emits
`file_open_requested` for files under the scoped project root and intentionally
does not expose destructive file actions yet.
